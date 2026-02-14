"""Team-level feature engineering utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Sequence

import pandas as pd

from .. import db
from ..config import DEFAULT_TEAM_FEATURE_WINDOWS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamFeatureConfig:
    """Configuration for team feature computation."""

    windows: Sequence[int] = DEFAULT_TEAM_FEATURE_WINDOWS
    attack_metrics: Sequence[str] = (
        "tries",
        "score",
        "run_metres",
        "line_breaks",
        "tackle_breaks",
        "offloads",
    )
    defence_metrics: Sequence[str] = (
        "tries_conceded",
        "score_conceded",
        "tackles",
        "missed_tackles",
    )
    control_metrics: Sequence[str] = (
        "completion_rate_percentage",
        "possession_percentage",
        "errors",
        "penalties_conceded",
    )
    include_recent_form: bool = True
    fillna_value: float | None = 0.0


def fetch_team_match_stats(
    connection: Connection, year: int | str
) -> pd.DataFrame:
    """Fetch per-match team statistics with round metadata and opponent stats.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to query.

    Returns
    -------
    pandas.DataFrame
        Team match stats joined to match rounds and opponent statistics.
    """
    normalized_year = db.normalize_year(year)
    team_stats = db.get_table("team_stats", normalized_year)
    matches = db.get_table("matches", normalized_year)

    query = f"""
        WITH team_match AS (
            SELECT
                ts.match_id,
                ts.squad_id,
                ts.score,
                ts.completion_rate_percentage,
                ts.line_breaks,
                ts.possession_percentage,
                ts.run_metres,
                ts.tackles,
                ts.errors,
                ts.missed_tackles,
                ts.post_contact_metres,
                ts.tries,
                ts.tackle_breaks,
                ts.offloads,
                ts.penalties_conceded,
                m.round_number,
                m.home_squad_id,
                m.away_squad_id
            FROM {team_stats} ts
            JOIN {matches} m ON ts.match_id = m.match_id
        )
        SELECT
            tm.match_id,
            tm.squad_id,
            tm.round_number,
            tm.home_squad_id,
            tm.away_squad_id,
            tm.score,
            tm.tries,
            tm.run_metres,
            tm.line_breaks,
            tm.tackle_breaks,
            tm.offloads,
            tm.tackles,
            tm.missed_tackles,
            tm.completion_rate_percentage,
            tm.possession_percentage,
            tm.errors,
            tm.penalties_conceded,
            tm.post_contact_metres,
            opp.score AS score_conceded,
            opp.tries AS tries_conceded
        FROM team_match tm
        LEFT JOIN team_match opp ON tm.match_id = opp.match_id
            AND tm.squad_id != opp.squad_id
        ORDER BY tm.squad_id, tm.round_number, tm.match_id
    """
    return db.fetch_df(connection, query)


def _rolling_feature(series: pd.Series, window: int, *, min_periods: int = 1) -> pd.Series:
    """Compute a shifted rolling mean for a series."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def compute_team_features(
    connection: Connection,
    year: int | str,
    *,
    config: TeamFeatureConfig | None = None,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute rolling team features using only historical data.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to compute features for.
    config : TeamFeatureConfig | None
        Feature configuration settings.
    as_of_round : int | None
        If provided, only return features for this round while using prior rounds
        for history. Useful for weekly incremental updates.

    Returns
    -------
    pandas.DataFrame
        Feature table keyed by match_id and squad_id.
    """
    feature_config = config or TeamFeatureConfig()
    stats = fetch_team_match_stats(connection, year)

    if stats.empty:
        raise ValueError("No team stats found for the requested season.")

    if as_of_round is not None:
        stats = stats[stats["round_number"] < as_of_round].copy()

    stats.sort_values(["squad_id", "round_number", "match_id"], inplace=True)

    grouped = stats.groupby("squad_id", group_keys=False)
    features = stats[["match_id", "squad_id", "round_number"]].copy()

    # Attack features
    for metric in feature_config.attack_metrics:
        if metric not in stats.columns:
            raise ValueError(f"Attack metric '{metric}' not found in team stats.")
        for window in feature_config.windows:
            feature_name = f"rolling_attack_{metric}_{window}"
            features[feature_name] = grouped[metric].transform(
                lambda s: _rolling_feature(s, window)
            )

    # Defence features
    for metric in feature_config.defence_metrics:
        if metric not in stats.columns:
            raise ValueError(f"Defence metric '{metric}' not found in team stats.")
        for window in feature_config.windows:
            feature_name = f"rolling_defence_{metric}_{window}"
            features[feature_name] = grouped[metric].transform(
                lambda s: _rolling_feature(s, window)
            )

    # Control/discipline features
    for metric in feature_config.control_metrics:
        if metric not in stats.columns:
            raise ValueError(f"Control metric '{metric}' not found in team stats.")
        for window in feature_config.windows:
            feature_name = f"rolling_{metric}_{window}"
            features[feature_name] = grouped[metric].transform(
                lambda s: _rolling_feature(s, window)
            )

    # Win rate features
    stats["is_win"] = (stats["score"] > stats["score_conceded"]).astype(float)
    for window in feature_config.windows:
        win_rate_name = f"rolling_win_rate_{window}"
        features[win_rate_name] = grouped["is_win"].transform(
            lambda s: _rolling_feature(s, window)
        )

    # Recent form indicators
    if feature_config.include_recent_form:
        features["recent_form_tries"] = grouped["tries"].transform(
            lambda s: _rolling_feature(s, 3)
        )
        features["recent_form_score"] = grouped["score"].transform(
            lambda s: _rolling_feature(s, 3)
        )
        features["recent_form_tries_conceded"] = grouped["tries_conceded"].transform(
            lambda s: _rolling_feature(s, 3)
        )

    # Match context
    features["matches_played"] = grouped["match_id"].cumcount()
    features["is_home"] = (stats["squad_id"] == stats["home_squad_id"]).astype(int)

    if feature_config.fillna_value is not None:
        features = features.fillna(feature_config.fillna_value)

    if as_of_round is not None:
        features = features[features["round_number"] == as_of_round]

    if features.empty:
        raise ValueError("No feature rows generated for the requested round.")

    LOGGER.info(
        "Computed team features for %s rows (season=%s, round=%s)",
        len(features),
        year,
        as_of_round,
    )
    return features.reset_index(drop=True)
