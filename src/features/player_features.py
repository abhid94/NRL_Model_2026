"""Player-level feature engineering utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Sequence

import pandas as pd

from .. import db
from ..config import DEFAULT_PLAYER_FEATURE_WINDOWS, position_from_jersey

LOGGER = logging.getLogger(__name__)


def _position_group_from_code(position_code: str) -> str:
    """Map position code to position group for modeling.

    Parameters
    ----------
    position_code : str
        Position code (FB, WG, CE, FE, HB, PR, HK, SR, LK, INT, RES).

    Returns
    -------
    str
        Position group category.
    """
    if position_code in ("FB", "WG", "CE"):
        return "Back"
    elif position_code in ("FE", "HB"):
        return "Halfback"
    elif position_code == "HK":
        return "Hooker"
    elif position_code in ("PR", "SR", "LK"):
        return "Forward"
    elif position_code == "INT":
        return "Interchange"
    else:
        return "Reserve"


@dataclass(frozen=True)
class PlayerFeatureConfig:
    """Configuration for player feature computation."""

    windows: Sequence[int] = DEFAULT_PLAYER_FEATURE_WINDOWS
    metrics: Sequence[str] = (
        "tries",
        "try_assists",
        "line_breaks",
        "run_metres",
    )
    include_recent_form: bool = True
    fillna_value: float | None = 0.0


def fetch_player_match_stats(
    connection: Connection, year: int | str
) -> pd.DataFrame:
    """Fetch per-match player statistics with round metadata.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to query.

    Returns
    -------
    pandas.DataFrame
        Player match stats joined to match rounds.
    """
    normalized_year = db.normalize_year(year)
    player_stats = db.get_table("player_stats", normalized_year)
    matches = db.get_table("matches", normalized_year)

    query = f"""
        SELECT
            ps.match_id,
            ps.player_id,
            ps.squad_id,
            ps.position,
            ps.jumper_number,
            ps.tries,
            ps.try_assists,
            ps.line_breaks,
            ps.run_metres,
            m.round_number,
            m.home_squad_id
        FROM {player_stats} ps
        JOIN {matches} m ON ps.match_id = m.match_id
        ORDER BY ps.player_id, m.round_number, ps.match_id
    """
    return db.fetch_df(connection, query)


def _rolling_feature(series: pd.Series, window: int, *, min_periods: int = 1) -> pd.Series:
    """Compute a shifted rolling mean for a series."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def compute_player_features(
    connection: Connection,
    year: int | str,
    *,
    config: PlayerFeatureConfig | None = None,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute rolling player features using only historical data.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to compute features for.
    config : PlayerFeatureConfig | None
        Feature configuration settings.
    as_of_round : int | None
        If provided, only return features for this round while using prior rounds
        for history. Useful for weekly incremental updates.

    Returns
    -------
    pandas.DataFrame
        Feature table keyed by match_id and player_id.
    """
    feature_config = config or PlayerFeatureConfig()
    stats = fetch_player_match_stats(connection, year)

    if stats.empty:
        raise ValueError("No player stats found for the requested season.")

    if as_of_round is not None:
        stats = stats[stats["round_number"] < as_of_round].copy()

    stats.sort_values(["player_id", "round_number", "match_id"], inplace=True)

    grouped = stats.groupby("player_id", group_keys=False)
    features = stats[["match_id", "player_id", "round_number", "squad_id"]].copy()

    for metric in feature_config.metrics:
        if metric not in stats.columns:
            raise ValueError(f"Metric '{metric}' not found in player stats.")
        for window in feature_config.windows:
            feature_name = f"rolling_{metric}_{window}"
            features[feature_name] = grouped[metric].transform(
                lambda s: _rolling_feature(s, window)
            )

    for window in feature_config.windows:
        try_rate_name = f"rolling_try_rate_{window}"
        features[try_rate_name] = grouped["tries"].transform(
            lambda s: _rolling_feature((s > 0).astype(float), window)
        )

    if feature_config.include_recent_form:
        features["recent_form_tries"] = grouped["tries"].transform(
            lambda s: _rolling_feature(s, 3)
        )

    features["matches_played"] = grouped["match_id"].cumcount()
    features["is_home"] = (stats["squad_id"] == stats["home_squad_id"]).astype(int)

    # Add position and starter features
    features["jumper_number"] = stats["jumper_number"]
    features["position_code"] = stats["jumper_number"].apply(
        lambda jn: position_from_jersey(jn).code
    )
    features["position_label"] = stats["jumper_number"].apply(
        lambda jn: position_from_jersey(jn).label
    )
    features["position_group"] = features["position_code"].apply(_position_group_from_code)
    features["is_starter"] = (stats["jumper_number"] <= 13).astype(int)

    if feature_config.fillna_value is not None:
        features = features.fillna(feature_config.fillna_value)

    if as_of_round is not None:
        features = features[features["round_number"] == as_of_round]

    if features.empty:
        raise ValueError("No feature rows generated for the requested round.")

    LOGGER.info(
        "Computed player features for %s rows (season=%s, round=%s)",
        len(features),
        year,
        as_of_round,
    )
    return features.reset_index(drop=True)
