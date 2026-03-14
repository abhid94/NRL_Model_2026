"""Player-level feature engineering utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Sequence

import numpy as np
import pandas as pd

from .. import db
from ..config import (
    BAYESIAN_SHRINKAGE_K,
    DEFAULT_PLAYER_FEATURE_WINDOWS,
    EWM_SPAN,
    POSITION_TRY_RATES,
    position_from_jersey,
)

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
        "line_break_assists",
        "run_metres",
        "tackle_breaks",
        "post_contact_metres",
        "offloads",
        "missed_tackles",
        "tackles",
        "errors",
        "passes",
        "possessions",
        "kicks_general_play",
        "kick_metres",
        "runs_hitup",
        "runs_dummy_half",
        "bomb_kicks_caught",
        "penalties_conceded",
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
            ps.line_break_assists,
            ps.run_metres,
            ps.tackle_breaks,
            ps.post_contact_metres,
            ps.offloads,
            ps.missed_tackles,
            ps.tackles,
            ps.errors,
            ps.passes,
            ps.possessions,
            ps.kicks_general_play,
            ps.kick_metres,
            ps.runs_hitup,
            ps.runs_dummy_half,
            ps.bomb_kicks_caught,
            ps.penalties_conceded,
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


def _ewm_feature(series: pd.Series, span: int) -> pd.Series:
    """Compute a shifted exponential weighted moving average.

    Recent values are weighted more heavily than older ones,
    capturing form changes faster than simple rolling averages.

    Parameters
    ----------
    series : pd.Series
        Raw metric series (one value per match, ordered chronologically).
    span : int
        EWM span (controls decay rate).

    Returns
    -------
    pd.Series
        Shifted EWM values. First observation is NaN (no prior data).
    """
    return series.shift(1).ewm(span=span, min_periods=1).mean()


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
        # Include the current round so rows exist for output filtering.
        # shift(1) in _rolling_feature ensures current round's stats don't
        # leak into rolling features — only prior rounds contribute.
        stats = stats[stats["round_number"] <= as_of_round].copy()

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

    # EWM (exponential weighted moving average) features for top SHAP metrics
    _EWM_METRICS = ("tries", "line_breaks", "tackle_breaks", "run_metres")
    for metric in _EWM_METRICS:
        if metric not in stats.columns:
            continue
        features[f"ewm_{metric}_{EWM_SPAN}"] = grouped[metric].transform(
            lambda s: _ewm_feature(s, EWM_SPAN)
        )

    # Season-to-date expanding window features
    # Mirror prior_season_* features so GBM can compare current vs historical form
    _EXPANDING_METRICS = ("tries", "line_breaks", "tackle_breaks", "run_metres", "try_assists")
    for metric in _EXPANDING_METRICS:
        features[f"season_avg_{metric}"] = grouped[metric].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
    features["season_try_rate"] = grouped["tries"].transform(
        lambda s: (s > 0).astype(float).shift(1).expanding(min_periods=1).mean()
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

    # Bayesian shrinkage try rate (computed after position_code and matches_played)
    # Formula: bayes_rate = (n * player_rate + k * position_prior) / (n + k)
    # Position prior from POSITION_TRY_RATES (training data averages, no leakage).
    features["bayesian_position_prior"] = features["position_code"].map(
        POSITION_TRY_RATES
    ).fillna(POSITION_TRY_RATES.get("RES", 0.04))

    n_matches = features["matches_played"].clip(lower=0)
    # NaN rolling_try_rate means no prior data → treat as 0 for Bayesian weighting
    player_rate = features.get(
        "rolling_try_rate_5", pd.Series(0.0, index=features.index)
    ).fillna(0.0)
    k = BAYESIAN_SHRINKAGE_K
    features["bayesian_try_rate"] = (
        (n_matches * player_rate + k * features["bayesian_position_prior"])
        / (n_matches + k)
    )
    features["bayesian_confidence"] = n_matches / (n_matches + k)

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


def compute_typical_position(
    connection: Connection,
    year: int | str,
    *,
    as_of_round: int | None = None,
    lookback: int = 10,
) -> pd.DataFrame:
    """Compute each player's typical jersey and position from recent history.

    Uses the mode of ``jumper_number`` over the last *lookback* matches
    (strictly before the current round) to determine the player's usual
    position.  This lets the model distinguish a natural winger listed as
    jersey 18 from a career bench player.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to compute for.
    as_of_round : int | None
        If provided, only uses matches from rounds < as_of_round.
    lookback : int
        Number of recent matches to consider for the mode.

    Returns
    -------
    pandas.DataFrame
        Columns: player_id, typical_jersey, typical_position_code,
        typical_position_label.  One row per player who has at least one
        prior match.
    """
    normalized_year = db.normalize_year(year)

    # Gather jumper_number history from current + prior season
    frames: list[pd.DataFrame] = []

    for yr in (normalized_year - 1, normalized_year):
        ps_table = db.get_table("player_stats", yr)
        m_table = db.get_table("matches", yr)
        try:
            round_filter = ""
            if yr == normalized_year and as_of_round is not None:
                round_filter = f"AND m.round_number < {int(as_of_round)}"

            q = f"""
                SELECT ps.player_id, ps.jumper_number, m.round_number,
                       {yr} AS season
                FROM {ps_table} ps
                JOIN {m_table} m ON ps.match_id = m.match_id
                WHERE 1=1 {round_filter}
                ORDER BY m.round_number DESC
            """
            frame = db.fetch_df(connection, q)
            if not frame.empty:
                frames.append(frame)
        except Exception:
            pass  # prior season table may not exist

    if not frames:
        return pd.DataFrame(
            columns=["player_id", "typical_jersey", "typical_position_code",
                     "typical_position_label"]
        )

    history = pd.concat(frames, ignore_index=True)
    # Sort newest first, then take last N matches per player
    history.sort_values(
        ["player_id", "season", "round_number"],
        ascending=[True, False, False],
        inplace=True,
    )
    recent = history.groupby("player_id").head(lookback)

    # Mode of jumper_number (most common jersey)
    typical = (
        recent.groupby("player_id")["jumper_number"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
        .rename(columns={"jumper_number": "typical_jersey"})
    )

    typical["typical_position_code"] = typical["typical_jersey"].apply(
        lambda j: position_from_jersey(int(j)).code
    )
    typical["typical_position_label"] = typical["typical_jersey"].apply(
        lambda j: position_from_jersey(int(j)).label
    )

    LOGGER.info(
        "Computed typical position for %d players (season=%s, as_of_round=%s)",
        len(typical), year, as_of_round,
    )
    return typical


def compute_cross_season_priors(
    connection: Connection,
    target_year: int | str,
) -> pd.DataFrame:
    """Compute prior-season aggregate stats for each player.

    For each player appearing in ``target_year``, computes full-season
    aggregates from ``target_year - 1``.  These fill the cold-start gap
    in early rounds where rolling windows are empty.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    target_year : int | str
        The season to produce priors for.

    Returns
    -------
    pandas.DataFrame
        Columns: player_id, prior_season_try_rate, prior_season_matches,
        prior_season_avg_line_breaks, prior_season_avg_tackle_breaks,
        prior_season_avg_run_metres, prior_season_avg_try_assists.
        One row per player.  Players without prior-season data are excluded.
    """
    prior_year = db.normalize_year(target_year) - 1
    prior_stats = db.get_table("player_stats", prior_year)
    prior_matches = db.get_table("matches", prior_year)

    query = f"""
        SELECT
            ps.player_id,
            ps.tries,
            ps.line_breaks,
            ps.tackle_breaks,
            ps.run_metres,
            ps.try_assists
        FROM {prior_stats} ps
        JOIN {prior_matches} m ON ps.match_id = m.match_id
        WHERE m.match_type = 'H'
    """
    try:
        prior = db.fetch_df(connection, query)
    except Exception:
        LOGGER.warning("No prior-season data for year %s", prior_year)
        return pd.DataFrame(columns=[
            "player_id",
            "prior_season_try_rate",
            "prior_season_matches",
            "prior_season_avg_line_breaks",
            "prior_season_avg_tackle_breaks",
            "prior_season_avg_run_metres",
            "prior_season_avg_try_assists",
        ])

    if prior.empty:
        return pd.DataFrame(columns=[
            "player_id",
            "prior_season_try_rate",
            "prior_season_matches",
            "prior_season_avg_line_breaks",
            "prior_season_avg_tackle_breaks",
            "prior_season_avg_run_metres",
            "prior_season_avg_try_assists",
        ])

    agg = prior.groupby("player_id").agg(
        prior_season_matches=("tries", "count"),
        prior_season_try_rate=("tries", lambda x: (x > 0).mean()),
        prior_season_avg_line_breaks=("line_breaks", "mean"),
        prior_season_avg_tackle_breaks=("tackle_breaks", "mean"),
        prior_season_avg_run_metres=("run_metres", "mean"),
        prior_season_avg_try_assists=("try_assists", "mean"),
    ).reset_index()

    LOGGER.info(
        "Computed cross-season priors for %d players (prior season=%s)",
        len(agg),
        prior_year,
    )
    return agg
