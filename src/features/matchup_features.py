"""Player-vs-opponent matchup feature engineering utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Sequence

import pandas as pd

from .. import db
from ..config import DEFAULT_PLAYER_FEATURE_WINDOWS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchupFeatureConfig:
    """Configuration for matchup feature computation."""

    windows: Sequence[int] = (3, 5)
    metrics: Sequence[str] = (
        "tries",
        "line_breaks",
        "run_metres",
        "try_assists",
        "tackle_breaks",
    )
    include_try_rate: bool = True
    fillna_value: float | None = None  # None = leave as NaN to indicate no history


def fetch_player_opponent_stats(
    connection: Connection, year: int | str
) -> pd.DataFrame:
    """Fetch player stats with opponent team identified.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to query.

    Returns
    -------
    pandas.DataFrame
        Player match stats with opponent squad identified.
    """
    normalized_year = db.normalize_year(year)
    player_stats = db.get_table("player_stats", normalized_year)
    matches = db.get_table("matches", normalized_year)

    query = f"""
        SELECT
            ps.match_id,
            ps.player_id,
            ps.squad_id,
            ps.opponent_squad_id,
            ps.position,
            ps.tries,
            ps.try_assists,
            ps.line_breaks,
            ps.run_metres,
            ps.tackle_breaks,
            ps.post_contact_metres,
            m.round_number,
            m.utc_start_time
        FROM {player_stats} ps
        JOIN {matches} m ON ps.match_id = m.match_id
        ORDER BY ps.player_id, m.round_number, ps.match_id
    """
    return db.fetch_df(connection, query)


def _rolling_matchup_feature(
    series: pd.Series, window: int, *, min_periods: int = 1
) -> pd.Series:
    """Compute a shifted rolling mean for matchup data.

    Uses shift(1) to ensure only prior matches are included.
    """
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def _compute_aggregate_matchup_stats(
    stats: pd.DataFrame, config: MatchupFeatureConfig
) -> pd.DataFrame:
    """Compute aggregate player-vs-opponent statistics.

    Parameters
    ----------
    stats : pandas.DataFrame
        Player stats with opponent_squad_id.
    config : MatchupFeatureConfig
        Feature configuration.

    Returns
    -------
    pandas.DataFrame
        Aggregate matchup features by player and opponent.
    """
    # Group by player and opponent
    grouped = stats.groupby(["player_id", "opponent_squad_id"], group_keys=False)

    # Base features - cumulative stats vs this opponent
    features = stats[
        ["match_id", "player_id", "squad_id", "opponent_squad_id", "round_number"]
    ].copy()

    # Count of prior games vs this opponent (excluding current match)
    features["matchup_games_vs_opp"] = (
        grouped["match_id"].cumcount()
    )  # 0 for first match, 1 for second, etc.

    # Aggregate stats vs this opponent (all prior matches)
    for metric in config.metrics:
        if metric not in stats.columns:
            raise ValueError(f"Metric '{metric}' not found in player stats.")

        # Cumulative average vs this opponent (excluding current match)
        agg_name = f"matchup_avg_{metric}_vs_opp"
        features[agg_name] = grouped[metric].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # Cumulative sum vs this opponent (excluding current match)
        sum_name = f"matchup_total_{metric}_vs_opp"
        features[sum_name] = grouped[metric].transform(
            lambda s: s.shift(1).expanding(min_periods=0).sum()
        )

    # Try rate vs this opponent (proportion of games with at least one try)
    if config.include_try_rate:
        features["matchup_try_rate_vs_opp"] = grouped["tries"].transform(
            lambda s: (s > 0).astype(float).shift(1).expanding(min_periods=1).mean()
        )

    return features


def _compute_rolling_matchup_stats(
    stats: pd.DataFrame, config: MatchupFeatureConfig
) -> pd.DataFrame:
    """Compute rolling window player-vs-opponent statistics.

    Parameters
    ----------
    stats : pandas.DataFrame
        Player stats with opponent_squad_id.
    config : MatchupFeatureConfig
        Feature configuration.

    Returns
    -------
    pandas.DataFrame
        Rolling matchup features by player and opponent.
    """
    # Group by player and opponent
    grouped = stats.groupby(["player_id", "opponent_squad_id"], group_keys=False)

    features = stats[
        ["match_id", "player_id", "squad_id", "opponent_squad_id", "round_number"]
    ].copy()

    # Rolling windows for recent form vs opponent
    for metric in config.metrics:
        if metric not in stats.columns:
            raise ValueError(f"Metric '{metric}' not found in player stats.")

        for window in config.windows:
            feature_name = f"matchup_rolling_{metric}_vs_opp_{window}"
            features[feature_name] = grouped[metric].transform(
                lambda s: _rolling_matchup_feature(s, window)
            )

    # Rolling try rate vs opponent
    if config.include_try_rate:
        for window in config.windows:
            try_rate_name = f"matchup_rolling_try_rate_vs_opp_{window}"
            features[try_rate_name] = grouped["tries"].transform(
                lambda s: _rolling_matchup_feature((s > 0).astype(float), window)
            )

    return features


def compute_matchup_features(
    connection: Connection,
    year: int | str,
    *,
    config: MatchupFeatureConfig | None = None,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute player-vs-opponent matchup features using only historical data.

    Features capture how a player has historically performed against a specific
    opponent team. Includes both aggregate (all-time vs opponent) and rolling
    (recent games vs opponent) statistics.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to compute features for.
    config : MatchupFeatureConfig | None
        Feature configuration settings.
    as_of_round : int | None
        If provided, only return features for this round while using prior rounds
        for history. Useful for weekly incremental updates.

    Returns
    -------
    pandas.DataFrame
        Matchup feature table keyed by match_id, player_id, and opponent_squad_id.

    Notes
    -----
    - Features are NaN when a player has no prior history vs the opponent
    - This is intentional to distinguish "no data" from "zero performance"
    - Downstream models should handle these NaNs appropriately (e.g., separate
      indicator feature or imputation based on overall player stats)
    """
    feature_config = config or MatchupFeatureConfig()
    stats = fetch_player_opponent_stats(connection, year)

    if stats.empty:
        raise ValueError("No player stats found for the requested season.")

    # Sort for proper temporal ordering
    # The shift(1) in feature computation ensures we only use prior data
    stats.sort_values(
        ["player_id", "opponent_squad_id", "round_number", "match_id"], inplace=True
    )

    # Compute aggregate features (all-time vs opponent)
    # shift(1).expanding() ensures only prior matches are used
    aggregate_features = _compute_aggregate_matchup_stats(stats, feature_config)

    # Compute rolling features (recent form vs opponent)
    # shift(1).rolling() ensures only prior matches are used
    rolling_features = _compute_rolling_matchup_stats(stats, feature_config)

    # Merge aggregate and rolling features
    features = aggregate_features.merge(
        rolling_features[
            [
                "match_id",
                "player_id",
            ]
            + [
                col
                for col in rolling_features.columns
                if col.startswith("matchup_rolling_")
            ]
        ],
        on=["match_id", "player_id"],
        how="left",
    )

    # Filter to specific round if requested
    if as_of_round is not None:
        features = features[features["round_number"] == as_of_round].copy()
        if features.empty:
            raise ValueError(
                f"No feature rows generated for round {as_of_round}. "
                f"Available rounds: {sorted(stats['round_number'].unique())}"
            )

    # Apply fillna if specified (None means leave as NaN to indicate no history)
    if feature_config.fillna_value is not None:
        matchup_cols = [col for col in features.columns if col.startswith("matchup_")]
        features[matchup_cols] = features[matchup_cols].fillna(
            feature_config.fillna_value
        )

    LOGGER.info(
        "Computed matchup features for %s rows (season=%s, round=%s)",
        len(features),
        year,
        as_of_round,
    )
    return features.reset_index(drop=True)
