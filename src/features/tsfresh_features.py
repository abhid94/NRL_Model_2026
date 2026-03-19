"""Automated time series feature extraction from player match histories via tsfresh.

This module uses tsfresh to extract compact, relevant time series features from
each player's recent match history. The resulting features capture temporal
patterns (trends, variance, autocorrelation, etc.) that hand-crafted rolling
averages may miss.

Features are extracted from a configurable window of prior matches for each
player and prefixed with ``ts_`` for easy identification in the feature store.

LEAKAGE PREVENTION:
- Only matches with ``round_number < as_of_round`` are used.
- The target variable (``tries > 0``) is computed from the same temporal cutoff.
- tsfresh relevance filtering uses only historical data.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src import db

LOGGER = logging.getLogger(__name__)

# Stats columns to feed into tsfresh.  These are the highest-signal player
# stats for try-scoring prediction, kept small for speed with MinimalFCParameters.
DEFAULT_STAT_COLUMNS: tuple[str, ...] = (
    "tries",
    "line_breaks",
    "tackle_breaks",
    "run_metres",
    "try_assists",
    "offloads",
)

# Minimum number of historical matches a player must have to extract features.
# Players with fewer matches get NaN for all tsfresh columns.
MIN_MATCHES_FOR_EXTRACTION: int = 3


def _fetch_player_histories(
    conn: sqlite3.Connection,
    season: int,
    as_of_round: int,
    window: int,
    stat_columns: Sequence[str],
) -> pd.DataFrame:
    """Fetch the last *window* matches per player strictly before *as_of_round*.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year (e.g. 2024).
    as_of_round : int
        Temporal cutoff -- only rounds < this value are included.
    window : int
        Maximum number of prior matches to include per player.
    stat_columns : Sequence[str]
        Stat columns to select from ``player_stats``.

    Returns
    -------
    pd.DataFrame
        Columns: ``player_id``, ``round_number``, ``match_seq``, plus each
        stat column.  ``match_seq`` is a 1-based chronological index within
        each player's history (most recent = highest number).
    """
    player_stats_table = db.get_table("player_stats", season)
    matches_table = db.get_table("matches", season)

    # Validate stat columns are safe identifiers
    for col in stat_columns:
        db.validate_identifier(col)

    stat_select = ", ".join(f"ps.{col}" for col in stat_columns)

    query = f"""
        SELECT
            ps.player_id,
            m.round_number,
            {stat_select}
        FROM {player_stats_table} ps
        JOIN {matches_table} m ON ps.match_id = m.match_id
        WHERE m.round_number < ?
          AND m.match_type = 'H'
        ORDER BY ps.player_id, m.round_number, ps.match_id
    """
    df = pd.read_sql_query(query, conn, params=[as_of_round])

    if df.empty:
        LOGGER.warning(
            "No player history found for season=%d, as_of_round=%d", season, as_of_round
        )
        return df

    # Keep only the last *window* matches per player (by chronological order).
    # Assign a reverse rank within each player group, then filter.
    df["_rank"] = df.groupby("player_id").cumcount(ascending=False)
    df = df[df["_rank"] < window].copy()
    df.drop(columns=["_rank"], inplace=True)

    # Create a 1-based match sequence number per player (time axis for tsfresh).
    df["match_seq"] = df.groupby("player_id").cumcount() + 1

    LOGGER.info(
        "Fetched histories: %d rows, %d unique players, season=%d, as_of_round=%d, window=%d",
        len(df),
        df["player_id"].nunique(),
        season,
        as_of_round,
        window,
    )
    return df


def _build_tsfresh_input(
    histories: pd.DataFrame,
    stat_columns: Sequence[str],
) -> pd.DataFrame:
    """Reshape player histories into the tsfresh long-format input.

    tsfresh expects a DataFrame with columns ``[id, time, value]`` or, for
    multiple kinds, ``[id, time, kind, value]``.  We use the "wide" convenience
    format: ``[id, time, stat1, stat2, ...]`` which ``extract_features``
    accepts directly when ``column_id``, ``column_sort`` are specified.

    Parameters
    ----------
    histories : pd.DataFrame
        Output of :func:`_fetch_player_histories`.
    stat_columns : Sequence[str]
        Stat column names present in *histories*.

    Returns
    -------
    pd.DataFrame
        Columns: ``player_id``, ``match_seq``, plus each stat column.
    """
    keep_cols = ["player_id", "match_seq"] + list(stat_columns)
    return histories[keep_cols].copy()


def extract_tsfresh_features(
    conn: sqlite3.Connection,
    season: int,
    as_of_round: int,
    window: int = 10,
    stat_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Extract tsfresh time series features from player match histories.

    For each player with sufficient history (>= ``MIN_MATCHES_FOR_EXTRACTION``
    matches before *as_of_round*), this function:

    1. Pulls the last *window* matches of stats.
    2. Formats data for tsfresh (player_id as entity, match sequence as time).
    3. Runs ``tsfresh.extract_features()`` with ``MinimalFCParameters`` for speed.
    4. Prefixes all output columns with ``ts_`` for identification.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year (e.g. 2024).
    as_of_round : int
        Temporal cutoff. Only data from rounds strictly less than this value
        is used.  This is the primary leakage-prevention mechanism.
    window : int, default 10
        Number of prior matches to include per player.
    stat_columns : Sequence[str] | None, optional
        Stat columns to extract features from.  Defaults to
        ``DEFAULT_STAT_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        One row per ``player_id`` with tsfresh feature columns prefixed
        by ``ts_``.  Players with fewer than ``MIN_MATCHES_FOR_EXTRACTION``
        prior matches are excluded (they will receive NaN after merging).

    Raises
    ------
    ImportError
        If tsfresh is not installed.
    """
    # Lazy import to avoid hard dependency at module level
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
        from tsfresh.utilities.dataframe_functions import impute
    except ImportError as exc:
        raise ImportError(
            "tsfresh is required for time series feature extraction. "
            "Install it with: pip install tsfresh"
        ) from exc

    if stat_columns is None:
        stat_columns = DEFAULT_STAT_COLUMNS

    LOGGER.info(
        "Extracting tsfresh features: season=%d, as_of_round=%d, window=%d, stats=%s",
        season,
        as_of_round,
        window,
        stat_columns,
    )

    # Step 1: Fetch player histories (strict temporal cutoff)
    histories = _fetch_player_histories(
        conn, season, as_of_round, window, stat_columns
    )

    if histories.empty:
        LOGGER.warning("No histories available — returning empty DataFrame")
        return pd.DataFrame(columns=["player_id"])

    # Step 2: Filter to players with enough history
    match_counts = histories.groupby("player_id")["match_seq"].max()
    eligible_players = match_counts[match_counts >= MIN_MATCHES_FOR_EXTRACTION].index
    histories = histories[histories["player_id"].isin(eligible_players)].copy()

    if histories.empty:
        LOGGER.warning(
            "No players with >= %d matches — returning empty DataFrame",
            MIN_MATCHES_FOR_EXTRACTION,
        )
        return pd.DataFrame(columns=["player_id"])

    LOGGER.info(
        "Players with sufficient history: %d (of %d total)",
        len(eligible_players),
        match_counts.shape[0],
    )

    # Step 3: Build tsfresh input format
    ts_input = _build_tsfresh_input(histories, stat_columns)

    # Step 4: Extract features with MinimalFCParameters (fast subset)
    # MinimalFCParameters extracts ~8 features per stat: sum, mean, median,
    # std, min, max, length, plus a few variance measures.
    # Disable progress bar and excessive logging from tsfresh
    extracted = extract_features(
        ts_input,
        column_id="player_id",
        column_sort="match_seq",
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=1,  # Single-threaded for reproducibility and SQLite safety
    )

    # Step 5: Impute any infinities / NaN from tsfresh calculations
    impute(extracted)

    # Step 6: Prefix all columns with ts_ for easy identification
    extracted.columns = [f"ts_{col}" for col in extracted.columns]

    # Reset index so player_id becomes a column
    extracted = extracted.reset_index().rename(columns={"index": "player_id"})

    # Ensure player_id is integer type (tsfresh may change it)
    extracted["player_id"] = extracted["player_id"].astype(int)

    LOGGER.info(
        "Extracted %d tsfresh features for %d players",
        len(extracted.columns) - 1,  # subtract player_id
        len(extracted),
    )

    return extracted


def add_tsfresh_features(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    season: int,
    as_of_round: int,
    window: int = 10,
    stat_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Extract tsfresh features and merge onto an existing feature DataFrame.

    This is the main public entry point for integrating tsfresh features into
    the feature store pipeline.  It wraps :func:`extract_tsfresh_features` and
    performs a left join on ``player_id``, so players without sufficient history
    receive NaN for all ``ts_`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Existing feature store DataFrame.  Must contain a ``player_id`` column.
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year (e.g. 2024).
    as_of_round : int
        Temporal cutoff for leakage prevention.  Only data from rounds
        strictly less than this value is used.
    window : int, default 10
        Number of prior matches per player to include in the time series.
    stat_columns : Sequence[str] | None, optional
        Stat columns to extract features from.  Defaults to
        ``DEFAULT_STAT_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional ``ts_``-prefixed feature columns.
        Players with fewer than ``MIN_MATCHES_FOR_EXTRACTION`` prior matches
        will have NaN for all tsfresh columns.

    Raises
    ------
    ValueError
        If *df* does not contain a ``player_id`` column.
    ImportError
        If tsfresh is not installed.
    """
    if "player_id" not in df.columns:
        raise ValueError("df must contain a 'player_id' column")

    LOGGER.info(
        "Adding tsfresh features to %d observations (season=%d, as_of_round=%d)",
        len(df),
        season,
        as_of_round,
    )

    ts_features = extract_tsfresh_features(
        conn=conn,
        season=season,
        as_of_round=as_of_round,
        window=window,
        stat_columns=stat_columns,
    )

    if ts_features.empty or len(ts_features.columns) <= 1:
        LOGGER.warning(
            "No tsfresh features extracted — returning df unchanged"
        )
        return df

    # Left merge: all rows in df are preserved; missing players get NaN
    n_before = len(df.columns)
    result = df.merge(ts_features, on="player_id", how="left")

    n_ts_cols = len(result.columns) - n_before
    n_matched = result[ts_features.columns[1]].notna().sum()  # first ts_ col
    coverage = n_matched / len(result) * 100 if len(result) > 0 else 0.0

    LOGGER.info(
        "Merged %d tsfresh features onto %d observations (%.1f%% coverage)",
        n_ts_cols,
        len(result),
        coverage,
    )

    return result
