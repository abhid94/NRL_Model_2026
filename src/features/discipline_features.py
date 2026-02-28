"""Discipline features from sin bins and on-report data.

Computes rolling discipline metrics for teams and players:
- team_rolling_sin_bins_N: team sin bin trend (from player_stats)
- player_rolling_sin_bins_N: player-level discipline
- opponent_rolling_sin_bins_N: opponent discipline (scoring opportunities)
- team_rolling_on_reports_N: team on-report trend (from match_reports)

LEAKAGE PREVENTION:
- All rolling windows use strict round < current_round filtering
- shift(1) applied before rolling to exclude current match
"""

import sqlite3
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_discipline_features(
    conn: sqlite3.Connection,
    season: int,
    as_of_round: Optional[int] = None,
    window: int = 5,
) -> pd.DataFrame:
    """Compute discipline features for each player-match observation.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    as_of_round : int, optional
        Only include matches up to this round.
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, player_id, plus discipline features.
    """
    round_filter = f"AND m.round_number <= {as_of_round}" if as_of_round else ""

    # Get sin bins per player per match from player_stats
    query = f"""
    SELECT
        ps.match_id,
        ps.player_id,
        ps.squad_id,
        ps.opponent_squad_id,
        m.round_number,
        ps.sin_bins
    FROM player_stats_{season} ps
    JOIN matches_{season} m ON ps.match_id = m.match_id
    WHERE m.match_type = 'H'
    {round_filter}
    ORDER BY m.round_number, ps.match_id, ps.player_id
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        return pd.DataFrame(columns=["match_id", "player_id"])

    # --- Team-level sin bins per match ---
    team_sinbins = (
        df.groupby(["match_id", "squad_id", "round_number"])["sin_bins"]
        .sum()
        .reset_index()
        .rename(columns={"sin_bins": "team_sin_bins"})
    )
    team_sinbins = team_sinbins.sort_values(["squad_id", "round_number", "match_id"])

    # Rolling team sin bins (shift to prevent leakage)
    team_sinbins[f"team_rolling_sin_bins_{window}"] = (
        team_sinbins
        .groupby("squad_id")["team_sin_bins"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    )

    # --- Opponent-level sin bins ---
    # Build opponent lookup
    match_opponents = (
        df[["match_id", "squad_id", "opponent_squad_id"]]
        .drop_duplicates()
    )

    opponent_sinbins = team_sinbins[["match_id", "squad_id", f"team_rolling_sin_bins_{window}"]].copy()
    opponent_sinbins = opponent_sinbins.rename(columns={
        "squad_id": "opponent_squad_id",
        f"team_rolling_sin_bins_{window}": f"opponent_rolling_sin_bins_{window}",
    })

    # --- Player-level sin bins ---
    player_sinbins = df[["match_id", "player_id", "squad_id", "round_number", "sin_bins"]].copy()
    player_sinbins = player_sinbins.sort_values(["player_id", "round_number", "match_id"])

    player_sinbins[f"player_rolling_sin_bins_{window}"] = (
        player_sinbins
        .groupby("player_id")["sin_bins"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    )

    # --- On-report features (from match_reports tables) ---
    on_report_features = _compute_on_report_features(conn, season, df, window)

    # --- Merge everything back to player-match level ---
    result = df[["match_id", "player_id", "squad_id", "opponent_squad_id"]].copy()

    # Team sin bins
    result = result.merge(
        team_sinbins[["match_id", "squad_id", f"team_rolling_sin_bins_{window}"]],
        on=["match_id", "squad_id"],
        how="left",
    )

    # Opponent sin bins
    result = result.merge(
        opponent_sinbins[["match_id", "opponent_squad_id", f"opponent_rolling_sin_bins_{window}"]],
        on=["match_id", "opponent_squad_id"],
        how="left",
    )

    # Player sin bins
    result = result.merge(
        player_sinbins[["match_id", "player_id", f"player_rolling_sin_bins_{window}"]],
        on=["match_id", "player_id"],
        how="left",
    )

    # On-report features
    if not on_report_features.empty:
        result = result.merge(
            on_report_features,
            on=["match_id", "squad_id"],
            how="left",
        )

    # Drop join keys that duplicate base columns
    result = result.drop(columns=["squad_id", "opponent_squad_id"], errors="ignore")

    n_features = len(result.columns) - 2  # minus match_id, player_id
    logger.info(
        "Discipline features: %d rows, %d features (window=%d)",
        len(result), n_features, window,
    )

    return result


def _compute_on_report_features(
    conn: sqlite3.Connection,
    season: int,
    base_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute on-report rolling features from match_reports tables.

    Parameters
    ----------
    conn : sqlite3.Connection
    season : int
    base_df : pd.DataFrame
        Base player observations with match_id, squad_id, round_number.
    window : int

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, team_rolling_on_reports_{window}
    """
    # Determine which match_reports table to use
    table_name = None
    for candidate in [f"match_reports_{season}", "match_reports"]:
        try:
            conn.execute(f"SELECT 1 FROM {candidate} LIMIT 1")
            table_name = candidate
            break
        except Exception:
            continue

    if table_name is None:
        logger.info("No match_reports table found for season %d", season)
        return pd.DataFrame(columns=["match_id", "squad_id"])

    # Get on-report counts per team per match
    # Only count on_reports for matches in this season
    match_ids = base_df["match_id"].unique().tolist()
    if not match_ids:
        return pd.DataFrame(columns=["match_id", "squad_id"])

    placeholders = ",".join("?" * len(match_ids))
    query = f"""
    SELECT match_id, squad_id, COUNT(*) as on_report_count
    FROM {table_name}
    WHERE match_id IN ({placeholders})
    GROUP BY match_id, squad_id
    """
    on_reports = pd.read_sql_query(query, conn, params=match_ids)

    if on_reports.empty:
        return pd.DataFrame(columns=["match_id", "squad_id"])

    # Get round_number for each match
    match_rounds = base_df[["match_id", "round_number"]].drop_duplicates()
    on_reports = on_reports.merge(match_rounds, on="match_id", how="inner")

    # Ensure all team-match combos exist (fill missing with 0)
    team_matches = base_df[["match_id", "squad_id", "round_number"]].drop_duplicates()
    on_reports = team_matches.merge(on_reports, on=["match_id", "squad_id", "round_number"], how="left")
    on_reports["on_report_count"] = on_reports["on_report_count"].fillna(0)

    on_reports = on_reports.sort_values(["squad_id", "round_number", "match_id"])

    on_reports[f"team_rolling_on_reports_{window}"] = (
        on_reports
        .groupby("squad_id")["on_report_count"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    )

    return on_reports[["match_id", "squad_id", f"team_rolling_on_reports_{window}"]].drop_duplicates()
