"""
Lineup Features for NRL ATS Model

This module computes features based on team lineups and teammate quality.

Key features:
1. Teammate playmaking quality — Rolling try assists from halves (6,7) and fullback (1)
2. Lineup stability — Changes from previous round, player continuity

All features respect temporal constraints for leakage prevention.
"""

import sqlite3
from typing import Optional
import pandas as pd
import numpy as np


def compute_teammate_playmaking_features(
    conn: sqlite3.Connection,
    year: int,
    as_of_round: Optional[int] = None,
    window: int = 5
) -> pd.DataFrame:
    """
    Compute rolling try assists from key playmaking teammates (jerseys 1, 6, 7).

    For each team-match, sums try assists from halves and fullback over the
    previous N matches. This measures the quality of attacking support.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    year : int
        Season year (e.g., 2024, 2025)
    as_of_round : int, optional
        If provided, only uses data from rounds < as_of_round (leakage prevention)
    window : int, default=5
        Number of previous matches to include in rolling window

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, round_number,
                 teammate_fullback_try_assists_rolling_N,
                 teammate_halves_try_assists_rolling_N,
                 teammate_playmakers_try_assists_rolling_N

    Notes
    -----
    - Playmakers are defined as jerseys 1 (fullback), 6 (five-eighth), 7 (halfback)
    - Uses STRICT temporal ordering (rounds < current round only)
    - Returns NaN for teams with < window prior matches
    - First match of season will have NaN (no history)
    """
    # Build WHERE clause for temporal constraint
    where_clause = ""
    if as_of_round is not None:
        where_clause = f"AND m.round_number < {as_of_round}"

    query = f"""
    WITH playmaker_stats AS (
        -- Get try assists from playmakers (jerseys 1, 6, 7) per match
        SELECT
            ps.match_id,
            ps.squad_id,
            m.round_number,
            m.utc_start_time,
            SUM(CASE WHEN ps.jumper_number = 1 THEN ps.try_assists ELSE 0 END) as fullback_try_assists,
            SUM(CASE WHEN ps.jumper_number IN (6, 7) THEN ps.try_assists ELSE 0 END) as halves_try_assists,
            SUM(CASE WHEN ps.jumper_number IN (1, 6, 7) THEN ps.try_assists ELSE 0 END) as playmakers_try_assists
        FROM player_stats_{year} ps
        JOIN matches_{year} m ON ps.match_id = m.match_id
        WHERE ps.jumper_number IN (1, 6, 7)
            {where_clause}
        GROUP BY ps.match_id, ps.squad_id, m.round_number, m.utc_start_time
    ),

    ranked_matches AS (
        -- Rank matches by time for each team to enable window selection
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY squad_id
                ORDER BY utc_start_time
            ) as match_sequence
        FROM playmaker_stats
    ),

    rolling_playmaker_stats AS (
        -- Compute rolling sum of try assists over previous {window} matches
        SELECT
            r1.match_id,
            r1.squad_id,
            r1.round_number,
            -- Sum try assists from previous matches within window
            SUM(r2.fullback_try_assists) as teammate_fullback_try_assists_rolling_{window},
            SUM(r2.halves_try_assists) as teammate_halves_try_assists_rolling_{window},
            SUM(r2.playmakers_try_assists) as teammate_playmakers_try_assists_rolling_{window},
            COUNT(r2.match_id) as matches_in_window
        FROM ranked_matches r1
        LEFT JOIN ranked_matches r2 ON
            r1.squad_id = r2.squad_id
            AND r2.match_sequence < r1.match_sequence
            AND r2.match_sequence >= r1.match_sequence - {window}
        GROUP BY r1.match_id, r1.squad_id, r1.round_number
    )

    SELECT
        match_id,
        squad_id,
        round_number,
        teammate_fullback_try_assists_rolling_{window},
        teammate_halves_try_assists_rolling_{window},
        teammate_playmakers_try_assists_rolling_{window},
        matches_in_window
    FROM rolling_playmaker_stats
    ORDER BY match_id, squad_id
    """

    df = pd.read_sql_query(query, conn)

    # Set to NaN if insufficient history (fewer than window matches)
    # This is intentional - models should learn that NaN = limited history
    insufficient_history = df['matches_in_window'] < window
    df.loc[insufficient_history, [
        f'teammate_fullback_try_assists_rolling_{window}',
        f'teammate_halves_try_assists_rolling_{window}',
        f'teammate_playmakers_try_assists_rolling_{window}'
    ]] = np.nan

    # Drop internal tracking column
    df = df.drop(columns=['matches_in_window'])

    return df


def compute_lineup_stability_features(
    conn: sqlite3.Connection,
    year: int,
    as_of_round: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute lineup stability features by comparing to previous round's lineup.

    For each team-match, counts how many players changed from the previous round.
    Also tracks whether each individual player was in the previous lineup.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    year : int
        Season year (e.g., 2024, 2025)
    as_of_round : int, optional
        If provided, only uses data from rounds <= as_of_round (lineups are announced pre-match)

    Returns
    -------
    pd.DataFrame
        Columns: match_id, player_id, squad_id, round_number,
                 lineup_changes_from_prev_round,
                 lineup_stability_pct,
                 player_was_in_prev_lineup

    Notes
    -----
    - First round of season will have NaN for all stability features (no previous lineup)
    - Compares starting 17 only (jerseys 1-17)
    - player_was_in_prev_lineup: 1 if player was in prev round, 0 if new, NaN if first round
    - lineup_changes_from_prev_round: count of new players (vs prev round's 17)
    - lineup_stability_pct: % of prev round's 17 that remain (0.0 to 1.0)
    """
    # Build WHERE clause for temporal constraint
    where_clause = ""
    if as_of_round is not None:
        where_clause = f"AND tl.round_number <= {as_of_round}"

    # Note: team_lists table may not exist for all years
    # Check if team_lists_{year} exists
    table_check = pd.read_sql_query(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='team_lists_{year}'",
        conn
    )

    if len(table_check) == 0:
        # No team_lists data for this year - return empty DataFrame
        return pd.DataFrame(columns=[
            'match_id', 'player_id', 'squad_id', 'round_number',
            'lineup_changes_from_prev_round', 'lineup_stability_pct',
            'player_was_in_prev_lineup'
        ])

    query = f"""
    WITH current_lineups AS (
        -- Get starting 17 for each match (jerseys 1-17)
        SELECT
            tl.match_id,
            tl.squad_id,
            tl.round_number,
            tl.player_id,
            tl.jersey_number,
            m.utc_start_time
        FROM team_lists_{year} tl
        JOIN matches_{year} m ON tl.match_id = m.match_id
        WHERE tl.jersey_number BETWEEN 1 AND 17
            {where_clause}
    ),

    team_matches_ranked AS (
        -- Rank each team's matches chronologically
        SELECT
            match_id,
            squad_id,
            round_number,
            utc_start_time,
            ROW_NUMBER() OVER (
                PARTITION BY squad_id
                ORDER BY utc_start_time
            ) as match_sequence
        FROM (
            SELECT DISTINCT match_id, squad_id, round_number, utc_start_time
            FROM current_lineups
        )
    ),

    lineups_with_prev_match AS (
        -- Join current lineup to previous match ID
        SELECT
            cl.*,
            tm_prev.match_id as prev_match_id
        FROM current_lineups cl
        JOIN team_matches_ranked tm_curr ON
            cl.match_id = tm_curr.match_id
            AND cl.squad_id = tm_curr.squad_id
        LEFT JOIN team_matches_ranked tm_prev ON
            tm_curr.squad_id = tm_prev.squad_id
            AND tm_prev.match_sequence = tm_curr.match_sequence - 1
    ),

    player_continuity AS (
        -- Check if each player was in previous lineup
        SELECT
            lp.match_id,
            lp.player_id,
            lp.squad_id,
            lp.round_number,
            CASE
                WHEN lp.prev_match_id IS NULL THEN NULL  -- No previous match
                WHEN prev_lineup.player_id IS NOT NULL THEN 1  -- Player was in prev lineup
                ELSE 0  -- Player was NOT in prev lineup (new to team)
            END as player_was_in_prev_lineup,
            lp.prev_match_id
        FROM lineups_with_prev_match lp
        LEFT JOIN team_lists_{year} prev_lineup ON
            lp.prev_match_id = prev_lineup.match_id
            AND lp.squad_id = prev_lineup.squad_id
            AND lp.player_id = prev_lineup.player_id
            AND prev_lineup.jersey_number BETWEEN 1 AND 17
    ),

    lineup_changes_agg AS (
        -- Aggregate changes per team-match
        SELECT
            match_id,
            squad_id,
            round_number,
            SUM(CASE WHEN player_was_in_prev_lineup = 0 THEN 1 ELSE 0 END) as lineup_changes_from_prev_round,
            COUNT(*) as current_lineup_size,
            -- Count players who were in prev lineup
            SUM(CASE WHEN player_was_in_prev_lineup = 1 THEN 1 ELSE 0 END) as retained_from_prev,
            -- Check if ANY player has prev_match_id (to distinguish first match from no changes)
            MAX(CASE WHEN prev_match_id IS NOT NULL THEN 1 ELSE 0 END) as has_prev_match
        FROM player_continuity
        GROUP BY match_id, squad_id, round_number
    )

    SELECT
        pc.match_id,
        pc.player_id,
        pc.squad_id,
        pc.round_number,
        lc.lineup_changes_from_prev_round,
        CASE
            WHEN lc.has_prev_match = 0 THEN NULL  -- First match, no previous lineup
            ELSE CAST(lc.retained_from_prev AS FLOAT) / lc.current_lineup_size
        END as lineup_stability_pct,
        pc.player_was_in_prev_lineup
    FROM player_continuity pc
    LEFT JOIN lineup_changes_agg lc ON
        pc.match_id = lc.match_id
        AND pc.squad_id = lc.squad_id
    ORDER BY pc.match_id, pc.squad_id, pc.player_id
    """

    df = pd.read_sql_query(query, conn)

    return df


def add_lineup_features_to_player_observations(
    player_df: pd.DataFrame,
    conn: sqlite3.Connection,
    year: int,
    as_of_round: Optional[int] = None,
    window: int = 5
) -> pd.DataFrame:
    """
    Add all lineup features to player-match observations.

    This is the main public function that joins both teammate playmaking
    and lineup stability features to a base player observations DataFrame.

    Parameters
    ----------
    player_df : pd.DataFrame
        Base player-match observations with columns: match_id, player_id, squad_id
    conn : sqlite3.Connection
        Database connection
    year : int
        Season year
    as_of_round : int, optional
        Temporal cutoff for leakage prevention
    window : int, default=5
        Rolling window size for playmaking features

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added lineup feature columns:
        - teammate_fullback_try_assists_rolling_N
        - teammate_halves_try_assists_rolling_N
        - teammate_playmakers_try_assists_rolling_N
        - lineup_changes_from_prev_round
        - lineup_stability_pct
        - player_was_in_prev_lineup

    Notes
    -----
    - Teammate features are at team-match level (same for all players on a team)
    - Stability features include both team-level and player-level features
    - Missing features will be NaN for first round or insufficient history
    """
    # Validate required columns
    required_cols = ['match_id', 'player_id', 'squad_id']
    missing_cols = set(required_cols) - set(player_df.columns)
    if missing_cols:
        raise ValueError(f"player_df missing required columns: {missing_cols}")

    # Compute teammate playmaking features (team-level)
    playmaking_features = compute_teammate_playmaking_features(
        conn=conn,
        year=year,
        as_of_round=as_of_round,
        window=window
    )

    # Compute lineup stability features (player-level)
    stability_features = compute_lineup_stability_features(
        conn=conn,
        year=year,
        as_of_round=as_of_round
    )

    # Join playmaking features (team-match level)
    result = player_df.merge(
        playmaking_features,
        on=['match_id', 'squad_id'],
        how='left',
        suffixes=('', '_dup')
    )

    # Drop duplicate round_number column if it exists
    if 'round_number_dup' in result.columns:
        result = result.drop(columns=['round_number_dup'])

    # Join stability features (player-match level)
    result = result.merge(
        stability_features,
        on=['match_id', 'player_id', 'squad_id'],
        how='left',
        suffixes=('', '_dup')
    )

    # Drop duplicate round_number column if it exists
    if 'round_number_dup' in result.columns:
        result = result.drop(columns=['round_number_dup'])

    return result


if __name__ == "__main__":
    # Demo usage
    import sys
    sys.path.append('/Users/abhidutta/Documents/repos/NRL_2026_Model')
    from src.db import get_connection

    conn = get_connection()

    # Example 1: Teammate playmaking features
    print("=== Teammate Playmaking Features (2024, Round 15) ===")
    playmaking = compute_teammate_playmaking_features(
        conn=conn,
        year=2024,
        as_of_round=15,
        window=5
    )
    print(playmaking.head(10))
    print(f"\nShape: {playmaking.shape}")
    print(f"Null counts:\n{playmaking.isnull().sum()}")

    # Example 2: Lineup stability features
    print("\n=== Lineup Stability Features (2025, Round 20) ===")
    stability = compute_lineup_stability_features(
        conn=conn,
        year=2025,
        as_of_round=20
    )
    print(stability.head(10))
    print(f"\nShape: {stability.shape}")
    print(f"Null counts:\n{stability.isnull().sum()}")

    # Example 3: Full integration
    print("\n=== Full Lineup Features Integration ===")
    # Get some player observations from player_stats
    base_query = """
    SELECT ps.match_id, ps.player_id, ps.squad_id, m.round_number
    FROM player_stats_2025 ps
    JOIN matches_2025 m ON ps.match_id = m.match_id
    WHERE m.round_number = 20
    LIMIT 50
    """
    base_df = pd.read_sql_query(base_query, conn)

    enriched = add_lineup_features_to_player_observations(
        player_df=base_df,
        conn=conn,
        year=2025,
        as_of_round=20,
        window=5
    )
    print(enriched.head(10))
    print(f"\nColumns: {list(enriched.columns)}")
    print(f"Shape: {enriched.shape}")

    conn.close()
