"""Edge-specific features for try scoring prediction.

This module exploits validated team edge attack patterns (e.g., Titans 64.2% left-edge tries)
by creating features that capture left/right/middle edge attack and defence profiles.

Edge mapping:
- Left edge: jerseys 2, 3, 11 (left wing, left centre, left second row)
- Right edge: jerseys 4, 5, 12 (right centre, right wing, right second row)
- Middle: jerseys 8, 9, 10, 13 (props, hooker, lock)
- Other: jerseys 1, 6, 7, 14+ (fullback, halves, interchange/reserves)

Key features:
- Team edge attack profiles: % of team tries scored by each edge (rolling)
- Team edge defence profiles: tries conceded to each edge (rolling)
- Player edge features: player's edge assignment + matchup scores
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src import config
from src.db import fetch_df, get_connection

logger = logging.getLogger(__name__)


def classify_jersey_to_edge(jersey_number: int | None) -> str:
    """Classify a jersey number to its edge (left/right/middle/other).

    Parameters
    ----------
    jersey_number : int | None
        Player's jersey number for the match.

    Returns
    -------
    str
        Edge classification: 'left', 'right', 'middle', or 'other'.
    """
    if jersey_number is None:
        return "other"
    return config.JERSEY_TO_EDGE.get(jersey_number, "other")


def compute_team_edge_attack_profiles(
    season: int, max_round: int | None = None, window: int = 5
) -> pd.DataFrame:
    """Compute rolling edge attack profiles for each team.

    For each team-match, calculates the percentage of team tries scored by
    each edge (left/right/middle) over the previous N matches.

    Parameters
    ----------
    season : int
        The season year (e.g., 2024).
    max_round : int | None
        If provided, only use data from rounds < max_round (for leakage prevention).
    window : int, default 5
        Number of previous matches to use for rolling calculation.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, left_edge_try_pct_rolling_{window},
                 right_edge_try_pct_rolling_{window}, middle_edge_try_pct_rolling_{window},
                 other_edge_try_pct_rolling_{window}, total_tries_rolling_{window}

    Notes
    -----
    - Uses score_flow to identify tries and player_stats to get jersey numbers
    - Percentages sum to 1.0 across all four edges for each team-match
    - Returns NaN for teams with no tries in the rolling window
    """
    logger.info(
        f"Computing team edge attack profiles for {season}, "
        f"max_round={max_round}, window={window}"
    )

    # Build WHERE clause for max_round filter
    round_filter = f"AND m.round_number < {max_round}" if max_round else ""

    # Query to get tries by edge for each team-match
    query = f"""
    WITH try_scores AS (
        SELECT
            sf.match_id,
            sf.squad_id AS scoring_team_id,
            sf.player_id,
            ps.jumper_number,
            m.round_number,
            m.utc_start_time
        FROM score_flow_{season} sf
        INNER JOIN matches_{season} m ON sf.match_id = m.match_id
        INNER JOIN player_stats_{season} ps
            ON sf.match_id = ps.match_id
            AND sf.player_id = ps.player_id
            AND sf.squad_id = ps.squad_id
        WHERE sf.score_name = 'try'
            {round_filter}
    )
    SELECT
        match_id,
        scoring_team_id AS squad_id,
        player_id,
        jumper_number,
        round_number,
        utc_start_time
    FROM try_scores
    ORDER BY squad_id, utc_start_time, match_id
    """

    with get_connection() as conn:
        df = fetch_df(conn, query)

    if df.empty:
        logger.warning(f"No try scores found for {season}")
        return pd.DataFrame()

    # Classify each try to an edge
    df["edge"] = df["jumper_number"].apply(classify_jersey_to_edge)

    # Count tries by edge for each team-match
    edge_counts = (
        df.groupby(["match_id", "squad_id", "edge"], as_index=False)
        .size()
        .rename(columns={"size": "tries"})
    )

    # Pivot to get edge columns
    edge_pivot = edge_counts.pivot(
        index=["match_id", "squad_id"], columns="edge", values="tries"
    ).fillna(0)

    # Ensure all edge columns exist
    for edge in ["left", "right", "middle", "other"]:
        if edge not in edge_pivot.columns:
            edge_pivot[edge] = 0

    edge_pivot = edge_pivot.reset_index()

    # Get all team-matches (including those with zero tries) for rolling window
    all_matches_query = f"""
    SELECT DISTINCT
        m.match_id,
        ts.squad_id,
        m.round_number,
        m.utc_start_time
    FROM matches_{season} m
    INNER JOIN team_stats_{season} ts ON m.match_id = ts.match_id
    WHERE 1=1
        {round_filter}
    ORDER BY ts.squad_id, m.utc_start_time, m.match_id
    """

    with get_connection() as conn:
        all_matches = fetch_df(conn, all_matches_query)

    # Merge to include zero-try matches
    merged = all_matches.merge(
        edge_pivot[["match_id", "squad_id", "left", "right", "middle", "other"]],
        on=["match_id", "squad_id"],
        how="left",
    ).fillna(0)

    # Compute rolling sums for each edge
    merged["total_tries"] = (
        merged["left"] + merged["right"] + merged["middle"] + merged["other"]
    )

    # Rolling window by team
    for col in ["left", "right", "middle", "other", "total_tries"]:
        merged[f"{col}_rolling"] = (
            merged.groupby("squad_id")[col]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # Compute percentages (avoid division by zero)
    for edge in ["left", "right", "middle", "other"]:
        merged[f"{edge}_edge_try_pct_rolling_{window}"] = (
            merged[f"{edge}_rolling"] / merged["total_tries_rolling"]
        ).fillna(0)

    # Select final columns
    result = merged[
        [
            "match_id",
            "squad_id",
            f"left_edge_try_pct_rolling_{window}",
            f"right_edge_try_pct_rolling_{window}",
            f"middle_edge_try_pct_rolling_{window}",
            f"other_edge_try_pct_rolling_{window}",
            "total_tries_rolling",
        ]
    ].copy()

    # Rename total_tries_rolling to include window size
    result = result.rename(columns={"total_tries_rolling": f"total_tries_rolling_{window}"})

    logger.info(
        f"Computed edge attack profiles: {len(result)} team-match observations"
    )
    return result


def compute_team_edge_defence_profiles(
    season: int, max_round: int | None = None, window: int = 5
) -> pd.DataFrame:
    """Compute rolling edge defence profiles for each team.

    For each team-match, calculates tries conceded to each opponent edge
    over the previous N matches.

    Parameters
    ----------
    season : int
        The season year (e.g., 2024).
    max_round : int | None
        If provided, only use data from rounds < max_round (for leakage prevention).
    window : int, default 5
        Number of previous matches to use for rolling calculation.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, conceded_to_left_edge_rolling_{window},
                 conceded_to_right_edge_rolling_{window},
                 conceded_to_middle_rolling_{window},
                 conceded_to_other_rolling_{window},
                 total_tries_conceded_rolling_{window}

    Notes
    -----
    - "Conceded to left edge" means tries scored by the opponent's left edge players
    - This measures defensive weakness on the defending team's right side
      (since opponent's left attacks defender's right)
    """
    logger.info(
        f"Computing team edge defence profiles for {season}, "
        f"max_round={max_round}, window={window}"
    )

    # First get edge attack profiles (tries scored by each team's edges)
    attack_profiles_full = compute_team_edge_attack_profiles(
        season=season, max_round=max_round, window=1
    )

    if attack_profiles_full.empty:
        return pd.DataFrame()

    # Get match opponent mapping
    round_filter = f"AND round_number < {max_round}" if max_round else ""
    opponent_query = f"""
    SELECT
        match_id,
        home_squad_id AS squad_id,
        away_squad_id AS opponent_squad_id
    FROM matches_{season}
    WHERE 1=1
        {round_filter}
    UNION ALL
    SELECT
        match_id,
        away_squad_id AS squad_id,
        home_squad_id AS opponent_squad_id
    FROM matches_{season}
    WHERE 1=1
        {round_filter}
    """

    with get_connection() as conn:
        opponents = fetch_df(conn, opponent_query)

    # Join opponent's tries by edge
    # For team A, opponent's left edge tries = tries conceded to left edge by team A
    # We need the raw try counts (not percentages) for this
    # Recompute edge try counts without rolling
    round_filter_clause = f"AND m.round_number < {max_round}" if max_round else ""
    edge_tries_query = f"""
    WITH try_scores AS (
        SELECT
            sf.match_id,
            sf.squad_id AS scoring_team_id,
            ps.jumper_number,
            m.round_number,
            m.utc_start_time
        FROM score_flow_{season} sf
        INNER JOIN matches_{season} m ON sf.match_id = m.match_id
        INNER JOIN player_stats_{season} ps
            ON sf.match_id = ps.match_id
            AND sf.player_id = ps.player_id
            AND sf.squad_id = ps.squad_id
        WHERE sf.score_name = 'try'
            {round_filter_clause}
    )
    SELECT
        match_id,
        scoring_team_id AS squad_id,
        jumper_number,
        utc_start_time
    FROM try_scores
    ORDER BY scoring_team_id, utc_start_time
    """

    with get_connection() as conn:
        edge_tries = fetch_df(conn, edge_tries_query)

    if edge_tries.empty:
        logger.warning(f"No tries found for defence profile computation")
        return pd.DataFrame()

    # Classify edges
    edge_tries["edge"] = edge_tries["jumper_number"].apply(classify_jersey_to_edge)

    # Count by edge
    edge_counts = (
        edge_tries.groupby(["match_id", "squad_id", "edge"], as_index=False)
        .size()
        .rename(columns={"size": "tries_scored"})
    )

    # Pivot
    edge_pivot = edge_counts.pivot(
        index=["match_id", "squad_id"], columns="edge", values="tries_scored"
    ).fillna(0)

    for edge in ["left", "right", "middle", "other"]:
        if edge not in edge_pivot.columns:
            edge_pivot[edge] = 0

    edge_pivot = edge_pivot.reset_index()
    edge_pivot.columns.name = None

    # Join with opponents
    conceded = opponents.merge(
        edge_pivot, left_on=["match_id", "opponent_squad_id"], right_on=["match_id", "squad_id"], how="left"
    )

    # Drop opponent squad_id column from edge pivot
    conceded = conceded.drop(columns=["squad_id_y"]).rename(columns={"squad_id_x": "squad_id"})

    # Fill missing values (matches where opponent scored zero tries)
    for edge in ["left", "right", "middle", "other"]:
        conceded[edge] = conceded[edge].fillna(0)

    # Get all team-matches for proper ordering
    all_matches_query = f"""
    SELECT DISTINCT
        m.match_id,
        ts.squad_id,
        m.round_number,
        m.utc_start_time
    FROM matches_{season} m
    INNER JOIN team_stats_{season} ts ON m.match_id = ts.match_id
    WHERE 1=1
        {round_filter}
    ORDER BY ts.squad_id, m.utc_start_time, m.match_id
    """

    with get_connection() as conn:
        all_matches = fetch_df(conn, all_matches_query)

    # Merge to preserve ordering
    result = all_matches.merge(
        conceded[["match_id", "squad_id", "left", "right", "middle", "other"]],
        on=["match_id", "squad_id"],
        how="left",
    ).fillna(0)

    result["total_conceded"] = (
        result["left"] + result["right"] + result["middle"] + result["other"]
    )

    # Compute rolling sums
    for col in ["left", "right", "middle", "other", "total_conceded"]:
        result[f"{col}_rolling"] = (
            result.groupby("squad_id")[col]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # Rename to conceded_to_*
    final = result[
        [
            "match_id",
            "squad_id",
        ]
    ].copy()

    final[f"conceded_to_left_edge_rolling_{window}"] = result["left_rolling"]
    final[f"conceded_to_right_edge_rolling_{window}"] = result["right_rolling"]
    final[f"conceded_to_middle_rolling_{window}"] = result["middle_rolling"]
    final[f"conceded_to_other_rolling_{window}"] = result["other_rolling"]
    final[f"total_tries_conceded_rolling_{window}"] = result["total_conceded_rolling"]

    logger.info(
        f"Computed edge defence profiles: {len(final)} team-match observations"
    )
    return final


def add_player_edge_features(
    player_df: pd.DataFrame,
    season: int,
    max_round: int | None = None,
    window: int = 5,
) -> pd.DataFrame:
    """Add edge-specific features to player-level data.

    For each player observation, adds:
    - player_edge: which edge the player belongs to (left/right/middle/other)
    - team_edge_try_share_rolling_{window}: player's team's try % for that edge
    - opponent_edge_conceded_rolling_{window}: opponent's tries conceded to that edge
    - edge_matchup_score_rolling_{window}: interaction of team edge strength × opponent edge weakness

    Parameters
    ----------
    player_df : pd.DataFrame
        Player-level data with columns: match_id, player_id, squad_id, jumper_number
        Must include jumper_number for edge classification.
    season : int
        The season year.
    max_round : int | None
        If provided, only use data from rounds < max_round.
    window : int, default 5
        Rolling window size for edge features.

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional edge feature columns.

    Notes
    -----
    - Requires jumper_number column in player_df
    - Edge matchup score measures how well team's edge attacks vs opponent's edge defence
    """
    logger.info(
        f"Adding player edge features for {len(player_df)} observations, "
        f"season={season}, window={window}"
    )

    if "jumper_number" not in player_df.columns:
        raise ValueError("player_df must include 'jumper_number' column")

    # Classify player edges
    player_df = player_df.copy()
    player_df["player_edge"] = player_df["jumper_number"].apply(classify_jersey_to_edge)

    # Get team edge attack profiles
    attack_profiles = compute_team_edge_attack_profiles(
        season=season, max_round=max_round, window=window
    )

    if attack_profiles.empty:
        logger.warning("No attack profiles computed, returning player_df unchanged")
        return player_df

    # Get team edge defence profiles
    defence_profiles = compute_team_edge_defence_profiles(
        season=season, max_round=max_round, window=window
    )

    # Get opponent squad_id for each player observation
    round_filter = f"AND round_number < {max_round}" if max_round else ""
    opponent_query = f"""
    SELECT
        match_id,
        home_squad_id AS squad_id,
        away_squad_id AS opponent_squad_id
    FROM matches_{season}
    WHERE 1=1
        {round_filter}
    UNION ALL
    SELECT
        match_id,
        away_squad_id AS squad_id,
        home_squad_id AS opponent_squad_id
    FROM matches_{season}
    WHERE 1=1
        {round_filter}
    """

    with get_connection() as conn:
        opponents = fetch_df(conn, opponent_query)

    # Join opponent info
    player_df = player_df.merge(opponents, on=["match_id", "squad_id"], how="left")

    # Join team's edge attack profiles
    player_df = player_df.merge(
        attack_profiles, on=["match_id", "squad_id"], how="left", suffixes=("", "_team")
    )

    # Join opponent's edge defence profiles
    player_df = player_df.merge(
        defence_profiles,
        left_on=["match_id", "opponent_squad_id"],
        right_on=["match_id", "squad_id"],
        how="left",
        suffixes=("", "_opponent"),
    )

    # Drop duplicate squad_id column
    if "squad_id_opponent" in player_df.columns:
        player_df = player_df.drop(columns=["squad_id_opponent"])

    # Create edge-specific features based on player's edge
    # team_edge_try_share: what % of team tries come from this player's edge
    # opponent_edge_conceded: how many tries does opponent concede to this edge
    def get_edge_feature(row: pd.Series, feature_type: str) -> float:
        """Extract edge-specific value based on player's edge."""
        edge = row["player_edge"]
        if edge == "other":
            col_name = f"other_{'edge_try_pct' if feature_type == 'attack' else ''}_rolling_{window}"
            if feature_type == "defence":
                col_name = f"conceded_to_other_rolling_{window}"
        else:
            if feature_type == "attack":
                col_name = f"{edge}_edge_try_pct_rolling_{window}"
            else:  # defence
                col_name = f"conceded_to_{edge}_edge_rolling_{window}"

        return row.get(col_name, 0.0)

    player_df[f"team_edge_try_share_rolling_{window}"] = player_df.apply(
        lambda row: get_edge_feature(row, "attack"), axis=1
    )

    player_df[f"opponent_edge_conceded_rolling_{window}"] = player_df.apply(
        lambda row: get_edge_feature(row, "defence"), axis=1
    )

    # Edge matchup score: team edge attack strength × opponent edge defensive weakness
    # Higher = favorable matchup for this player's edge
    # Normalize by average: if team scores 50% tries from left (vs 25% avg), and opponent concedes
    # more to left edge, this is a positive signal
    player_df[f"edge_matchup_score_rolling_{window}"] = (
        player_df[f"team_edge_try_share_rolling_{window}"]
        * player_df[f"opponent_edge_conceded_rolling_{window}"]
    )

    # Clean up intermediate columns if desired (keep for transparency)
    # Drop the full attack/defence profile columns to reduce clutter
    cols_to_drop = [col for col in player_df.columns if col.endswith("_team") or col.endswith("_opponent")]
    # Actually, let's keep them for now since they don't have those suffixes

    logger.info(f"Added edge features: {player_df['player_edge'].value_counts().to_dict()}")

    return player_df
