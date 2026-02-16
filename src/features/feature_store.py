"""
Feature Store — Consolidates all features for model training.

This module joins features from:
- player_features.py
- team_features.py
- matchup_features.py
- game_context_features.py
- edge_features.py
- lineup_features.py
- odds/betfair.py

Outputs a single training-ready DataFrame with one row per (match_id, player_id).

LEAKAGE PREVENTION:
- All features use ONLY data from rounds < current_round
- Target variable is kept separate until final join
- as_of_round parameter enforces temporal cutoff
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

from src.features.player_features import compute_player_features
from src.features.team_features import compute_team_features
from src.features.matchup_features import compute_matchup_features
from src.features.game_context_features import compute_game_context_features
from src.features.edge_features import add_player_edge_features
from src.features.lineup_features import add_lineup_features_to_player_observations
from src.odds.betfair import add_betfair_odds_features

logger = logging.getLogger(__name__)


def build_feature_store(
    conn: sqlite3.Connection,
    season: int,
    as_of_round: Optional[int] = None,
    include_target: bool = True
) -> pd.DataFrame:
    """
    Build consolidated feature store for a single season.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    season : int
        Season year (e.g., 2024)
    as_of_round : int, optional
        Only include matches up to this round (for backtesting).
        If None, includes all rounds.
    include_target : bool, default=True
        Whether to include the target variable (scored_try).
        Set to False when building features for prediction.

    Returns
    -------
    pd.DataFrame
        Consolidated feature store with one row per (match_id, player_id).

    Notes
    -----
    - All features use ONLY pre-match data (no leakage)
    - Features computed with as_of_round parameter for backtesting
    - Target variable is binary: 1 if player scored >= 1 try, 0 otherwise
    """
    logger.info(f"Building feature store for {season} season (as_of_round={as_of_round})")

    # 1. Load base observations (all player-match records)
    base_df = _get_base_observations(conn, season, as_of_round)
    logger.info(f"Base observations: {len(base_df)} player-match records")

    # 2. Compute player features
    player_df = compute_player_features(conn, season, as_of_round=as_of_round)
    logger.info(f"Player features: {len(player_df)} rows")

    # 3. Compute team features (own team)
    team_df = compute_team_features(conn, season, as_of_round=as_of_round)
    logger.info(f"Team features: {len(team_df)} rows")

    # 4. Create opponent team features (rename columns for clarity)
    opponent_team_df = team_df.copy()
    opponent_team_df.columns = [
        col if col == 'match_id'
        else 'opponent_squad_id' if col == 'squad_id'
        else f'opponent_{col}'
        for col in opponent_team_df.columns
    ]
    logger.info(f"Opponent team features: {len(opponent_team_df)} rows")

    # 5. Compute matchup features
    matchup_df = compute_matchup_features(conn, season, as_of_round=as_of_round)
    logger.info(f"Matchup features: {len(matchup_df)} rows")

    # 6. Compute game context features
    game_context_df = compute_game_context_features(conn, season, as_of_round=as_of_round)
    logger.info(f"Game context features: {len(game_context_df)} rows")

    # 7. Join the core features on (match_id, player_id) or (match_id, squad_id)
    # Strategy: Keep only essential columns from base_df, let feature DataFrames provide the rest
    # Need opponent_squad_id for opponent team features merge
    df = base_df[['match_id', 'player_id', 'opponent_squad_id', 'tries']].copy()

    # Join player features (provides squad_id, opponent_squad_id, round_number, is_home, position, etc.)
    df = df.merge(
        player_df,
        on=['match_id', 'player_id'],
        how='left'
    )

    # Join own team features (join on match_id + squad_id)
    # Extract squad_id from player features for joining
    df = df.merge(
        team_df,
        on=['match_id', 'squad_id'],
        how='left',
        suffixes=('', '_team_own')
    )

    # Join opponent team features (join on match_id + opponent_squad_id)
    # opponent_squad_id comes from player features
    df = df.merge(
        opponent_team_df,
        on=['match_id', 'opponent_squad_id'],
        how='left',
        suffixes=('', '_team_opp')
    )

    # Join matchup features
    df = df.merge(
        matchup_df,
        on=['match_id', 'player_id'],
        how='left',
        suffixes=('', '_matchup')
    )

    # Join game context features
    df = df.merge(
        game_context_df,
        on=['match_id', 'player_id'],
        how='left',
        suffixes=('', '_context')
    )

    # 8. Add edge features (augments the DataFrame)
    df = add_player_edge_features(df, season=season, max_round=as_of_round, window=5)
    logger.info(f"After edge features: {len(df)} rows")

    # 9. Add lineup features (augments the DataFrame)
    df = add_lineup_features_to_player_observations(df, conn, season, as_of_round=as_of_round, window=5)
    logger.info(f"After lineup features: {len(df)} rows")

    # 10. Add Betfair odds features (augments the DataFrame)
    df = add_betfair_odds_features(df, conn, season)
    logger.info(f"After odds features: {len(df)} rows")

    # Add season column
    df['season'] = season

    # Add target variable if requested
    if include_target:
        df['scored_try'] = (df['tries'] >= 1).astype(int)

    logger.info(f"Final feature store: {len(df)} rows × {len(df.columns)} columns")

    # Validate output
    _validate_feature_store(df, season, include_target)

    return df


def _get_base_observations(
    conn: sqlite3.Connection,
    season: int,
    as_of_round: Optional[int] = None
) -> pd.DataFrame:
    """
    Get base player-match observations with metadata.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    season : int
        Season year
    as_of_round : int, optional
        If provided, return observations for ONLY this round (for weekly updates).
        If None, return observations for ALL rounds in the season.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, player_id, squad_id, opponent_squad_id,
                 round_number, is_home, tries (for target)
    """
    round_filter = f"AND m.round_number = {as_of_round}" if as_of_round else ""

    query = f"""
    SELECT
        ps.match_id,
        ps.player_id,
        ps.squad_id,
        ps.opponent_squad_id,
        m.round_number,
        CASE WHEN ps.side = 'home' THEN 1 ELSE 0 END as is_home,
        ps.tries
    FROM player_stats_{season} ps
    JOIN matches_{season} m ON ps.match_id = m.match_id
    WHERE m.match_type = 'H'  -- H = Home and Away season (regular season)
    {round_filter}
    ORDER BY m.round_number, ps.match_id, ps.player_id
    """

    df = pd.read_sql_query(query, conn)
    return df


def _validate_feature_store(df: pd.DataFrame, season: int, include_target: bool):
    """
    Validate feature store data quality.

    Raises
    ------
    ValueError
        If validation fails
    """
    # Check for duplicates
    duplicates = df.duplicated(subset=['match_id', 'player_id']).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate (match_id, player_id) pairs")

    # Check for missing match_id or player_id
    if df['match_id'].isna().any():
        raise ValueError("Found missing match_id values")
    if df['player_id'].isna().any():
        raise ValueError("Found missing player_id values")

    # Check target variable if included
    if include_target:
        if 'scored_try' not in df.columns:
            raise ValueError("Target variable 'scored_try' not found")
        if df['scored_try'].isna().any():
            raise ValueError("Target variable has missing values")
        if not df['scored_try'].isin([0, 1]).all():
            raise ValueError("Target variable must be binary (0 or 1)")

    # Check season column
    if 'season' not in df.columns:
        raise ValueError("Season column not found")
    if (df['season'] != season).any():
        raise ValueError(f"Season mismatch: expected {season}")

    logger.info(f"✅ Feature store validation passed for {season} season")


def save_feature_store(df: pd.DataFrame, output_path: str):
    """
    Save feature store to Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        Feature store DataFrame
    output_path : str
        Path to output Parquet file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved feature store to {output_path} ({file_size_mb:.2f} MB)")


def load_feature_store(input_path: str) -> pd.DataFrame:
    """
    Load feature store from Parquet format.

    Parameters
    ----------
    input_path : str
        Path to Parquet file

    Returns
    -------
    pd.DataFrame
        Feature store DataFrame
    """
    df = pd.read_parquet(input_path, engine='pyarrow')
    logger.info(f"✅ Loaded feature store from {input_path}: {len(df)} rows × {len(df.columns)} columns")
    return df


def build_multi_season_feature_store(
    conn: sqlite3.Connection,
    seasons: List[int],
    output_dir: str,
    as_of_round: Optional[int] = None,
    save_combined: bool = True
) -> Dict[int, pd.DataFrame]:
    """
    Build feature stores for multiple seasons.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    seasons : List[int]
        List of season years (e.g., [2024, 2025])
    output_dir : str
        Directory to save Parquet files
    as_of_round : int, optional
        Only include matches up to this round (applied to all seasons)
    save_combined : bool, default=True
        Whether to save a combined multi-season file

    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping season → DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    season_dfs = {}

    for season in seasons:
        logger.info(f"Building feature store for {season} season...")
        df = build_feature_store(conn, season, as_of_round)
        season_dfs[season] = df

        # Save individual season file
        output_path = output_dir / f"feature_store_{season}.parquet"
        save_feature_store(df, output_path)

    # Save combined file if requested
    if save_combined and len(season_dfs) > 1:
        combined_df = pd.concat(season_dfs.values(), ignore_index=True)
        combined_path = output_dir / "feature_store_combined.parquet"
        save_feature_store(combined_df, combined_path)
        logger.info(f"✅ Combined feature store: {len(combined_df)} total rows")

    return season_dfs


def get_feature_metadata() -> pd.DataFrame:
    """
    Get metadata about all features in the feature store.

    Returns
    -------
    pd.DataFrame
        Columns: feature_name, module, description, feature_type
    """
    metadata = [
        # Base columns
        {'feature_name': 'match_id', 'module': 'base', 'description': 'Match identifier', 'feature_type': 'id'},
        {'feature_name': 'player_id', 'module': 'base', 'description': 'Player identifier', 'feature_type': 'id'},
        {'feature_name': 'squad_id', 'module': 'base', 'description': "Player's team", 'feature_type': 'id'},
        {'feature_name': 'opponent_squad_id', 'module': 'base', 'description': 'Opponent team', 'feature_type': 'id'},
        {'feature_name': 'round_number', 'module': 'base', 'description': 'Round number', 'feature_type': 'metadata'},
        {'feature_name': 'is_home', 'module': 'base', 'description': 'Home field advantage', 'feature_type': 'feature'},
        {'feature_name': 'season', 'module': 'base', 'description': 'Season year', 'feature_type': 'metadata'},
        {'feature_name': 'scored_try', 'module': 'base', 'description': 'Target: 1 if scored try, 0 otherwise', 'feature_type': 'target'},

        # Player features (actual column names from compute_player_features)
        {'feature_name': 'rolling_tries_3', 'module': 'player', 'description': 'Player tries (last 3 matches)', 'feature_type': 'feature'},
        {'feature_name': 'rolling_try_rate_3', 'module': 'player', 'description': 'Player try rate (last 3)', 'feature_type': 'feature'},
        {'feature_name': 'jumper_number', 'module': 'player', 'description': 'Jersey number', 'feature_type': 'feature'},
        {'feature_name': 'position_code', 'module': 'player', 'description': 'Position code', 'feature_type': 'feature'},
        {'feature_name': 'position_label', 'module': 'player', 'description': 'Position label', 'feature_type': 'feature'},
        {'feature_name': 'position_group', 'module': 'player', 'description': 'Position group', 'feature_type': 'feature'},
        {'feature_name': 'is_starter', 'module': 'player', 'description': '1 if jersey 1-13, 0 otherwise', 'feature_type': 'feature'},

        # Team features (actual column names from compute_team_features)
        {'feature_name': 'rolling_tries_5', 'module': 'team', 'description': 'Team tries scored (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'rolling_tries_conceded_5', 'module': 'team', 'description': 'Team tries conceded (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'rolling_completion_rate_percentage_5', 'module': 'team', 'description': 'Team completion % (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'opponent_rolling_tries_5', 'module': 'team', 'description': 'Opponent tries scored (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'opponent_rolling_tries_conceded_5', 'module': 'team', 'description': 'Opponent tries conceded (last 5)', 'feature_type': 'feature'},

        # Game context features (actual column names)
        {'feature_name': 'expected_team_tries_5', 'module': 'game_context', 'description': 'Team attack × opponent defence weakness', 'feature_type': 'feature'},
        {'feature_name': 'player_try_share_5', 'module': 'game_context', 'description': 'Player tries / team tries (last 5)', 'feature_type': 'feature'},

        # Edge features (actual column names)
        {'feature_name': 'player_edge', 'module': 'edge', 'description': 'Left/Right/Middle/Other edge classification', 'feature_type': 'feature'},
        {'feature_name': 'left_edge_try_pct_rolling_5', 'module': 'edge', 'description': 'Team left edge try % (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'right_edge_try_pct_rolling_5', 'module': 'edge', 'description': 'Team right edge try % (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'conceded_to_left_edge_rolling_5', 'module': 'edge', 'description': 'Opponent tries conceded to left edge (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'conceded_to_right_edge_rolling_5', 'module': 'edge', 'description': 'Opponent tries conceded to right edge (last 5)', 'feature_type': 'feature'},
        {'feature_name': 'edge_matchup_score_rolling_5', 'module': 'edge', 'description': 'Edge attack strength × opponent edge weakness', 'feature_type': 'feature'},

        # Matchup features (actual column names)
        {'feature_name': 'player_vs_opponent_tries', 'module': 'matchup', 'description': 'Player tries vs this opponent (career)', 'feature_type': 'feature'},
        {'feature_name': 'player_vs_opponent_try_rate', 'module': 'matchup', 'description': 'Player try rate vs this opponent', 'feature_type': 'feature'},
        {'feature_name': 'player_vs_opponent_games', 'module': 'matchup', 'description': 'Games vs this opponent', 'feature_type': 'feature'},

        # Lineup features (actual column names)
        {'feature_name': 'playmaker_quality_rolling_5', 'module': 'lineup', 'description': 'Teammate try assists (halves + fullback, last 5)', 'feature_type': 'feature'},
        {'feature_name': 'lineup_stability_pct', 'module': 'lineup', 'description': '% unchanged from previous round', 'feature_type': 'feature'},
        {'feature_name': 'player_was_in_prev_lineup', 'module': 'lineup', 'description': '1 if in previous lineup, 0 if new', 'feature_type': 'feature'},

        # Odds features
        {'feature_name': 'betfair_closing_odds', 'module': 'odds', 'description': 'Betfair TO_SCORE closing decimal odds', 'feature_type': 'feature'},
        {'feature_name': 'betfair_implied_prob', 'module': 'odds', 'description': '1 / closing_odds', 'feature_type': 'feature'},
        {'feature_name': 'betfair_spread', 'module': 'odds', 'description': 'Best back - best lay price', 'feature_type': 'feature'},
        {'feature_name': 'betfair_total_matched_volume', 'module': 'odds', 'description': 'Total $ matched', 'feature_type': 'feature'},
    ]

    return pd.DataFrame(metadata)


def get_train_val_split(
    df: pd.DataFrame,
    train_seasons: List[int],
    val_seasons: List[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature store into train/validation sets by season.

    Parameters
    ----------
    df : pd.DataFrame
        Combined feature store (multiple seasons)
    train_seasons : List[int]
        Seasons to use for training
    val_seasons : List[int]
        Seasons to use for validation

    Returns
    -------
    train_df : pd.DataFrame
        Training set
    val_df : pd.DataFrame
        Validation set

    Notes
    -----
    This is a TEMPORAL split — validation must come after training chronologically.
    """
    train_df = df[df['season'].isin(train_seasons)].copy()
    val_df = df[df['season'].isin(val_seasons)].copy()

    logger.info(f"Train/val split: {len(train_df)} train rows, {len(val_df)} val rows")
    logger.info(f"Train seasons: {train_seasons}, Val seasons: {val_seasons}")

    return train_df, val_df
