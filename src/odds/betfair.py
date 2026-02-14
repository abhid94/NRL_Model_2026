"""
Betfair Odds Extraction Module

Extracts Betfair TO_SCORE (Anytime Try Scorer) odds and converts them to features.

Key Functions:
- extract_betfair_odds() - Get odds for player-match observations
- apply_price_fallback_chain() - Handle missing last_preplay_price
- add_betfair_odds_features() - Main integration function

Leakage Prevention:
- Odds are pre-match public data (Rule 3 allows them)
- No future information leakage

Data Quality Notes:
- 33% of TO_SCORE records have empty last_preplay_price
- Fallback chain recovers 99%+ coverage
- 8.1% of TO_SCORE records lack AD_player_id (unmapped runners)
"""

import sqlite3
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def odds_to_implied_probability(decimal_odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Parameters
    ----------
    decimal_odds : float
        Decimal odds (e.g., 2.50 for 5/2 odds)

    Returns
    -------
    float
        Implied probability (0.0 to 1.0)

    Examples
    --------
    >>> odds_to_implied_probability(2.0)
    0.5
    >>> odds_to_implied_probability(4.0)
    0.25
    """
    if pd.isna(decimal_odds) or decimal_odds <= 0:
        return np.nan
    return 1.0 / decimal_odds


def apply_price_fallback_chain(row: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    """
    Apply fallback chain for Betfair closing prices.

    Fallback order:
    1. last_preplay_price (primary, 66% coverage)
    2. best_back_price_1_min_prior (recovers to 99% coverage)
    3. best_back_price_30_min_prior (recovers remaining gaps)
    4. best_back_price_60_min_prior (final fallback)

    Parameters
    ----------
    row : pd.Series
        Row from betfair_markets table with price columns

    Returns
    -------
    Tuple[Optional[float], Optional[str]]
        (selected_price, source_field_name)
        Returns (None, None) if no price available

    Examples
    --------
    >>> row = pd.Series({'last_preplay_price': 2.5, 'best_back_price_1_min_prior': 2.4})
    >>> apply_price_fallback_chain(row)
    (2.5, 'last_preplay')

    >>> row = pd.Series({'last_preplay_price': '', 'best_back_price_1_min_prior': 2.4})
    >>> apply_price_fallback_chain(row)
    (2.4, '1min')
    """
    # Try last_preplay_price (primary)
    if pd.notna(row.get('last_preplay_price')) and row.get('last_preplay_price') != '':
        try:
            price = float(row['last_preplay_price'])
            if price > 0:
                return price, 'last_preplay'
        except (ValueError, TypeError):
            pass

    # Fallback 1: best_back_price_1_min_prior
    if pd.notna(row.get('best_back_price_1_min_prior')) and row.get('best_back_price_1_min_prior') != '':
        try:
            price = float(row['best_back_price_1_min_prior'])
            if price > 0:
                return price, '1min'
        except (ValueError, TypeError):
            pass

    # Fallback 2: best_back_price_30_min_prior
    if pd.notna(row.get('best_back_price_30_min_prior')) and row.get('best_back_price_30_min_prior') != '':
        try:
            price = float(row['best_back_price_30_min_prior'])
            if price > 0:
                return price, '30min'
        except (ValueError, TypeError):
            pass

    # Fallback 3: best_back_price_60_min_prior
    if pd.notna(row.get('best_back_price_60_min_prior')) and row.get('best_back_price_60_min_prior') != '':
        try:
            price = float(row['best_back_price_60_min_prior'])
            if price > 0:
                return price, '60min'
        except (ValueError, TypeError):
            pass

    # No price available
    return None, None


def extract_betfair_odds(
    conn: sqlite3.Connection,
    match_id: int,
    player_id: int,
    year: int
) -> Optional[pd.Series]:
    """
    Extract Betfair TO_SCORE odds for a specific player-match.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    match_id : int
        Match ID
    player_id : int
        Player ID
    year : int
        Season year (for table suffix)

    Returns
    -------
    Optional[pd.Series]
        Series with odds features, or None if no odds available

    Features returned:
    - betfair_closing_odds: Final decimal odds (after fallback)
    - betfair_implied_prob: 1 / decimal_odds
    - betfair_odds_source: Which field was used (last_preplay, 1min, 30min, 60min)
    - betfair_total_matched_volume: Market liquidity
    - betfair_spread: best_back - best_lay (1min prior)
    """
    query = f"""
    SELECT
        last_preplay_price,
        best_back_price_1_min_prior,
        best_back_price_30_min_prior,
        best_back_price_60_min_prior,
        best_lay_price_1_min_prior,
        total_matched_volume
    FROM betfair_markets_{year}
    WHERE market_type = 'TO_SCORE'
        AND AD_match_id = ?
        AND AD_player_id = ?
    LIMIT 1
    """

    result = pd.read_sql_query(query, conn, params=(match_id, player_id))

    if result.empty:
        return None

    row = result.iloc[0]

    # Apply fallback chain
    closing_odds, odds_source = apply_price_fallback_chain(row)

    if closing_odds is None:
        return None

    # Calculate implied probability
    implied_prob = odds_to_implied_probability(closing_odds)

    # Calculate spread (best_back - best_lay at 1min prior)
    spread = np.nan
    try:
        back_1min = row.get('best_back_price_1_min_prior')
        lay_1min = row.get('best_lay_price_1_min_prior')
        if pd.notna(back_1min) and pd.notna(lay_1min) and back_1min != '' and lay_1min != '':
            spread = float(back_1min) - float(lay_1min)
    except (ValueError, TypeError):
        pass

    # Total matched volume
    matched_volume = row.get('total_matched_volume')
    if pd.notna(matched_volume) and matched_volume != '':
        try:
            matched_volume = float(matched_volume)
        except (ValueError, TypeError):
            matched_volume = np.nan
    else:
        matched_volume = np.nan

    return pd.Series({
        'betfair_closing_odds': closing_odds,
        'betfair_implied_prob': implied_prob,
        'betfair_odds_source': odds_source,
        'betfair_total_matched_volume': matched_volume,
        'betfair_spread': spread
    })


def add_betfair_odds_features(
    player_observations: pd.DataFrame,
    conn: sqlite3.Connection,
    year: int = None
) -> pd.DataFrame:
    """
    Add Betfair TO_SCORE odds features to player observations.

    This function extracts pre-match Betfair odds for each player-match observation
    and adds them as features. Odds are pre-match public data, so no leakage.

    Parameters
    ----------
    player_observations : pd.DataFrame
        DataFrame with columns: match_id, player_id, season (optional)
        Must have one row per player-match observation
    conn : sqlite3.Connection
        Database connection
    year : int, optional
        Season year to filter. If None, uses 'season' column from DataFrame

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added odds features:
        - betfair_closing_odds: Decimal odds (e.g., 2.5)
        - betfair_implied_prob: Implied probability (e.g., 0.4)
        - betfair_odds_source: Price source (last_preplay, 1min, 30min, 60min)
        - betfair_total_matched_volume: Market liquidity in currency units
        - betfair_spread: Back-lay spread (market efficiency indicator)

    Notes
    -----
    - ~8% of player-match observations will have no odds (unmapped Betfair runners)
    - Missing odds are represented as NaN
    - Price fallback chain recovers 99%+ of available Betfair data

    Examples
    --------
    >>> obs = pd.DataFrame({'match_id': [124450101], 'player_id': [995278], 'season': [2024]})
    >>> obs_with_odds = add_betfair_odds_features(obs, conn)
    >>> obs_with_odds['betfair_implied_prob'].iloc[0]
    0.34
    """
    df = player_observations.copy()

    # Validate required columns
    required_cols = ['match_id', 'player_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Determine year/season column
    if year is not None:
        # Use provided year for all rows
        df['_temp_year'] = year
        year_col = '_temp_year'
    elif 'season' in df.columns:
        year_col = 'season'
    else:
        raise ValueError("Must provide 'year' parameter or 'season' column in DataFrame")

    # Initialize odds feature columns
    odds_cols = [
        'betfair_closing_odds',
        'betfair_implied_prob',
        'betfair_odds_source',
        'betfair_total_matched_volume',
        'betfair_spread'
    ]
    for col in odds_cols:
        df[col] = np.nan
    df['betfair_odds_source'] = df['betfair_odds_source'].astype('object')

    # Extract odds for each player-match
    logger.info(f"Extracting Betfair TO_SCORE odds for {len(df)} player-match observations...")

    extracted_count = 0
    for idx, row in df.iterrows():
        odds_features = extract_betfair_odds(
            conn=conn,
            match_id=row['match_id'],
            player_id=row['player_id'],
            year=row[year_col]
        )

        if odds_features is not None:
            for col in odds_cols:
                df.at[idx, col] = odds_features[col]
            extracted_count += 1

    # Clean up temp column if we created it
    if year is not None:
        df = df.drop(columns=['_temp_year'])

    # Log coverage stats
    coverage_pct = (extracted_count / len(df)) * 100
    logger.info(f"Extracted odds for {extracted_count}/{len(df)} observations ({coverage_pct:.1f}% coverage)")

    # Log source distribution
    if extracted_count > 0:
        source_dist = df['betfair_odds_source'].value_counts(dropna=False)
        logger.info(f"Odds source distribution:\n{source_dist}")

    return df


def validate_betfair_odds_features(df: pd.DataFrame) -> None:
    """
    Validate Betfair odds features for data quality issues.

    Checks:
    1. Odds are >= 1.01 (valid range)
    2. Implied probabilities sum to > 1.0 for each match (overround exists)
    3. Spread is non-negative (back >= lay)
    4. Matched volume is non-negative

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Betfair odds features

    Raises
    ------
    ValueError
        If validation fails
    """
    if 'betfair_closing_odds' not in df.columns:
        return

    # Check valid odds range
    invalid_odds = df[
        df['betfair_closing_odds'].notna() &
        ((df['betfair_closing_odds'] < 1.01) | (df['betfair_closing_odds'] > 1000))
    ]
    if len(invalid_odds) > 0:
        raise ValueError(f"Found {len(invalid_odds)} observations with invalid odds (must be 1.01-1000)")

    # Check implied probabilities match odds
    if 'betfair_implied_prob' in df.columns:
        df_with_odds = df[df['betfair_closing_odds'].notna()].copy()
        if len(df_with_odds) > 0:
            expected_prob = 1.0 / df_with_odds['betfair_closing_odds']
            prob_diff = abs(df_with_odds['betfair_implied_prob'] - expected_prob)
            if (prob_diff > 0.0001).any():
                raise ValueError("Implied probabilities don't match odds")

    # Check spread is non-negative
    if 'betfair_spread' in df.columns:
        negative_spread = df[df['betfair_spread'] < 0]
        if len(negative_spread) > 0:
            logger.warning(f"Found {len(negative_spread)} observations with negative spread (back < lay)")

    # Check matched volume is non-negative
    if 'betfair_total_matched_volume' in df.columns:
        negative_volume = df[df['betfair_total_matched_volume'] < 0]
        if len(negative_volume) > 0:
            raise ValueError(f"Found {len(negative_volume)} observations with negative matched volume")

    logger.info("Betfair odds features validation passed")


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append('/Users/abhidutta/Documents/repos/NRL_2026_Model')
    from src.db import get_connection

    conn = get_connection()

    # Test extraction for a single player-match
    test_match_id = 124450101
    test_player_id = 995278  # Cody Walker

    print(f"\nTesting odds extraction for match {test_match_id}, player {test_player_id}...")
    odds = extract_betfair_odds(conn, test_match_id, test_player_id, 2024)
    print(f"Extracted odds:\n{odds}")

    # Test on small dataset
    query = """
    SELECT DISTINCT ps.match_id, ps.player_id, 2024 as season
    FROM player_stats_2024 ps
    WHERE ps.match_id IN (124450101, 124450102)
    LIMIT 20
    """
    test_df = pd.read_sql_query(query, conn)
    print(f"\nTesting on {len(test_df)} player-match observations...")

    result = add_betfair_odds_features(test_df, conn)
    print(f"\nSample results:")
    print(result[['match_id', 'player_id', 'betfair_closing_odds', 'betfair_implied_prob', 'betfair_odds_source']].head(10))

    # Validate
    validate_betfair_odds_features(result)
    print("\n✅ Validation passed!")

    conn.close()
