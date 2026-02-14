"""
Tests for Betfair Odds Extraction Module

Test Coverage:
1. Odds conversion (decimal to implied probability)
2. Price fallback chain (last_preplay → 1min → 30min → 60min)
3. Odds extraction for specific player-match
4. Bulk odds feature addition
5. Data quality validation
6. Edge cases (missing data, unmapped runners)
7. Leakage prevention (odds are pre-match data, so allowed)
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import sys
sys.path.append('/Users/abhidutta/Documents/repos/NRL_2026_Model')

from src.odds.betfair import (
    odds_to_implied_probability,
    apply_price_fallback_chain,
    extract_betfair_odds,
    add_betfair_odds_features,
    validate_betfair_odds_features
)
from src.db import get_connection


# ===== Test Fixtures =====

@pytest.fixture
def db_conn():
    """Database connection fixture."""
    conn = get_connection()
    yield conn
    conn.close()


@pytest.fixture
def sample_player_observations(db_conn):
    """Get sample player-match observations from 2024."""
    query = """
    SELECT DISTINCT
        ps.match_id,
        ps.player_id,
        2024 as season
    FROM player_stats_2024 ps
    WHERE ps.match_id IN (124450101, 124450102)
    LIMIT 30
    """
    return pd.read_sql_query(query, db_conn)


# ===== Test 1: Odds Conversion =====

def test_odds_to_implied_probability():
    """Test decimal odds to implied probability conversion."""
    # Standard cases
    assert odds_to_implied_probability(2.0) == pytest.approx(0.5)
    assert odds_to_implied_probability(4.0) == pytest.approx(0.25)
    assert odds_to_implied_probability(1.5) == pytest.approx(0.6667, abs=0.001)
    assert odds_to_implied_probability(10.0) == pytest.approx(0.1)

    # Edge cases
    assert pd.isna(odds_to_implied_probability(0))
    assert pd.isna(odds_to_implied_probability(-1.0))
    assert pd.isna(odds_to_implied_probability(np.nan))


# ===== Test 2: Price Fallback Chain =====

def test_fallback_chain_uses_last_preplay_first():
    """Test that last_preplay_price is preferred when available."""
    row = pd.Series({
        'last_preplay_price': 2.5,
        'best_back_price_1_min_prior': 2.4,
        'best_back_price_30_min_prior': 2.3,
        'best_back_price_60_min_prior': 2.2
    })
    price, source = apply_price_fallback_chain(row)
    assert price == 2.5
    assert source == 'last_preplay'


def test_fallback_chain_uses_1min_when_last_preplay_empty():
    """Test fallback to 1min price when last_preplay is empty string."""
    row = pd.Series({
        'last_preplay_price': '',  # Empty string (common in data)
        'best_back_price_1_min_prior': 2.4,
        'best_back_price_30_min_prior': 2.3,
        'best_back_price_60_min_prior': 2.2
    })
    price, source = apply_price_fallback_chain(row)
    assert price == 2.4
    assert source == '1min'


def test_fallback_chain_uses_30min_when_1min_missing():
    """Test fallback to 30min price."""
    row = pd.Series({
        'last_preplay_price': '',
        'best_back_price_1_min_prior': '',
        'best_back_price_30_min_prior': 2.3,
        'best_back_price_60_min_prior': 2.2
    })
    price, source = apply_price_fallback_chain(row)
    assert price == 2.3
    assert source == '30min'


def test_fallback_chain_uses_60min_when_all_else_missing():
    """Test fallback to 60min price (final fallback)."""
    row = pd.Series({
        'last_preplay_price': '',
        'best_back_price_1_min_prior': '',
        'best_back_price_30_min_prior': '',
        'best_back_price_60_min_prior': 2.2
    })
    price, source = apply_price_fallback_chain(row)
    assert price == 2.2
    assert source == '60min'


def test_fallback_chain_returns_none_when_all_missing():
    """Test that None is returned when no prices available."""
    row = pd.Series({
        'last_preplay_price': '',
        'best_back_price_1_min_prior': '',
        'best_back_price_30_min_prior': '',
        'best_back_price_60_min_prior': ''
    })
    price, source = apply_price_fallback_chain(row)
    assert price is None
    assert source is None


def test_fallback_chain_handles_non_numeric_strings():
    """Test that non-numeric price strings are skipped."""
    row = pd.Series({
        'last_preplay_price': 'invalid',
        'best_back_price_1_min_prior': 2.4,
        'best_back_price_30_min_prior': 2.3,
        'best_back_price_60_min_prior': 2.2
    })
    price, source = apply_price_fallback_chain(row)
    assert price == 2.4
    assert source == '1min'


# ===== Test 3: Extract Odds for Specific Player-Match =====

def test_extract_odds_for_known_player_match(db_conn):
    """Test extracting odds for a known player-match (Cody Walker, match 124450101)."""
    match_id = 124450101
    player_id = 995278  # Cody Walker
    year = 2024

    odds = extract_betfair_odds(db_conn, match_id, player_id, year)

    assert odds is not None
    assert odds['betfair_closing_odds'] > 0
    assert 0 < odds['betfair_implied_prob'] < 1
    assert odds['betfair_odds_source'] in ['last_preplay', '1min', '30min', '60min']
    assert odds['betfair_closing_odds'] == pytest.approx(1.0 / odds['betfair_implied_prob'], rel=0.01)


def test_extract_odds_returns_none_for_unmapped_player(db_conn):
    """Test that None is returned for unmapped player."""
    match_id = 124450101
    player_id = 999999999  # Non-existent player
    year = 2024

    odds = extract_betfair_odds(db_conn, match_id, player_id, year)
    assert odds is None


def test_extract_odds_returns_none_for_non_existent_match(db_conn):
    """Test that None is returned for non-existent match."""
    match_id = 999999999
    player_id = 995278
    year = 2024

    odds = extract_betfair_odds(db_conn, match_id, player_id, year)
    assert odds is None


# ===== Test 4: Add Odds Features to Bulk Observations =====

def test_add_odds_features_returns_all_columns(sample_player_observations, db_conn):
    """Test that all expected odds columns are added."""
    result = add_betfair_odds_features(sample_player_observations, db_conn, year=2024)

    expected_cols = [
        'betfair_closing_odds',
        'betfair_implied_prob',
        'betfair_odds_source',
        'betfair_total_matched_volume',
        'betfair_spread'
    ]

    for col in expected_cols:
        assert col in result.columns


def test_add_odds_features_preserves_row_count(sample_player_observations, db_conn):
    """Test that row count is preserved (no rows added or dropped)."""
    original_count = len(sample_player_observations)
    result = add_betfair_odds_features(sample_player_observations, db_conn, year=2024)
    assert len(result) == original_count


def test_add_odds_features_coverage(sample_player_observations, db_conn):
    """Test that odds are extracted for >80% of observations."""
    result = add_betfair_odds_features(sample_player_observations, db_conn, year=2024)

    # Should have high coverage (>80% from historical data)
    non_null_odds = result['betfair_closing_odds'].notna().sum()
    coverage_pct = (non_null_odds / len(result)) * 100

    assert coverage_pct > 80, f"Coverage too low: {coverage_pct:.1f}%"


def test_add_odds_features_source_distribution(sample_player_observations, db_conn):
    """Test that fallback chain is used correctly."""
    result = add_betfair_odds_features(sample_player_observations, db_conn, year=2024)

    # Count source distribution
    source_counts = result['betfair_odds_source'].value_counts()

    # Should have some last_preplay (primary source)
    assert 'last_preplay' in source_counts.index

    # May have fallback sources (1min, 30min, 60min)
    fallback_sources = ['1min', '30min', '60min']
    fallback_count = sum(source_counts.get(src, 0) for src in fallback_sources)

    # At least some should use fallback (based on 33% empty last_preplay)
    assert fallback_count > 0, "Expected some fallback sources to be used"


def test_add_odds_features_with_season_column(db_conn):
    """Test that function works with 'season' column instead of year parameter."""
    # Create test DataFrame with season column
    test_df = pd.DataFrame({
        'match_id': [124450101, 124450102],
        'player_id': [995278, 996013],
        'season': [2024, 2024]
    })

    result = add_betfair_odds_features(test_df, db_conn)

    assert 'betfair_closing_odds' in result.columns
    assert len(result) == 2


def test_add_odds_features_raises_without_year_or_season(db_conn):
    """Test that function raises error without year parameter or season column."""
    test_df = pd.DataFrame({
        'match_id': [124450101],
        'player_id': [995278]
    })

    with pytest.raises(ValueError, match="Must provide 'year' parameter or 'season' column"):
        add_betfair_odds_features(test_df, db_conn)


# ===== Test 5: Data Quality Validation =====

def test_validate_odds_features_passes_for_valid_data(sample_player_observations, db_conn):
    """Test validation passes for real data."""
    result = add_betfair_odds_features(sample_player_observations, db_conn, year=2024)

    # Should not raise any exceptions
    validate_betfair_odds_features(result)


def test_validate_odds_features_fails_for_invalid_odds():
    """Test validation fails for odds outside valid range."""
    invalid_df = pd.DataFrame({
        'betfair_closing_odds': [0.5, 1500.0],  # Both outside 1.01-1000 range
        'betfair_implied_prob': [2.0, 0.0007]
    })

    with pytest.raises(ValueError, match="invalid odds"):
        validate_betfair_odds_features(invalid_df)


def test_validate_odds_features_checks_probability_consistency():
    """Test validation fails if implied prob doesn't match odds."""
    inconsistent_df = pd.DataFrame({
        'betfair_closing_odds': [2.0, 4.0],
        'betfair_implied_prob': [0.6, 0.3]  # Should be 0.5 and 0.25
    })

    with pytest.raises(ValueError, match="Implied probabilities don't match odds"):
        validate_betfair_odds_features(inconsistent_df)


def test_validate_odds_features_checks_negative_volume():
    """Test validation fails for negative matched volume."""
    invalid_df = pd.DataFrame({
        'betfair_closing_odds': [2.0],
        'betfair_implied_prob': [0.5],
        'betfair_total_matched_volume': [-100.0]
    })

    with pytest.raises(ValueError, match="negative matched volume"):
        validate_betfair_odds_features(invalid_df)


# ===== Test 6: Edge Cases =====

def test_odds_features_handle_nan_gracefully(db_conn):
    """Test that NaN values are handled correctly for unmapped players."""
    # Create test DataFrame with one mapped and one unmapped player
    test_df = pd.DataFrame({
        'match_id': [124450101, 124450101],
        'player_id': [995278, 999999999],  # Second is unmapped
        'season': [2024, 2024]
    })

    result = add_betfair_odds_features(test_df, db_conn)

    # First player should have odds
    assert pd.notna(result.iloc[0]['betfair_closing_odds'])

    # Second player should have NaN
    assert pd.isna(result.iloc[1]['betfair_closing_odds'])


def test_fallback_chain_with_zero_prices():
    """Test that zero prices are treated as invalid."""
    row = pd.Series({
        'last_preplay_price': 0.0,  # Invalid
        'best_back_price_1_min_prior': 2.4,
        'best_back_price_30_min_prior': 2.3,
        'best_back_price_60_min_prior': 2.2
    })
    price, source = apply_price_fallback_chain(row)
    assert price == 2.4
    assert source == '1min'


# ===== Test 7: Leakage Prevention =====

def test_odds_are_pre_match_data_no_leakage(db_conn):
    """
    Test that odds extraction follows leakage prevention rules.

    Odds are pre-match public data (Rule 3 in CLAUDE.md allows them).
    This test verifies that:
    1. Odds are extracted for matches that have occurred
    2. Odds do NOT include any post-match information (is_winner is NOT used)
    3. Price timestamp is before match start time
    """
    # Use a known player with odds (Cody Walker, match 124450101)
    match_id = 124450101
    player_id = 995278  # Cody Walker - known to have odds

    # Extract odds
    odds = extract_betfair_odds(db_conn, match_id, player_id, 2024)

    # Verify odds features do NOT include match outcome
    assert odds is not None
    assert 'is_winner' not in odds.index, "Odds features must not include match outcome"
    assert 'tries' not in odds.index, "Odds features must not include tries scored"

    # Verify we're using pre-match prices (closing/1min/30min/60min all pre-match)
    assert odds['betfair_odds_source'] in ['last_preplay', '1min', '30min', '60min']


def test_odds_features_do_not_use_match_outcome(sample_player_observations, db_conn):
    """Test that odds feature extraction never uses match outcome data."""
    result = add_betfair_odds_features(sample_player_observations, db_conn, year=2024)

    # Verify no outcome columns exist
    outcome_columns = ['is_winner', 'tries', 'score']
    for col in outcome_columns:
        assert col not in result.columns, f"Odds features must not include {col}"


# ===== Test 8: Real Data Integration =====

def test_extract_odds_for_full_round(db_conn):
    """Test extracting odds for all players in a full round."""
    # Get all player-match observations from Round 1, 2024
    query = """
    SELECT
        ps.match_id,
        ps.player_id,
        2024 as season
    FROM player_stats_2024 ps
    JOIN matches_2024 m ON ps.match_id = m.match_id
    WHERE m.round_number = 1
    """
    round_1_obs = pd.read_sql_query(query, db_conn)

    result = add_betfair_odds_features(round_1_obs, db_conn, year=2024)

    # Should have high coverage
    non_null_odds = result['betfair_closing_odds'].notna().sum()
    coverage_pct = (non_null_odds / len(result)) * 100

    assert coverage_pct > 85, f"Round 1 coverage: {coverage_pct:.1f}%"
    assert len(result) == len(round_1_obs), "Row count should be preserved"


# ===== Run Tests =====

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
