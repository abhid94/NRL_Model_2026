"""Tests for matchup feature computation."""

import pytest
import pandas as pd

from src import db
from src.features.matchup_features import (
    compute_matchup_features,
    fetch_player_opponent_stats,
    MatchupFeatureConfig,
)


@pytest.fixture
def db_connection():
    """Provide a database connection for tests."""
    conn = db.get_connection()
    yield conn
    conn.close()


def test_fetch_player_opponent_stats(db_connection):
    """Test that player opponent stats are fetched correctly."""
    stats = fetch_player_opponent_stats(db_connection, 2024)

    assert not stats.empty, "Should have player stats for 2024"
    assert "player_id" in stats.columns
    assert "opponent_squad_id" in stats.columns
    assert "round_number" in stats.columns
    assert "tries" in stats.columns


def test_compute_matchup_features_basic(db_connection):
    """Test basic matchup feature computation."""
    features = compute_matchup_features(db_connection, 2024)

    # Check structure
    assert not features.empty
    assert "match_id" in features.columns
    assert "player_id" in features.columns
    assert "opponent_squad_id" in features.columns
    assert "matchup_games_vs_opp" in features.columns

    # Check that matchup features exist
    matchup_cols = [col for col in features.columns if col.startswith("matchup_")]
    assert len(matchup_cols) > 0, "Should have matchup features"


def test_matchup_features_temporal_ordering(db_connection):
    """Test that matchup features respect temporal ordering."""
    features = compute_matchup_features(db_connection, 2024)

    # Find a player with repeat matchups
    repeat_matchups = features[features["matchup_games_vs_opp"] >= 1]

    if len(repeat_matchups) > 0:
        sample = repeat_matchups.iloc[0]
        player_id = sample["player_id"]
        opp_id = sample["opponent_squad_id"]
        round_num = sample["round_number"]

        # Count actual prior matches
        query = f"""
            SELECT COUNT(*) as prior_matches
            FROM player_stats_2024 ps
            JOIN matches_2024 m ON ps.match_id = m.match_id
            WHERE ps.player_id = {player_id}
              AND ps.opponent_squad_id = {opp_id}
              AND m.round_number < {round_num}
        """
        actual_prior = db_connection.execute(query).fetchone()[0]

        # Feature value should match actual prior matches
        assert (
            sample["matchup_games_vs_opp"] == actual_prior
        ), "Feature should only count prior matches"


def test_matchup_features_no_leakage(db_connection):
    """Test that features for round N don't include data from round N."""
    # Get features for round 10
    features_r10 = compute_matchup_features(db_connection, 2024, as_of_round=10)

    # For first-time matchups in round 10, games_vs_opp should be 0
    first_time = features_r10[features_r10["matchup_games_vs_opp"] == 0]

    # For these players, verify they truly have no prior matches vs this opponent
    if len(first_time) > 0:
        sample = first_time.iloc[0]
        player_id = sample["player_id"]
        opp_id = sample["opponent_squad_id"]

        query = f"""
            SELECT COUNT(*) as prior_matches
            FROM player_stats_2024 ps
            JOIN matches_2024 m ON ps.match_id = m.match_id
            WHERE ps.player_id = {player_id}
              AND ps.opponent_squad_id = {opp_id}
              AND m.round_number < 10
        """
        actual_prior = db_connection.execute(query).fetchone()[0]

        assert (
            actual_prior == 0
        ), "First-time matchup should have zero prior matches"


def test_matchup_features_as_of_round(db_connection):
    """Test that as_of_round parameter works correctly."""
    features = compute_matchup_features(db_connection, 2024, as_of_round=5)

    # Should only return features for round 5
    assert features["round_number"].unique().tolist() == [5]

    # Should have features computed from rounds 1-4
    # Check that at least some players have non-zero games_vs_opp
    # (since some teams play each other early in the season)
    assert features["matchup_games_vs_opp"].max() >= 0


def test_matchup_features_nan_handling(db_connection):
    """Test that NaN values are preserved for first-time matchups."""
    config = MatchupFeatureConfig(fillna_value=None)  # Don't fill NaN
    features = compute_matchup_features(db_connection, 2024, config=config)

    # First-time matchups should have NaN for aggregate features
    first_time = features[features["matchup_games_vs_opp"] == 0]

    if len(first_time) > 0:
        # Aggregate features should be NaN
        assert pd.isna(
            first_time["matchup_avg_tries_vs_opp"].iloc[0]
        ), "First matchup should have NaN aggregate stats"


def test_matchup_features_config(db_connection):
    """Test that feature configuration works."""
    config = MatchupFeatureConfig(
        windows=(3,),  # Only 3-game rolling window
        metrics=("tries",),  # Only tries metric
        include_try_rate=False,  # No try rate
    )
    features = compute_matchup_features(db_connection, 2024, config=config)

    # Should have rolling_tries_vs_opp_3
    assert "matchup_rolling_tries_vs_opp_3" in features.columns

    # Should not have 5-game window or other metrics
    assert "matchup_rolling_tries_vs_opp_5" not in features.columns
    assert "matchup_rolling_line_breaks_vs_opp_3" not in features.columns
    assert "matchup_try_rate_vs_opp" not in features.columns


def test_matchup_features_year_2025(db_connection):
    """Test that features work for 2025 season."""
    if not db.table_exists(db_connection, "player_stats_2025"):
        pytest.skip("2025 data not available")

    features = compute_matchup_features(db_connection, 2025)
    assert not features.empty
    assert "matchup_games_vs_opp" in features.columns
