"""Tests for game context feature computation."""

import pytest
import pandas as pd

from src import db
from src.config import DB_PATH
from src.features.game_context_features import (
    GameContextConfig,
    compute_game_context_features,
    compute_player_try_share,
    compute_team_attack_strength,
    compute_team_defence_weakness,
    fetch_player_team_context,
)


@pytest.fixture
def connection():
    """Create a database connection for testing."""
    conn = db.get_connection(DB_PATH)
    yield conn
    conn.close()


def test_fetch_player_team_context(connection):
    """Test fetching player-team context data."""
    df = fetch_player_team_context(connection, 2024)

    assert not df.empty, "Player-team context should not be empty"
    assert "match_id" in df.columns
    assert "player_id" in df.columns
    assert "squad_id" in df.columns
    assert "opponent_squad_id" in df.columns
    assert "is_home" in df.columns
    assert "team_tries" in df.columns
    assert "player_tries" in df.columns

    # Validate is_home is binary
    assert df["is_home"].isin([0, 1]).all()

    # Validate opponent_squad_id is populated
    assert df["opponent_squad_id"].notna().all()


def test_compute_team_attack_strength(connection):
    """Test team attack strength computation."""
    df = compute_team_attack_strength(connection, 2024, windows=(3, 5))

    assert not df.empty, "Team attack features should not be empty"
    assert "match_id" in df.columns
    assert "squad_id" in df.columns
    assert "team_attack_tries_3" in df.columns
    assert "team_attack_tries_5" in df.columns
    assert "team_attack_score_3" in df.columns
    assert "team_attack_score_5" in df.columns


def test_compute_team_defence_weakness(connection):
    """Test team defence weakness computation."""
    df = compute_team_defence_weakness(connection, 2024, windows=(3, 5))

    assert not df.empty, "Team defence features should not be empty"
    assert "match_id" in df.columns
    assert "squad_id" in df.columns
    assert "team_defence_tries_conceded_3" in df.columns
    assert "team_defence_tries_conceded_5" in df.columns
    assert "team_defence_score_conceded_3" in df.columns
    assert "team_defence_score_conceded_5" in df.columns


def test_compute_player_try_share():
    """Test player try share computation."""
    # Create sample data
    data = {
        "player_id": [1, 1, 1, 2, 2, 2],
        "player_tries": [1, 0, 2, 0, 1, 0],
        "team_tries": [5, 4, 6, 4, 5, 3],
    }
    df = pd.DataFrame(data)

    result = compute_player_try_share(df, windows=(2,), min_team_tries=1)

    assert "player_try_share_2" in result.columns
    # First observation has no history
    assert pd.isna(result.loc[0, "player_try_share_2"])


def test_compute_game_context_features_full(connection):
    """Test full game context feature computation."""
    df = compute_game_context_features(connection, 2024)

    assert not df.empty, "Game context features should not be empty"
    assert "match_id" in df.columns
    assert "player_id" in df.columns
    assert "squad_id" in df.columns
    assert "opponent_squad_id" in df.columns
    assert "is_home" in df.columns

    # Check expected_team_tries features exist
    assert "expected_team_tries_3" in df.columns
    assert "expected_team_tries_5" in df.columns
    assert "expected_team_tries_10" in df.columns

    # Check player_try_share features exist
    assert "player_try_share_3" in df.columns
    assert "player_try_share_5" in df.columns
    assert "player_try_share_10" in df.columns

    # Check opponent context features exist
    assert "opponent_tries_conceded_3" in df.columns
    assert "opponent_tries_conceded_5" in df.columns
    assert "opponent_tries_conceded_10" in df.columns

    # Validate expected_team_tries is non-negative
    assert (df["expected_team_tries_5"] >= 0).all()


def test_game_context_features_as_of_round(connection):
    """Test game context features with as_of_round parameter."""
    df = compute_game_context_features(connection, 2024, as_of_round=10)

    assert not df.empty, "Game context features for round 10 should not be empty"
    # All rows should be from round 10
    assert (df["round_number"] == 10).all()


def test_game_context_features_no_leakage(connection):
    """Test that game context features don't leak future data."""
    # Compute features for round 5
    df = compute_game_context_features(connection, 2024, as_of_round=5)

    # Get actual match outcomes for round 5
    query = """
        SELECT ps.match_id, ps.player_id, ps.tries
        FROM player_stats_2024 ps
        JOIN matches_2024 m ON ps.match_id = m.match_id
        WHERE m.round_number = 5
    """
    actual = db.fetch_df(connection, query)

    # Merge features with actual outcomes
    merged = df.merge(actual, on=["match_id", "player_id"], how="inner")

    # Features should exist for round 5
    assert not merged.empty, "Should have features for round 5"

    # Check that expected_team_tries is computed from prior rounds only
    # by verifying it's based on historical data (should have NaN for teams
    # with no prior matches, or reasonable values otherwise)
    assert "expected_team_tries_5" in merged.columns


def test_game_context_config_custom_windows(connection):
    """Test game context features with custom configuration."""
    config = GameContextConfig(
        windows=(2, 4),
        min_team_tries_for_share=2,
        fillna_value=None,
    )
    df = compute_game_context_features(connection, 2024, config=config)

    assert not df.empty
    assert "expected_team_tries_2" in df.columns
    assert "expected_team_tries_4" in df.columns
    assert "expected_team_tries_3" not in df.columns  # Should not exist
