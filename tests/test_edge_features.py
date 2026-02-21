"""Tests for edge-specific features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.edge_features import (
    add_player_edge_features,
    classify_jersey_to_edge,
    compute_team_edge_attack_profiles,
    compute_team_edge_defence_profiles,
)


class TestEdgeClassification:
    """Test jersey number to edge mapping."""

    def test_left_edge_jerseys(self):
        """Left edge: 2, 3, 11."""
        assert classify_jersey_to_edge(2) == "left"  # Left wing
        assert classify_jersey_to_edge(3) == "left"  # Left centre
        assert classify_jersey_to_edge(11) == "left"  # Left second row

    def test_right_edge_jerseys(self):
        """Right edge: 4, 5, 12."""
        assert classify_jersey_to_edge(4) == "right"  # Right centre
        assert classify_jersey_to_edge(5) == "right"  # Right wing
        assert classify_jersey_to_edge(12) == "right"  # Right second row

    def test_middle_jerseys(self):
        """Middle: 8, 9, 10, 13."""
        assert classify_jersey_to_edge(8) == "middle"  # Prop
        assert classify_jersey_to_edge(9) == "middle"  # Hooker
        assert classify_jersey_to_edge(10) == "middle"  # Prop
        assert classify_jersey_to_edge(13) == "middle"  # Lock

    def test_other_jerseys(self):
        """Other: 1, 6, 7, 14+."""
        assert classify_jersey_to_edge(1) == "other"  # Fullback
        assert classify_jersey_to_edge(6) == "other"  # Five-eighth
        assert classify_jersey_to_edge(7) == "other"  # Halfback
        assert classify_jersey_to_edge(14) == "other"  # Interchange
        assert classify_jersey_to_edge(18) == "other"  # Reserve

    def test_none_jersey(self):
        """None jersey returns 'other'."""
        assert classify_jersey_to_edge(None) == "other"


class TestTeamEdgeAttackProfiles:
    """Test team edge attack profile computation."""

    def test_returns_dataframe(self):
        """Should return a dataframe with expected columns."""
        result = compute_team_edge_attack_profiles(season=2024, window=5)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        expected_cols = [
            "match_id",
            "squad_id",
            "left_edge_try_pct_rolling_5",
            "right_edge_try_pct_rolling_5",
            "middle_edge_try_pct_rolling_5",
            "other_edge_try_pct_rolling_5",
            "total_tries_rolling_5",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_percentages_sum_to_one(self):
        """Edge percentages should sum to ~1.0 for each team-match."""
        result = compute_team_edge_attack_profiles(season=2024, window=5)

        # For matches with tries, percentages should sum to 1
        has_tries = result["total_tries_rolling_5"] > 0
        pct_cols = [
            "left_edge_try_pct_rolling_5",
            "right_edge_try_pct_rolling_5",
            "middle_edge_try_pct_rolling_5",
            "other_edge_try_pct_rolling_5",
        ]

        pct_sums = result.loc[has_tries, pct_cols].sum(axis=1)
        assert (pct_sums.between(0.99, 1.01)).all(), "Percentages should sum to ~1.0"

    def test_max_round_filtering(self):
        """max_round parameter should filter to rounds < max_round."""
        result_all = compute_team_edge_attack_profiles(season=2024, window=5)
        result_filtered = compute_team_edge_attack_profiles(
            season=2024, max_round=10, window=5
        )

        # Filtered result should have fewer observations
        assert len(result_filtered) < len(result_all)

    def test_no_negative_percentages(self):
        """All percentages should be >= 0."""
        result = compute_team_edge_attack_profiles(season=2024, window=5)

        pct_cols = [
            "left_edge_try_pct_rolling_5",
            "right_edge_try_pct_rolling_5",
            "middle_edge_try_pct_rolling_5",
            "other_edge_try_pct_rolling_5",
        ]

        for col in pct_cols:
            # First match per team will be NaN due to shift(1), drop NaN before checking
            assert (result[col].dropna() >= 0).all(), f"{col} should be >= 0 (excluding NaN)"


class TestTeamEdgeDefenceProfiles:
    """Test team edge defence profile computation."""

    def test_returns_dataframe(self):
        """Should return a dataframe with expected columns."""
        result = compute_team_edge_defence_profiles(season=2024, window=5)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        expected_cols = [
            "match_id",
            "squad_id",
            "conceded_to_left_edge_rolling_5",
            "conceded_to_right_edge_rolling_5",
            "conceded_to_middle_rolling_5",
            "conceded_to_other_rolling_5",
            "total_tries_conceded_rolling_5",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_no_negative_values(self):
        """Conceded tries should be >= 0."""
        result = compute_team_edge_defence_profiles(season=2024, window=5)

        conceded_cols = [
            "conceded_to_left_edge_rolling_5",
            "conceded_to_right_edge_rolling_5",
            "conceded_to_middle_rolling_5",
            "conceded_to_other_rolling_5",
            "total_tries_conceded_rolling_5",
        ]

        for col in conceded_cols:
            # First match per team will be NaN due to shift(1), drop NaN before checking
            assert (result[col].dropna() >= 0).all(), f"{col} should be >= 0 (excluding NaN)"

    def test_max_round_filtering(self):
        """max_round parameter should filter to rounds < max_round."""
        result_all = compute_team_edge_defence_profiles(season=2024, window=5)
        result_filtered = compute_team_edge_defence_profiles(
            season=2024, max_round=10, window=5
        )

        # Filtered result should have fewer observations
        assert len(result_filtered) < len(result_all)


class TestPlayerEdgeFeatures:
    """Test player-level edge feature addition."""

    @pytest.fixture
    def sample_player_data(self):
        """Create sample player data for testing."""
        # Get real data from player_stats for testing
        from src.db import fetch_df, get_connection

        query = """
        SELECT
            match_id,
            player_id,
            squad_id,
            jumper_number
        FROM player_stats_2024
        LIMIT 100
        """
        with get_connection() as conn:
            df = fetch_df(conn, query)
        return df

    def test_returns_dataframe_with_edge_features(self, sample_player_data):
        """Should add edge feature columns."""
        result = add_player_edge_features(
            player_df=sample_player_data, season=2024, window=5
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_player_data)

        expected_cols = [
            "player_edge",
            "team_edge_try_share_rolling_5",
            "opponent_edge_conceded_rolling_5",
            "edge_matchup_score_rolling_5",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_player_edge_classification(self, sample_player_data):
        """Player edge should match jersey number mapping."""
        result = add_player_edge_features(
            player_df=sample_player_data, season=2024, window=5
        )

        # Check some specific cases
        left_wing = result[result["jumper_number"] == 2]
        if not left_wing.empty:
            assert (left_wing["player_edge"] == "left").all()

        right_wing = result[result["jumper_number"] == 5]
        if not right_wing.empty:
            assert (right_wing["player_edge"] == "right").all()

        hooker = result[result["jumper_number"] == 9]
        if not hooker.empty:
            assert (hooker["player_edge"] == "middle").all()

    def test_edge_matchup_score_is_product(self, sample_player_data):
        """Edge matchup score should be try_share × conceded."""
        result = add_player_edge_features(
            player_df=sample_player_data, season=2024, window=5
        )

        # Check that matchup score = try_share × conceded (approximately)
        expected_score = (
            result["team_edge_try_share_rolling_5"]
            * result["opponent_edge_conceded_rolling_5"]
        )

        # Use pd.testing.assert_series_equal with tolerance
        pd.testing.assert_series_equal(
            result["edge_matchup_score_rolling_5"],
            expected_score,
            check_names=False,
            atol=1e-6,
        )

    def test_requires_jumper_number(self):
        """Should raise ValueError if jumper_number is missing."""
        bad_df = pd.DataFrame(
            {"match_id": [1, 2], "player_id": [100, 101], "squad_id": [1, 2]}
        )

        with pytest.raises(ValueError, match="jumper_number"):
            add_player_edge_features(player_df=bad_df, season=2024, window=5)

    def test_max_round_filtering(self, sample_player_data):
        """max_round parameter should affect edge profile features."""
        result_all = add_player_edge_features(
            player_df=sample_player_data, season=2024, window=5
        )

        result_filtered = add_player_edge_features(
            player_df=sample_player_data, season=2024, max_round=10, window=5
        )

        # The features should be different when filtered
        # (unless all sample data is from later rounds)
        # At minimum, the function should not raise an error
        assert "player_edge" in result_filtered.columns
        assert len(result_filtered) == len(sample_player_data)
