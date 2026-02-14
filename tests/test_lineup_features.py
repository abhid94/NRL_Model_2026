"""
Tests for lineup features module.

Validates:
1. Teammate playmaking features computation
2. Lineup stability features computation
3. Leakage prevention (as_of_round parameter)
4. Correct handling of edge cases (first match, no lineup data)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.db import get_connection
from src.features.lineup_features import (
    compute_teammate_playmaking_features,
    compute_lineup_stability_features,
    add_lineup_features_to_player_observations
)


@pytest.fixture
def db_conn():
    """Get database connection for tests."""
    conn = get_connection()
    yield conn
    conn.close()


class TestTeammatePlaymakingFeatures:
    """Tests for teammate playmaking quality features."""

    def test_playmaking_features_shape_and_columns(self, db_conn):
        """Test that playmaking features returns expected columns."""
        df = compute_teammate_playmaking_features(db_conn, year=2024, window=5)

        # Should have one row per team-match
        assert len(df) > 0
        assert 'match_id' in df.columns
        assert 'squad_id' in df.columns
        assert 'round_number' in df.columns
        assert 'teammate_fullback_try_assists_rolling_5' in df.columns
        assert 'teammate_halves_try_assists_rolling_5' in df.columns
        assert 'teammate_playmakers_try_assists_rolling_5' in df.columns

        # Check data types
        assert df['teammate_fullback_try_assists_rolling_5'].dtype in [np.float64, float]
        assert df['teammate_playmakers_try_assists_rolling_5'].dtype in [np.float64, float]

    def test_playmaking_features_non_negative(self, db_conn):
        """Test that try assists are non-negative."""
        df = compute_teammate_playmaking_features(db_conn, year=2024, window=5)

        # Remove NaN values (early-season matches)
        df_valid = df[df['teammate_playmakers_try_assists_rolling_5'].notna()]

        assert (df_valid['teammate_fullback_try_assists_rolling_5'] >= 0).all()
        assert (df_valid['teammate_halves_try_assists_rolling_5'] >= 0).all()
        assert (df_valid['teammate_playmakers_try_assists_rolling_5'] >= 0).all()

    def test_playmaking_features_leakage_prevention(self, db_conn):
        """Test that as_of_round parameter correctly limits data."""
        df_round_10 = compute_teammate_playmaking_features(
            db_conn, year=2024, as_of_round=10, window=5
        )
        df_round_20 = compute_teammate_playmaking_features(
            db_conn, year=2024, as_of_round=20, window=5
        )

        # Round 10 should have fewer matches than round 20
        assert len(df_round_10) < len(df_round_20)

        # Round 10 should only have rounds < 10
        assert (df_round_10['round_number'] < 10).all()

        # Round 20 should only have rounds < 20
        assert (df_round_20['round_number'] < 20).all()

    def test_playmaking_features_early_season_nans(self, db_conn):
        """Test that early-season matches have NaN (insufficient history)."""
        df = compute_teammate_playmaking_features(
            db_conn, year=2024, as_of_round=6, window=5
        )

        # Early rounds should have high % of NaN
        nan_pct = df['teammate_playmakers_try_assists_rolling_5'].isna().sum() / len(df)
        assert nan_pct > 0.3  # At least 30% should be NaN for first 6 rounds

    def test_playmaking_features_later_season_fewer_nans(self, db_conn):
        """Test that later-season matches have fewer NaN."""
        df = compute_teammate_playmaking_features(
            db_conn, year=2024, as_of_round=27, window=5
        )

        # Later rounds should have lower % of NaN
        nan_pct = df['teammate_playmakers_try_assists_rolling_5'].isna().sum() / len(df)
        assert nan_pct < 0.3  # Less than 30% should be NaN by end of season

    def test_playmaking_sum_equals_total(self, db_conn):
        """Test that fullback + halves = total playmakers."""
        df = compute_teammate_playmaking_features(db_conn, year=2024, window=5)

        # Filter to non-NaN rows
        df_valid = df[df['teammate_playmakers_try_assists_rolling_5'].notna()].copy()

        # Playmakers = fullback + halves
        df_valid['calculated_total'] = (
            df_valid['teammate_fullback_try_assists_rolling_5'] +
            df_valid['teammate_halves_try_assists_rolling_5']
        )

        # Should match (within floating point tolerance)
        assert np.allclose(
            df_valid['calculated_total'],
            df_valid['teammate_playmakers_try_assists_rolling_5'],
            rtol=0.001
        )


class TestLineupStabilityFeatures:
    """Tests for lineup stability features."""

    def test_stability_features_shape_and_columns(self, db_conn):
        """Test that stability features returns expected columns."""
        df = compute_lineup_stability_features(db_conn, year=2025)

        assert len(df) > 0
        assert 'match_id' in df.columns
        assert 'player_id' in df.columns
        assert 'squad_id' in df.columns
        assert 'round_number' in df.columns
        assert 'lineup_changes_from_prev_round' in df.columns
        assert 'lineup_stability_pct' in df.columns
        assert 'player_was_in_prev_lineup' in df.columns

    def test_stability_features_valid_ranges(self, db_conn):
        """Test that stability features are in valid ranges."""
        df = compute_lineup_stability_features(db_conn, year=2025)

        # Filter to non-NaN rows
        df_valid = df[df['lineup_stability_pct'].notna()]

        # Stability should be between 0 and 1
        assert (df_valid['lineup_stability_pct'] >= 0).all()
        assert (df_valid['lineup_stability_pct'] <= 1).all()

        # Changes should be non-negative
        assert (df_valid['lineup_changes_from_prev_round'] >= 0).all()

        # player_was_in_prev_lineup should be 0 or 1 (when not NaN)
        df_player_valid = df[df['player_was_in_prev_lineup'].notna()]
        assert df_player_valid['player_was_in_prev_lineup'].isin([0, 1]).all()

    def test_stability_features_leakage_prevention(self, db_conn):
        """Test that as_of_round parameter correctly limits data."""
        df_round_12 = compute_lineup_stability_features(
            db_conn, year=2025, as_of_round=12
        )
        df_round_20 = compute_lineup_stability_features(
            db_conn, year=2025, as_of_round=20
        )

        # Round 12 should have fewer player-matches than round 20
        assert len(df_round_12) < len(df_round_20)

        # Round 12 should only have rounds <= 12
        assert (df_round_12['round_number'] <= 12).all()

        # Round 20 should only have rounds <= 20
        assert (df_round_20['round_number'] <= 20).all()

    def test_stability_features_first_round_nans(self, db_conn):
        """Test that first round has NaN (no previous lineup)."""
        df = compute_lineup_stability_features(db_conn, year=2025, as_of_round=11)

        # Round 10 is first round of team_lists_2025, so it should be all NaN
        round_10 = df[df['round_number'] == 10]
        assert round_10['lineup_stability_pct'].isna().all()
        assert round_10['player_was_in_prev_lineup'].isna().all()

    def test_stability_features_later_rounds_have_values(self, db_conn):
        """Test that later rounds have non-NaN values."""
        df = compute_lineup_stability_features(db_conn, year=2025, as_of_round=20)

        # Round 15+ should have mostly non-NaN values
        round_15_plus = df[df['round_number'] >= 15]
        nan_pct = round_15_plus['lineup_stability_pct'].isna().sum() / len(round_15_plus)
        assert nan_pct < 0.1  # Less than 10% NaN

    def test_stability_no_team_lists_returns_empty(self, db_conn):
        """Test that years without team_lists data return empty DataFrame."""
        df = compute_lineup_stability_features(db_conn, year=2024)

        # 2024 has no team_lists data, should return empty with correct columns
        assert len(df) == 0
        assert 'lineup_changes_from_prev_round' in df.columns
        assert 'lineup_stability_pct' in df.columns


class TestLineupFeaturesIntegration:
    """Tests for full lineup features integration."""

    def test_add_lineup_features_integration(self, db_conn):
        """Test that lineup features can be added to player observations."""
        # Get some player observations from 2025 (has team_lists data)
        base_query = """
        SELECT ps.match_id, ps.player_id, ps.squad_id, m.round_number
        FROM player_stats_2025 ps
        JOIN matches_2025 m ON ps.match_id = m.match_id
        WHERE m.round_number = 15
        LIMIT 100
        """
        base_df = pd.read_sql_query(base_query, db_conn)

        # Add lineup features
        enriched = add_lineup_features_to_player_observations(
            base_df, db_conn, year=2025, as_of_round=15, window=5
        )

        # Should have same number of rows
        assert len(enriched) == len(base_df)

        # Should have additional columns
        assert 'teammate_fullback_try_assists_rolling_5' in enriched.columns
        assert 'teammate_halves_try_assists_rolling_5' in enriched.columns
        assert 'teammate_playmakers_try_assists_rolling_5' in enriched.columns
        assert 'lineup_changes_from_prev_round' in enriched.columns
        assert 'lineup_stability_pct' in enriched.columns
        assert 'player_was_in_prev_lineup' in enriched.columns

    def test_add_lineup_features_missing_columns_raises_error(self, db_conn):
        """Test that missing required columns raises an error."""
        # Missing player_id column
        bad_df = pd.DataFrame({
            'match_id': [123],
            'squad_id': [321]
        })

        with pytest.raises(ValueError, match="missing required columns"):
            add_lineup_features_to_player_observations(
                bad_df, db_conn, year=2025
            )

    def test_add_lineup_features_preserves_original_columns(self, db_conn):
        """Test that original columns are preserved."""
        base_query = """
        SELECT ps.match_id, ps.player_id, ps.squad_id, ps.tries, m.round_number
        FROM player_stats_2025 ps
        JOIN matches_2025 m ON ps.match_id = m.match_id
        WHERE m.round_number = 20
        LIMIT 50
        """
        base_df = pd.read_sql_query(base_query, db_conn)

        enriched = add_lineup_features_to_player_observations(
            base_df, db_conn, year=2025, as_of_round=20
        )

        # All original columns should be present
        for col in base_df.columns:
            assert col in enriched.columns

        # Original data should be unchanged
        assert (enriched['tries'] == base_df['tries']).all()


class TestLineupFeaturesRealWorldScenarios:
    """Tests for real-world scenarios and edge cases."""

    def test_team_with_no_changes(self, db_conn):
        """Test that a team with no lineup changes has 100% stability."""
        df = compute_lineup_stability_features(db_conn, year=2025, as_of_round=20)

        # Find teams with 100% stability (no changes)
        df_stable = df[df['lineup_stability_pct'] == 1.0]

        # These teams should have 0 changes
        assert (df_stable['lineup_changes_from_prev_round'] == 0).all()

        # All players should have been in prev lineup
        assert (df_stable['player_was_in_prev_lineup'] == 1).all()

    def test_playmaking_features_consistent_across_team(self, db_conn):
        """Test that playmaking features are same for all players on same team-match."""
        df_full = compute_teammate_playmaking_features(
            db_conn, year=2024, as_of_round=15, window=5
        )

        # Pick a specific match-team
        sample = df_full.iloc[0]
        match_id = sample['match_id']
        squad_id = sample['squad_id']

        # Should only be one row per team-match
        team_match_df = df_full[
            (df_full['match_id'] == match_id) &
            (df_full['squad_id'] == squad_id)
        ]
        assert len(team_match_df) == 1

    def test_stability_features_new_player_flagged(self, db_conn):
        """Test that new players (not in prev lineup) are flagged correctly."""
        df = compute_lineup_stability_features(db_conn, year=2025, as_of_round=20)

        # Find players who were NOT in previous lineup
        new_players = df[df['player_was_in_prev_lineup'] == 0]

        # These should exist (teams make changes)
        assert len(new_players) > 0

        # Their teams should have > 0 changes
        for idx, row in new_players.head(10).iterrows():
            assert row['lineup_changes_from_prev_round'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
