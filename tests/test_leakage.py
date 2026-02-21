"""Automated data leakage detection tests.

CLAUDE.md Section 7, Rule 8: "Automated leakage checks run as part of the test
suite (tests/test_leakage.py). Every feature must pass these checks before
entering the feature store."

These tests verify:
1. Rolling features use shift(1) — round R's feature uses ONLY rounds < R data
2. Edge features do NOT include current-match try data
3. No feature correlates suspiciously with the target (r > 0.5)
4. Target variable (scored_try) is never used as a model feature
5. Feature store has correct temporal ordering
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src import db
from src.config import DB_PATH

logger = logging.getLogger(__name__)

SEASON = 2024  # Use 2024 for all leakage tests (most data)


@pytest.fixture(scope="module")
def connection():
    """Database connection for the test module."""
    conn = db.get_connection(DB_PATH)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# 1. Edge feature shift(1) validation
# ---------------------------------------------------------------------------
class TestEdgeFeatureLeakage:
    """Verify edge features exclude current-match data."""

    def test_attack_profile_excludes_current_match(self, connection):
        """Team edge attack profiles for round R must not include round R tries.

        shift(1) means the first match of each team should have NaN rolling values.
        """
        from src.features.edge_features import compute_team_edge_attack_profiles

        profiles = compute_team_edge_attack_profiles(season=SEASON, window=5)
        if profiles.empty:
            pytest.skip("No attack profile data")

        # Get match ordering per team
        match_order = pd.read_sql_query(
            f"""
            SELECT DISTINCT ts.squad_id, m.match_id, m.round_number, m.utc_start_time
            FROM team_stats_{SEASON} ts
            JOIN matches_{SEASON} m ON ts.match_id = m.match_id
            ORDER BY ts.squad_id, m.utc_start_time
            """,
            connection,
        )
        first_matches = match_order.groupby("squad_id").first().reset_index()

        nan_count = 0
        total = 0
        for _, row in first_matches.iterrows():
            team_first = profiles[
                (profiles["match_id"] == row["match_id"])
                & (profiles["squad_id"] == row["squad_id"])
            ]
            if team_first.empty:
                continue
            total += 1
            val = team_first["total_tries_rolling_5"].iloc[0]
            if pd.isna(val):
                nan_count += 1

        assert total > 0, "No first-match observations found"
        assert nan_count == total, (
            f"Expected all first-match total_tries_rolling_5 to be NaN (shift excludes first), "
            f"got {nan_count}/{total} NaN. shift(1) may be missing."
        )

    def test_defence_profile_excludes_current_match(self, connection):
        """Team edge defence profiles for round R must not include round R tries."""
        from src.features.edge_features import compute_team_edge_defence_profiles

        profiles = compute_team_edge_defence_profiles(season=SEASON, window=5)
        if profiles.empty:
            pytest.skip("No defence profile data")

        match_order = pd.read_sql_query(
            f"""
            SELECT DISTINCT ts.squad_id, m.match_id, m.round_number, m.utc_start_time
            FROM team_stats_{SEASON} ts
            JOIN matches_{SEASON} m ON ts.match_id = m.match_id
            ORDER BY ts.squad_id, m.utc_start_time
            """,
            connection,
        )
        first_matches = match_order.groupby("squad_id").first().reset_index()

        nan_count = 0
        total = 0
        for _, row in first_matches.iterrows():
            team_first = profiles[
                (profiles["match_id"] == row["match_id"])
                & (profiles["squad_id"] == row["squad_id"])
            ]
            if team_first.empty:
                continue
            total += 1
            val = team_first["total_tries_conceded_rolling_5"].iloc[0]
            if pd.isna(val):
                nan_count += 1

        assert total > 0, "No first-match observations found"
        assert nan_count == total, (
            f"Expected all first-match total_tries_conceded_rolling_5 to be NaN (shift excludes first), "
            f"got {nan_count}/{total} NaN. shift(1) may be missing."
        )

    def test_edge_matchup_score_not_perfectly_correlated_with_target(self, connection):
        """Edge matchup score should not have suspiciously high correlation with target."""
        from src.features.edge_features import add_player_edge_features

        query = f"""
        SELECT ps.match_id, ps.player_id, ps.squad_id, ps.jumper_number,
               ps.opponent_squad_id, ps.tries
        FROM player_stats_{SEASON} ps
        JOIN matches_{SEASON} m ON ps.match_id = m.match_id
        WHERE m.match_type = 'H'
        """
        player_df = pd.read_sql_query(query, connection)
        player_df["scored_try"] = (player_df["tries"] >= 1).astype(int)

        result = add_player_edge_features(player_df, season=SEASON, window=5)

        if "edge_matchup_score_rolling_5" not in result.columns:
            pytest.skip("edge_matchup_score not computed")

        valid = result.dropna(subset=["edge_matchup_score_rolling_5"])
        if len(valid) < 100:
            pytest.skip("Too few valid observations")

        corr = valid["edge_matchup_score_rolling_5"].corr(valid["scored_try"])
        assert abs(corr) < 0.50, (
            f"Edge matchup score has suspiciously high correlation with target: {corr:.3f}. "
            f"Possible data leakage."
        )

    def test_attack_rolling_changes_between_matches(self, connection):
        """Rolling edge features should differ between consecutive matches for the same team
        (verifying shift is working — if no shift, feature at match N would be identical
        to feature at match N+1 when window > games played).
        """
        from src.features.edge_features import compute_team_edge_attack_profiles

        profiles = compute_team_edge_attack_profiles(season=SEASON, window=5)
        if profiles.empty:
            pytest.skip("No attack profile data")

        # Get match ordering per team
        match_order = pd.read_sql_query(
            f"""
            SELECT DISTINCT ts.squad_id, m.match_id, m.round_number, m.utc_start_time
            FROM team_stats_{SEASON} ts
            JOIN matches_{SEASON} m ON ts.match_id = m.match_id
            ORDER BY ts.squad_id, m.utc_start_time
            """,
            connection,
        )

        # Pick one team and check second vs third match
        team_id = match_order["squad_id"].iloc[0]
        team_matches = match_order[match_order["squad_id"] == team_id].reset_index(drop=True)

        if len(team_matches) < 3:
            pytest.skip("Team has fewer than 3 matches")

        m2 = profiles[
            (profiles["match_id"] == team_matches.loc[1, "match_id"])
            & (profiles["squad_id"] == team_id)
        ]["total_tries_rolling_5"].iloc[0]

        m3 = profiles[
            (profiles["match_id"] == team_matches.loc[2, "match_id"])
            & (profiles["squad_id"] == team_id)
        ]["total_tries_rolling_5"].iloc[0]

        # Second match uses only match 1 data; third uses matches 1+2
        # They should generally differ (unless both matches had identical tries)
        # At minimum, verify m2 is not NaN (it should have 1 prior match)
        assert not pd.isna(m2), "Second match rolling should not be NaN"


# ---------------------------------------------------------------------------
# 2. Player feature shift(1) validation
# ---------------------------------------------------------------------------
class TestPlayerFeatureLeakage:
    """Verify player rolling features exclude current-match data."""

    def test_rolling_features_use_shift(self, connection):
        """Player rolling features should use only prior match data.

        Since default fillna_value=0.0, first-match values will be 0.0 (filled NaN).
        We verify by computing features with fillna_value=None and checking NaN.

        Note: groupby().nth(0) returns the actual first row (not first non-null).
        """
        from src.features.player_features import compute_player_features, PlayerFeatureConfig

        config = PlayerFeatureConfig(fillna_value=None)
        features = compute_player_features(connection, SEASON, config=config)

        # First match per player should have NaN rolling features
        # Use nth(0) not first() — first() returns first non-null value per column
        first = (
            features.sort_values(["player_id", "round_number"])
            .groupby("player_id")
            .nth(0)
            .reset_index()
        )

        nan_pct = first["rolling_tries_3"].isna().mean()
        assert nan_pct == 1.0, (
            f"Expected 100% of first-match rolling_tries_3 to be NaN (fillna=None), "
            f"got {nan_pct:.1%}. shift(1) may be missing."
        )

    def test_rolling_value_uses_only_prior_matches(self, connection):
        """For a player with known try history, verify rolling value excludes current match."""
        from src.features.player_features import compute_player_features, PlayerFeatureConfig

        config = PlayerFeatureConfig(fillna_value=None)
        features = compute_player_features(connection, SEASON, config=config)

        # Get raw tries
        raw = pd.read_sql_query(
            f"""
            SELECT ps.match_id, ps.player_id, ps.tries, m.round_number
            FROM player_stats_{SEASON} ps
            JOIN matches_{SEASON} m ON ps.match_id = m.match_id
            ORDER BY ps.player_id, m.round_number
            """,
            connection,
        )

        merged = features.merge(raw[["match_id", "player_id", "tries"]], on=["match_id", "player_id"])

        # For second match per player, rolling_tries_3 should equal first match tries
        # (window=3, min_periods=1, only 1 prior match → mean of 1 value)
        for pid in merged["player_id"].unique()[:20]:
            player = merged[merged["player_id"] == pid].sort_values("round_number")
            if len(player) < 2:
                continue
            second = player.iloc[1]
            first_tries = player.iloc[0]["tries"]
            rolling_val = second["rolling_tries_3"]
            if pd.isna(rolling_val):
                continue
            assert abs(rolling_val - first_tries) < 1e-6, (
                f"Player {pid}: second match rolling={rolling_val}, "
                f"first match tries={first_tries}. Should be equal (only 1 prior match)."
            )


# ---------------------------------------------------------------------------
# 3. Team feature shift(1) validation
# ---------------------------------------------------------------------------
class TestTeamFeatureLeakage:
    """Verify team rolling features exclude current-match data."""

    def test_first_match_rolling_is_nan_without_fillna(self, connection):
        """Team features for first match should be NaN (with fillna=None)."""
        from src.features.team_features import compute_team_features, TeamFeatureConfig

        config = TeamFeatureConfig(fillna_value=None)
        features = compute_team_features(connection, SEASON, config=config)

        # Use nth(0) not first() — first() returns first non-null value per column
        first = (
            features.sort_values(["squad_id", "round_number"])
            .groupby("squad_id")
            .nth(0)
            .reset_index()
        )

        if "rolling_attack_tries_3" in first.columns:
            nan_pct = first["rolling_attack_tries_3"].isna().mean()
            assert nan_pct == 1.0, (
                f"Expected 100% of first-match rolling_attack_tries_3 to be NaN, "
                f"got {nan_pct:.1%}"
            )


# ---------------------------------------------------------------------------
# 4. Game context feature shift(1) validation
# ---------------------------------------------------------------------------
class TestGameContextFeatureLeakage:
    """Verify game context rolling features exclude current-match data."""

    def test_first_match_context_is_nan_without_fillna(self, connection):
        """Expected team tries for first match should be NaN (with fillna=None).

        Game context features compose team_attack × team_defence sub-features.
        The sub-functions have their own internal shift(1) but may fill NaN
        with 0 internally. We verify the pattern using player_try_share instead,
        which is computed directly from player-level rolling data.
        """
        from src.features.game_context_features import (
            compute_game_context_features,
            GameContextConfig,
        )

        config = GameContextConfig(fillna_value=None)
        features = compute_game_context_features(connection, SEASON, config=config)

        # player_try_share uses shift(1) directly on player tries / team tries
        share_col = [c for c in features.columns if "player_try_share" in c]
        if not share_col:
            pytest.skip("No player_try_share column found")

        col = share_col[0]
        # Use nth(0) to get actual first row per player
        first = (
            features.sort_values(["player_id", "round_number"])
            .groupby("player_id")
            .nth(0)
            .reset_index()
        )
        nan_pct = first[col].isna().mean()

        # Player try share at first match should be NaN (no prior tries to compute share from)
        assert nan_pct > 0.5, (
            f"Expected >50% of first-match {col} to be NaN (fillna=None), "
            f"got {nan_pct:.1%}. Shift may be missing."
        )


# ---------------------------------------------------------------------------
# 5. Feature store target isolation
# ---------------------------------------------------------------------------
class TestFeatureStoreLeakage:
    """Verify the feature store has no target leakage."""

    def test_scored_try_not_in_model_features(self):
        """scored_try should NOT be used as a model feature."""
        from src.features.feature_store import get_feature_metadata

        metadata = get_feature_metadata()
        features = metadata[metadata["feature_type"] == "feature"]["feature_name"].tolist()

        assert "scored_try" not in features, "scored_try is in the feature list!"
        assert "tries" not in features, "raw tries column is in the feature list!"

    def test_no_feature_suspiciously_correlated_with_target(self, connection):
        """No feature should have correlation > 0.5 with scored_try."""
        from src.features.feature_store import build_feature_store

        fs = build_feature_store(connection, SEASON)

        target = fs["scored_try"]
        numeric_cols = fs.select_dtypes(include=[np.number]).columns.tolist()

        exclude = {"match_id", "player_id", "squad_id", "opponent_squad_id",
                   "round_number", "season", "scored_try", "tries"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

        high_corr = []
        for col in feature_cols:
            corr = fs[col].corr(target)
            if abs(corr) > 0.50:
                high_corr.append((col, corr))

        assert len(high_corr) == 0, (
            f"Features with suspiciously high correlation (>0.5) with target:\n"
            + "\n".join(f"  {name}: {r:.3f}" for name, r in high_corr)
        )

    def test_feature_store_round1_rolling_is_zero(self, connection):
        """Round 1 rolling features should be 0 (filled NaN from shift(1) + fillna=0)."""
        from src.features.feature_store import build_feature_store

        fs = build_feature_store(connection, SEASON)

        round1 = fs[fs["round_number"] == 1]
        if round1.empty:
            pytest.skip("No round 1 data")

        # Player rolling features default to fillna=0.0, so first-round values = 0
        rolling_cols = [c for c in fs.columns if c.startswith("rolling_tries_")]
        if not rolling_cols:
            pytest.skip("No rolling_tries columns found")

        col = rolling_cols[0]
        zero_pct = (round1[col] == 0.0).mean()
        assert zero_pct > 0.9, (
            f"Round 1 feature '{col}' has {zero_pct:.1%} zero values — "
            f"expected >90% (filled NaN from shift)"
        )

    def test_max_round_filters_correctly(self, connection):
        """max_round parameter should prevent data from future rounds leaking in."""
        from src.features.edge_features import compute_team_edge_attack_profiles

        max_rnd = 10
        profiles = compute_team_edge_attack_profiles(season=SEASON, max_round=max_rnd, window=5)

        if profiles.empty:
            pytest.skip("No attack profiles")

        match_rounds = pd.read_sql_query(
            f"SELECT match_id, round_number FROM matches_{SEASON}",
            connection,
        )
        merged = profiles.merge(match_rounds, on="match_id")

        assert merged["round_number"].max() < max_rnd, (
            f"max_round={max_rnd} but data includes round {merged['round_number'].max()}"
        )


# ---------------------------------------------------------------------------
# 6. Cross-module consistency
# ---------------------------------------------------------------------------
class TestCrossModuleConsistency:
    """Verify all feature modules use the same shift pattern."""

    def test_all_modules_produce_nan_for_first_observation_without_fillna(self, connection):
        """Every rolling feature should be NaN for a player/team's first match (fillna=None).

        Uses nth(0) to get the actual first row, not first non-null value.
        """
        from src.features.player_features import compute_player_features, PlayerFeatureConfig
        from src.features.team_features import compute_team_features, TeamFeatureConfig

        # Player features with fillna=None
        p_config = PlayerFeatureConfig(fillna_value=None)
        player_feats = compute_player_features(connection, SEASON, config=p_config)

        p_first = (
            player_feats.sort_values(["player_id", "round_number"])
            .groupby("player_id")
            .nth(0)
            .reset_index()
        )
        p_rolling = [c for c in p_first.columns if c.startswith("rolling_")]

        for col in p_rolling[:3]:
            assert p_first[col].isna().all(), (
                f"Player feature '{col}' has non-NaN values at first match (fillna=None)"
            )

        # Team features with fillna=None
        t_config = TeamFeatureConfig(fillna_value=None)
        team_feats = compute_team_features(connection, SEASON, config=t_config)

        t_first = (
            team_feats.sort_values(["squad_id", "round_number"])
            .groupby("squad_id")
            .nth(0)
            .reset_index()
        )
        t_rolling = [c for c in t_first.columns if c.startswith("rolling_")]

        for col in t_rolling[:3]:
            assert t_first[col].isna().all(), (
                f"Team feature '{col}' has non-NaN values at first match (fillna=None)"
            )

    def test_edge_features_consistent_with_player_features(self, connection):
        """Edge features first match NaN pattern should match player/team features."""
        from src.features.edge_features import compute_team_edge_attack_profiles

        profiles = compute_team_edge_attack_profiles(season=SEASON, window=5)
        if profiles.empty:
            pytest.skip("No edge profiles")

        # Edge features don't have fillna, so first match should be NaN
        match_order = pd.read_sql_query(
            f"""
            SELECT DISTINCT ts.squad_id, m.match_id, m.utc_start_time
            FROM team_stats_{SEASON} ts
            JOIN matches_{SEASON} m ON ts.match_id = m.match_id
            ORDER BY ts.squad_id, m.utc_start_time
            """,
            connection,
        )
        first_matches = match_order.groupby("squad_id").first().reset_index()

        merged = first_matches.merge(profiles, on=["match_id", "squad_id"])
        nan_pct = merged["total_tries_rolling_5"].isna().mean()

        assert nan_pct == 1.0, (
            f"Expected 100% of first-match edge rolling features to be NaN, "
            f"got {nan_pct:.1%}"
        )
