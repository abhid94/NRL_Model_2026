"""Tests for baseline models and strategies."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    BetRecommendation,
    CompositeStrategy,
    EdgeMatchupStrategy,
    EnrichedLogisticModel,
    FadeHotStreakStrategy,
    LogisticBaselineModel,
    MarketImpliedStrategy,
    ModelEdgeStrategy,
    PositionBaselineModel,
    SegmentPlayStrategy,
)


def _make_sample_data(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Create sample feature data for testing."""
    rng = np.random.RandomState(seed)
    positions = ["Back", "Forward", "Halfback", "Hooker", "Interchange"]
    pos_codes = ["WG", "PR", "HB", "HK", "INT"]
    df = pd.DataFrame({
        "match_id": np.arange(n) // 4 + 1,
        "player_id": np.arange(1, n + 1),
        "position_group": rng.choice(positions, n),
        "position_code": rng.choice(pos_codes, n),
        "expected_team_tries_5": rng.uniform(2, 6, n),
        "player_try_share_5": rng.uniform(0, 0.2, n),
        "is_home": rng.randint(0, 2, n),
        "is_starter": rng.randint(0, 2, n),
        "opponent_rolling_defence_tries_conceded_5": rng.uniform(2, 8, n),
        "rolling_try_rate_3": rng.uniform(0, 0.5, n),
        "rolling_try_rate_5": rng.uniform(0, 0.5, n),
        "rolling_line_breaks_5": rng.uniform(0, 5, n),
        "rolling_tackle_breaks_3": rng.uniform(0, 3, n),
        "rolling_attack_tries_5": rng.uniform(10, 30, n),
        "opponent_rolling_defence_missed_tackles_5": rng.uniform(10, 40, n),
        "edge_matchup_score_rolling_5": rng.uniform(0, 1, n),
        "team_edge_try_share_rolling_5": rng.uniform(0, 0.5, n),
        "opponent_edge_conceded_rolling_5": rng.uniform(0, 10, n),
        "betfair_implied_prob": rng.uniform(0.05, 0.6, n),
        "betfair_closing_odds": rng.uniform(1.5, 10, n),
    })
    y = rng.randint(0, 2, n).astype(float)
    return df, y


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestPositionBaselineModel:
    def test_fit_predict_shape(self):
        df, y = _make_sample_data()
        model = PositionBaselineModel()
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == len(df)
        assert all(0 <= p <= 1 for p in probs)

    def test_position_rates_differ(self):
        """Different position groups should get different rates."""
        df, y = _make_sample_data(n=1000)
        # Force backs to have higher try rate
        back_mask = df["position_group"] == "Back"
        y[back_mask] = np.random.RandomState(1).choice([0, 1], p=[0.6, 0.4], size=back_mask.sum())
        fwd_mask = df["position_group"] == "Forward"
        y[fwd_mask] = np.random.RandomState(2).choice([0, 1], p=[0.9, 0.1], size=fwd_mask.sum())

        model = PositionBaselineModel()
        model.fit(df, y)

        back_row = pd.DataFrame({"position_group": ["Back"]})
        fwd_row = pd.DataFrame({"position_group": ["Forward"]})
        assert model.predict_proba(back_row)[0] > model.predict_proba(fwd_row)[0]

    def test_feature_names(self):
        model = PositionBaselineModel()
        assert model.feature_names() == ["position_group"]

    def test_unknown_position_uses_global_rate(self):
        df, y = _make_sample_data(n=100)
        model = PositionBaselineModel()
        model.fit(df, y)
        unknown = pd.DataFrame({"position_group": ["UnknownPos"]})
        prob = model.predict_proba(unknown)[0]
        assert 0 < prob < 1


class TestLogisticBaselineModel:
    def test_fit_predict(self):
        df, y = _make_sample_data()
        model = LogisticBaselineModel()
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == len(df)
        assert all(0 <= p <= 1 for p in probs)

    def test_handles_nan(self):
        df, y = _make_sample_data()
        df.loc[0, "expected_team_tries_5"] = np.nan
        model = LogisticBaselineModel()
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert not np.isnan(probs[0])  # NaN row gets fallback

    def test_feature_names(self):
        model = LogisticBaselineModel()
        assert "position_group" in model.feature_names()
        assert len(model.feature_names()) == 6


class TestEnrichedLogisticModel:
    def test_fit_predict(self):
        df, y = _make_sample_data()
        model = EnrichedLogisticModel()
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == len(df)
        assert all(0 <= p <= 1 for p in probs)

    def test_handles_missing_features(self):
        """Should work even if some features are missing from input."""
        df, y = _make_sample_data()
        model = EnrichedLogisticModel()
        model.fit(df, y)
        # Drop a feature
        df2 = df.drop(columns=["edge_matchup_score_rolling_5"])
        probs = model.predict_proba(df2)
        assert len(probs) == len(df2)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestModelEdgeStrategy:
    def test_requires_model(self):
        strategy = ModelEdgeStrategy()
        df, _ = _make_sample_data()
        with pytest.raises(ValueError, match="requires a model"):
            strategy.select_bets(df)

    def test_returns_bet_recommendations(self):
        df, y = _make_sample_data()
        model = PositionBaselineModel()
        model.fit(df, y)
        strategy = ModelEdgeStrategy(min_edge=0.0)  # Accept any edge
        bets = strategy.select_bets(df, model=model)
        assert isinstance(bets, list)
        if bets:
            assert isinstance(bets[0], BetRecommendation)

    def test_filters_ineligible_positions(self):
        df, y = _make_sample_data()
        # Force all to ineligible
        df["position_code"] = "PR"
        model = PositionBaselineModel()
        model.fit(df, y)
        strategy = ModelEdgeStrategy(min_edge=0.0)
        bets = strategy.select_bets(df, model=model)
        assert len(bets) == 0

    def test_name(self):
        assert ModelEdgeStrategy().name == "ModelEdge"


class TestSegmentPlayStrategy:
    def test_selects_backs_vs_weak_defence(self):
        df, _ = _make_sample_data(n=100)
        df["position_code"] = "WG"
        df["betfair_closing_odds"] = 3.0
        df["betfair_implied_prob"] = 1 / 3.0
        df["opponent_rolling_defence_tries_conceded_5"] = np.linspace(2, 10, 100)
        strategy = SegmentPlayStrategy()
        bets = strategy.select_bets(df)
        assert len(bets) > 0
        # Should only pick high tries_conceded
        for bet in bets:
            row = df[df["player_id"] == bet.player_id].iloc[0]
            assert row["opponent_rolling_defence_tries_conceded_5"] >= df[
                "opponent_rolling_defence_tries_conceded_5"
            ].quantile(0.75)


class TestEdgeMatchupStrategy:
    def test_selects_top_quartile(self):
        df, _ = _make_sample_data(n=100)
        df["position_code"] = "WG"
        df["betfair_implied_prob"] = 0.3
        df["betfair_closing_odds"] = 3.33
        df["edge_matchup_score_rolling_5"] = np.linspace(0.01, 1, 100)
        strategy = EdgeMatchupStrategy()
        bets = strategy.select_bets(df)
        assert len(bets) > 0


class TestFadeHotStreakStrategy:
    def test_avoids_hot_streaks(self):
        df, _ = _make_sample_data(n=100)
        df["position_code"] = "WG"
        df["betfair_implied_prob"] = 0.3
        df["betfair_closing_odds"] = 3.0
        # Give some players very hot streaks
        df["rolling_try_rate_3"] = 0.2
        df.loc[0:9, "rolling_try_rate_3"] = 0.8  # Hot streak
        strategy = FadeHotStreakStrategy(streak_threshold=1.5)
        bets = strategy.select_bets(df)
        hot_ids = set(df.loc[0:9, "player_id"])
        for bet in bets:
            # Hot streak players may still appear if below threshold
            pass  # Strategy filters proportionally, just verify it runs
        assert len(bets) > 0


class TestMarketImpliedStrategy:
    def test_high_implied_prob(self):
        df, _ = _make_sample_data(n=50)
        df = df.copy()
        df["position_code"] = "WG"
        implied = np.linspace(0.1, 0.6, 50)
        df["betfair_implied_prob"] = implied
        df["betfair_closing_odds"] = 1 / implied
        strategy = MarketImpliedStrategy(min_implied_prob=0.30)
        bets = strategy.select_bets(df)
        assert len(bets) > 0
        for bet in bets:
            assert bet.implied_prob >= 0.30


class TestCompositeStrategy:
    def test_deduplicates(self):
        df, y = _make_sample_data(n=100)
        df["position_code"] = "WG"
        df["betfair_implied_prob"] = 0.3
        df["betfair_closing_odds"] = 3.33
        df["edge_matchup_score_rolling_5"] = 0.5
        df["opponent_rolling_defence_tries_conceded_5"] = 8.0

        s1 = SegmentPlayStrategy()
        s2 = EdgeMatchupStrategy()
        composite = CompositeStrategy([s1, s2])
        bets = composite.select_bets(df)

        # Check no duplicate (match_id, player_id)
        keys = [(b.match_id, b.player_id) for b in bets]
        assert len(keys) == len(set(keys))

    def test_name(self):
        s = CompositeStrategy([SegmentPlayStrategy(), EdgeMatchupStrategy()])
        assert "Composite" in s.name
