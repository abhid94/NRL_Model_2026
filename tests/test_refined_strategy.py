"""Tests for RefinedEdgeStrategy and Sprint 4C edge analysis functions."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.edge_analysis import (
    conditional_edge_analysis,
    cross_season_stability,
    model_vs_market_disagreement,
    stability_analysis,
    two_way_segment_roi,
)
from src.models.baseline import BaseModel, RefinedEdgeStrategy


# ---------------------------------------------------------------------------
# Simple model for testing strategy
# ---------------------------------------------------------------------------

class _FixedProbModel(BaseModel):
    """Returns pre-set probabilities for testing."""

    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._probs[:len(X)]

    def feature_names(self) -> list[str]:
        return ["dummy"]


@pytest.fixture
def predictions_df():
    """Sample predictions DataFrame for strategy testing."""
    n = 20
    return pd.DataFrame({
        "match_id": np.repeat([1, 2, 3, 4], 5),
        "player_id": range(1, n + 1),
        "position_code": ["FB", "WG", "CE", "PR", "HB"] * 4,
        "betfair_implied_prob": [0.20] * n,
        "betfair_closing_odds": [3.0, 2.5, 4.0, 2.0, 3.5] * 4,
        "scored_try": [1, 0, 1, 0, 0] * 4,
        "expected_team_tries_5": [4.5, 3.0, 5.0, 2.5, 4.2] * 4,
    })


@pytest.fixture
def sample_bet_df():
    """Sample bet DataFrame for edge analysis testing."""
    np.random.seed(42)
    n = 100
    odds = np.random.uniform(1.5, 8.0, n)
    won = np.random.binomial(1, 0.3, n)
    stake = np.full(n, 100.0)
    payout = np.where(won, stake * odds, 0.0)
    return pd.DataFrame({
        "match_id": np.arange(1, n + 1),
        "player_id": np.arange(1001, 1001 + n),
        "position_code": np.random.choice(["WG", "CE", "FB", "SR"], n),
        "model_prob": np.random.uniform(0.15, 0.55, n),
        "implied_prob": np.random.uniform(0.10, 0.50, n),
        "odds": odds,
        "edge": np.random.uniform(0.02, 0.15, n),
        "stake": stake,
        "payout": payout,
        "won": won,
        "season": np.random.choice([2024, 2025], n),
        "round_number": np.random.choice(range(3, 28), n),
    })


@pytest.fixture
def sample_feature_store():
    """Feature store for enrichment."""
    n = 100
    return pd.DataFrame({
        "match_id": np.arange(1, n + 1),
        "player_id": np.arange(1001, 1001 + n),
        "expected_team_tries_5": np.random.uniform(2.0, 6.0, n),
    })


# ---------------------------------------------------------------------------
# RefinedEdgeStrategy tests
# ---------------------------------------------------------------------------

class TestRefinedEdgeStrategy:
    def test_name(self):
        s = RefinedEdgeStrategy()
        assert s.name == "RefinedEdge"

    def test_requires_model(self, predictions_df):
        s = RefinedEdgeStrategy()
        with pytest.raises(ValueError, match="requires a model"):
            s.select_bets(predictions_df)

    def test_filters_by_position(self, predictions_df):
        # Give high probs to all players to ensure edge > threshold
        probs = np.full(20, 0.40)
        model = _FixedProbModel(probs)
        s = RefinedEdgeStrategy(
            min_edge=0.05,
            positions=frozenset({"WG"}),
            min_odds=1.0,
            max_odds=10.0,
            min_team_tries=0.0,
        )
        bets = s.select_bets(predictions_df, model=model)
        for bet in bets:
            assert bet.position_code == "WG"

    def test_filters_by_odds_range(self, predictions_df):
        probs = np.full(20, 0.40)
        model = _FixedProbModel(probs)
        s = RefinedEdgeStrategy(
            min_edge=0.05,
            min_odds=3.0,
            max_odds=4.0,
            min_team_tries=0.0,
        )
        bets = s.select_bets(predictions_df, model=model)
        for bet in bets:
            assert 3.0 <= bet.odds <= 4.0

    def test_filters_by_edge(self, predictions_df):
        # Low probs -> small edge -> no bets
        probs = np.full(20, 0.22)  # edge = 0.02 < 0.05
        model = _FixedProbModel(probs)
        s = RefinedEdgeStrategy(min_edge=0.05)
        bets = s.select_bets(predictions_df, model=model)
        assert len(bets) == 0

    def test_filters_by_team_tries(self, predictions_df):
        probs = np.full(20, 0.40)
        model = _FixedProbModel(probs)
        s = RefinedEdgeStrategy(
            min_edge=0.05,
            min_odds=1.0,
            max_odds=10.0,
            min_team_tries=4.0,
        )
        bets = s.select_bets(predictions_df, model=model)
        # Only players with expected_team_tries_5 >= 4.0 should be selected
        for bet in bets:
            row = predictions_df[predictions_df["player_id"] == bet.player_id].iloc[0]
            assert row["expected_team_tries_5"] >= 4.0

    def test_bet_fields_populated(self, predictions_df):
        probs = np.full(20, 0.40)
        model = _FixedProbModel(probs)
        s = RefinedEdgeStrategy(
            min_edge=0.05, min_odds=1.0, max_odds=10.0, min_team_tries=0.0,
        )
        bets = s.select_bets(predictions_df, model=model)
        assert len(bets) > 0
        for bet in bets:
            assert bet.model_prob > 0
            assert bet.implied_prob > 0
            assert bet.odds > 1.0
            assert bet.edge >= 0.05


# ---------------------------------------------------------------------------
# model_vs_market_disagreement tests
# ---------------------------------------------------------------------------

class TestModelVsMarketDisagreement:
    def test_returns_quartiles(self, sample_bet_df):
        result = model_vs_market_disagreement(sample_bet_df)
        assert not result.empty
        assert "quartile" in result.columns
        assert "avg_disagreement" in result.columns

    def test_empty_df(self):
        result = model_vs_market_disagreement(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# stability_analysis tests
# ---------------------------------------------------------------------------

class TestStabilityAnalysis:
    def test_returns_bootstrap_ci(self, sample_bet_df):
        result = stability_analysis(sample_bet_df, n_bootstrap=500)
        assert "roi" in result
        assert "roi_ci_lower" in result
        assert "roi_ci_upper" in result
        assert "p_positive_roi" in result
        assert result["roi_ci_lower"] <= result["roi"] <= result["roi_ci_upper"]

    def test_empty_df(self):
        result = stability_analysis(pd.DataFrame())
        assert result["n_bets"] == 0

    def test_p_positive_roi_range(self, sample_bet_df):
        result = stability_analysis(sample_bet_df)
        assert 0.0 <= result["p_positive_roi"] <= 1.0


# ---------------------------------------------------------------------------
# cross_season_stability tests
# ---------------------------------------------------------------------------

class TestCrossSeasonStability:
    def test_returns_per_season(self, sample_bet_df):
        result = cross_season_stability(sample_bet_df)
        assert not result.empty
        assert "season" in result.columns
        assert len(result) == 2  # 2024 and 2025

    def test_empty_df(self):
        result = cross_season_stability(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# two_way_segment_roi tests
# ---------------------------------------------------------------------------

class TestTwoWaySegmentRoi:
    def test_basic_pivot(self, sample_bet_df):
        result = two_way_segment_roi(
            sample_bet_df,
            row_col="position_code",
            col_col="odds",
            col_bins=[1.0, 3.0, 5.0, 10.0],
            col_labels=["short", "mid", "long"],
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_empty_df(self):
        result = two_way_segment_roi(pd.DataFrame(), "a", "b")
        assert result.empty

    def test_categorical_both(self, sample_bet_df):
        sample_bet_df["odds_band"] = pd.cut(
            sample_bet_df["odds"],
            bins=[0, 3, 5, 100],
            labels=["short", "mid", "long"],
        )
        result = two_way_segment_roi(
            sample_bet_df,
            row_col="position_code",
            col_col="odds_band",
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# conditional_edge_analysis tests
# ---------------------------------------------------------------------------

class TestConditionalEdgeAnalysis:
    def test_with_position_filter(self, sample_bet_df, sample_feature_store):
        result = conditional_edge_analysis(
            sample_bet_df, sample_feature_store,
            position_filter=["WG"],
        )
        assert "n_bets" in result
        assert result["n_bets"] > 0

    def test_with_odds_filter(self, sample_bet_df, sample_feature_store):
        result = conditional_edge_analysis(
            sample_bet_df, sample_feature_store,
            min_odds=2.0, max_odds=4.0,
        )
        assert "n_bets" in result
        assert result["n_bets"] > 0

    def test_empty(self, sample_feature_store):
        result = conditional_edge_analysis(
            pd.DataFrame(), sample_feature_store,
        )
        assert result == {}
