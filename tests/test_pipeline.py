"""Tests for the weekly pipeline modules."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import BaseModel
from src.pipeline.bet_recommendations import BetCard, generate_bet_card
from src.pipeline.predict_round import predict_round
from src.pipeline.weekly_pipeline import check_drawdown


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _ConstantModel(BaseModel):
    """Returns constant probabilities for testing."""

    def __init__(self, prob: float = 0.30):
        self._prob = prob

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._prob)

    def feature_names(self) -> list[str]:
        return ["dummy"]


@pytest.fixture
def feature_store():
    """Minimal feature store for testing."""
    n = 40
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "match_id": np.repeat([1, 2, 3, 4], 10),
        "player_id": list(range(1, n + 1)),
        "squad_id": np.repeat([100, 200, 300, 400], 10),
        "opponent_squad_id": np.repeat([200, 100, 400, 300], 10),
        "round_number": 5,
        "season": 2025,
        "position_code": (["FB", "WG", "CE", "FE", "HB", "SR", "LK", "PR", "HK", "INT"] * 4),
        "position_group": (["Back", "Back", "Back", "Halfback", "Halfback",
                            "Forward", "Forward", "Forward", "Hooker", "Interchange"] * 4),
        "is_starter": ([1] * 7 + [1, 1, 0]) * 4,
        "is_home": np.repeat([1, 0, 1, 0], 10),
        "betfair_implied_prob": rng.uniform(0.05, 0.50, n),
        "betfair_closing_odds": rng.uniform(1.5, 10.0, n),
        "scored_try": rng.choice([0, 1], n, p=[0.81, 0.19]),
    })


# ---------------------------------------------------------------------------
# predict_round tests
# ---------------------------------------------------------------------------

class TestPredictRound:
    def test_basic_prediction(self, feature_store):
        model = _ConstantModel(0.30)
        result = predict_round(model, feature_store, 2025, 5)
        assert not result.empty
        assert "model_prob" in result.columns
        assert "edge" in result.columns
        assert "is_eligible" in result.columns

    def test_model_prob_values(self, feature_store):
        model = _ConstantModel(0.30)
        result = predict_round(model, feature_store, 2025, 5)
        assert (result["model_prob"] == 0.30).all()

    def test_edge_computation(self, feature_store):
        model = _ConstantModel(0.30)
        result = predict_round(model, feature_store, 2025, 5)
        # Edge = model_prob - implied_prob
        expected_edge = 0.30 - result["betfair_implied_prob"]
        np.testing.assert_array_almost_equal(
            result["edge"].values, expected_edge.values, decimal=10,
        )

    def test_eligibility_excludes_pr_hk_int(self, feature_store):
        model = _ConstantModel(0.30)
        result = predict_round(model, feature_store, 2025, 5)
        ineligible = result[~result["is_eligible"]]
        eligible = result[result["is_eligible"]]
        # PR, HK, INT should NOT be eligible
        assert all(pos in {"PR", "HK", "INT"} for pos in ineligible["position_code"])
        # FB, WG, CE, FE, HB, SR, LK should be eligible (if they have odds)
        for pos in ["FB", "WG", "CE", "FE", "HB", "SR", "LK"]:
            assert pos in eligible["position_code"].values

    def test_empty_round(self, feature_store):
        model = _ConstantModel(0.30)
        result = predict_round(model, feature_store, 2025, 99)
        assert result.empty

    def test_sorted_by_edge(self, feature_store):
        model = _ConstantModel(0.30)
        result = predict_round(model, feature_store, 2025, 5)
        # Non-NaN edges should be descending
        edges = result["edge"].dropna()
        assert (edges.diff().dropna() <= 0).all()


# ---------------------------------------------------------------------------
# generate_bet_card tests
# ---------------------------------------------------------------------------

class TestGenerateBetCard:
    def test_basic_bet_card(self, feature_store):
        model = _ConstantModel(0.40)
        preds = predict_round(model, feature_store, 2025, 5)
        card = generate_bet_card(preds, bankroll=10000)
        assert isinstance(card, BetCard)
        assert card.bankroll == 10000
        assert card.season == 2025
        assert card.round_number == 5

    def test_no_bets_when_no_edge(self, feature_store):
        model = _ConstantModel(0.05)  # Very low prob -> negative edge
        preds = predict_round(model, feature_store, 2025, 5)
        card = generate_bet_card(preds, bankroll=10000)
        assert len(card.bets) == 0
        assert card.total_staked == 0.0

    def test_max_bets_per_match(self, feature_store):
        model = _ConstantModel(0.60)  # High prob -> big edge for most
        preds = predict_round(model, feature_store, 2025, 5)
        card = generate_bet_card(preds, bankroll=10000, max_bets_per_match=2)
        # Count bets per match
        df = card.to_dataframe()
        if not df.empty:
            bets_per_match = df.groupby("match_id").size()
            assert (bets_per_match <= 2).all()

    def test_max_bets_per_round(self, feature_store):
        model = _ConstantModel(0.60)
        preds = predict_round(model, feature_store, 2025, 5)
        card = generate_bet_card(preds, bankroll=10000, max_bets_per_round=3)
        assert len(card.bets) <= 3

    def test_exposure_cap(self, feature_store):
        model = _ConstantModel(0.60)
        preds = predict_round(model, feature_store, 2025, 5)
        card = generate_bet_card(preds, bankroll=10000, max_round_exposure_pct=0.10)
        assert card.total_staked <= 10000 * 0.10 + 1  # Small tolerance

    def test_flat_stake_mode(self, feature_store):
        model = _ConstantModel(0.40)
        preds = predict_round(model, feature_store, 2025, 5)
        card = generate_bet_card(preds, bankroll=10000, flat_stake=50.0)
        df = card.to_dataframe()
        if not df.empty:
            assert (df["stake"] == 50.0).all()

    def test_summary_string(self, feature_store):
        model = _ConstantModel(0.40)
        preds = predict_round(model, feature_store, 2025, 5)
        card = generate_bet_card(preds, bankroll=10000)
        summary = card.summary()
        assert "Bet Card" in summary
        assert "2025" in summary

    def test_empty_predictions(self):
        card = generate_bet_card(pd.DataFrame(), bankroll=10000)
        assert len(card.bets) == 0


# ---------------------------------------------------------------------------
# check_drawdown tests
# ---------------------------------------------------------------------------

class TestCheckDrawdown:
    def test_ok_status(self):
        result = check_drawdown(9500, 10000)
        assert result["status"] == "OK"
        assert result["kelly_adjustment"] == 1.0

    def test_warning_status(self):
        result = check_drawdown(8400, 10000)
        assert result["status"] == "WARNING"
        assert result["kelly_adjustment"] < 1.0

    def test_halt_status(self):
        result = check_drawdown(7400, 10000)
        assert result["status"] == "HALT"
        assert result["kelly_adjustment"] == 0.0

    def test_stop_status(self):
        result = check_drawdown(5900, 10000)
        assert result["status"] == "STOP"
        assert result["kelly_adjustment"] == 0.0

    def test_no_drawdown(self):
        result = check_drawdown(10500, 10000)
        assert result["status"] == "OK"
        assert result["drawdown_pct"] < 0

    def test_drawdown_pct_accurate(self):
        result = check_drawdown(8000, 10000)
        assert abs(result["drawdown_pct"] - 0.20) < 0.001
