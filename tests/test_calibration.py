"""Tests for probability calibration wrapper."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_calibration_error
from src.models.baseline import PositionBaselineModel
from src.models.calibration import CalibratedModel


def _make_sample_data(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Create sample data with round_number for temporal splitting."""
    rng = np.random.RandomState(seed)
    positions = ["Back", "Forward", "Halfback", "Hooker", "Interchange"]
    rounds = np.repeat(np.arange(1, 26), n // 25 + 1)[:n]
    df = pd.DataFrame({
        "round_number": rounds,
        "position_group": rng.choice(positions, n),
    })
    # Target correlated with position
    y = np.zeros(n)
    for i, pos in enumerate(df["position_group"]):
        if pos == "Back":
            y[i] = rng.choice([0, 1], p=[0.6, 0.4])
        elif pos == "Forward":
            y[i] = rng.choice([0, 1], p=[0.9, 0.1])
        else:
            y[i] = rng.choice([0, 1], p=[0.75, 0.25])
    return df, y


class TestCalibratedModel:
    def test_isotonic_produces_valid_probs(self):
        df, y = _make_sample_data()
        base = PositionBaselineModel()
        model = CalibratedModel(base, method="isotonic", cal_rounds=5)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == len(df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_sigmoid_produces_valid_probs(self):
        df, y = _make_sample_data()
        base = PositionBaselineModel()
        model = CalibratedModel(base, method="sigmoid", cal_rounds=5)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == len(df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_invalid_method_raises(self):
        base = PositionBaselineModel()
        with pytest.raises(ValueError, match="method must be"):
            CalibratedModel(base, method="invalid")

    def test_feature_names_from_base(self):
        df, y = _make_sample_data()
        base = PositionBaselineModel()
        model = CalibratedModel(base, method="isotonic")
        model.fit(df, y)
        assert model.feature_names() == base.feature_names()

    def test_temporal_split_respects_rounds(self):
        """Last cal_rounds should be held out for calibration."""
        df, y = _make_sample_data(n=500)
        base = PositionBaselineModel()
        cal_rounds = 5
        model = CalibratedModel(base, method="isotonic", cal_rounds=cal_rounds)
        model.fit(df, y)
        # The calibrator should exist (enough data for split)
        assert model._calibrator is not None

    def test_small_data_fallback(self):
        """With very few rounds, should still work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "round_number": [1] * 10,
            "position_group": rng.choice(["Back", "Forward"], 10),
        })
        y = rng.randint(0, 2, 10).astype(float)
        base = PositionBaselineModel()
        model = CalibratedModel(base, method="isotonic", cal_rounds=5)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == 10

    def test_no_round_column_fallback(self):
        """Without round_number, should use 80/20 split."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "position_group": rng.choice(["Back", "Forward", "Halfback"], 100),
        })
        y = rng.randint(0, 2, 100).astype(float)
        base = PositionBaselineModel()
        model = CalibratedModel(base, method="isotonic", cal_rounds=5)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == 100


class TestFlatStakeBacktest:
    """Test flat-stake mode in backtest config."""

    def test_flat_stake_config_default_none(self):
        from src.evaluation.backtest import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.flat_stake is None

    def test_flat_stake_config_set(self):
        from src.evaluation.backtest import BacktestConfig
        cfg = BacktestConfig(flat_stake=100.0)
        assert cfg.flat_stake == 100.0

    def test_flat_stake_sizing(self):
        from src.evaluation.backtest import BacktestConfig, apply_staking
        from src.models.baseline import BetRecommendation

        config = BacktestConfig(flat_stake=100.0)
        bets = [
            BetRecommendation(
                match_id=1, player_id=i,
                model_prob=0.35, implied_prob=0.25,
                odds=4.0, edge=0.10,
            )
            for i in range(5)
        ]
        result = apply_staking(bets, bankroll=10000, config=config)
        for bet in result:
            assert bet.stake == 100.0

    def test_flat_stake_respects_match_cap(self):
        from src.evaluation.backtest import BacktestConfig, apply_staking
        from src.models.baseline import BetRecommendation

        config = BacktestConfig(flat_stake=100.0, max_bets_per_match=2)
        # 5 bets in same match
        bets = [
            BetRecommendation(
                match_id=1, player_id=i,
                model_prob=0.35, implied_prob=0.25,
                odds=4.0, edge=0.10 - i * 0.01,
            )
            for i in range(5)
        ]
        result = apply_staking(bets, bankroll=10000, config=config)
        assert len(result) <= 2

    def test_flat_stake_respects_round_exposure(self):
        from src.evaluation.backtest import BacktestConfig, apply_staking
        from src.models.baseline import BetRecommendation

        # max_round_exposure = 20% of 1000 = 200, but 5 bets × 100 = 500
        config = BacktestConfig(flat_stake=100.0, max_round_exposure_pct=0.20)
        bets = [
            BetRecommendation(
                match_id=i, player_id=i,
                model_prob=0.35, implied_prob=0.25,
                odds=4.0, edge=0.10,
            )
            for i in range(5)
        ]
        result = apply_staking(bets, bankroll=1000, config=config)
        total = sum(b.stake for b in result)
        assert total <= 200.0 + 0.01  # Small tolerance

    def test_flat_stake_zero_edge_excluded(self):
        from src.evaluation.backtest import BacktestConfig, apply_staking
        from src.models.baseline import BetRecommendation

        config = BacktestConfig(flat_stake=100.0)
        bets = [
            BetRecommendation(
                match_id=1, player_id=1,
                model_prob=0.25, implied_prob=0.25,
                odds=4.0, edge=0.0,
            ),
        ]
        result = apply_staking(bets, bankroll=10000, config=config)
        assert len(result) == 0
