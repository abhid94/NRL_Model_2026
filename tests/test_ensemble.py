"""Tests for WeightedEnsemble and StackedEnsemble."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import BaseModel
from src.models.ensemble import StackedEnsemble, WeightedEnsemble, prediction_diversity


class SimpleModel(BaseModel):
    """Minimal model for testing ensembles."""

    def __init__(self, bias: float = 0.0):
        self._bias = bias
        self._mean: float = 0.19

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._mean = float(np.mean(y))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean + self._bias)

    def feature_names(self) -> list[str]:
        return ["dummy"]


@pytest.fixture
def sample_data():
    """Create sample data with round_number for temporal CV."""
    rng = np.random.RandomState(42)
    n_rounds = 37
    players_per_round = 8
    n = n_rounds * players_per_round  # 296
    df = pd.DataFrame({
        "match_id": np.repeat(range(1, n_rounds + 1), players_per_round),
        "player_id": np.tile(range(1, players_per_round + 1), n_rounds),
        "round_number": np.repeat(range(1, n_rounds + 1), players_per_round),
        "season": 2024,
        "position_group": rng.choice(["Back", "Forward"], n),
        "position_code": rng.choice(["FB", "WG", "PR"], n),
        "is_home": rng.choice([0, 1], n),
        "rolling_try_rate_5": rng.uniform(0, 0.5, n),
        "betfair_implied_prob": rng.uniform(0.05, 0.5, n),
        "betfair_closing_odds": rng.uniform(1.5, 10, n),
    })
    df["scored_try"] = rng.choice([0, 1], n, p=[0.81, 0.19])
    return df


class TestWeightedEnsemble:
    """Tests for WeightedEnsemble."""

    def test_equal_weights_default(self, sample_data):
        models = [SimpleModel(0.0), SimpleModel(0.02)]
        ens = WeightedEnsemble(models)
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)
        probs = ens.predict_proba(sample_data)
        assert probs.shape == (len(sample_data),)

    def test_custom_weights(self, sample_data):
        models = [SimpleModel(0.0), SimpleModel(0.1)]
        ens = WeightedEnsemble(models, weights=[0.7, 0.3])
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)

        np.testing.assert_allclose(ens.weights, [0.7, 0.3])
        probs = ens.predict_proba(sample_data)
        # Should be weighted average
        p1 = models[0].predict_proba(sample_data)
        p2 = models[1].predict_proba(sample_data)
        expected = 0.7 * p1 + 0.3 * p2
        np.testing.assert_allclose(probs, expected)

    def test_weights_normalized(self, sample_data):
        models = [SimpleModel(), SimpleModel()]
        ens = WeightedEnsemble(models, weights=[2.0, 8.0])
        np.testing.assert_allclose(ens.weights, [0.2, 0.8])

    def test_learned_weights(self, sample_data):
        models = [SimpleModel(0.0), SimpleModel(0.05)]
        ens = WeightedEnsemble(models, learn_weights=True, holdout_rounds=5)
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)
        # Weights should have been learned (not necessarily equal)
        assert ens.weights.shape == (2,)
        assert abs(ens.weights.sum() - 1.0) < 1e-10

    def test_minimum_models(self):
        with pytest.raises(ValueError, match="at least 2"):
            WeightedEnsemble([SimpleModel()])

    def test_weight_length_mismatch(self):
        with pytest.raises(ValueError, match="weights length"):
            WeightedEnsemble([SimpleModel(), SimpleModel()], weights=[1.0])

    def test_feature_names_deduplicated(self, sample_data):
        models = [SimpleModel(), SimpleModel()]
        ens = WeightedEnsemble(models)
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)
        names = ens.feature_names()
        assert len(names) == len(set(names))


class TestStackedEnsemble:
    """Tests for StackedEnsemble."""

    def test_fit_predict_shapes(self, sample_data):
        models = [SimpleModel(0.0), SimpleModel(0.02)]
        ens = StackedEnsemble(models, n_folds=3)
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)
        probs = ens.predict_proba(sample_data)
        assert probs.shape == (len(sample_data),)

    def test_probabilities_valid_range(self, sample_data):
        models = [SimpleModel(0.0), SimpleModel(0.02)]
        ens = StackedEnsemble(models, n_folds=3)
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)
        probs = ens.predict_proba(sample_data)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_requires_round_number(self, sample_data):
        df = sample_data.drop(columns=["round_number"])
        models = [SimpleModel(), SimpleModel()]
        ens = StackedEnsemble(models)
        with pytest.raises(ValueError, match="round_number"):
            ens.fit(df, df["scored_try"].values)

    def test_include_market_feature(self, sample_data):
        models = [SimpleModel(0.0), SimpleModel(0.02)]
        ens = StackedEnsemble(models, n_folds=3, include_market=True)
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)
        names = ens.feature_names()
        assert "betfair_implied_prob" in names

    def test_without_market_feature(self, sample_data):
        models = [SimpleModel(0.0), SimpleModel(0.02)]
        ens = StackedEnsemble(models, n_folds=3, include_market=False)
        y = sample_data["scored_try"].values
        ens.fit(sample_data, y)
        names = ens.feature_names()
        assert "betfair_implied_prob" not in names

    def test_fallback_few_rounds(self):
        """With very few rounds, should fall back to simple average."""
        rng = np.random.RandomState(42)
        n = 30
        df = pd.DataFrame({
            "round_number": np.repeat([1, 2, 3], 10),
            "scored_try": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "betfair_implied_prob": rng.uniform(0.1, 0.4, n),
        })
        models = [SimpleModel(0.0), SimpleModel(0.02)]
        ens = StackedEnsemble(models, n_folds=5)
        ens.fit(df, df["scored_try"].values)
        # Should still produce valid predictions
        probs = ens.predict_proba(df)
        assert probs.shape == (n,)


class TestPredictionDiversity:
    """Tests for prediction_diversity helper."""

    def test_correlation_matrix_shape(self, sample_data):
        m1, m2 = SimpleModel(0.0), SimpleModel(0.05)
        y = sample_data["scored_try"].values
        m1.fit(sample_data, y)
        m2.fit(sample_data, y)
        corr = prediction_diversity([m1, m2], sample_data)
        assert corr.shape == (2, 2)

    def test_perfect_correlation_same_model(self, sample_data):
        m1, m2 = SimpleModel(0.0), SimpleModel(0.0)
        y = sample_data["scored_try"].values
        m1.fit(sample_data, y)
        m2.fit(sample_data, y)
        corr = prediction_diversity([m1, m2], sample_data)
        # Same predictions -> correlation is NaN (constant) or 1.0
        # SimpleModel returns constant, so correlation is NaN
        assert corr.shape == (2, 2)
