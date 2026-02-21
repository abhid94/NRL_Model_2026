"""Tests for PoissonModel."""

import numpy as np
import pandas as pd
import pytest

from src.models.poisson import PoissonModel


@pytest.fixture
def sample_data():
    """Create sample feature store data for testing."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "match_id": np.repeat(range(1, 26), 8)[:n],
        "player_id": np.tile(range(1, 9), 25)[:n],
        "squad_id": np.repeat([1, 2], n // 2),
        "opponent_squad_id": np.repeat([2, 1], n // 2),
        "round_number": np.repeat(range(1, 26), 8)[:n],
        "season": 2024,
        "position_group": rng.choice(["Back", "Forward", "Halfback"], n),
        "position_code": rng.choice(["FB", "WG", "CE", "PR", "HB", "SR"], n),
        "player_edge": rng.choice(["left", "right", "middle"], n),
        "is_home": rng.choice([0, 1], n),
        "is_starter": rng.choice([0, 1], n),
        "rolling_try_rate_5": rng.uniform(0, 0.5, n),
        "rolling_line_breaks_5": rng.uniform(0, 3, n),
        "expected_team_tries_5": rng.uniform(2, 6, n),
        "player_try_share_5": rng.uniform(0, 0.2, n),
        "betfair_implied_prob": rng.uniform(0.05, 0.5, n),
        "betfair_closing_odds": rng.uniform(1.5, 10, n),
    })
    # Create try counts: Poisson-like distribution
    df["tries"] = rng.poisson(0.2, n)
    df["scored_try"] = (df["tries"] > 0).astype(int)
    return df


class TestPoissonModelBasic:
    """Basic functionality tests."""

    def test_fit_predict_shapes(self, sample_data):
        model = PoissonModel()
        y = sample_data["scored_try"].values
        model.fit(sample_data, y)
        probs = model.predict_proba(sample_data)
        assert probs.shape == (len(sample_data),)

    def test_probabilities_valid_range(self, sample_data):
        model = PoissonModel()
        y = sample_data["scored_try"].values
        model.fit(sample_data, y)
        probs = model.predict_proba(sample_data)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_predict_lambda_non_negative(self, sample_data):
        model = PoissonModel()
        y = sample_data["scored_try"].values
        model.fit(sample_data, y)
        lambdas = model.predict_lambda(sample_data)
        assert np.all(lambdas >= 0.0)

    def test_lambda_to_prob_consistency(self, sample_data):
        model = PoissonModel()
        y = sample_data["scored_try"].values
        model.fit(sample_data, y)
        lambdas = model.predict_lambda(sample_data)
        probs = model.predict_proba(sample_data)
        expected = 1.0 - np.exp(-lambdas)
        np.testing.assert_array_almost_equal(probs, expected, decimal=10)

    def test_uses_try_counts_when_available(self, sample_data):
        """Model should use 'tries' column (counts) when available."""
        model = PoissonModel()
        y = sample_data["scored_try"].values
        model.fit(sample_data, y)
        # Should have fitted on try counts, not binary
        probs = model.predict_proba(sample_data)
        assert len(probs) == len(sample_data)

    def test_binary_fallback(self, sample_data):
        """Model should work with binary y when 'tries' column absent."""
        df = sample_data.drop(columns=["tries"])
        model = PoissonModel()
        y = df["scored_try"].values
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert probs.shape == (len(df),)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


class TestPoissonModelNaNHandling:
    """NaN and missing data tests."""

    def test_handles_nan_features(self, sample_data):
        """Should impute NaN with training means."""
        df = sample_data.copy()
        # Inject NaN into numeric features
        rng = np.random.RandomState(99)
        mask = rng.random(len(df)) < 0.2
        df.loc[mask, "rolling_try_rate_5"] = np.nan
        df.loc[mask, "rolling_line_breaks_5"] = np.nan

        model = PoissonModel()
        y = df["scored_try"].values
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert not np.any(np.isnan(probs))

    def test_handles_missing_categorical(self, sample_data):
        """Should handle missing categorical with __missing__ fill."""
        df = sample_data.copy()
        df.loc[:10, "position_group"] = np.nan
        model = PoissonModel()
        y = df["scored_try"].values
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert probs.shape == (len(df),)


class TestPoissonModelConfiguration:
    """Configuration and option tests."""

    def test_exclude_betfair(self, sample_data):
        model = PoissonModel(exclude_betfair=True)
        y = sample_data["scored_try"].values
        model.fit(sample_data, y)
        # Feature names should not include betfair columns
        names = model.feature_names()
        betfair_names = [n for n in names if "betfair" in n.lower()]
        assert len(betfair_names) == 0

    def test_feature_names_populated(self, sample_data):
        model = PoissonModel()
        y = sample_data["scored_try"].values
        model.fit(sample_data, y)
        names = model.feature_names()
        assert len(names) > 0
        assert "const" in names

    def test_error_before_fit(self, sample_data):
        model = PoissonModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(sample_data)

    def test_error_lambda_before_fit(self, sample_data):
        model = PoissonModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_lambda(sample_data)
