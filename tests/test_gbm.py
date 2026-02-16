"""Tests for LightGBM models."""

import numpy as np
import pandas as pd
import pytest

from src.models.gbm import (
    CATEGORICAL_COLS,
    EXCLUDE_COLS,
    GBMModel,
    GBMModelNoBetfair,
    _detect_features,
)


def _make_sample_data(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Create sample feature data for GBM testing."""
    rng = np.random.RandomState(seed)
    positions = ["Back", "Forward", "Halfback", "Hooker", "Interchange"]
    pos_codes = ["WG", "PR", "HB", "HK", "INT"]
    edges = ["left", "right", "middle"]
    df = pd.DataFrame({
        "match_id": np.arange(n) // 4 + 1,
        "player_id": np.arange(1, n + 1),
        "squad_id": rng.randint(1, 17, n),
        "opponent_squad_id": rng.randint(1, 17, n),
        "round_number": rng.randint(1, 28, n),
        "season": rng.choice([2024, 2025], n),
        "scored_try": rng.randint(0, 2, n),
        "position_group": rng.choice(positions, n),
        "position_code": rng.choice(pos_codes, n),
        "player_edge": rng.choice(edges, n),
        "is_home": rng.randint(0, 2, n),
        "is_starter": rng.randint(0, 2, n),
        "expected_team_tries_5": rng.uniform(2, 6, n),
        "player_try_share_5": rng.uniform(0, 0.2, n),
        "rolling_try_rate_5": rng.uniform(0, 0.5, n),
        "rolling_line_breaks_5": rng.uniform(0, 5, n),
        "edge_matchup_score_rolling_5": rng.uniform(0, 1, n),
        "betfair_implied_prob": rng.uniform(0.05, 0.6, n),
        "betfair_closing_odds": rng.uniform(1.5, 10, n),
        "betfair_total_matched_volume": rng.uniform(0, 1000, n),
    })
    # Add some NaN values
    df.loc[rng.choice(n, 30, replace=False), "expected_team_tries_5"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "edge_matchup_score_rolling_5"] = np.nan
    y = df["scored_try"].values
    return df, y


class TestDetectFeatures:
    def test_excludes_id_cols(self):
        df, _ = _make_sample_data()
        features = _detect_features(df)
        for col in EXCLUDE_COLS:
            assert col not in features, f"{col} should be excluded"

    def test_includes_numeric_features(self):
        df, _ = _make_sample_data()
        features = _detect_features(df)
        assert "expected_team_tries_5" in features
        assert "rolling_try_rate_5" in features
        assert "is_home" in features

    def test_includes_categorical_features(self):
        df, _ = _make_sample_data()
        features = _detect_features(df)
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                assert col in features

    def test_exclude_betfair(self):
        df, _ = _make_sample_data()
        features = _detect_features(df, exclude_betfair=True)
        for f in features:
            assert not f.startswith("betfair_"), f"Betfair feature {f} should be excluded"

    def test_include_betfair_by_default(self):
        df, _ = _make_sample_data()
        features = _detect_features(df, exclude_betfair=False)
        betfair_feats = [f for f in features if f.startswith("betfair_")]
        assert len(betfair_feats) > 0


class TestGBMModel:
    def test_fit_predict_shape(self):
        df, y = _make_sample_data()
        model = GBMModel(n_estimators=10)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == len(df)

    def test_probabilities_valid_range(self):
        df, y = _make_sample_data()
        model = GBMModel(n_estimators=10)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_handles_nan(self):
        """GBM should handle NaN without errors."""
        df, y = _make_sample_data()
        # Add more NaN
        df.loc[0:50, "rolling_line_breaks_5"] = np.nan
        model = GBMModel(n_estimators=10)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert not np.any(np.isnan(probs))

    def test_feature_names_correct(self):
        df, y = _make_sample_data()
        model = GBMModel(n_estimators=10)
        model.fit(df, y)
        names = model.feature_names()
        assert len(names) > 0
        for col in EXCLUDE_COLS:
            assert col not in names

    def test_feature_importance(self):
        df, y = _make_sample_data()
        model = GBMModel(n_estimators=10)
        model.fit(df, y)
        imp = model.feature_importance()
        assert "feature" in imp.columns
        assert "importance" in imp.columns
        assert len(imp) == len(model.feature_names())

    def test_predict_before_fit_raises(self):
        model = GBMModel()
        df, _ = _make_sample_data(n=300)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(df)

    def test_includes_betfair_features(self):
        df, y = _make_sample_data()
        model = GBMModel(n_estimators=10)
        model.fit(df, y)
        names = model.feature_names()
        betfair = [n for n in names if n.startswith("betfair_")]
        assert len(betfair) > 0


class TestGBMModelNoBetfair:
    def test_excludes_betfair(self):
        df, y = _make_sample_data()
        model = GBMModelNoBetfair(n_estimators=10)
        model.fit(df, y)
        names = model.feature_names()
        betfair = [n for n in names if n.startswith("betfair_")]
        assert len(betfair) == 0

    def test_fit_predict_works(self):
        df, y = _make_sample_data()
        model = GBMModelNoBetfair(n_estimators=10)
        model.fit(df, y)
        probs = model.predict_proba(df)
        assert len(probs) == len(df)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_fewer_features_than_full(self):
        df, y = _make_sample_data()
        full_model = GBMModel(n_estimators=10)
        no_bf_model = GBMModelNoBetfair(n_estimators=10)
        full_model.fit(df, y)
        no_bf_model.fit(df, y)
        assert len(no_bf_model.feature_names()) < len(full_model.feature_names())
