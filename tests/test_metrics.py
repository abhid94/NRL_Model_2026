"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    build_evaluation_report,
    compute_auc,
    compute_brier_score,
    compute_calibration_error,
    compute_calibration_curve,
    compute_clv,
    compute_log_loss,
    compute_max_drawdown,
    compute_pnl_series,
    compute_pr_auc,
    compute_roi,
    compute_segment_metrics,
    compute_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------

class TestAUC:
    def test_perfect_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_prob) == 1.0

    def test_random_auc_near_half(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=10000)
        y_prob = rng.rand(10000)
        auc = compute_auc(y_true, y_prob)
        assert 0.45 < auc < 0.55

    def test_single_class_returns_nan(self):
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.5, 0.6, 0.7])
        assert np.isnan(compute_auc(y_true, y_prob))


class TestPRAUC:
    def test_perfect_pr_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_pr_auc(y_true, y_prob) == pytest.approx(1.0)

    def test_single_class_returns_nan(self):
        assert np.isnan(compute_pr_auc(np.array([0, 0]), np.array([0.1, 0.2])))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert compute_brier_score(y_true, y_prob) == 0.0

    def test_worst_predictions(self):
        y_true = np.array([0, 1])
        y_prob = np.array([1.0, 0.0])
        assert compute_brier_score(y_true, y_prob) == 1.0


class TestCalibrationError:
    def test_perfectly_calibrated(self):
        # Create a large sample where predicted = actual rate in each bin
        rng = np.random.RandomState(42)
        n = 10000
        y_prob = rng.rand(n)
        y_true = (rng.rand(n) < y_prob).astype(float)
        ece = compute_calibration_error(y_true, y_prob, n_bins=10)
        assert ece < 0.03  # Should be near zero with large sample

    def test_empty_input(self):
        assert compute_calibration_error(np.array([]), np.array([])) == 0.0


class TestCalibrationCurve:
    def test_returns_dataframe(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        df = compute_calibration_curve(y_true, y_prob, n_bins=5)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"bin_mid", "avg_predicted", "avg_actual", "count"}


class TestLogLoss:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.0, 1.0])
        # With clipping, not exactly 0 but very small
        assert compute_log_loss(y_true, y_prob) < 0.001


# ---------------------------------------------------------------------------
# Economic metrics
# ---------------------------------------------------------------------------

class TestROI:
    def test_breakeven(self):
        stakes = np.array([100.0, 100.0])
        payouts = np.array([200.0, 0.0])
        assert compute_roi(stakes, payouts) == 0.0

    def test_positive_roi(self):
        stakes = np.array([100.0])
        payouts = np.array([150.0])
        assert compute_roi(stakes, payouts) == pytest.approx(0.5)

    def test_zero_stakes(self):
        assert compute_roi(np.array([]), np.array([])) == 0.0


class TestCLV:
    def test_positive_clv(self):
        model = np.array([0.30, 0.25])
        implied = np.array([0.20, 0.20])
        assert compute_clv(model, implied) == pytest.approx(0.075)

    def test_empty(self):
        assert compute_clv(np.array([]), np.array([])) == 0.0


class TestPnLSeries:
    def test_cumulative(self):
        stakes = np.array([10.0, 10.0, 10.0])
        payouts = np.array([20.0, 0.0, 15.0])
        pnl = compute_pnl_series(stakes, payouts)
        np.testing.assert_array_almost_equal(pnl, [10.0, 0.0, 5.0])


class TestMaxDrawdown:
    def test_no_drawdown(self):
        pnl = np.array([1.0, 2.0, 3.0])
        assert compute_max_drawdown(pnl) == 0.0

    def test_known_drawdown(self):
        pnl = np.array([10.0, 5.0, 8.0, 3.0, 12.0])
        # Peak at 10, then drops to 3 -> drawdown = 7
        assert compute_max_drawdown(pnl) == 7.0

    def test_empty(self):
        assert compute_max_drawdown(np.array([])) == 0.0


class TestSharpeRatio:
    def test_no_variance(self):
        stakes = np.array([10.0, 10.0])
        payouts = np.array([10.0, 10.0])
        assert compute_sharpe_ratio(stakes, payouts) == 0.0

    def test_positive_sharpe(self):
        stakes = np.array([10.0, 10.0, 10.0])
        payouts = np.array([15.0, 12.0, 14.0])
        assert compute_sharpe_ratio(stakes, payouts) > 0


# ---------------------------------------------------------------------------
# Segment metrics
# ---------------------------------------------------------------------------

class TestSegmentMetrics:
    def test_basic_segments(self):
        df = pd.DataFrame({
            "position": ["WG", "WG", "CE", "CE"],
            "stake": [10.0, 10.0, 10.0, 10.0],
            "payout": [25.0, 0.0, 30.0, 0.0],
            "won": [1, 0, 1, 0],
        })
        result = compute_segment_metrics(df, "position")
        assert len(result) == 2
        wg = result[result["segment"] == "WG"].iloc[0]
        assert wg["n_bets"] == 2
        assert wg["roi"] == pytest.approx(0.25)
        assert wg["hit_rate"] == pytest.approx(0.5)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"stake": [1], "payout": [2], "won": [1]})
        with pytest.raises(ValueError, match="not found"):
            compute_segment_metrics(df, "missing_col")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    def test_basic_report(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        report = build_evaluation_report(y_true, y_prob)
        assert "auc" in report
        assert "brier_score" in report
        assert report["n_predictions"] == 4

    def test_with_economic_metrics(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        stakes = np.array([10.0, 10.0])
        payouts = np.array([0.0, 25.0])
        report = build_evaluation_report(
            y_true, y_prob, stakes=stakes, payouts=payouts,
            model_prob=np.array([0.4]), implied_prob=np.array([0.3]),
        )
        assert "roi" in report
        assert "clv" in report
        assert report["n_bets"] == 2
