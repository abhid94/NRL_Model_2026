"""Tests for devigging module."""

import numpy as np
import pytest

from src.odds.devig import (
    compute_overround,
    devig_binary,
    devig_bookmaker_ats,
    devig_multiplicative,
    devig_shin,
)


class TestDevigBinary:
    """Test binary (back/lay) devigging."""

    def test_back_lay_midpoint(self):
        """Binary devig should be midpoint of back/lay implied."""
        # Back 3.0 (33.3%), Lay 3.2 (31.25%)
        result = devig_binary(3.0, 3.2)
        expected = (1 / 3.0 + 1 / 3.2) / 2
        assert abs(result - expected) < 0.001

    def test_back_only_fallback(self):
        """Without lay, should apply ~2% reduction."""
        result = devig_binary(3.0)
        naive = 1 / 3.0
        assert result < naive  # Should be lower than naive
        assert result > naive * 0.95  # But not by much

    def test_true_prob_lower_than_naive(self):
        """Devigged prob should always be <= naive implied."""
        for odds in [2.0, 3.0, 5.0, 10.0, 20.0]:
            naive = 1 / odds
            result = devig_binary(odds, odds * 1.05)  # 5% spread
            assert result <= naive, f"Devigged {result} > naive {naive} for odds {odds}"

    def test_invalid_odds(self):
        """Invalid odds should return NaN."""
        assert np.isnan(devig_binary(0.5))
        assert np.isnan(devig_binary(-1.0))
        assert np.isnan(devig_binary(np.nan))

    def test_short_odds(self):
        """Short odds (favourite) should devig correctly."""
        result = devig_binary(1.5, 1.6)
        assert 0.6 < result < 0.7  # ~65% range

    def test_long_odds(self):
        """Long odds should devig correctly."""
        result = devig_binary(50.0, 55.0)
        assert 0.01 < result < 0.025

    def test_clipped_to_0_1(self):
        """Probabilities should be clipped to [0, 1]."""
        result = devig_binary(1.01, 1.01)  # Very short odds
        assert 0.0 <= result <= 1.0


class TestDevigBookmakerATS:
    """Test bookmaker ATS devigging."""

    def test_margin_correction(self):
        """Should apply correction factor correctly."""
        result = devig_bookmaker_ats(3.0, bookmaker_correction=0.88)
        expected = (1 / 3.0) * 0.88
        assert abs(result - expected) < 0.001

    def test_default_fallback(self):
        """Without correction, should use 0.88 default."""
        result = devig_bookmaker_ats(3.0)
        expected = (1 / 3.0) * 0.88
        assert abs(result - expected) < 0.001

    def test_always_less_than_naive(self):
        """Devigged prob should be less than naive for positive margin."""
        for odds in [2.0, 3.0, 5.0, 10.0]:
            result = devig_bookmaker_ats(odds, bookmaker_correction=0.88)
            assert result < 1 / odds

    def test_invalid_odds(self):
        """Invalid odds should return NaN."""
        assert np.isnan(devig_bookmaker_ats(0.5))
        assert np.isnan(devig_bookmaker_ats(-1.0))

    def test_clipped(self):
        """Result should be in [0, 1]."""
        result = devig_bookmaker_ats(1.01, bookmaker_correction=1.0)
        assert 0.0 <= result <= 1.0


class TestDevigShin:
    """Test Shin's method for mutually exclusive markets."""

    def test_sums_to_one(self):
        """Shin probs should sum to 1.0."""
        odds = [2.0, 3.0, 5.0]
        result = devig_shin(odds)
        valid = result[~np.isnan(result)]
        assert abs(valid.sum() - 1.0) < 0.01

    def test_preserves_ranking(self):
        """Shorter odds should have higher prob."""
        odds = [2.0, 3.0, 5.0, 10.0]
        result = devig_shin(odds)
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1]

    def test_reduces_from_naive(self):
        """Devigged probs should be less than naive (since market has margin)."""
        odds = [2.0, 3.0, 5.0]
        result = devig_shin(odds)
        naive = np.array([1 / o for o in odds])
        # At least some should be reduced (total goes from >1 to 1)
        assert result.sum() < naive.sum()

    def test_too_few_selections(self):
        """Should return NaN with fewer than 2 selections."""
        result = devig_shin([3.0])
        assert np.all(np.isnan(result))


class TestDevigMultiplicative:
    """Test standard multiplicative devigging."""

    def test_sums_to_one(self):
        """Should sum to 1.0."""
        odds = [2.0, 3.0, 5.0]
        result = devig_multiplicative(odds)
        valid = result[~np.isnan(result)]
        assert abs(valid.sum() - 1.0) < 1e-6

    def test_proportional_scaling(self):
        """Ratios between probs should be same as ratios of naive."""
        odds = [2.0, 4.0]
        result = devig_multiplicative(odds)
        # 2.0 should be exactly 2x the prob of 4.0
        assert abs(result[0] / result[1] - 2.0) < 1e-6


class TestComputeOverround:
    """Test overround calculation."""

    def test_fair_market(self):
        """Odds summing to exactly 1.0 implied."""
        odds = [2.0, 2.0]  # 50% + 50% = 100%
        assert abs(compute_overround(odds) - 1.0) < 1e-6

    def test_standard_margin(self):
        """Typical bookmaker margin."""
        odds = [1.9, 1.9]  # ~105% overround
        result = compute_overround(odds)
        assert result > 1.0

    def test_empty(self):
        """Empty list should return 0."""
        assert compute_overround([]) == 0.0

    def test_ignores_invalid(self):
        """Should skip invalid odds."""
        odds = [2.0, -1.0, 0.5, 3.0]
        result = compute_overround(odds)
        expected = 1 / 2.0 + 1 / 3.0
        assert abs(result - expected) < 1e-6
