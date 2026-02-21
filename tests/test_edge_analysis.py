"""Tests for edge discovery and segment analysis functions."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.edge_analysis import (
    ODDS_BANDS,
    ODDS_LABELS,
    TEAM_TRIES_BINS,
    TEAM_TRIES_LABELS,
    _compute_segment_stats,
    _enrich_bets,
    calibration_by_position,
    clv_analysis,
    cumulative_pnl_by_round,
    generate_edge_report,
    odds_band_roi,
    season_breakdown,
    segment_roi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bet_df():
    """Sample bet DataFrame mimicking BacktestResult.to_bet_dataframe()."""
    np.random.seed(42)
    n = 100
    positions = np.random.choice(["WG", "CE", "FB", "SR", "HB", "FE", "LK"], n)
    odds = np.random.uniform(1.5, 8.0, n)
    won = np.random.binomial(1, 0.3, n)
    stake = np.full(n, 100.0)
    payout = np.where(won, stake * odds, 0.0)

    return pd.DataFrame({
        "match_id": np.arange(1, n + 1),
        "player_id": np.arange(1001, 1001 + n),
        "position_code": positions,
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
    """Sample feature store for enrichment testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "match_id": np.arange(1, n + 1),
        "player_id": np.arange(1001, 1001 + n),
        "edge_matchup_score_rolling_5": np.random.uniform(0.0, 2.0, n),
        "expected_team_tries_5": np.random.uniform(2.0, 6.0, n),
        "player_edge": np.random.choice(["left", "right", "middle", "none"], n),
        "team_edge_try_share_rolling_5": np.random.uniform(0.0, 0.5, n),
    })


# ---------------------------------------------------------------------------
# _compute_segment_stats
# ---------------------------------------------------------------------------

class TestComputeSegmentStats:
    def test_basic_stats(self, sample_bet_df):
        stats = _compute_segment_stats(sample_bet_df)
        assert stats["n_bets"] == 100
        assert stats["total_staked"] == 10000.0
        assert stats["hit_rate"] > 0
        assert stats["avg_odds"] > 1.0
        assert isinstance(stats["roi"], float)

    def test_empty_group(self):
        empty = pd.DataFrame({
            "stake": [], "payout": [], "won": [], "odds": [], "edge": [],
        })
        stats = _compute_segment_stats(empty)
        assert stats["n_bets"] == 0
        assert stats["roi"] == 0.0
        assert stats["hit_rate"] == 0.0

    def test_all_winners(self):
        df = pd.DataFrame({
            "stake": [100.0, 100.0],
            "payout": [300.0, 400.0],
            "won": [1, 1],
            "odds": [3.0, 4.0],
            "edge": [0.10, 0.15],
        })
        stats = _compute_segment_stats(df)
        assert stats["hit_rate"] == 1.0
        assert stats["roi"] == 2.5  # (700-200)/200

    def test_all_losers(self):
        df = pd.DataFrame({
            "stake": [100.0, 100.0],
            "payout": [0.0, 0.0],
            "won": [0, 0],
            "odds": [3.0, 4.0],
            "edge": [0.10, 0.15],
        })
        stats = _compute_segment_stats(df)
        assert stats["hit_rate"] == 0.0
        assert stats["roi"] == -1.0


# ---------------------------------------------------------------------------
# _enrich_bets
# ---------------------------------------------------------------------------

class TestEnrichBets:
    def test_enrichment_adds_columns(self, sample_bet_df, sample_feature_store):
        enriched = _enrich_bets(
            sample_bet_df, sample_feature_store,
            ["edge_matchup_score_rolling_5", "expected_team_tries_5"],
        )
        assert "edge_matchup_score_rolling_5" in enriched.columns
        assert "expected_team_tries_5" in enriched.columns
        assert len(enriched) == len(sample_bet_df)

    def test_enrichment_empty_bet_df(self, sample_feature_store):
        empty = pd.DataFrame()
        result = _enrich_bets(empty, sample_feature_store, ["edge_matchup_score_rolling_5"])
        assert result.empty

    def test_enrichment_missing_column(self, sample_bet_df, sample_feature_store):
        enriched = _enrich_bets(
            sample_bet_df, sample_feature_store, ["nonexistent_col"],
        )
        assert "nonexistent_col" not in enriched.columns
        assert len(enriched) == len(sample_bet_df)


# ---------------------------------------------------------------------------
# segment_roi
# ---------------------------------------------------------------------------

class TestSegmentRoi:
    def test_by_position(self, sample_bet_df):
        result = segment_roi(sample_bet_df, "position_code")
        assert not result.empty
        assert "segment" in result.columns
        assert "roi" in result.columns
        assert "n_bets" in result.columns
        # All positions should be represented
        assert set(result["segment"]) <= {"WG", "CE", "FB", "SR", "HB", "FE", "LK"}
        # Total bets should sum to 100
        assert result["n_bets"].sum() == 100

    def test_with_bins(self, sample_bet_df):
        result = segment_roi(
            sample_bet_df,
            segment_col="odds",
            bins=ODDS_BANDS,
            labels=ODDS_LABELS,
        )
        assert not result.empty
        assert all(s in ODDS_LABELS for s in result["segment"])

    def test_empty_df(self):
        result = segment_roi(pd.DataFrame(), "position_code")
        assert result.empty

    def test_missing_column_raises(self, sample_bet_df):
        with pytest.raises(ValueError, match="not found"):
            segment_roi(sample_bet_df, "nonexistent_col")

    def test_single_segment(self):
        df = pd.DataFrame({
            "position_code": ["WG", "WG", "WG"],
            "stake": [100.0, 100.0, 100.0],
            "payout": [300.0, 0.0, 0.0],
            "won": [1, 0, 0],
            "odds": [3.0, 3.0, 3.0],
            "edge": [0.10, 0.10, 0.10],
        })
        result = segment_roi(df, "position_code")
        assert len(result) == 1
        assert result.iloc[0]["segment"] == "WG"
        assert result.iloc[0]["n_bets"] == 3


# ---------------------------------------------------------------------------
# odds_band_roi
# ---------------------------------------------------------------------------

class TestOddsBandRoi:
    def test_returns_bands(self, sample_bet_df):
        result = odds_band_roi(sample_bet_df)
        assert not result.empty
        assert all(s in ODDS_LABELS for s in result["segment"])

    def test_all_odds_in_one_band(self):
        df = pd.DataFrame({
            "odds": [2.5, 2.8, 2.2],
            "stake": [100.0, 100.0, 100.0],
            "payout": [250.0, 0.0, 220.0],
            "won": [1, 0, 1],
            "edge": [0.10, 0.10, 0.10],
        })
        result = odds_band_roi(df)
        assert len(result) == 1
        assert result.iloc[0]["segment"] == "$2.00-3.00"


# ---------------------------------------------------------------------------
# season_breakdown
# ---------------------------------------------------------------------------

class TestSeasonBreakdown:
    def test_two_seasons(self, sample_bet_df):
        result = season_breakdown(sample_bet_df)
        assert len(result) == 2
        assert set(result["segment"]) == {"2024", "2025"}

    def test_single_season(self):
        df = pd.DataFrame({
            "season": [2024, 2024],
            "stake": [100.0, 100.0],
            "payout": [0.0, 300.0],
            "won": [0, 1],
            "odds": [3.0, 3.0],
            "edge": [0.10, 0.10],
        })
        result = season_breakdown(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# clv_analysis
# ---------------------------------------------------------------------------

class TestClvAnalysis:
    def test_returns_three_groups(self, sample_bet_df):
        result = clv_analysis(sample_bet_df)
        assert len(result) == 3
        assert set(result["group"]) == {"winners", "losers", "overall"}

    def test_all_columns_present(self, sample_bet_df):
        result = clv_analysis(sample_bet_df)
        for col in ["group", "n_bets", "avg_clv", "avg_model_prob", "avg_implied_prob"]:
            assert col in result.columns

    def test_empty_df(self):
        result = clv_analysis(pd.DataFrame())
        assert result.empty

    def test_overall_count_matches(self, sample_bet_df):
        result = clv_analysis(sample_bet_df)
        overall = result[result["group"] == "overall"].iloc[0]
        assert overall["n_bets"] == len(sample_bet_df)


# ---------------------------------------------------------------------------
# calibration_by_position
# ---------------------------------------------------------------------------

class TestCalibrationByPosition:
    def test_returns_per_position(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.6, 0.3, 0.7, 0.2, 0.5, 0.4, 0.8, 0.1, 0.6, 0.3])
        positions = np.array(["WG", "WG", "WG", "WG", "WG", "CE", "CE", "CE", "CE", "CE"])
        result = calibration_by_position(y_true, y_prob, positions)
        assert len(result) == 2
        assert set(result["position"]) == {"CE", "WG"}
        assert "ece" in result.columns
        assert "n_obs" in result.columns

    def test_skips_small_groups(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.6, 0.3, 0.7, 0.2, 0.5, 0.4, 0.8])
        positions = np.array(["WG", "WG", "WG", "WG", "WG", "CE", "CE"])
        result = calibration_by_position(y_true, y_prob, positions)
        # CE has only 2 observations, should be skipped (< 5)
        assert len(result) == 1
        assert result.iloc[0]["position"] == "WG"


# ---------------------------------------------------------------------------
# cumulative_pnl_by_round
# ---------------------------------------------------------------------------

class TestCumulativePnlByRound:
    def test_basic(self, sample_bet_df):
        result = cumulative_pnl_by_round(sample_bet_df)
        assert not result.empty
        assert "cumulative_pnl" in result.columns
        assert "round_profit" in result.columns
        # Cumulative PNL should be running sum of round profits
        expected_cum = result["round_profit"].cumsum()
        np.testing.assert_array_almost_equal(
            result["cumulative_pnl"].values,
            expected_cum.values,
        )

    def test_empty(self):
        result = cumulative_pnl_by_round(pd.DataFrame())
        assert result.empty

    def test_sorted_by_season_round(self, sample_bet_df):
        result = cumulative_pnl_by_round(sample_bet_df)
        # Check sorted
        for i in range(1, len(result)):
            prev = (result.iloc[i - 1]["season"], result.iloc[i - 1]["round_number"])
            curr = (result.iloc[i]["season"], result.iloc[i]["round_number"])
            assert curr >= prev


# ---------------------------------------------------------------------------
# generate_edge_report
# ---------------------------------------------------------------------------

class TestGenerateEdgeReport:
    def test_report_keys(self, sample_bet_df, sample_feature_store):
        report = generate_edge_report(sample_bet_df, sample_feature_store)
        assert isinstance(report, dict)
        # Must have core sections
        assert "position_roi" in report
        assert "odds_band_roi" in report
        assert "season_roi" in report
        assert "clv" in report
        assert "cumulative_pnl" in report

    def test_report_enrichment(self, sample_bet_df, sample_feature_store):
        report = generate_edge_report(sample_bet_df, sample_feature_store)
        # Should have matchup and team tries analysis from enrichment
        assert "matchup_quartile_roi" in report
        assert "team_tries_roi" in report
        assert "edge_zone_roi" in report

    def test_empty_bets(self, sample_feature_store):
        report = generate_edge_report(pd.DataFrame(), sample_feature_store)
        assert report == {}

    def test_all_dataframes(self, sample_bet_df, sample_feature_store):
        report = generate_edge_report(sample_bet_df, sample_feature_store)
        for key, df in report.items():
            assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"
