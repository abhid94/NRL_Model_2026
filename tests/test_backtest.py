"""Tests for the walk-forward backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_INITIAL_BANKROLL,
    DEFAULT_KELLY_FRACTION,
    MAX_BETS_PER_MATCH,
    MAX_BETS_PER_ROUND,
    MAX_ROUND_EXPOSURE_PCT,
    MAX_STAKE_PCT,
    MIN_STAKE,
)
from src.evaluation.backtest import (
    BacktestConfig,
    BacktestResult,
    RoundResult,
    apply_staking,
    compare_backtests,
    run_backtest,
)
from src.models.baseline import (
    BetRecommendation,
    ModelEdgeStrategy,
    PositionBaselineModel,
    SegmentPlayStrategy,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestBacktestConfig:
    def test_defaults_match_claude_md(self):
        cfg = BacktestConfig()
        assert cfg.initial_bankroll == DEFAULT_INITIAL_BANKROLL
        assert cfg.kelly_fraction == DEFAULT_KELLY_FRACTION
        assert cfg.max_stake_pct == MAX_STAKE_PCT
        assert cfg.max_round_exposure_pct == MAX_ROUND_EXPOSURE_PCT
        assert cfg.max_bets_per_round == MAX_BETS_PER_ROUND
        assert cfg.max_bets_per_match == MAX_BETS_PER_MATCH
        assert cfg.min_stake == MIN_STAKE

    def test_custom_config(self):
        cfg = BacktestConfig(initial_bankroll=5000, kelly_fraction=0.1)
        assert cfg.initial_bankroll == 5000
        assert cfg.kelly_fraction == 0.1


# ---------------------------------------------------------------------------
# Staking constraints
# ---------------------------------------------------------------------------

class TestApplyStaking:
    def _make_bet(self, match_id=1, player_id=1, edge=0.10, odds=3.0):
        return BetRecommendation(
            match_id=match_id,
            player_id=player_id,
            model_prob=0.40,
            implied_prob=0.30,
            odds=odds,
            edge=edge,
        )

    def test_kelly_sizing(self):
        config = BacktestConfig()
        bet = self._make_bet(edge=0.10, odds=3.0)
        bets = apply_staking([bet], bankroll=10000, config=config)
        assert len(bets) == 1
        # Kelly = 0.25 * 0.10 / (3.0 - 1) * 10000 = 125
        assert bets[0].stake == pytest.approx(125.0)

    def test_per_bet_cap(self):
        config = BacktestConfig()
        bet = self._make_bet(edge=0.50, odds=1.5)  # Very high Kelly
        bets = apply_staking([bet], bankroll=10000, config=config)
        assert bets[0].stake <= config.max_stake_pct * 10000

    def test_min_stake_filter(self):
        config = BacktestConfig()
        bet = self._make_bet(edge=0.001, odds=5.0)  # Tiny edge -> tiny stake
        bets = apply_staking([bet], bankroll=100, config=config)
        # With bankroll=100, Kelly stake would be tiny
        assert all(b.stake >= config.min_stake for b in bets) or len(bets) == 0

    def test_per_match_cap(self):
        config = BacktestConfig()
        # 6 bets on same match
        bets = [self._make_bet(match_id=1, player_id=i, edge=0.10) for i in range(6)]
        result = apply_staking(bets, bankroll=10000, config=config)
        match_counts = {}
        for b in result:
            match_counts[b.match_id] = match_counts.get(b.match_id, 0) + 1
        assert all(c <= config.max_bets_per_match for c in match_counts.values())

    def test_per_round_exposure_cap(self):
        config = BacktestConfig()
        # Many large bets
        bets = [
            self._make_bet(match_id=i, player_id=i, edge=0.20, odds=2.0)
            for i in range(20)
        ]
        result = apply_staking(bets, bankroll=10000, config=config)
        total = sum(b.stake for b in result)
        assert total <= config.max_round_exposure_pct * 10000 + 1.0  # Small float tolerance

    def test_bet_count_cap(self):
        config = BacktestConfig()
        bets = [
            self._make_bet(match_id=i, player_id=i, edge=0.10)
            for i in range(30)
        ]
        result = apply_staking(bets, bankroll=10000, config=config)
        assert len(result) <= config.max_bets_per_round

    def test_zero_edge_dropped(self):
        config = BacktestConfig()
        bet = self._make_bet(edge=0.0)
        bets = apply_staking([bet], bankroll=10000, config=config)
        assert len(bets) == 0

    def test_empty_bets(self):
        assert apply_staking([], 10000, BacktestConfig()) == []

    def test_zero_bankroll(self):
        bet = self._make_bet()
        assert apply_staking([bet], 0, BacktestConfig()) == []


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_summary(self):
        result = BacktestResult(
            strategy_name="test",
            model_name="test_model",
            config=BacktestConfig(),
            round_results=[
                RoundResult(
                    season=2024, round_number=3,
                    bets=[{"won": 1, "stake": 100, "payout": 300, "odds": 3.0, "edge": 0.1}],
                    n_bets=1, total_staked=100, total_payout=300,
                    profit=200, bankroll_after=10200,
                ),
            ],
        )
        s = result.summary()
        assert s["n_bets"] == 1
        assert s["roi"] == pytest.approx(2.0)
        assert s["final_bankroll"] == 10200

    def test_to_bet_dataframe(self):
        result = BacktestResult(
            strategy_name="test",
            model_name="m",
            config=BacktestConfig(),
            round_results=[
                RoundResult(
                    season=2024, round_number=3,
                    bets=[{"match_id": 1, "player_id": 10, "won": 1, "stake": 50, "payout": 150}],
                    n_bets=1, total_staked=50, total_payout=150,
                    profit=100, bankroll_after=10100,
                ),
            ],
        )
        df = result.to_bet_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["season"] == 2024


# ---------------------------------------------------------------------------
# Temporal integrity
# ---------------------------------------------------------------------------

def _make_feature_store(n_rounds: int = 10, n_players_per_round: int = 20) -> pd.DataFrame:
    """Build a synthetic feature store for backtest testing."""
    rng = np.random.RandomState(42)
    rows = []
    for rnd in range(1, n_rounds + 1):
        for p in range(1, n_players_per_round + 1):
            rows.append({
                "match_id": rnd * 100 + (p - 1) // 4,
                "player_id": p,
                "round_number": rnd,
                "season": 2024,
                "position_group": rng.choice(["Back", "Forward", "Halfback"]),
                "position_code": rng.choice(["WG", "CE", "FB", "PR", "HB"]),
                "is_home": rng.randint(0, 2),
                "is_starter": rng.randint(0, 2),
                "expected_team_tries_5": rng.uniform(2, 6),
                "player_try_share_5": rng.uniform(0, 0.2),
                "opponent_rolling_defence_tries_conceded_5": rng.uniform(2, 8),
                "rolling_try_rate_3": rng.uniform(0, 0.5),
                "betfair_implied_prob": rng.uniform(0.05, 0.5),
                "betfair_closing_odds": rng.uniform(2, 10),
                "edge_matchup_score_rolling_5": rng.uniform(0, 0.5),
                "scored_try": rng.randint(0, 2),
            })
    return pd.DataFrame(rows)


class TestBacktestTemporalIntegrity:
    def test_training_data_before_prediction(self):
        """Verify model never sees future data."""
        fs = _make_feature_store()
        model = PositionBaselineModel()
        strategy = ModelEdgeStrategy(min_edge=0.0)
        config = BacktestConfig()

        # We can't directly inspect training splits in run_backtest,
        # but we can verify the backtest completes without error
        # and produces round results only for rounds >= min_round
        result = run_backtest(fs, strategy, model=model, config=config, min_round=3)
        for rr in result.round_results:
            assert rr.round_number >= 3

    def test_cross_season_training(self):
        """When backtesting 2025, training includes all of 2024."""
        fs_2024 = _make_feature_store(n_rounds=5)
        fs_2025 = _make_feature_store(n_rounds=5)
        fs_2025["season"] = 2025
        fs_2025["match_id"] = fs_2025["match_id"] + 10000  # Avoid collisions
        fs = pd.concat([fs_2024, fs_2025], ignore_index=True)

        model = PositionBaselineModel()
        strategy = ModelEdgeStrategy(min_edge=0.0)
        result = run_backtest(fs, strategy, model=model, seasons=[2024, 2025], min_round=3)

        # Should have results from both seasons
        seasons_seen = {rr.season for rr in result.round_results}
        assert 2024 in seasons_seen
        assert 2025 in seasons_seen


class TestBacktestBankroll:
    def test_bankroll_accounting(self):
        """Bankroll should update correctly across rounds."""
        fs = _make_feature_store(n_rounds=5, n_players_per_round=10)
        model = PositionBaselineModel()
        strategy = ModelEdgeStrategy(min_edge=0.0)
        config = BacktestConfig(initial_bankroll=10000)

        result = run_backtest(fs, strategy, model=model, config=config, min_round=3)

        if result.round_results:
            expected_bankroll = config.initial_bankroll
            for rr in result.round_results:
                expected_bankroll += rr.profit
                assert rr.bankroll_after == pytest.approx(expected_bankroll, abs=0.01)


class TestBacktestRuleBased:
    def test_rule_based_strategy_no_model(self):
        """Rule-based strategies should work without a model."""
        fs = _make_feature_store()
        strategy = SegmentPlayStrategy()
        result = run_backtest(fs, strategy, model=None, min_round=3)
        # Should complete without error
        assert isinstance(result, BacktestResult)


class TestCompareBacktests:
    def test_comparison_table(self):
        r1 = BacktestResult("s1", "m1", BacktestConfig())
        r2 = BacktestResult("s2", "m2", BacktestConfig())
        df = compare_backtests([r1, r2])
        assert len(df) == 2
        assert "strategy" in df.columns
        assert "roi" in df.columns


# ---------------------------------------------------------------------------
# Integration test with real feature store
# ---------------------------------------------------------------------------

class TestBacktestIntegration:
    @pytest.fixture
    def feature_store(self):
        """Load real feature store if available."""
        try:
            df = pd.read_parquet("data/feature_store/feature_store_2024.parquet")
            return df
        except FileNotFoundError:
            pytest.skip("Feature store not available")

    def test_position_baseline_2024(self, feature_store):
        """Run position baseline on 2024 data — sanity check."""
        model = PositionBaselineModel()
        strategy = ModelEdgeStrategy(min_edge=0.05)
        config = BacktestConfig(initial_bankroll=10000)

        result = run_backtest(
            feature_store, strategy, model=model,
            config=config, seasons=[2024], min_round=5,
        )
        summary = result.summary()

        # Sanity checks — not testing for profit, just that it runs correctly
        assert summary["n_rounds"] > 0
        assert summary["n_bets"] >= 0
        assert summary["final_bankroll"] > 0
        # Position baseline should NOT dramatically profit — it's the floor
        assert -0.5 < summary["roi"] < 1.0  # Wide range, just sanity
