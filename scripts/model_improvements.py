"""Comprehensive model improvement comparison.

Runs multiple configurations and generates a comparison report:
1. Kelly staking vs flat stake (quarter, half, eighth Kelly)
2. PositionCalibratedModel vs global CalibratedModel
3. Ensemble + MarketBlended vs solo GBM + MarketBlended

Output: data/backtest_results/model_improvements_report.xlsx
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURE_STORE_DIR, BACKTEST_RESULTS_DIR
from src.features.feature_store import load_feature_store
from src.models.gbm import GBMModel
from src.models.calibration import CalibratedModel, PositionCalibratedModel
from src.models.poisson import PoissonModel
from src.models.baseline import (
    EnrichedLogisticModel,
    MarketBlendedStrategy,
)
from src.models.ensemble import WeightedEnsemble, StackedEnsemble
from src.evaluation.backtest import run_backtest, BacktestConfig, compare_backtests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_best_gbm():
    """Create the best GBM model from Phase 5B."""
    return GBMModel(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=80,
        reg_alpha=3.0,
        reg_lambda=3.0,
    )


def make_best_strategy():
    """Create the best strategy from Phase 5B."""
    return MarketBlendedStrategy(alpha=0.25, min_edge=0.03)


# ──────────────────────────────────────────────────────────────
# Step 1: Kelly staking comparison
# ──────────────────────────────────────────────────────────────

def run_kelly_comparison(fs: pd.DataFrame) -> list[dict]:
    """Compare flat $100, quarter-Kelly, half-Kelly, eighth-Kelly."""
    logger.info("=" * 60)
    logger.info("STEP 1: Kelly Staking Comparison")
    logger.info("=" * 60)

    configs = [
        ("Flat $100", BacktestConfig(flat_stake=100.0)),
        ("Quarter Kelly (0.25)", BacktestConfig(kelly_fraction=0.25)),
        ("Eighth Kelly (0.125)", BacktestConfig(kelly_fraction=0.125)),
        ("Sixteenth Kelly (0.0625)", BacktestConfig(kelly_fraction=0.0625)),
    ]

    results = []
    for label, config in configs:
        logger.info("Running: %s", label)
        model = CalibratedModel(base_model=make_best_gbm(), method="isotonic", cal_rounds=5)
        strategy = make_best_strategy()
        bt = run_backtest(fs, strategy, model, config, seasons=[2024, 2025], min_round=3)
        s = bt.summary()
        s["config_label"] = label
        s["staking"] = label

        # Per-season breakdown
        for season in [2024, 2025]:
            season_bets = [r for r in bt.round_results if r.season == season]
            staked = sum(r.total_staked for r in season_bets)
            payout = sum(r.total_payout for r in season_bets)
            profit = payout - staked
            n = sum(r.n_bets for r in season_bets)
            s[f"roi_{season}"] = profit / staked if staked > 0 else 0.0
            s[f"profit_{season}"] = profit
            s[f"n_bets_{season}"] = n

        # Sharpe ratio (round-level)
        rnd_df = bt.to_round_dataframe()
        if len(rnd_df) > 1 and rnd_df["profit"].std() > 0:
            s["sharpe"] = rnd_df["profit"].mean() / rnd_df["profit"].std()
        else:
            s["sharpe"] = 0.0

        results.append(s)
        logger.info(
            "  %s: %d bets, ROI=%.1f%%, profit=$%.0f, max_dd=$%.0f, sharpe=%.2f",
            label, s["n_bets"], s["roi"] * 100, s["profit"],
            s.get("max_drawdown", 0), s["sharpe"],
        )

    return results


# ──────────────────────────────────────────────────────────────
# Step 2: PositionCalibratedModel comparison
# ──────────────────────────────────────────────────────────────

def run_position_calibration_comparison(fs: pd.DataFrame) -> list[dict]:
    """Compare global vs position-specific calibration."""
    logger.info("=" * 60)
    logger.info("STEP 2: PositionCalibratedModel Comparison")
    logger.info("=" * 60)

    configs = [
        ("Global Isotonic", CalibratedModel(
            base_model=make_best_gbm(), method="isotonic", cal_rounds=5,
        )),
        ("Position Isotonic", PositionCalibratedModel(
            base_model=make_best_gbm(), method="isotonic", cal_rounds=5,
        )),
        ("Position Isotonic (min=20)", PositionCalibratedModel(
            base_model=make_best_gbm(), method="isotonic", cal_rounds=5,
            min_samples_per_group=20,
        )),
    ]

    bt_config = BacktestConfig(flat_stake=100.0)
    results = []
    for label, model in configs:
        logger.info("Running: %s", label)
        strategy = make_best_strategy()
        bt = run_backtest(fs, strategy, model, bt_config, seasons=[2024, 2025], min_round=3)
        s = bt.summary()
        s["config_label"] = label
        s["calibration"] = label

        # Per-season breakdown
        for season in [2024, 2025]:
            season_bets = [r for r in bt.round_results if r.season == season]
            staked = sum(r.total_staked for r in season_bets)
            payout = sum(r.total_payout for r in season_bets)
            profit = payout - staked
            n = sum(r.n_bets for r in season_bets)
            s[f"roi_{season}"] = profit / staked if staked > 0 else 0.0
            s[f"profit_{season}"] = profit
            s[f"n_bets_{season}"] = n

        # Hit rate by position from bet DataFrame
        bet_df = bt.to_bet_dataframe()
        if not bet_df.empty:
            pos_stats = bet_df.groupby("position_code").agg(
                bets=("won", "count"),
                wins=("won", "sum"),
                avg_edge=("edge", "mean"),
            )
            pos_stats["hit_rate"] = pos_stats["wins"] / pos_stats["bets"]
            s["position_breakdown"] = pos_stats.to_dict()

        results.append(s)
        logger.info(
            "  %s: %d bets, ROI=%.1f%%, profit=$%.0f",
            label, s["n_bets"], s["roi"] * 100, s["profit"],
        )

    return results


# ──────────────────────────────────────────────────────────────
# Step 3: Ensemble + MarketBlended comparison
# ──────────────────────────────────────────────────────────────

def run_ensemble_comparison(fs: pd.DataFrame) -> list[dict]:
    """Test ensemble models with MarketBlended strategy."""
    logger.info("=" * 60)
    logger.info("STEP 3: Ensemble + MarketBlended Comparison")
    logger.info("=" * 60)

    bt_config = BacktestConfig(flat_stake=100.0)
    results = []

    # Baseline: solo GBM
    logger.info("Running: Solo CalibratedGBM (baseline)")
    model = CalibratedModel(base_model=make_best_gbm(), method="isotonic", cal_rounds=5)
    strategy = make_best_strategy()
    bt = run_backtest(fs, strategy, model, bt_config, seasons=[2024, 2025], min_round=3)
    s = bt.summary()
    s["config_label"] = "Solo CalibratedGBM"
    s["ensemble_type"] = "None (baseline)"
    for season in [2024, 2025]:
        season_bets = [r for r in bt.round_results if r.season == season]
        staked = sum(r.total_staked for r in season_bets)
        payout = sum(r.total_payout for r in season_bets)
        s[f"roi_{season}"] = (payout - staked) / staked if staked > 0 else 0.0
        s[f"profit_{season}"] = payout - staked
        s[f"n_bets_{season}"] = sum(r.n_bets for r in season_bets)
    results.append(s)

    # Weighted ensemble: GBM + Logistic
    logger.info("Running: WeightedEnsemble(GBM, Logistic)")
    try:
        base_models = [make_best_gbm(), EnrichedLogisticModel(C=0.1)]
        ensemble = WeightedEnsemble(base_models, learn_weights=True, holdout_rounds=5)
        cal_ensemble = CalibratedModel(base_model=ensemble, method="isotonic", cal_rounds=5)
        strategy = make_best_strategy()
        bt = run_backtest(fs, strategy, cal_ensemble, bt_config, seasons=[2024, 2025], min_round=3)
        s = bt.summary()
        s["config_label"] = "Weighted(GBM+Logistic)"
        s["ensemble_type"] = "WeightedEnsemble"
        for season in [2024, 2025]:
            season_bets = [r for r in bt.round_results if r.season == season]
            staked = sum(r.total_staked for r in season_bets)
            payout = sum(r.total_payout for r in season_bets)
            s[f"roi_{season}"] = (payout - staked) / staked if staked > 0 else 0.0
            s[f"profit_{season}"] = payout - staked
            s[f"n_bets_{season}"] = sum(r.n_bets for r in season_bets)
        results.append(s)
    except Exception as e:
        logger.warning("WeightedEnsemble(GBM+Logistic) failed: %s", e)

    # Weighted ensemble: GBM + Poisson
    logger.info("Running: WeightedEnsemble(GBM, Poisson)")
    try:
        base_models = [make_best_gbm(), PoissonModel(reg_alpha=1.0)]
        ensemble = WeightedEnsemble(base_models, learn_weights=True, holdout_rounds=5)
        cal_ensemble = CalibratedModel(base_model=ensemble, method="isotonic", cal_rounds=5)
        strategy = make_best_strategy()
        bt = run_backtest(fs, strategy, cal_ensemble, bt_config, seasons=[2024, 2025], min_round=3)
        s = bt.summary()
        s["config_label"] = "Weighted(GBM+Poisson)"
        s["ensemble_type"] = "WeightedEnsemble"
        for season in [2024, 2025]:
            season_bets = [r for r in bt.round_results if r.season == season]
            staked = sum(r.total_staked for r in season_bets)
            payout = sum(r.total_payout for r in season_bets)
            s[f"roi_{season}"] = (payout - staked) / staked if staked > 0 else 0.0
            s[f"profit_{season}"] = payout - staked
            s[f"n_bets_{season}"] = sum(r.n_bets for r in season_bets)
        results.append(s)
    except Exception as e:
        logger.warning("WeightedEnsemble(GBM+Poisson) failed: %s", e)

    # Weighted ensemble: GBM + Logistic + Poisson (3-model)
    logger.info("Running: WeightedEnsemble(GBM, Logistic, Poisson)")
    try:
        base_models = [make_best_gbm(), EnrichedLogisticModel(C=0.1), PoissonModel(reg_alpha=1.0)]
        ensemble = WeightedEnsemble(base_models, learn_weights=True, holdout_rounds=5)
        cal_ensemble = CalibratedModel(base_model=ensemble, method="isotonic", cal_rounds=5)
        strategy = make_best_strategy()
        bt = run_backtest(fs, strategy, cal_ensemble, bt_config, seasons=[2024, 2025], min_round=3)
        s = bt.summary()
        s["config_label"] = "Weighted(GBM+Logistic+Poisson)"
        s["ensemble_type"] = "WeightedEnsemble (3-model)"
        for season in [2024, 2025]:
            season_bets = [r for r in bt.round_results if r.season == season]
            staked = sum(r.total_staked for r in season_bets)
            payout = sum(r.total_payout for r in season_bets)
            s[f"roi_{season}"] = (payout - staked) / staked if staked > 0 else 0.0
            s[f"profit_{season}"] = payout - staked
            s[f"n_bets_{season}"] = sum(r.n_bets for r in season_bets)
        results.append(s)
    except Exception as e:
        logger.warning("WeightedEnsemble(GBM+Logistic+Poisson) failed: %s", e)

    # Stacked ensemble: GBM + Logistic
    logger.info("Running: StackedEnsemble(GBM, Logistic)")
    try:
        base_models = [make_best_gbm(), EnrichedLogisticModel(C=0.1)]
        stacked = StackedEnsemble(base_models, n_folds=5, include_market=True)
        cal_stacked = CalibratedModel(base_model=stacked, method="isotonic", cal_rounds=5)
        strategy = make_best_strategy()
        bt = run_backtest(fs, strategy, cal_stacked, bt_config, seasons=[2024, 2025], min_round=3)
        s = bt.summary()
        s["config_label"] = "Stacked(GBM+Logistic)"
        s["ensemble_type"] = "StackedEnsemble"
        for season in [2024, 2025]:
            season_bets = [r for r in bt.round_results if r.season == season]
            staked = sum(r.total_staked for r in season_bets)
            payout = sum(r.total_payout for r in season_bets)
            s[f"roi_{season}"] = (payout - staked) / staked if staked > 0 else 0.0
            s[f"profit_{season}"] = payout - staked
            s[f"n_bets_{season}"] = sum(r.n_bets for r in season_bets)
        results.append(s)
    except Exception as e:
        logger.warning("StackedEnsemble(GBM+Logistic) failed: %s", e)

    for s in results:
        logger.info(
            "  %s: %d bets, ROI=%.1f%%, profit=$%.0f",
            s["config_label"], s["n_bets"], s["roi"] * 100, s["profit"],
        )

    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    start = time.time()

    # Load feature store
    fs_path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    logger.info("Loading feature store from %s", fs_path)
    fs = load_feature_store(str(fs_path))
    logger.info("Feature store: %d rows x %d cols", len(fs), len(fs.columns))

    # Run all comparisons
    kelly_results = run_kelly_comparison(fs)
    position_results = run_position_calibration_comparison(fs)
    ensemble_results = run_ensemble_comparison(fs)

    # ──────────────────────────────────────────────────────────
    # Build comparison report
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Building Comparison Report")
    logger.info("=" * 60)

    BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BACKTEST_RESULTS_DIR / "model_improvements_report.xlsx"

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        # Kelly comparison sheet
        kelly_df = pd.DataFrame(kelly_results)
        kelly_cols = [
            "config_label", "n_bets", "roi", "profit", "hit_rate", "avg_odds",
            "avg_edge", "max_drawdown", "sharpe", "final_bankroll",
            "roi_2024", "profit_2024", "n_bets_2024",
            "roi_2025", "profit_2025", "n_bets_2025",
        ]
        kelly_show = kelly_df[[c for c in kelly_cols if c in kelly_df.columns]].copy()
        for c in ["roi", "hit_rate", "avg_edge", "roi_2024", "roi_2025"]:
            if c in kelly_show.columns:
                kelly_show[c] = (kelly_show[c] * 100).round(1)
        for c in ["profit", "max_drawdown", "final_bankroll", "profit_2024", "profit_2025"]:
            if c in kelly_show.columns:
                kelly_show[c] = kelly_show[c].round(2)
        kelly_show["sharpe"] = kelly_show["sharpe"].round(3)
        kelly_show.columns = [
            "Config", "Bets", "ROI %", "Profit $", "Hit Rate %", "Avg Odds",
            "Avg Edge %", "Max Drawdown $", "Sharpe", "Final Bankroll $",
            "ROI 2024 %", "Profit 2024 $", "Bets 2024",
            "ROI 2025 %", "Profit 2025 $", "Bets 2025",
        ]
        kelly_show.to_excel(writer, sheet_name="Kelly Comparison", index=False)

        # Position calibration sheet
        pos_df = pd.DataFrame(position_results)
        pos_cols = [
            "config_label", "n_bets", "roi", "profit", "hit_rate", "avg_odds",
            "avg_edge", "max_drawdown",
            "roi_2024", "profit_2024", "n_bets_2024",
            "roi_2025", "profit_2025", "n_bets_2025",
        ]
        pos_show = pos_df[[c for c in pos_cols if c in pos_df.columns]].copy()
        for c in ["roi", "hit_rate", "avg_edge", "roi_2024", "roi_2025"]:
            if c in pos_show.columns:
                pos_show[c] = (pos_show[c] * 100).round(1)
        for c in ["profit", "max_drawdown", "profit_2024", "profit_2025"]:
            if c in pos_show.columns:
                pos_show[c] = pos_show[c].round(2)
        pos_show.columns = [
            "Config", "Bets", "ROI %", "Profit $", "Hit Rate %", "Avg Odds",
            "Avg Edge %", "Max Drawdown $",
            "ROI 2024 %", "Profit 2024 $", "Bets 2024",
            "ROI 2025 %", "Profit 2025 $", "Bets 2025",
        ]
        pos_show.to_excel(writer, sheet_name="Position Calibration", index=False)

        # Ensemble comparison sheet
        ens_df = pd.DataFrame(ensemble_results)
        ens_cols = [
            "config_label", "n_bets", "roi", "profit", "hit_rate", "avg_odds",
            "avg_edge", "max_drawdown",
            "roi_2024", "profit_2024", "n_bets_2024",
            "roi_2025", "profit_2025", "n_bets_2025",
        ]
        ens_show = ens_df[[c for c in ens_cols if c in ens_df.columns]].copy()
        for c in ["roi", "hit_rate", "avg_edge", "roi_2024", "roi_2025"]:
            if c in ens_show.columns:
                ens_show[c] = (ens_show[c] * 100).round(1)
        for c in ["profit", "max_drawdown", "profit_2024", "profit_2025"]:
            if c in ens_show.columns:
                ens_show[c] = ens_show[c].round(2)
        ens_show.columns = [
            "Config", "Bets", "ROI %", "Profit $", "Hit Rate %", "Avg Odds",
            "Avg Edge %", "Max Drawdown $",
            "ROI 2024 %", "Profit 2024 $", "Bets 2024",
            "ROI 2025 %", "Profit 2025 $", "Bets 2025",
        ]
        ens_show.to_excel(writer, sheet_name="Ensemble Comparison", index=False)

        # Overall summary sheet
        all_results = kelly_results + position_results + ensemble_results
        all_df = pd.DataFrame(all_results)
        summary_cols = ["config_label", "n_bets", "roi", "profit", "hit_rate",
                        "max_drawdown", "roi_2024", "roi_2025"]
        summary = all_df[[c for c in summary_cols if c in all_df.columns]].copy()
        for c in ["roi", "hit_rate", "roi_2024", "roi_2025"]:
            if c in summary.columns:
                summary[c] = (summary[c] * 100).round(1)
        for c in ["profit", "max_drawdown"]:
            if c in summary.columns:
                summary[c] = summary[c].round(2)
        summary = summary.sort_values("roi", ascending=False).reset_index(drop=True)
        summary.columns = [
            "Config", "Bets", "ROI %", "Profit $", "Hit Rate %",
            "Max Drawdown $", "ROI 2024 %", "ROI 2025 %",
        ]
        summary.to_excel(writer, sheet_name="Overall Ranking", index=False)

    elapsed = time.time() - start
    logger.info("Report saved to %s (%.0fs elapsed)", output_path, elapsed)

    # Print top-line results
    print("\n" + "=" * 70)
    print("MODEL IMPROVEMENTS COMPARISON RESULTS")
    print("=" * 70)
    all_results_sorted = sorted(all_results, key=lambda x: x["roi"], reverse=True)
    for i, r in enumerate(all_results_sorted[:10], 1):
        roi_24 = r.get("roi_2024", 0) * 100
        roi_25 = r.get("roi_2025", 0) * 100
        print(
            f"  {i}. {r['config_label']:<40s} "
            f"ROI={r['roi']*100:+6.1f}%  "
            f"Bets={r['n_bets']:3d}  "
            f"2024={roi_24:+5.1f}%  "
            f"2025={roi_25:+5.1f}%  "
            f"DD=${r.get('max_drawdown', 0):,.0f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
