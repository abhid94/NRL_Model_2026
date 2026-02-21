"""Sprint 4B: Ensemble model walk-forward backtest.

Compares weighted and stacked ensembles vs individual models.
Tests whether combining diverse models improves stability across seasons.
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import FEATURE_STORE_DIR
from src.evaluation.backtest import BacktestConfig, compare_backtests, run_backtest
from src.models.baseline import EnrichedLogisticModel, ModelEdgeStrategy
from src.models.calibration import CalibratedModel
from src.models.ensemble import (
    StackedEnsemble,
    WeightedEnsemble,
    prediction_diversity,
)
from src.models.gbm import GBMModel
from src.models.poisson import PoissonModel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("Sprint 4B: Ensemble Model Backtest")
    LOGGER.info("=" * 70)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)
    LOGGER.info("Loaded feature store: %d rows x %d cols", len(fs), len(fs.columns))

    flat_config = BacktestConfig(flat_stake=100.0)
    strategy = ModelEdgeStrategy()

    # Base models for ensembling
    def make_gbm():
        return CalibratedModel(GBMModel(n_estimators=200), method="isotonic")

    def make_poisson():
        return CalibratedModel(PoissonModel(reg_alpha=1.0), method="isotonic")

    def make_logistic():
        return EnrichedLogisticModel()

    # Define model configs
    models = {
        "CalibratedGBM (baseline)": make_gbm(),
        "CalibratedPoisson": make_poisson(),
        "EnrichedLogistic": make_logistic(),
        "WeightedEnsemble (equal)": WeightedEnsemble(
            [make_gbm(), make_poisson(), make_logistic()],
        ),
        "WeightedEnsemble (learned)": WeightedEnsemble(
            [make_gbm(), make_poisson(), make_logistic()],
            learn_weights=True, holdout_rounds=5,
        ),
        "StackedEnsemble": StackedEnsemble(
            [make_gbm(), make_poisson(), make_logistic()],
            n_folds=5, include_market=True,
        ),
        "StackedEnsemble (no market)": StackedEnsemble(
            [make_gbm(), make_poisson(), make_logistic()],
            n_folds=5, include_market=False,
        ),
    }

    results = []
    print("\n" + "=" * 95)
    print(f"{'Model':<35} {'Bets':>6} {'Hit%':>7} {'ROI%':>8} {'Profit':>10} {'MaxDD':>10}")
    print("=" * 95)

    for label, model in models.items():
        try:
            result = run_backtest(fs, strategy, model, flat_config, min_round=3)
            summary = result.summary()

            n_bets = summary.get("n_bets", 0)
            hit_rate = summary.get("hit_rate", 0) * 100
            roi = summary.get("roi", 0) * 100
            profit = summary.get("profit", 0)
            max_dd = summary.get("max_drawdown", 0)

            print(f"{label:<35} {n_bets:>6} {hit_rate:>6.1f}% {roi:>7.1f}% {profit:>9.0f} {max_dd:>9.0f}")

            # Season breakdown
            bet_df = result.to_bet_dataframe()
            if not bet_df.empty and "season" in bet_df.columns:
                for season in sorted(bet_df["season"].unique()):
                    s_df = bet_df[bet_df["season"] == season]
                    if s_df["stake"].sum() > 0:
                        s_roi = (s_df["payout"].sum() - s_df["stake"].sum()) / s_df["stake"].sum() * 100
                        s_bets = len(s_df)
                        s_hit = s_df["won"].mean() * 100
                        print(f"  └─ {season}: {s_bets} bets, {s_hit:.1f}% hit, {s_roi:+.1f}% ROI")

            results.append(result)
        except Exception as e:
            print(f"{label:<35} ERROR: {e}")
            LOGGER.exception("Model %s failed", label)

    print("=" * 95)

    # Comparison table
    if results:
        comparison = compare_backtests(results)
        print("\nComparison table:")
        print(comparison.to_string(index=False))

    # Diversity analysis — fit all base models on full data
    print("\n" + "=" * 70)
    print("Prediction Diversity (pairwise correlation on full dataset)")
    print("=" * 70)
    try:
        div_models = {
            "CalibratedGBM": make_gbm(),
            "CalibratedPoisson": make_poisson(),
            "EnrichedLogistic": make_logistic(),
        }
        y = fs["scored_try"].values
        fitted = []
        names = []
        for name, model in div_models.items():
            model.fit(fs, y)
            fitted.append(model)
            names.append(name)

        corr = prediction_diversity(fitted, fs)
        corr.index = names
        corr.columns = names
        print(corr.round(3).to_string())
    except Exception as e:
        print(f"Diversity analysis failed: {e}")

    # Save results
    out_dir = FEATURE_STORE_DIR.parent / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if results:
        comparison = compare_backtests(results)
        comparison.to_csv(out_dir / "sprint_4b_ensemble_comparison.csv", index=False)
        LOGGER.info("Saved to %s", out_dir / "sprint_4b_ensemble_comparison.csv")


if __name__ == "__main__":
    main()
