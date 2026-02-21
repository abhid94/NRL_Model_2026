"""Sprint 4A: Poisson model walk-forward backtest.

Compares PoissonModel + ModelEdge vs CalibratedPoisson + ModelEdge
against the CalibratedGBM baseline (+6.3% ROI).
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import FEATURE_STORE_DIR
from src.evaluation.backtest import BacktestConfig, compare_backtests, run_backtest
from src.evaluation.metrics import (
    build_evaluation_report,
    compute_auc,
    compute_brier_score,
    compute_calibration_error,
)
from src.models.baseline import ModelEdgeStrategy
from src.models.calibration import CalibratedModel
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
    LOGGER.info("Sprint 4A: Poisson Model Backtest")
    LOGGER.info("=" * 70)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)
    LOGGER.info("Loaded feature store: %d rows x %d cols", len(fs), len(fs.columns))

    flat_config = BacktestConfig(flat_stake=100.0)
    strategy = ModelEdgeStrategy()

    # Define models to compare
    models = {
        "Poisson": PoissonModel(reg_alpha=1.0),
        "CalibratedPoisson": CalibratedModel(
            PoissonModel(reg_alpha=1.0), method="isotonic",
        ),
        "CalibratedGBM (baseline)": CalibratedModel(
            GBMModel(n_estimators=200), method="isotonic",
        ),
    }

    results = []
    print("\n" + "=" * 95)
    print(f"{'Model':<30} {'Bets':>6} {'Hit%':>7} {'ROI%':>8} {'Profit':>10} {'MaxDD':>10}")
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

            print(f"{label:<30} {n_bets:>6} {hit_rate:>6.1f}% {roi:>7.1f}% {profit:>9.0f} {max_dd:>9.0f}")

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
            print(f"{label:<30} ERROR: {e}")
            LOGGER.exception("Model %s failed", label)

    print("=" * 95)

    # Comparison table
    if results:
        comparison = compare_backtests(results)
        print("\nComparison table:")
        print(comparison.to_string(index=False))

    # Save results
    out_dir = FEATURE_STORE_DIR.parent / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if results:
        comparison.to_csv(out_dir / "sprint_4a_poisson_comparison.csv", index=False)
        LOGGER.info("Saved to %s", out_dir / "sprint_4a_poisson_comparison.csv")


if __name__ == "__main__":
    main()
