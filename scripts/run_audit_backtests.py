"""Audit backtest runner: Re-run all strategies after edge leakage fix.

Compares corrected ROI numbers against previously reported values.
Uses flat-stake ($100) for fair comparison.
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import FEATURE_STORE_DIR
from src.evaluation.backtest import BacktestConfig, run_backtest
from src.models.baseline import (
    EdgeMatchupStrategy,
    ModelEdgeStrategy,
    SegmentPlayStrategy,
    FadeHotStreakStrategy,
    MarketImpliedStrategy,
    CompositeStrategy,
)
from src.models.gbm import GBMModel, GBMModelNoBetfair

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Previous (leaked) ROI values for comparison
PREVIOUS_ROI = {
    "GBM_NoBetfair+ModelEdge": 50.5,
    "GBM+ModelEdge": 44.3,
    "EdgeMatchup": 40.1,
    "EnrichedLogistic+ModelEdge": 53.7,
    "Composite": 11.0,
    "SegmentPlay": -5.7,
    "FadeHotStreak": -14.1,
    "MarketImplied": 0.0,
}


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("AUDIT: Re-running backtests after edge leakage fix")
    LOGGER.info("=" * 70)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)
    LOGGER.info("Loaded feature store: %d rows x %d cols", len(fs), len(fs.columns))

    flat_config = BacktestConfig(flat_stake=100.0)

    strategies = [
        ("GBM_NoBetfair+ModelEdge", ModelEdgeStrategy(), GBMModelNoBetfair(n_estimators=200)),
        ("GBM+ModelEdge", ModelEdgeStrategy(), GBMModel(n_estimators=200)),
        ("EdgeMatchup", EdgeMatchupStrategy(), None),
        ("SegmentPlay", SegmentPlayStrategy(), None),
        ("FadeHotStreak", FadeHotStreakStrategy(), None),
        ("MarketImplied", MarketImpliedStrategy(), None),
        ("Composite", CompositeStrategy([ModelEdgeStrategy(), EdgeMatchupStrategy()]), None),
    ]

    # Try to add logistic models if available
    try:
        from src.models.baseline import LogisticBaseline, EnrichedLogisticModel
        strategies.append(("EnrichedLogistic+ModelEdge", ModelEdgeStrategy(), EnrichedLogisticModel()))
    except ImportError:
        LOGGER.warning("EnrichedLogisticModel not available, skipping")

    try:
        from src.models.calibration import CalibratedModel
        from src.models.gbm import GBMModel as GBM
        cal_model = CalibratedModel(GBM(n_estimators=200), method="isotonic")
        strategies.append(("CalibratedGBM+ModelEdge", ModelEdgeStrategy(), cal_model))
    except ImportError:
        LOGGER.warning("CalibratedModel not available, skipping")

    results = []
    print("\n" + "=" * 90)
    print(f"{'Strategy':<35} {'Bets':>6} {'Hit%':>7} {'ROI%':>8} {'Profit':>10} {'MaxDD':>10} {'Prev ROI%':>10} {'Delta':>8}")
    print("=" * 90)

    for label, strategy, model in strategies:
        try:
            result = run_backtest(fs, strategy, model, flat_config, min_round=3)
            summary = result.summary()

            n_bets = summary.get("n_bets", 0)
            hit_rate = summary.get("hit_rate", 0) * 100
            roi = summary.get("roi", 0) * 100
            profit = summary.get("profit", 0)
            max_dd = summary.get("max_drawdown", 0)

            prev = PREVIOUS_ROI.get(label, float("nan"))
            delta = roi - prev if not pd.isna(prev) else float("nan")

            print(f"{label:<35} {n_bets:>6} {hit_rate:>6.1f}% {roi:>7.1f}% {profit:>9.0f} {max_dd:>9.0f} {prev:>9.1f}% {delta:>+7.1f}pp")

            results.append({
                "strategy": label,
                "n_bets": n_bets,
                "hit_rate": hit_rate,
                "roi": roi,
                "profit": profit,
                "max_drawdown": max_dd,
                "prev_roi": prev,
                "delta_pp": delta,
            })

            # Season breakdown
            bet_df = result.to_bet_dataframe()
            if not bet_df.empty and "season" in bet_df.columns:
                for season in sorted(bet_df["season"].unique()):
                    s_df = bet_df[bet_df["season"] == season]
                    s_roi = (s_df["payout"].sum() - s_df["stake"].sum()) / s_df["stake"].sum() * 100
                    s_bets = len(s_df)
                    s_hit = s_df["won"].mean() * 100
                    print(f"  └─ {season}: {s_bets} bets, {s_hit:.1f}% hit, {s_roi:+.1f}% ROI")

        except Exception as e:
            print(f"{label:<35} ERROR: {e}")
            LOGGER.exception("Strategy %s failed", label)

    print("=" * 90)
    print("\npp = percentage points change from previous (leaked) results")
    print("Negative delta = leakage was inflating results")
    print()

    # Save results
    results_df = pd.DataFrame(results)
    out_path = FEATURE_STORE_DIR.parent / "backtest_results" / "audit_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    LOGGER.info("Saved comparison to %s", out_path)


if __name__ == "__main__":
    main()
