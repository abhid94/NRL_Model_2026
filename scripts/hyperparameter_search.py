"""Systematic hyperparameter grid search for the best model config.

Grid over: n_estimators, max_depth, reg_lambda, min_child_samples, alpha.
Walk-forward evaluation on both 2024 and 2025 seasons.

Output: data/backtest_results/hyperparameter_search.xlsx
"""

import itertools
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURE_STORE_DIR, BACKTEST_RESULTS_DIR
from src.features.feature_store import load_feature_store
from src.models.gbm import GBMModel
from src.models.calibration import CalibratedModel
from src.models.baseline import MarketBlendedStrategy
from src.evaluation.backtest import run_backtest, BacktestConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Hyperparameter grid
GRID = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 4, 5],
    "reg_lambda": [2.0, 3.0, 5.0],
    "min_child_samples": [60, 80, 100],
    "alpha": [0.15, 0.25, 0.35],
}


def main():
    start = time.time()

    # Load feature store
    fs_path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    logger.info("Loading feature store from %s", fs_path)
    fs = load_feature_store(str(fs_path))
    logger.info("Feature store: %d rows x %d cols", len(fs), len(fs.columns))

    bt_config = BacktestConfig(flat_stake=100.0)

    # Generate all combinations
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    total = len(combos)
    logger.info("Grid search: %d combinations", total)

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        # Create model
        gbm = GBMModel(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=0.05,
            min_child_samples=params["min_child_samples"],
            reg_alpha=params["reg_lambda"],
            reg_lambda=params["reg_lambda"],
        )
        model = CalibratedModel(base_model=gbm, method="isotonic", cal_rounds=5)

        # Create strategy
        strategy = MarketBlendedStrategy(alpha=params["alpha"], min_edge=0.03)

        # Run backtest
        try:
            bt = run_backtest(fs, strategy, model, bt_config, seasons=[2024, 2025], min_round=3)
            s = bt.summary()
        except Exception as e:
            logger.warning("Config %d/%d failed: %s", i, total, e)
            continue

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

        # Check profitable in BOTH seasons
        s["profitable_both"] = (s.get("roi_2024", 0) > 0) and (s.get("roi_2025", 0) > 0)

        # Store hyperparameters
        s.update(params)

        results.append(s)

        if i % 10 == 0 or i == total:
            profitable = sum(1 for r in results if r.get("profitable_both", False))
            best_roi = max((r["roi"] for r in results), default=0)
            logger.info(
                "Progress: %d/%d configs tested, %d profitable in both seasons, best ROI=%.1f%%",
                i, total, profitable, best_roi * 100,
            )

    # Build results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("roi", ascending=False).reset_index(drop=True)

    # Save to Excel
    BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BACKTEST_RESULTS_DIR / "hyperparameter_search.xlsx"

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        # All results
        show_cols = [
            "n_estimators", "max_depth", "reg_lambda", "min_child_samples", "alpha",
            "n_bets", "roi", "profit", "hit_rate", "avg_odds", "avg_edge",
            "max_drawdown", "roi_2024", "roi_2025", "profitable_both",
            "n_bets_2024", "n_bets_2025",
        ]
        show = results_df[[c for c in show_cols if c in results_df.columns]].copy()
        for c in ["roi", "hit_rate", "avg_edge", "roi_2024", "roi_2025"]:
            if c in show.columns:
                show[c] = (show[c] * 100).round(1)
        for c in ["profit", "max_drawdown"]:
            if c in show.columns:
                show[c] = show[c].round(2)
        show.to_excel(writer, sheet_name="All Configs", index=False)

        # Top 20 profitable in both seasons
        both = results_df[results_df["profitable_both"] == True].head(20)
        if not both.empty:
            both_show = both[[c for c in show_cols if c in both.columns]].copy()
            for c in ["roi", "hit_rate", "avg_edge", "roi_2024", "roi_2025"]:
                if c in both_show.columns:
                    both_show[c] = (both_show[c] * 100).round(1)
            for c in ["profit", "max_drawdown"]:
                if c in both_show.columns:
                    both_show[c] = both_show[c].round(2)
            both_show.to_excel(writer, sheet_name="Profitable Both Seasons", index=False)

    elapsed = time.time() - start
    logger.info("Results saved to %s (%.0fs elapsed)", output_path, elapsed)

    # Print top results
    profitable = results_df[results_df["profitable_both"] == True]
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH RESULTS ({len(results_df)} configs tested)")
    print(f"{'='*80}")
    print(f"Profitable in BOTH seasons: {len(profitable)}/{len(results_df)}")
    print()
    if not profitable.empty:
        for i, (_, r) in enumerate(profitable.head(10).iterrows(), 1):
            print(
                f"  {i}. n={int(r['n_estimators'])}, d={int(r['max_depth'])}, "
                f"reg={r['reg_lambda']}, mcs={int(r['min_child_samples'])}, "
                f"alpha={r['alpha']:.2f}  |  "
                f"ROI={r['roi']*100:+.1f}%  Bets={int(r['n_bets'])}  "
                f"2024={r['roi_2024']*100:+.1f}%  2025={r['roi_2025']*100:+.1f}%  "
                f"DD=${r['max_drawdown']:,.0f}"
            )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
