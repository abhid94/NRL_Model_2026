"""Run Optuna hyperparameter optimization on historical data.

Finds optimal GBM parameters by maximizing walk-forward ROI on the
combined 2024+2025 feature store.

Usage:
    python scripts/run_optuna.py --trials 100
    python scripts/run_optuna.py --trials 200 --metric roi --folds 4
    python scripts/run_optuna.py --trials 50 --timeout 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import FEATURE_STORE_DIR, MODEL_ARTIFACTS_DIR
from src.models.hyperopt import optimize_gbm_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for ATS GBM model",
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--metric", default="roi", choices=["roi", "log_loss", "brier"],
                        help="Optimization metric")
    parser.add_argument("--folds", type=int, default=3, help="Walk-forward folds")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds")
    parser.add_argument("--min-edge", type=float, default=0.03, help="Min edge for ROI calc")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load feature stores
    dfs = []
    for year in [2024, 2025]:
        path = FEATURE_STORE_DIR / f"feature_store_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            dfs.append(df)
            print(f"Loaded {len(df):,} rows from {path.name}")

    if not dfs:
        print("ERROR: No feature store files found. Run rebuild_feature_store.py first.")
        sys.exit(1)

    feature_store = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(feature_store):,} rows, {len(feature_store.columns)} columns")

    # Run optimization
    print(f"\nStarting Optuna: {args.trials} trials, {args.folds} folds, "
          f"metric={args.metric}, timeout={args.timeout}s")
    print("=" * 60)

    result = optimize_gbm_params(
        feature_store,
        n_trials=args.trials,
        n_walk_forward_folds=args.folds,
        metric=args.metric,
        min_edge=args.min_edge,
        timeout=args.timeout,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"Best {args.metric}: {result['value']:.4f}")
    print(f"Trials completed: {result['n_trials_completed']}")
    print(f"\nBest parameters:")
    for k, v in sorted(result["params"].items()):
        print(f"  {k}: {v}")

    # Save to file
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_ARTIFACTS_DIR / "optuna_best_params.json"
    with open(output_path, "w") as f:
        json.dump({
            "params": result["params"],
            "value": result["value"],
            "metric": args.metric,
            "n_trials": result["n_trials_completed"],
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
