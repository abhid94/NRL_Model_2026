"""Diagnose why 2025 is consistently unprofitable.

Investigates:
1. Model calibration per season
2. Per-round ROI (is 2025 uniformly bad or a few rounds?)
3. GBM hyperparameter grid search
4. Feature importance shift between seasons
5. Platt vs isotonic calibration
6. Different training data configs (2024-only training, etc.)
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.config import FEATURE_STORE_DIR
from src.evaluation.backtest import BacktestConfig, run_backtest
from src.evaluation.metrics import compute_brier_score, compute_auc, compute_calibration_error
from src.models.baseline import ModelEdgeStrategy
from src.models.calibration import CalibratedModel
from src.models.gbm import GBMModel

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def per_season_roi(result):
    bdf = result.to_bet_dataframe()
    if bdf.empty or "season" not in bdf.columns:
        return {}
    out = {}
    for season in sorted(bdf["season"].unique()):
        sdf = bdf[bdf["season"] == season]
        total_staked = sdf["stake"].sum()
        total_payout = sdf["payout"].sum()
        if total_staked > 0:
            out[int(season)] = {
                "n_bets": len(sdf),
                "roi": (total_payout - total_staked) / total_staked,
            }
    return out


def per_round_roi(result):
    bdf = result.to_bet_dataframe()
    if bdf.empty:
        return pd.DataFrame()
    rows = []
    for (season, rnd), grp in bdf.groupby(["season", "round_number"]):
        staked = grp["stake"].sum()
        payout = grp["payout"].sum()
        rows.append({
            "season": int(season),
            "round": int(rnd),
            "n_bets": len(grp),
            "staked": staked,
            "profit": payout - staked,
            "roi": (payout - staked) / staked if staked > 0 else 0,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("DIAGNOSIS: Why is 2025 consistently unprofitable?")
    print("=" * 70)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)

    flat_config = BacktestConfig(flat_stake=100.0)
    strategy = ModelEdgeStrategy()

    # -----------------------------------------------------------------------
    # 1. Per-round ROI breakdown
    # -----------------------------------------------------------------------
    print("\n--- Part 1: Per-round ROI (CalGBM 200, flat $100) ---")
    model = CalibratedModel(GBMModel(n_estimators=200), method="isotonic")
    result = run_backtest(fs, strategy, model, flat_config, min_round=3)
    rnd_df = per_round_roi(result)

    for season in [2024, 2025]:
        sdf = rnd_df[rnd_df["season"] == season].sort_values("round")
        wins = (sdf["profit"] > 0).sum()
        total = len(sdf)
        total_profit = sdf["profit"].sum()
        print(f"\n  {season}: {wins}/{total} profitable rounds, total profit ${total_profit:.0f}")
        for _, row in sdf.iterrows():
            marker = "+" if row["profit"] > 0 else "-"
            print(f"    R{row['round']:>2}: {row['n_bets']:>2} bets, ${row['profit']:>+7.0f} ({row['roi']*100:>+6.1f}%) {marker}")

    # -----------------------------------------------------------------------
    # 2. GBM hyperparameter grid search
    # -----------------------------------------------------------------------
    print("\n--- Part 2: GBM Hyperparameter Grid ---")
    print(f"{'Config':<45} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8}")
    print("-" * 80)

    hp_configs = [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 50, "learning_rate": 0.2, "max_depth": 3},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        # Very shallow / regularized
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2},
    ]

    best_both = None
    best_min_roi = -float("inf")

    for hp in hp_configs:
        label = f"GBM(n={hp['n_estimators']},lr={hp['learning_rate']},d={hp['max_depth']})"
        try:
            gbm = GBMModel(
                n_estimators=hp["n_estimators"],
                learning_rate=hp.get("learning_rate", 0.1),
                max_depth=hp.get("max_depth", 4),
            )
            model = CalibratedModel(gbm, method="isotonic")
            result = run_backtest(fs, strategy, model, flat_config, min_round=3)
            s = result.summary()
            season_data = per_season_roi(result)
            r24 = season_data.get(2024, {}).get("roi", float("nan"))
            r25 = season_data.get(2025, {}).get("roi", float("nan"))
            n_bets = s.get("n_bets", 0)
            roi = s.get("roi", 0)

            min_season_roi = min(r24, r25) if not (np.isnan(r24) or np.isnan(r25)) else float("-inf")
            marker = " *" if min_season_roi > best_min_roi else ""
            if min_season_roi > best_min_roi:
                best_min_roi = min_season_roi
                best_both = label

            print(f"{label:<45} {n_bets:>5} {roi*100:>+6.1f}% {r24*100:>+7.1f}% {r25*100:>+7.1f}%{marker}")
        except Exception as e:
            print(f"{label:<45} ERROR: {e}")

    if best_both:
        print(f"\n  Best min(2024, 2025) ROI: {best_both} at {best_min_roi*100:+.1f}%")

    # -----------------------------------------------------------------------
    # 3. Platt vs Isotonic calibration
    # -----------------------------------------------------------------------
    print("\n--- Part 3: Calibration Method Comparison ---")
    print(f"{'Config':<45} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8}")
    print("-" * 80)

    for cal_method in ["isotonic", "sigmoid"]:
        for cal_rounds in [3, 5, 8]:
            label = f"GBM(200)+{cal_method}(cal_rounds={cal_rounds})"
            model = CalibratedModel(
                GBMModel(n_estimators=200),
                method=cal_method,
                cal_rounds=cal_rounds,
            )
            result = run_backtest(fs, strategy, model, flat_config, min_round=3)
            s = result.summary()
            season_data = per_season_roi(result)
            r24 = season_data.get(2024, {}).get("roi", float("nan"))
            r25 = season_data.get(2025, {}).get("roi", float("nan"))
            print(f"{label:<45} {s.get('n_bets', 0):>5} {s.get('roi', 0)*100:>+6.1f}% {r24*100:>+7.1f}% {r25*100:>+7.1f}%")

    # -----------------------------------------------------------------------
    # 4. Raw GBM (no calibration) — is calibration hurting?
    # -----------------------------------------------------------------------
    print("\n--- Part 4: Raw GBM (no calibration) ---")
    print(f"{'Config':<45} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8}")
    print("-" * 80)

    for n_est in [50, 100, 200, 300]:
        label = f"RawGBM(n={n_est})"
        model = GBMModel(n_estimators=n_est)
        result = run_backtest(fs, strategy, model, flat_config, min_round=3)
        s = result.summary()
        season_data = per_season_roi(result)
        r24 = season_data.get(2024, {}).get("roi", float("nan"))
        r25 = season_data.get(2025, {}).get("roi", float("nan"))
        print(f"{label:<45} {s.get('n_bets', 0):>5} {s.get('roi', 0)*100:>+6.1f}% {r24*100:>+7.1f}% {r25*100:>+7.1f}%")

    # -----------------------------------------------------------------------
    # 5. Feature ablation: exclude Betfair features
    # -----------------------------------------------------------------------
    print("\n--- Part 5: Feature Ablation ---")
    print(f"{'Config':<45} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8}")
    print("-" * 80)

    for exclude_betfair in [False, True]:
        label = f"CalGBM(200, excl_betfair={exclude_betfair})"
        model = CalibratedModel(
            GBMModel(n_estimators=200, exclude_betfair=exclude_betfair),
            method="isotonic",
        )
        result = run_backtest(fs, strategy, model, flat_config, min_round=3)
        s = result.summary()
        season_data = per_season_roi(result)
        r24 = season_data.get(2024, {}).get("roi", float("nan"))
        r25 = season_data.get(2025, {}).get("roi", float("nan"))
        print(f"{label:<45} {s.get('n_bets', 0):>5} {s.get('roi', 0)*100:>+6.1f}% {r24*100:>+7.1f}% {r25*100:>+7.1f}%")

    # -----------------------------------------------------------------------
    # 6. Training data experiment: train on 2024 only (predict 2025)
    # -----------------------------------------------------------------------
    print("\n--- Part 6: Training Data Experiments ---")
    print(f"{'Config':<45} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8}")
    print("-" * 80)

    # Train-on-2024-only config: still uses walk-forward but with only 2024 data
    fs_2024 = fs[fs["season"] == 2024].copy()
    fs_2025 = fs[fs["season"] == 2025].copy()

    # 2024 only
    model = CalibratedModel(GBMModel(n_estimators=200), method="isotonic")
    result = run_backtest(fs_2024, strategy, model, flat_config, min_round=3)
    s = result.summary()
    print(f"{'CalGBM(200) train+eval 2024 only':<45} {s.get('n_bets', 0):>5} {s.get('roi', 0)*100:>+6.1f}%")

    # 2025 only
    model = CalibratedModel(GBMModel(n_estimators=200), method="isotonic")
    result = run_backtest(fs_2025, strategy, model, flat_config, min_round=3)
    s = result.summary()
    print(f"{'CalGBM(200) train+eval 2025 only':<45} {s.get('n_bets', 0):>5} {s.get('roi', 0)*100:>+6.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
