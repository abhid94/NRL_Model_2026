"""Deep validation of the best candidate model: GBM(n=300, lr=0.05, d=4).

Runs:
1. Bootstrap stability analysis with 2000 samples
2. Cross-season bootstrap CIs
3. Nearby hyperparameter neighborhood search
4. Combination with RefinedEdge strategies
5. Per-position and per-odds-band ROI
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
from src.evaluation.edge_analysis import (
    stability_analysis,
    cross_season_stability,
)
from src.models.baseline import ModelEdgeStrategy, RefinedEdgeStrategy
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


def run_config(fs, strategy, model, config, label):
    result = run_backtest(fs, strategy, model, config, min_round=3)
    s = result.summary()
    season_data = per_season_roi(result)
    r24 = season_data.get(2024, {}).get("roi", float("nan"))
    r25 = season_data.get(2025, {}).get("roi", float("nan"))
    n24 = season_data.get(2024, {}).get("n_bets", 0)
    n25 = season_data.get(2025, {}).get("n_bets", 0)
    both = (r24 > 0) and (r25 > 0)
    return {
        "label": label,
        "n_bets": s.get("n_bets", 0),
        "roi": s.get("roi", 0),
        "r24": r24, "r25": r25,
        "n24": n24, "n25": n25,
        "both_positive": both,
        "result": result,
    }


def main():
    print("=" * 70)
    print("DEEP VALIDATION: GBM(n=300, lr=0.05, d=4) + Calibrated")
    print("=" * 70)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)
    flat_config = BacktestConfig(flat_stake=100.0)
    strategy = ModelEdgeStrategy()

    # -----------------------------------------------------------------------
    # 1. Bootstrap stability of the best candidate
    # -----------------------------------------------------------------------
    print("\n--- Part 1: Bootstrap Stability (2000 samples) ---")
    model = CalibratedModel(GBMModel(n_estimators=300, learning_rate=0.05, max_depth=4), method="isotonic")
    result = run_backtest(fs, strategy, model, flat_config, min_round=3)
    bdf = result.to_bet_dataframe()
    s = result.summary()

    print(f"  Overall: {s.get('n_bets', 0)} bets, {s.get('roi', 0)*100:+.1f}% ROI")

    stab = stability_analysis(bdf, n_bootstrap=2000)
    print(f"  Bootstrap ROI: {stab['roi']*100:+.1f}%")
    print(f"  95% CI: [{stab['roi_ci_lower']*100:.1f}%, {stab['roi_ci_upper']*100:.1f}%]")
    print(f"  P(ROI > 0): {stab['p_positive_roi']*100:.0f}%")

    cross = cross_season_stability(bdf, n_bootstrap=2000)
    if not cross.empty:
        for _, row in cross.iterrows():
            print(f"  {int(row['season'])}: {int(row['n_bets'])} bets, ROI={row['roi']*100:+.1f}%, "
                  f"CI=[{row['roi_ci_lower']*100:.1f}%, {row['roi_ci_upper']*100:.1f}%], "
                  f"P(ROI>0)={row['p_positive_roi']*100:.0f}%")

    # -----------------------------------------------------------------------
    # 2. Neighborhood hyperparameter search
    # -----------------------------------------------------------------------
    print("\n--- Part 2: Neighborhood Search ---")
    print(f"{'Config':<50} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8} {'Both':>5}")
    print("-" * 85)

    neighbors = [
        (250, 0.05, 4), (300, 0.05, 4), (350, 0.05, 4), (400, 0.05, 4),
        (300, 0.03, 4), (300, 0.07, 4), (300, 0.10, 4),
        (300, 0.05, 3), (300, 0.05, 5), (300, 0.05, 6),
        (250, 0.05, 3), (250, 0.05, 5),
        (350, 0.05, 3), (350, 0.05, 5),
        (400, 0.03, 4), (400, 0.05, 3),
        (300, 0.03, 3), (300, 0.03, 5),
    ]

    # Remove duplicates
    seen = set()
    unique_neighbors = []
    for n, lr, d in neighbors:
        key = (n, lr, d)
        if key not in seen:
            seen.add(key)
            unique_neighbors.append(key)

    profitable = []
    for n_est, lr, depth in unique_neighbors:
        label = f"CalGBM(n={n_est},lr={lr},d={depth})"
        model = CalibratedModel(
            GBMModel(n_estimators=n_est, learning_rate=lr, max_depth=depth),
            method="isotonic",
        )
        info = run_config(fs, strategy, model, flat_config, label)
        marker = " YES" if info["both_positive"] else ""
        print(f"{label:<50} {info['n_bets']:>5} {info['roi']*100:>+6.1f}% "
              f"{info['r24']*100:>+7.1f}% {info['r25']*100:>+7.1f}%{marker:>5}")
        if info["both_positive"]:
            profitable.append(info)

    # -----------------------------------------------------------------------
    # 3. Best configs + RefinedEdge
    # -----------------------------------------------------------------------
    print("\n--- Part 3: Best Configs + RefinedEdge Strategies ---")
    print(f"{'Config':<55} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8} {'Both':>5}")
    print("-" * 90)

    # Use all configs that were profitable in both seasons
    if profitable:
        best_configs = profitable
    else:
        # Fall back to the original candidate
        best_configs = [{"label": "CalGBM(n=300,lr=0.05,d=4)", "result": result}]

    refined_strategies = [
        ("RefinedEdge(default)", RefinedEdgeStrategy()),
        ("RefinedEdge(backs)", RefinedEdgeStrategy(positions=frozenset(["FB", "WG", "CE"]))),
        ("RefinedEdge(backs+halves)", RefinedEdgeStrategy(positions=frozenset(["FB", "WG", "CE", "FE", "HB"]))),
        ("RefinedEdge(e>0.07)", RefinedEdgeStrategy(min_edge=0.07)),
        ("RefinedEdge(backs,o2-5)", RefinedEdgeStrategy(
            positions=frozenset(["FB", "WG", "CE"]), min_odds=2.0, max_odds=5.0)),
    ]

    for n_est, lr, depth in [(300, 0.05, 4)] + [(p["label"].split("(")[1].rstrip(")"),) for p in profitable if p["label"] != "CalGBM(n=300,lr=0.05,d=4)"][:3]:
        if isinstance(n_est, tuple):
            continue  # skip complex parsing
        for strat_name, strat in refined_strategies:
            label = f"CalGBM({n_est},{lr},{depth})+{strat_name}"
            model = CalibratedModel(
                GBMModel(n_estimators=int(n_est), learning_rate=float(lr), max_depth=int(depth)),
                method="isotonic",
            )
            info = run_config(fs, strat, model, flat_config, label)
            marker = " YES" if info["both_positive"] else ""
            print(f"{label:<55} {info['n_bets']:>5} {info['roi']*100:>+6.1f}% "
                  f"{info['r24']*100:>+7.1f}% {info['r25']*100:>+7.1f}%{marker:>5}")

    # -----------------------------------------------------------------------
    # 4. Per-position ROI of best candidate
    # -----------------------------------------------------------------------
    print("\n--- Part 4: Per-position ROI (CalGBM 300/0.05/d4) ---")
    model = CalibratedModel(GBMModel(n_estimators=300, learning_rate=0.05, max_depth=4), method="isotonic")
    result = run_backtest(fs, strategy, model, flat_config, min_round=3)
    bdf = result.to_bet_dataframe()

    if not bdf.empty and "position_code" in bdf.columns:
        print(f"{'Position':>10} {'Bets':>6} {'Hit%':>7} {'ROI%':>8} {'Profit':>9}")
        print("-" * 45)
        for pos, grp in bdf.groupby("position_code"):
            n = len(grp)
            staked = grp["stake"].sum()
            payout = grp["payout"].sum()
            hit = grp["hit"].mean() * 100 if "hit" in grp.columns else 0
            roi = (payout - staked) / staked * 100 if staked > 0 else 0
            profit = payout - staked
            print(f"{pos:>10} {n:>6} {hit:>6.1f}% {roi:>+7.1f}% ${profit:>8.0f}")

    # -----------------------------------------------------------------------
    # 5. Per-odds-band ROI
    # -----------------------------------------------------------------------
    print("\n--- Part 5: Per-odds-band ROI ---")
    if not bdf.empty and "odds" in bdf.columns:
        bdf["odds_band"] = pd.cut(bdf["odds"], bins=[0, 2, 3, 4, 5, 7, 100],
                                   labels=["<2", "2-3", "3-4", "4-5", "5-7", "7+"])
        print(f"{'Odds Band':>10} {'Bets':>6} {'Hit%':>7} {'ROI%':>8} {'Profit':>9}")
        print("-" * 45)
        for band, grp in bdf.groupby("odds_band", observed=True):
            n = len(grp)
            staked = grp["stake"].sum()
            payout = grp["payout"].sum()
            hit = grp["hit"].mean() * 100 if "hit" in grp.columns else 0
            roi = (payout - staked) / staked * 100 if staked > 0 else 0
            profit = payout - staked
            print(f"{band:>10} {n:>6} {hit:>6.1f}% {roi:>+7.1f}% ${profit:>8.0f}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
