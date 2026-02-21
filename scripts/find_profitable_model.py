"""Focused profitability search: find a model/strategy profitable in BOTH seasons.

Runs fast backtests with targeted configurations and reports per-season ROI.
Goal: find any configuration where ROI > 0 in BOTH 2024 AND 2025.
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
from src.evaluation.edge_analysis import stability_analysis, cross_season_stability
from src.models.baseline import (
    ModelEdgeStrategy,
    RefinedEdgeStrategy,
    SegmentPlayStrategy,
)
from src.models.calibration import CalibratedModel
from src.models.gbm import GBMModel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def per_season_roi(result) -> dict[int, dict]:
    """Get per-season ROI from a backtest result."""
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
                "profit": total_payout - total_staked,
                "hit_rate": sdf["hit"].mean() if "hit" in sdf.columns else np.nan,
            }
    return out


def main():
    print("=" * 70)
    print("PROFITABILITY SEARCH: Find model profitable in BOTH seasons")
    print("=" * 70)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)
    print(f"Data: {len(fs)} rows x {len(fs.columns)} cols")

    flat_config = BacktestConfig(flat_stake=100.0)

    # -----------------------------------------------------------------------
    # Configuration matrix: model × strategy × parameters
    # -----------------------------------------------------------------------

    configs = []

    # --- CalibratedGBM variations ---
    for n_est in [100, 200, 300]:
        configs.append((
            f"CalGBM({n_est})+ModelEdge",
            ModelEdgeStrategy(),
            CalibratedModel(GBMModel(n_estimators=n_est), method="isotonic"),
        ))

    # --- RefinedEdge with different parameters ---
    refined_params = [
        {"min_edge": 0.05},  # default
        {"min_edge": 0.07},
        {"min_edge": 0.10},
        {"min_edge": 0.05, "positions": frozenset(["FB", "WG", "CE"])},
        {"min_edge": 0.05, "positions": frozenset(["FB", "WG", "CE", "FE", "HB"])},
        {"min_edge": 0.05, "min_odds": 2.0, "max_odds": 5.0},
        {"min_edge": 0.05, "min_odds": 2.0, "max_odds": 4.0},
        {"min_edge": 0.07, "positions": frozenset(["FB", "WG", "CE"]), "min_odds": 2.0, "max_odds": 5.0},
        {"min_edge": 0.05, "positions": frozenset(["WG"]), "min_odds": 2.0, "max_odds": 5.0},
        {"min_edge": 0.05, "min_team_tries": 4.0},
        {"min_edge": 0.05, "positions": frozenset(["FB", "WG", "CE"]), "min_team_tries": 4.0},
        {"min_edge": 0.07, "positions": frozenset(["FB", "WG", "CE"]), "min_odds": 2.0, "max_odds": 5.0, "min_team_tries": 3.5},
    ]

    for i, params in enumerate(refined_params):
        label_parts = []
        if "min_edge" in params:
            label_parts.append(f"e>{params['min_edge']}")
        if "positions" in params:
            pos_str = "+".join(sorted(params["positions"]))
            label_parts.append(pos_str)
        if "min_odds" in params:
            label_parts.append(f"o{params.get('min_odds', 1)}-{params.get('max_odds', 99)}")
        if "min_team_tries" in params:
            label_parts.append(f"tt>{params['min_team_tries']}")
        label = f"CalGBM+Refined({','.join(label_parts)})"

        configs.append((
            label,
            RefinedEdgeStrategy(**params),
            CalibratedModel(GBMModel(n_estimators=200), method="isotonic"),
        ))

    # --- Kelly staking variants ---
    kelly_config = BacktestConfig()  # uses Kelly staking
    configs_with_staking = []
    for label, strat, model in configs[:3]:  # Only GBM variations
        configs_with_staking.append((label + " [Kelly]", strat, model, kelly_config))

    # -----------------------------------------------------------------------
    # Run backtests
    # -----------------------------------------------------------------------

    print(f"\nRunning {len(configs)} configurations...")
    print(f"\n{'Config':<55} {'Bets':>5} {'ROI%':>7} {'2024':>8} {'2025':>8} {'Both+?':>6}")
    print("-" * 95)

    profitable_both = []

    for label, strat, model in configs:
        try:
            result = run_backtest(fs, strat, model, flat_config, min_round=3)
            s = result.summary()
            n_bets = s.get("n_bets", 0)
            roi = s.get("roi", 0)

            season_data = per_season_roi(result)
            r24 = season_data.get(2024, {}).get("roi", float("nan"))
            r25 = season_data.get(2025, {}).get("roi", float("nan"))
            both_positive = (r24 > 0) and (r25 > 0)

            marker = " YES" if both_positive else "  no"
            print(f"{label:<55} {n_bets:>5} {roi*100:>+6.1f}% {r24*100:>+7.1f}% {r25*100:>+7.1f}% {marker}")

            if both_positive:
                profitable_both.append({
                    "label": label,
                    "n_bets": n_bets,
                    "overall_roi": roi,
                    "roi_2024": r24,
                    "roi_2025": r25,
                    "n_2024": season_data.get(2024, {}).get("n_bets", 0),
                    "n_2025": season_data.get(2025, {}).get("n_bets", 0),
                    "result": result,
                })

        except Exception as e:
            print(f"{label:<55} ERROR: {e}")

    # Kelly variants
    if configs_with_staking:
        print("\n--- Kelly Staking Variants ---")
        for label, strat, model, cfg in configs_with_staking:
            try:
                result = run_backtest(fs, strat, model, cfg, min_round=3)
                s = result.summary()
                n_bets = s.get("n_bets", 0)
                roi = s.get("roi", 0)

                season_data = per_season_roi(result)
                r24 = season_data.get(2024, {}).get("roi", float("nan"))
                r25 = season_data.get(2025, {}).get("roi", float("nan"))
                both_positive = (r24 > 0) and (r25 > 0)

                marker = " YES" if both_positive else "  no"
                print(f"{label:<55} {n_bets:>5} {roi*100:>+6.1f}% {r24*100:>+7.1f}% {r25*100:>+7.1f}% {marker}")

                if both_positive:
                    profitable_both.append({
                        "label": label,
                        "n_bets": n_bets,
                        "overall_roi": roi,
                        "roi_2024": r24,
                        "roi_2025": r25,
                        "n_2024": season_data.get(2024, {}).get("n_bets", 0),
                        "n_2025": season_data.get(2025, {}).get("n_bets", 0),
                        "result": result,
                    })
            except Exception as e:
                print(f"{label:<55} ERROR: {e}")

    # -----------------------------------------------------------------------
    # Detailed analysis of profitable configs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    if profitable_both:
        print(f"FOUND {len(profitable_both)} configurations profitable in BOTH seasons!")
        print("=" * 70)

        for cfg in sorted(profitable_both, key=lambda x: x["overall_roi"], reverse=True):
            print(f"\n--- {cfg['label']} ---")
            print(f"  Overall: {cfg['n_bets']} bets, {cfg['overall_roi']*100:+.1f}% ROI")
            print(f"  2024: {cfg['n_2024']} bets, {cfg['roi_2024']*100:+.1f}% ROI")
            print(f"  2025: {cfg['n_2025']} bets, {cfg['roi_2025']*100:+.1f}% ROI")

            # Bootstrap stability
            bdf = cfg["result"].to_bet_dataframe()
            if len(bdf) >= 20:
                stab = stability_analysis(bdf, n_bootstrap=1000)
                print(f"  Bootstrap P(ROI > 0): {stab['p_positive_roi']*100:.0f}%")
                print(f"  95% CI: [{stab['roi_ci_lower']*100:.1f}%, {stab['roi_ci_upper']*100:.1f}%]")

                cross = cross_season_stability(bdf, n_bootstrap=1000)
                if not cross.empty:
                    for _, row in cross.iterrows():
                        print(f"  {int(row['season'])}: P(ROI>0)={row['p_positive_roi']*100:.0f}%, "
                              f"CI=[{row['roi_ci_lower']*100:.1f}%, {row['roi_ci_upper']*100:.1f}%]")
    else:
        print("NO configuration found profitable in BOTH seasons.")
        print("=" * 70)
        print("\nClosest candidates (best combined ROI):")
        # Show top 5 by overall ROI even if not both positive
        all_results = []
        # Re-collect from the loop above... not ideal but let's just print conclusion
        print("Review the per-season columns above for the least negative combos.")

    print("\nDone.")


if __name__ == "__main__":
    main()
