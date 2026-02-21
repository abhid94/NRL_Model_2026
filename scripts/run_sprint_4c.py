"""Sprint 4C: Deep edge analysis and refined strategy backtest.

Runs segment profitability analysis with bootstrap CIs,
cross-season stability validation, two-way interactions,
and RefinedEdgeStrategy backtest vs all prior strategies.
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import FEATURE_STORE_DIR
from src.evaluation.backtest import BacktestConfig, compare_backtests, run_backtest
from src.evaluation.edge_analysis import (
    ODDS_BANDS,
    ODDS_LABELS,
    conditional_edge_analysis,
    cross_season_stability,
    model_vs_market_disagreement,
    stability_analysis,
    two_way_segment_roi,
)
from src.models.baseline import (
    ModelEdgeStrategy,
    RefinedEdgeStrategy,
    SegmentPlayStrategy,
    FadeHotStreakStrategy,
)
from src.models.calibration import CalibratedModel
from src.models.ensemble import WeightedEnsemble
from src.models.gbm import GBMModel
from src.models.poisson import PoissonModel
from src.models.baseline import EnrichedLogisticModel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("Sprint 4C: Deep Edge Analysis & Refined Strategy")
    LOGGER.info("=" * 70)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)
    LOGGER.info("Loaded feature store: %d rows x %d cols", len(fs), len(fs.columns))

    flat_config = BacktestConfig(flat_stake=100.0)

    # -----------------------------------------------------------------------
    # Part 1: Run CalibratedGBM baseline to get bet data for analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 1: Baseline CalibratedGBM for edge analysis")
    print("=" * 70)

    cal_gbm = CalibratedModel(GBMModel(n_estimators=200), method="isotonic")
    strategy = ModelEdgeStrategy()
    baseline_result = run_backtest(fs, strategy, cal_gbm, flat_config, min_round=3)
    bet_df = baseline_result.to_bet_dataframe()

    summary = baseline_result.summary()
    print(f"Baseline: {summary.get('n_bets', 0)} bets, "
          f"{summary.get('roi', 0)*100:.1f}% ROI, "
          f"${summary.get('profit', 0):.0f} profit")

    # -----------------------------------------------------------------------
    # Part 2: Model vs Market Disagreement Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 2: Model vs Market Disagreement")
    print("=" * 70)

    disagreement = model_vs_market_disagreement(bet_df)
    if not disagreement.empty:
        print(disagreement.to_string(index=False))
    else:
        print("  No disagreement data available")

    # -----------------------------------------------------------------------
    # Part 3: Bootstrap Stability Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 3: Bootstrap Stability (1000 samples)")
    print("=" * 70)

    stability = stability_analysis(bet_df, n_bootstrap=1000)
    print(f"  ROI: {stability['roi']*100:.1f}%")
    print(f"  95% CI: [{stability['roi_ci_lower']*100:.1f}%, {stability['roi_ci_upper']*100:.1f}%]")
    print(f"  P(ROI > 0): {stability['p_positive_roi']*100:.1f}%")
    print(f"  N bets: {stability['n_bets']}")

    # -----------------------------------------------------------------------
    # Part 4: Cross-season Stability
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 4: Cross-season Stability")
    print("=" * 70)

    cross = cross_season_stability(bet_df)
    if not cross.empty:
        for _, row in cross.iterrows():
            print(f"  {int(row['season'])}: ROI={row['roi']*100:.1f}%, "
                  f"CI=[{row['roi_ci_lower']*100:.1f}%, {row['roi_ci_upper']*100:.1f}%], "
                  f"P(ROI>0)={row['p_positive_roi']*100:.1f}%, "
                  f"N={row['n_bets']}")

    # -----------------------------------------------------------------------
    # Part 5: Conditional Edge Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 5: Conditional Edge Analysis")
    print("=" * 70)

    conditions = [
        ("Backs only", {"position_filter": ["FB", "WG", "CE"]}),
        ("Backs, odds 2-4", {"position_filter": ["FB", "WG", "CE"], "min_odds": 2.0, "max_odds": 4.0}),
        ("Backs, odds 2-4, team tries > 4", {
            "position_filter": ["FB", "WG", "CE"],
            "min_odds": 2.0, "max_odds": 4.0,
            "min_team_tries": 4.0,
        }),
        ("Halves + back-row", {"position_filter": ["FE", "HB", "SR", "LK"]}),
        ("Wings only, odds 2-5", {"position_filter": ["WG"], "min_odds": 2.0, "max_odds": 5.0}),
    ]

    for label, kwargs in conditions:
        result = conditional_edge_analysis(bet_df, fs, **kwargs)
        n = result.get("n_bets", 0)
        roi = result.get("roi", 0)
        hit = result.get("hit_rate", 0)
        print(f"  {label:<40} N={n:>4}, ROI={roi*100:>+6.1f}%, Hit={hit*100:>5.1f}%")

    # -----------------------------------------------------------------------
    # Part 6: Two-way Interaction Analysis (Position x Odds)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 6: Two-way ROI (Position x Odds Band)")
    print("=" * 70)

    if not bet_df.empty:
        enriched = bet_df.copy()
        interaction = two_way_segment_roi(
            enriched,
            row_col="position_code",
            col_col="odds",
            col_bins=ODDS_BANDS,
            col_labels=ODDS_LABELS,
        )
        if not interaction.empty:
            print((interaction * 100).round(1).to_string())
        else:
            print("  No interaction data")

    # -----------------------------------------------------------------------
    # Part 7: RefinedEdgeStrategy Backtest vs Prior Strategies
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 7: Strategy Comparison (including RefinedEdge)")
    print("=" * 70)

    def make_gbm():
        return CalibratedModel(GBMModel(n_estimators=200), method="isotonic")

    def make_poisson():
        return CalibratedModel(PoissonModel(reg_alpha=1.0), method="isotonic")

    def make_logistic():
        return EnrichedLogisticModel()

    strategies_models = [
        ("CalibratedGBM+ModelEdge", ModelEdgeStrategy(), make_gbm()),
        ("CalibratedGBM+RefinedEdge", RefinedEdgeStrategy(), make_gbm()),
        ("CalibratedGBM+RefinedEdge(tight)", RefinedEdgeStrategy(
            min_edge=0.07, min_odds=2.0, max_odds=4.0, min_team_tries=4.0,
        ), make_gbm()),
        ("WeightedEnsemble+ModelEdge", ModelEdgeStrategy(), WeightedEnsemble(
            [make_gbm(), make_poisson(), make_logistic()],
        )),
        ("WeightedEnsemble+RefinedEdge", RefinedEdgeStrategy(), WeightedEnsemble(
            [make_gbm(), make_poisson(), make_logistic()],
        )),
        ("SegmentPlay", SegmentPlayStrategy(), None),
        ("FadeHotStreak", FadeHotStreakStrategy(), None),
    ]

    results = []
    print(f"\n{'Strategy':<40} {'Bets':>6} {'Hit%':>7} {'ROI%':>8} {'Profit':>10} {'MaxDD':>10}")
    print("-" * 80)

    for label, strat, model in strategies_models:
        try:
            result = run_backtest(fs, strat, model, flat_config, min_round=3)
            s = result.summary()
            n_bets = s.get("n_bets", 0)
            hit_rate = s.get("hit_rate", 0) * 100
            roi = s.get("roi", 0) * 100
            profit = s.get("profit", 0)
            max_dd = s.get("max_drawdown", 0)

            print(f"{label:<40} {n_bets:>6} {hit_rate:>6.1f}% {roi:>7.1f}% {profit:>9.0f} {max_dd:>9.0f}")

            # Season breakdown
            bdf = result.to_bet_dataframe()
            if not bdf.empty and "season" in bdf.columns:
                for season in sorted(bdf["season"].unique()):
                    s_df = bdf[bdf["season"] == season]
                    if s_df["stake"].sum() > 0:
                        s_roi = (s_df["payout"].sum() - s_df["stake"].sum()) / s_df["stake"].sum() * 100
                        s_bets = len(s_df)
                        print(f"  └─ {season}: {s_bets} bets, {s_roi:+.1f}% ROI")

            # Bootstrap stability for each strategy
            bdf = result.to_bet_dataframe()
            if not bdf.empty:
                stab = stability_analysis(bdf, n_bootstrap=500)
                print(f"  └─ P(ROI>0): {stab['p_positive_roi']*100:.0f}%, "
                      f"CI: [{stab['roi_ci_lower']*100:.1f}%, {stab['roi_ci_upper']*100:.1f}%]")

            results.append(result)
        except Exception as e:
            print(f"{label:<40} ERROR: {e}")
            LOGGER.exception("Strategy %s failed", label)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_dir = FEATURE_STORE_DIR.parent / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if results:
        comparison = compare_backtests(results)
        comparison.to_csv(out_dir / "sprint_4c_refined_comparison.csv", index=False)
        LOGGER.info("Saved to %s", out_dir / "sprint_4c_refined_comparison.csv")

    print("\n" + "=" * 70)
    print("Sprint 4C Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
