"""Sprint 3B experiment runner.

Runs all GBM backtests, calibration experiments, SHAP analysis,
and flat-stake re-evaluation of Sprint 3A strategies.

Outputs:
- Comparison table (all strategies, flat-stake and Kelly)
- SHAP feature importance CSV + plots
- Per-season breakdowns
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.config import FEATURE_STORE_DIR, MODEL_ARTIFACTS_DIR
from src.evaluation.backtest import BacktestConfig, compare_backtests, run_backtest
from src.evaluation.metrics import (
    build_evaluation_report,
    compute_calibration_error,
)
from src.models.baseline import (
    BaseModel,
    CompositeStrategy,
    EdgeMatchupStrategy,
    EnrichedLogisticModel,
    FadeHotStreakStrategy,
    LogisticBaselineModel,
    MarketImpliedStrategy,
    ModelEdgeStrategy,
    PositionBaselineModel,
    SegmentPlayStrategy,
)
from src.models.calibration import CalibratedModel
from src.models.gbm import GBMModel, GBMModelNoBetfair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_combined_feature_store() -> pd.DataFrame:
    """Load the combined 2024+2025 feature store."""
    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    df = pd.read_parquet(path)
    LOGGER.info("Loaded feature store: %d rows x %d cols", len(df), len(df.columns))
    return df


def run_flat_stake_3a_strategies(fs: pd.DataFrame) -> list:
    """Re-run Sprint 3A strategies with $100 flat stakes."""
    flat_config = BacktestConfig(flat_stake=100.0)
    results = []

    # Strategy 1: ModelEdge + PositionBaseline
    r = run_backtest(fs, ModelEdgeStrategy(), PositionBaselineModel(), flat_config, min_round=3)
    results.append(r)
    LOGGER.info("[Flat 3A] %s/%s: ROI=%.1f%%, bets=%d", r.strategy_name, r.model_name, r.roi * 100, r.n_bets)

    # Strategy 2: ModelEdge + LogisticBaseline
    r = run_backtest(fs, ModelEdgeStrategy(), LogisticBaselineModel(), flat_config, min_round=3)
    results.append(r)
    LOGGER.info("[Flat 3A] %s/%s: ROI=%.1f%%, bets=%d", r.strategy_name, r.model_name, r.roi * 100, r.n_bets)

    # Strategy 3: ModelEdge + EnrichedLogistic
    r = run_backtest(fs, ModelEdgeStrategy(), EnrichedLogisticModel(), flat_config, min_round=3)
    results.append(r)
    LOGGER.info("[Flat 3A] %s/%s: ROI=%.1f%%, bets=%d", r.strategy_name, r.model_name, r.roi * 100, r.n_bets)

    # Strategy 4: SegmentPlay (rule-based)
    r = run_backtest(fs, SegmentPlayStrategy(), None, flat_config, min_round=3)
    results.append(r)
    LOGGER.info("[Flat 3A] %s: ROI=%.1f%%, bets=%d", r.strategy_name, r.roi * 100, r.n_bets)

    # Strategy 5: EdgeMatchup (rule-based)
    r = run_backtest(fs, EdgeMatchupStrategy(), None, flat_config, min_round=3)
    results.append(r)
    LOGGER.info("[Flat 3A] %s: ROI=%.1f%%, bets=%d", r.strategy_name, r.roi * 100, r.n_bets)

    # Strategy 6: FadeHotStreak (rule-based)
    r = run_backtest(fs, FadeHotStreakStrategy(), None, flat_config, min_round=3)
    results.append(r)
    LOGGER.info("[Flat 3A] %s: ROI=%.1f%%, bets=%d", r.strategy_name, r.roi * 100, r.n_bets)

    # Strategy 7: Composite (ModelEdge+EdgeMatchup+SegmentPlay) + EnrichedLogistic
    composite = CompositeStrategy([ModelEdgeStrategy(), EdgeMatchupStrategy(), SegmentPlayStrategy()])
    r = run_backtest(fs, composite, EnrichedLogisticModel(), flat_config, min_round=3)
    results.append(r)
    LOGGER.info("[Flat 3A] %s/%s: ROI=%.1f%%, bets=%d", r.strategy_name, r.model_name, r.roi * 100, r.n_bets)

    return results


def run_gbm_backtests(fs: pd.DataFrame) -> list:
    """Run Sprint 3B GBM backtests."""
    results = []

    # 8. GBM (all features), flat $100
    LOGGER.info("=" * 60)
    LOGGER.info("Backtest 8: GBM (all features), $100 flat stake")
    flat_config = BacktestConfig(flat_stake=100.0)
    r = run_backtest(fs, ModelEdgeStrategy(), GBMModel(n_estimators=200), flat_config, min_round=3)
    results.append(r)
    LOGGER.info("  ROI=%.1f%%, bets=%d, staked=$%.0f", r.roi * 100, r.n_bets, r.total_staked)

    # 9. GBM (no Betfair), flat $100
    LOGGER.info("Backtest 9: GBM (no Betfair), $100 flat stake")
    r = run_backtest(fs, ModelEdgeStrategy(), GBMModelNoBetfair(n_estimators=200), flat_config, min_round=3)
    results.append(r)
    LOGGER.info("  ROI=%.1f%%, bets=%d, staked=$%.0f", r.roi * 100, r.n_bets, r.total_staked)

    # 10. GBM + Isotonic calibration, flat $100
    LOGGER.info("Backtest 10: GBM + Isotonic calibration, $100 flat stake")
    cal_model = CalibratedModel(GBMModel(n_estimators=200), method="isotonic", cal_rounds=5)
    r = run_backtest(fs, ModelEdgeStrategy(), cal_model, flat_config, min_round=3)
    results.append(r)
    LOGGER.info("  ROI=%.1f%%, bets=%d, staked=$%.0f", r.roi * 100, r.n_bets, r.total_staked)

    # 11. GBM + Isotonic, Kelly staking
    LOGGER.info("Backtest 11: GBM + Isotonic, Kelly staking")
    kelly_config = BacktestConfig()
    cal_model = CalibratedModel(GBMModel(n_estimators=200), method="isotonic", cal_rounds=5)
    r = run_backtest(fs, ModelEdgeStrategy(), cal_model, kelly_config, min_round=3)
    results.append(r)
    LOGGER.info("  ROI=%.1f%%, bets=%d, final_bankroll=$%.0f", r.roi * 100, r.n_bets,
                r.round_results[-1].bankroll_after if r.round_results else 10000)

    # 12. GBM + Isotonic + Composite strategy, Kelly
    LOGGER.info("Backtest 12: GBM + Isotonic + Composite, Kelly staking")
    composite = CompositeStrategy([ModelEdgeStrategy(), EdgeMatchupStrategy(), SegmentPlayStrategy()])
    cal_model = CalibratedModel(GBMModel(n_estimators=200), method="isotonic", cal_rounds=5)
    r = run_backtest(fs, composite, cal_model, kelly_config, min_round=3)
    results.append(r)
    LOGGER.info("  ROI=%.1f%%, bets=%d, final_bankroll=$%.0f", r.roi * 100, r.n_bets,
                r.round_results[-1].bankroll_after if r.round_results else 10000)

    return results


def run_per_season_breakdown(fs: pd.DataFrame) -> None:
    """Run best models per-season to check consistency."""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("PER-SEASON BREAKDOWN")
    LOGGER.info("=" * 60)

    flat_config = BacktestConfig(flat_stake=100.0)

    for season in [2024, 2025]:
        LOGGER.info("\n--- Season %d ---", season)

        # GBM all features
        r = run_backtest(fs, ModelEdgeStrategy(), GBMModel(n_estimators=200),
                         flat_config, seasons=[season], min_round=3)
        LOGGER.info("  GBM (all): ROI=%.1f%%, bets=%d", r.roi * 100, r.n_bets)

        # GBM no Betfair
        r = run_backtest(fs, ModelEdgeStrategy(), GBMModelNoBetfair(n_estimators=200),
                         flat_config, seasons=[season], min_round=3)
        LOGGER.info("  GBM (no BF): ROI=%.1f%%, bets=%d", r.roi * 100, r.n_bets)

        # EdgeMatchup (unchanged)
        r = run_backtest(fs, EdgeMatchupStrategy(), None,
                         flat_config, seasons=[season], min_round=3)
        LOGGER.info("  EdgeMatchup: ROI=%.1f%%, bets=%d", r.roi * 100, r.n_bets)


def run_shap_analysis(fs: pd.DataFrame) -> None:
    """SHAP analysis: fit GBM on 2024, explain 2025 predictions."""
    try:
        import shap
    except ImportError:
        LOGGER.warning("SHAP not installed — skipping SHAP analysis. Install with: pip install shap")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("SHAP ANALYSIS")
    LOGGER.info("=" * 60)

    # Train on 2024, explain 2025
    train = fs[fs["season"] == 2024].copy()
    test = fs[fs["season"] == 2025].copy()

    if train.empty or test.empty:
        LOGGER.warning("Not enough data for SHAP (need both 2024 and 2025)")
        return

    model = GBMModel(n_estimators=200)
    y_train = train["scored_try"].values
    model.fit(train, y_train)

    # Prepare test features
    features = model.feature_names()
    X_test = test[features].copy()
    for col in [c for c in features if c in {"position_group", "position_code", "player_edge"}]:
        X_test[col] = X_test[col].astype("category")

    # SHAP values
    LOGGER.info("Computing SHAP values for %d test observations...", len(X_test))
    explainer = shap.TreeExplainer(model._model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # Positive class
    else:
        sv = shap_values

    # Feature importance by mean |SHAP|
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Save
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(MODEL_ARTIFACTS_DIR / "shap_importance.csv", index=False)
    LOGGER.info("Saved SHAP importance to %s", MODEL_ARTIFACTS_DIR / "shap_importance.csv")

    # Print top 20
    LOGGER.info("\nTop 20 features by mean |SHAP|:")
    for _, row in importance_df.head(20).iterrows():
        LOGGER.info("  %s: %.4f", row["feature"], row["mean_abs_shap"])

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = importance_df.head(20)
    ax.barh(range(len(top20)), top20["mean_abs_shap"].values[::-1])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Feature Importance (SHAP) — GBM Model")
    plt.tight_layout()
    fig.savefig(MODEL_ARTIFACTS_DIR / "shap_importance_bar.png", dpi=150)
    plt.close(fig)
    LOGGER.info("Saved bar chart to %s", MODEL_ARTIFACTS_DIR / "shap_importance_bar.png")

    # Beeswarm plot
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(sv, X_test, feature_names=features, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(MODEL_ARTIFACTS_DIR / "shap_beeswarm.png", dpi=150)
        plt.close()
        LOGGER.info("Saved beeswarm plot to %s", MODEL_ARTIFACTS_DIR / "shap_beeswarm.png")
    except Exception as e:
        LOGGER.warning("Beeswarm plot failed: %s", e)


def main() -> None:
    """Run all Sprint 3B experiments."""
    LOGGER.info("=" * 60)
    LOGGER.info("SPRINT 3B: GBM + Calibration + SHAP + Flat-Stake Validation")
    LOGGER.info("=" * 60)

    fs = load_combined_feature_store()
    all_results = []

    # Part 1: Re-run Sprint 3A with flat stakes
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("PART 1: Sprint 3A strategies with $100 flat stakes")
    LOGGER.info("=" * 60)
    flat_3a = run_flat_stake_3a_strategies(fs)
    all_results.extend(flat_3a)

    # Part 2: Sprint 3B GBM experiments
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("PART 2: Sprint 3B GBM experiments")
    LOGGER.info("=" * 60)
    gbm_results = run_gbm_backtests(fs)
    all_results.extend(gbm_results)

    # Comparison table
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("COMPARISON TABLE (ALL STRATEGIES)")
    LOGGER.info("=" * 60)
    comparison = compare_backtests(all_results)
    # Add staking type column
    n_flat_3a = len(flat_3a)
    staking = (
        ["$100 Flat"] * n_flat_3a
        + ["$100 Flat"] * 3  # Backtests 8,9,10
        + ["Kelly"] * 2       # Backtests 11,12
    )
    comparison["staking"] = staking[:len(comparison)]

    display_cols = [
        "strategy", "model", "staking", "n_bets", "total_staked",
        "profit", "roi", "hit_rate", "avg_odds", "avg_edge", "max_drawdown",
    ]
    available_cols = [c for c in display_cols if c in comparison.columns]
    print("\n" + comparison[available_cols].to_string(index=False, float_format="%.3f"))

    # Save comparison
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(MODEL_ARTIFACTS_DIR / "sprint_3b_comparison.csv", index=False)
    LOGGER.info("Saved comparison to %s", MODEL_ARTIFACTS_DIR / "sprint_3b_comparison.csv")

    # Part 3: Per-season breakdown
    run_per_season_breakdown(fs)

    # Part 4: SHAP analysis
    run_shap_analysis(fs)

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("SPRINT 3B COMPLETE")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
