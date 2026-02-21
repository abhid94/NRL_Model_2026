"""Sprint 3C experiment runner: Edge Discovery & Segment Analysis.

Runs the top 3 profitable strategies through comprehensive segment analysis:
1. GBM NoBetfair + ModelEdge (best flat ROI)
2. GBM + ModelEdge
3. EdgeMatchup rule-based

For each strategy, produces:
- ROI by position, matchup quartile, team tries, odds band, season
- CLV analysis (winners vs losers)
- Round-by-round P&L curves
- Saves CSV outputs to data/backtest_results/
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config import BACKTEST_RESULTS_DIR, FEATURE_STORE_DIR
from src.evaluation.backtest import BacktestConfig, run_backtest
from src.evaluation.edge_analysis import generate_edge_report
from src.models.baseline import EdgeMatchupStrategy, ModelEdgeStrategy
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


def print_report_table(name: str, df: pd.DataFrame) -> None:
    """Pretty-print a report DataFrame."""
    if df.empty:
        LOGGER.info("  [%s] — no data", name)
        return
    print(f"\n  {name}:")
    print("  " + "-" * 80)
    print(df.to_string(index=False))
    print()


def run_strategy_analysis(
    label: str,
    fs: pd.DataFrame,
    strategy,
    model=None,
) -> None:
    """Run backtest + full edge analysis for one strategy."""
    LOGGER.info("=" * 70)
    LOGGER.info("STRATEGY: %s", label)
    LOGGER.info("=" * 70)

    flat_config = BacktestConfig(flat_stake=100.0)
    result = run_backtest(fs, strategy, model, flat_config, min_round=3)
    bet_df = result.to_bet_dataframe()

    summary = result.summary()
    LOGGER.info(
        "  %d bets | ROI: %.1f%% | Hit rate: %.1f%% | Profit: $%.0f | Max DD: $%.0f",
        summary.get("n_bets", 0),
        summary.get("roi", 0) * 100,
        summary.get("hit_rate", 0) * 100,
        summary.get("profit", 0),
        summary.get("max_drawdown", 0),
    )

    if bet_df.empty:
        LOGGER.warning("  No bets placed — skipping edge analysis")
        return

    # Generate full report
    report = generate_edge_report(bet_df, fs)

    # Print all tables
    for section_name, section_df in report.items():
        print_report_table(section_name, section_df)

    # Save CSVs
    out_dir = BACKTEST_RESULTS_DIR / label.lower().replace(" ", "_").replace("+", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    for section_name, section_df in report.items():
        if not section_df.empty:
            path = out_dir / f"{section_name}.csv"
            section_df.to_csv(path, index=False)
    LOGGER.info("  Saved %d CSV files to %s", len(report), out_dir)


def main() -> None:
    """Run all Sprint 3C edge discovery experiments."""
    LOGGER.info("=" * 70)
    LOGGER.info("SPRINT 3C: Edge Discovery & Segment Analysis")
    LOGGER.info("=" * 70)

    fs = load_combined_feature_store()

    # Strategy 1: GBM NoBetfair + ModelEdge
    run_strategy_analysis(
        "GBM_NoBetfair_ModelEdge",
        fs,
        ModelEdgeStrategy(),
        GBMModelNoBetfair(n_estimators=200),
    )

    # Strategy 2: GBM + ModelEdge
    run_strategy_analysis(
        "GBM_ModelEdge",
        fs,
        ModelEdgeStrategy(),
        GBMModel(n_estimators=200),
    )

    # Strategy 3: EdgeMatchup rule-based
    run_strategy_analysis(
        "EdgeMatchup",
        fs,
        EdgeMatchupStrategy(),
        None,
    )

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("SPRINT 3C COMPLETE")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
