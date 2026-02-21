"""CLI entry point for the weekly prediction pipeline.

Usage examples:
    # Full pipeline: rebuild features, train, predict, recommend
    python scripts/run_weekly_pipeline.py --season 2026 --round 5

    # Use cached features (faster)
    python scripts/run_weekly_pipeline.py --season 2026 --round 5 --no-rebuild

    # Custom bankroll and flat staking
    python scripts/run_weekly_pipeline.py --season 2026 --round 5 --bankroll 15000 --flat-stake 100

    # Dry run on historical data (2025 season)
    python scripts/run_weekly_pipeline.py --season 2025 --round 20 --bankroll 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_INITIAL_BANKROLL
from src.pipeline.weekly_pipeline import run_weekly_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NRL ATS Weekly Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g. 2026)")
    parser.add_argument("--round", type=int, required=True, help="Round number to predict")
    parser.add_argument(
        "--bankroll", type=float, default=DEFAULT_INITIAL_BANKROLL,
        help=f"Current bankroll (default: ${DEFAULT_INITIAL_BANKROLL:,.0f})",
    )
    parser.add_argument(
        "--no-rebuild", action="store_true",
        help="Skip feature store rebuild (use cached parquet files)",
    )
    parser.add_argument(
        "--flat-stake", type=float, default=None,
        help="Use flat staking instead of Kelly (e.g. 100 for $100 per bet)",
    )
    parser.add_argument(
        "--training-seasons", type=int, nargs="+", default=None,
        help="Seasons to include in training (default: 2024 2025 + current)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_weekly_pipeline(
        season=args.season,
        round_number=args.round,
        bankroll=args.bankroll,
        training_seasons=args.training_seasons,
        rebuild_features=not args.no_rebuild,
        flat_stake=args.flat_stake,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(result["bet_card"].summary())
    print("=" * 60)

    ds = result["drawdown_status"]
    print(f"\nDrawdown: {ds['drawdown_pct']*100:.1f}% — {ds['status']}")
    if ds["status"] != "OK":
        print(f"  {ds['message']}")

    print(f"\nTraining rows: {result['training_rows']:,}")
    print(f"Pipeline time: {result['elapsed_seconds']:.1f}s")
    print(f"Log saved: {result['log_entry'].get('timestamp', 'N/A')}")


if __name__ == "__main__":
    main()
