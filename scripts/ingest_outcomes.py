"""CLI entry point for outcome ingestion and CLV tracking.

Run this AFTER a round completes (Monday/Tuesday) to:
1. Ingest actual match results from Champion Data
2. Compare predictions to actuals
3. Record CLV for adaptive Kelly adjustments
4. Print round P&L summary

Usage:
    python scripts/ingest_outcomes.py --season 2026 --round 2
    python scripts/ingest_outcomes.py --season 2026 --round 2 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.weekly_pipeline import ingest_outcomes_and_clv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest round outcomes and record CLV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--round", type=int, required=True, help="Completed round number")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = ingest_outcomes_and_clv(
        season=args.season,
        round_number=args.round,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"Outcome Ingestion: Season {args.season} Round {args.round}")
    print("=" * 60)

    ing = result.get("ingestion", {})
    if "error" in ing:
        print(f"\nIngestion ERROR: {ing['error']}")
    else:
        print(f"\nMatches ingested: {ing.get('n_ingested', 0)}")
        if ing.get("n_failed", 0) > 0:
            print(f"Matches failed: {ing['n_failed']}")

    print(f"CLV records: {result.get('clv_records', 0)}")

    ev = result.get("evaluation", {})
    if ev:
        print(f"\nRound P&L:")
        print(f"  Bets: {ev['n_bets']} ({ev['n_wins']} wins, {ev['win_rate']:.0%} rate)")
        print(f"  Staked: ${ev['total_staked']:,.0f}")
        print(f"  Payout: ${ev['total_payout']:,.0f}")
        print(f"  Profit: ${ev['profit']:+,.0f} ({ev['roi']:+.1%} ROI)")
    else:
        print("\nNo bet evaluation data available.")


if __name__ == "__main__":
    main()
