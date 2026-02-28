"""CLI runner: ingest the 2026 NRL fixture from Champion Data.

Usage::

    python scripts/ingest_2026_fixture.py
    python scripts/ingest_2026_fixture.py --year 2026
    python scripts/ingest_2026_fixture.py --url "https://..." --year 2026

After running, verify with::

    SELECT COUNT(*) FROM matches_2026;
    SELECT round_number, COUNT(*) FROM matches_2026
    GROUP BY round_number ORDER BY round_number;
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable regardless
# of the working directory from which this script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Default Champion Data fixture URL for the 2026 NRL season
# ---------------------------------------------------------------------------
_DEFAULT_URL = (
    "https://mc.championdata.com/data/12999/fixture.json"
    "?uuid=c2ca04ae-ed5d-46e3-856d-04b80972e1c0&_=1771938144697"
)
_DEFAULT_YEAR = 2026


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest the NRL fixture from Champion Data into nrl_data.db"
    )
    parser.add_argument(
        "--url",
        default=_DEFAULT_URL,
        help="Champion Data fixture JSON URL (default: 2026 season URL)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=_DEFAULT_YEAR,
        help=f"Season year (default: {_DEFAULT_YEAR})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Ingesting %d fixture from Champion Data", args.year)
    logger.info("URL: %s", args.url)

    from src.ingestion.ingest_champion_data import ingest_fixture

    try:
        summary = ingest_fixture(url=args.url, year=args.year)
    except RuntimeError as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 55)
    print(f"  Fixture Ingestion Summary — {args.year} NRL Season")
    print("=" * 55)
    print(f"  Matches written  : {summary['n_matches']}")
    print(f"  Rounds covered   : {summary['n_rounds']}")
    print(f"  Teams upserted   : {summary['n_teams']}")
    sample = summary.get("match_ids_sample", [])
    if sample:
        sample_str = ", ".join(str(m) for m in sample)
        print(f"  Sample match IDs : {sample_str} ...")
    print("=" * 55)

    # ---------------------------------------------------------------------------
    # Quick verification query
    # ---------------------------------------------------------------------------
    print()
    print("Running quick verification queries …")

    from src.config import DB_PATH
    from src.db import get_connection, get_table

    conn = get_connection(DB_PATH)
    table = get_table("matches", args.year)

    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  matches_{args.year} row count : {total}")

    rounds_rows = conn.execute(
        f"SELECT round_number, COUNT(*) FROM {table} "
        f"GROUP BY round_number ORDER BY round_number"
    ).fetchall()
    print(f"\n  Round distribution ({len(rounds_rows)} rounds):")
    for rnd, cnt in rounds_rows:
        bar = "#" * cnt
        print(f"    Round {rnd:>3} : {cnt:>2} matches  {bar}")

    team_count = conn.execute(
        f"""
        SELECT COUNT(DISTINCT sq) FROM (
            SELECT home_squad_id AS sq FROM {table}
            UNION
            SELECT away_squad_id AS sq FROM {table}
        )
        """
    ).fetchone()[0]
    print(f"\n  Distinct teams : {team_count}")

    # Orphan check: squad_ids not in teams table
    orphan_count = conn.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT home_squad_id AS sq FROM {table}
            UNION
            SELECT away_squad_id AS sq FROM {table}
        ) src
        LEFT JOIN teams t ON src.sq = t.squad_id
        WHERE t.squad_id IS NULL
        """
    ).fetchone()[0]
    if orphan_count == 0:
        print("  Orphan squad_ids: none (all present in teams table)")
    else:
        print(f"  WARNING: {orphan_count} squad_ids missing from teams table!")

    conn.close()
    print()
    print("Done. matches_{} is ready for downstream pipeline use.".format(args.year))


if __name__ == "__main__":
    main()
