"""Ingest NRL fixture data from the Champion Data API.

Main entry point: ingest_fixture(url, year)

Pipeline steps:
    1. Fetch fixture JSON from Champion Data endpoint
    2. Parse each match record into DB-column format
    3. Create matches_{year} table if it does not exist
    4. Upsert teams extracted from the fixture into the teams table
    5. Upsert match rows into matches_{year}

Usage::

    from src.ingestion.ingest_champion_data import ingest_fixture

    summary = ingest_fixture(
        url="https://mc.championdata.com/data/12999/fixture.json?...",
        year=2026,
    )
    # summary: {n_matches, n_rounds, n_teams, match_ids_sample}
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.request
from typing import Any, Optional

from src.config import DB_PATH
from src.db import get_connection, get_table, normalize_year, table_exists

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_MATCHES_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    match_id INTEGER PRIMARY KEY,
    match_number INTEGER,
    round_number INTEGER,
    match_type TEXT,
    match_status TEXT,
    utc_start_time TEXT,
    local_start_time TEXT,
    home_squad_id INTEGER,
    away_squad_id INTEGER,
    venue_id INTEGER,
    venue_name TEXT,
    venue_code TEXT,
    period_completed INTEGER,
    period_seconds INTEGER,
    final_code TEXT,
    final_short_code TEXT
)
"""


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_fixture(url: str) -> list[dict[str, Any]]:
    """Fetch the fixture JSON from a Champion Data endpoint.

    Parameters
    ----------
    url : str
        Champion Data fixture URL.

    Returns
    -------
    list[dict[str, Any]]
        Raw match dicts from ``fixture.match``.

    Raises
    ------
    RuntimeError
        On HTTP errors or unexpected JSON structure.
    """
    LOGGER.info("Fetching fixture from %s", url)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} fetching fixture from {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching fixture: {exc.reason}") from exc

    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in fixture response: {exc}") from exc

    try:
        matches = payload["fixture"]["match"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected fixture JSON structure — expected fixture.match list: {exc}"
        ) from exc

    if not isinstance(matches, list):
        raise RuntimeError(
            f"fixture.match is not a list (got {type(matches).__name__})"
        )

    LOGGER.info("Fetched %d matches from Champion Data", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def _parse_match(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a raw Champion Data match dict to DB column format.

    Parameters
    ----------
    raw : dict[str, Any]
        Single match entry from the Champion Data fixture JSON.

    Returns
    -------
    dict[str, Any]
        Row dict keyed by ``matches_{year}`` column names.

    Raises
    ------
    KeyError
        If a required field is missing from the raw dict.
    """
    # Mandatory fields — raise loudly if missing
    match_id = int(raw["matchId"])
    home_squad_id = int(raw["homeSquadId"])
    away_squad_id = int(raw["awaySquadId"])

    def _none_if_empty(val: Any) -> Optional[str]:
        """Return None for empty strings."""
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None

    return {
        "match_id": match_id,
        "match_number": raw.get("matchNumber"),
        "round_number": raw.get("roundNumber"),
        "match_type": _none_if_empty(raw.get("matchType")),
        "match_status": _none_if_empty(raw.get("matchStatus")),
        "utc_start_time": _none_if_empty(raw.get("utcStartTime")),
        "local_start_time": _none_if_empty(raw.get("localStartTime")),
        "home_squad_id": home_squad_id,
        "away_squad_id": away_squad_id,
        "venue_id": raw.get("venueId"),
        "venue_name": _none_if_empty(raw.get("venueName")),
        "venue_code": _none_if_empty(raw.get("venueCode")),
        # Not present in the fixture endpoint (only in live/completed match data)
        "period_completed": None,
        "period_seconds": None,
        "final_code": _none_if_empty(raw.get("finalCode")),
        "final_short_code": _none_if_empty(raw.get("finalShortCode")),
    }


# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------


def create_matches_table(conn: sqlite3.Connection, year: int) -> None:
    """Create matches_{year} table if it does not exist.

    The schema matches matches_2024 / matches_2025 exactly.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    year : int
        Season year.
    """
    table = get_table("matches", year)
    conn.execute(_MATCHES_DDL.format(table=table))
    conn.commit()
    LOGGER.info("Table ready: %s", table)


# ---------------------------------------------------------------------------
# Upserts
# ---------------------------------------------------------------------------


def upsert_teams_from_fixture(
    conn: sqlite3.Connection,
    raw_matches: list[dict[str, Any]],
) -> int:
    """Extract team data from raw fixture matches and upsert into teams table.

    Extracts both home and away team info from each match. Uses
    INSERT OR IGNORE so existing teams are not overwritten.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    raw_matches : list[dict[str, Any]]
        Raw match dicts from the Champion Data fixture (before parsing).

    Returns
    -------
    int
        Number of unique teams processed.
    """
    seen: dict[int, dict[str, Any]] = {}
    for raw in raw_matches:
        for side in ("home", "away"):
            squad_id = raw.get(f"{side}SquadId")
            if squad_id is None:
                continue
            squad_id = int(squad_id)
            if squad_id not in seen:
                seen[squad_id] = {
                    "squad_id": squad_id,
                    "squad_name": raw.get(f"{side}SquadName", ""),
                    "squad_nickname": raw.get(f"{side}SquadNickname", ""),
                    "squad_code": raw.get(f"{side}SquadCode", ""),
                }

    if not seen:
        LOGGER.warning("No team data found in fixture")
        return 0

    teams = list(seen.values())

    # INSERT OR IGNORE preserves existing rows; follow up with UPDATE for name
    # fields so we can refresh if Champion Data changes a display name.
    conn.executemany(
        """
        INSERT OR IGNORE INTO teams (squad_id, squad_name, squad_nickname, squad_code)
        VALUES (?, ?, ?, ?)
        """,
        [
            (t["squad_id"], t["squad_name"], t["squad_nickname"], t["squad_code"])
            for t in teams
        ],
    )
    # Update display fields for existing rows (squad_id already present)
    conn.executemany(
        """
        UPDATE teams
        SET squad_name = ?,
            squad_nickname = ?,
            squad_code = ?
        WHERE squad_id = ?
          AND (squad_name != ? OR squad_nickname != ? OR squad_code != ?)
        """,
        [
            (
                t["squad_name"], t["squad_nickname"], t["squad_code"],
                t["squad_id"],
                t["squad_name"], t["squad_nickname"], t["squad_code"],
            )
            for t in teams
        ],
    )
    conn.commit()
    LOGGER.info("Upserted %d teams from fixture", len(teams))
    return len(teams)


def upsert_matches(
    conn: sqlite3.Connection,
    matches: list[dict[str, Any]],
    year: int,
) -> int:
    """Upsert parsed match rows into matches_{year}.

    Uses INSERT OR REPLACE to handle both initial load and re-runs
    (idempotent — re-running with the same fixture produces the same rows).

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    matches : list[dict[str, Any]]
        Parsed match dicts (output of ``_parse_match``).
    year : int
        Season year.

    Returns
    -------
    int
        Number of rows written.
    """
    table = get_table("matches", year)
    rows = [
        (
            m["match_id"],
            m["match_number"],
            m["round_number"],
            m["match_type"],
            m["match_status"],
            m["utc_start_time"],
            m["local_start_time"],
            m["home_squad_id"],
            m["away_squad_id"],
            m["venue_id"],
            m["venue_name"],
            m["venue_code"],
            m["period_completed"],
            m["period_seconds"],
            m["final_code"],
            m["final_short_code"],
        )
        for m in matches
    ]

    conn.executemany(
        f"""
        INSERT OR REPLACE INTO {table} (
            match_id, match_number, round_number, match_type, match_status,
            utc_start_time, local_start_time, home_squad_id, away_squad_id,
            venue_id, venue_name, venue_code,
            period_completed, period_seconds, final_code, final_short_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    LOGGER.info("Upserted %d matches into %s", len(rows), table)
    return len(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ingest_fixture(
    url: str,
    year: int,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Ingest the 2026 (or any season) Champion Data fixture into the DB.

    Steps:
        1. Fetch fixture JSON from Champion Data
        2. Parse each match record
        3. Create matches_{year} table if missing
        4. Upsert teams into teams table
        5. Upsert match rows into matches_{year}

    Parameters
    ----------
    url : str
        Champion Data fixture endpoint URL.
    year : int
        Season year (e.g. 2026).
    conn : sqlite3.Connection, optional
        Existing DB connection. If None, a new connection is opened and
        closed at the end of this call.

    Returns
    -------
    dict[str, Any]
        Summary with keys:

        - ``n_matches`` — total match rows written
        - ``n_rounds`` — number of distinct round numbers
        - ``n_teams`` — number of unique teams upserted
        - ``match_ids_sample`` — first 5 match_ids for spot-checking
    """
    year = normalize_year(year)
    LOGGER.info("Starting fixture ingestion for season %d", year)

    _own_conn = conn is None
    if _own_conn:
        conn = get_connection(DB_PATH)

    try:
        # Step 1: Fetch
        raw_matches = fetch_fixture(url)

        if not raw_matches:
            LOGGER.warning("No matches returned from fixture endpoint")
            return {
                "n_matches": 0,
                "n_rounds": 0,
                "n_teams": 0,
                "match_ids_sample": [],
            }

        # Step 2: Parse
        parsed: list[dict[str, Any]] = []
        for raw in raw_matches:
            try:
                parsed.append(_parse_match(raw))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to parse match record {raw.get('matchId', '?')}: {exc}"
                ) from exc

        # Step 3: Create table
        create_matches_table(conn, year)

        # Step 4: Upsert teams (defensive — they should already exist)
        n_teams = upsert_teams_from_fixture(conn, raw_matches)

        # Step 5: Upsert matches
        n_matches = upsert_matches(conn, parsed, year)

        rounds = sorted({m["round_number"] for m in parsed if m["round_number"] is not None})
        match_ids_sample = [m["match_id"] for m in parsed[:5]]

        summary: dict[str, Any] = {
            "n_matches": n_matches,
            "n_rounds": len(rounds),
            "n_teams": n_teams,
            "match_ids_sample": match_ids_sample,
        }

        LOGGER.info(
            "Fixture ingestion complete: %d matches, %d rounds, %d teams",
            n_matches,
            len(rounds),
            n_teams,
        )
        return summary

    finally:
        if _own_conn and conn is not None:
            conn.close()
