"""Ingest scraped team lists into the database.

Main entry point: ingest_round_team_lists(round_number, year)

Pipeline steps:
    1. Scrape team lists from LeagueUnlimited (primary) or nrl.com (fallback)
    2. Map team names → squad_ids via teams table
    3. Match player names → player_ids via player_matcher (4-tier matching)
    4. Resolve match_ids from matches_{year} by round_number + squad_id
    5. Upsert into team_lists_{year}
    6. Upsert player mappings into nrl_player_mapping
    7. Log unmatched players as warnings

Usage:
    from src.ingestion.ingest_team_lists import ingest_round_team_lists
    summary = ingest_round_team_lists(round_number=1, year=2026)
    # summary: {n_scraped, n_matched, n_unmatched, n_inserted}
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

import pandas as pd

from src.config import DB_PATH
from src.db import get_connection, get_table, normalize_year, table_exists
from src.ingestion.player_matcher import build_player_mapping, save_player_mapping
from src.ingestion.team_list_scraper import fetch_team_lists

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# DDL helpers
# ------------------------------------------------------------------

_TEAM_LISTS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    match_id INTEGER,
    round_number INTEGER NOT NULL,
    squad_id INTEGER NOT NULL,
    squad_name TEXT,
    player_name TEXT NOT NULL,
    player_id INTEGER,
    jersey_number INTEGER NOT NULL,
    position TEXT,
    scraped_position TEXT,
    PRIMARY KEY (round_number, squad_id, jersey_number)
)
"""

_NRL_PLAYER_MAPPING_DDL = """
CREATE TABLE IF NOT EXISTS nrl_player_mapping (
    mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_name TEXT NOT NULL,
    squad_id INTEGER,
    official_player_id INTEGER,
    match_type TEXT,
    confidence_score REAL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scraped_name, squad_id),
    FOREIGN KEY (official_player_id) REFERENCES players_2025(player_id)
)
"""

_NRL_PLAYER_MAPPING_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_nrl_player_mapping_player_id ON nrl_player_mapping(official_player_id)",
    "CREATE INDEX IF NOT EXISTS idx_nrl_player_mapping_name ON nrl_player_mapping(scraped_name)",
    "CREATE INDEX IF NOT EXISTS idx_nrl_player_mapping_squad ON nrl_player_mapping(squad_id)",
]


def create_team_lists_table(conn: sqlite3.Connection, year: int) -> None:
    """Create team_lists_{year} table if it does not exist.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    year : int
        Season year.
    """
    year = normalize_year(year)
    table = get_table("team_lists", year)
    conn.execute(_TEAM_LISTS_DDL.format(table=table))
    conn.commit()
    LOGGER.info("Table ready: %s", table)


def create_nrl_player_mapping_table(conn: sqlite3.Connection) -> None:
    """Create nrl_player_mapping table and indexes if they do not exist.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    """
    conn.execute(_NRL_PLAYER_MAPPING_DDL)
    for idx_sql in _NRL_PLAYER_MAPPING_INDEXES:
        conn.execute(idx_sql)
    conn.commit()
    LOGGER.info("Table ready: nrl_player_mapping")


# ------------------------------------------------------------------
# Team name → squad_id resolution
# ------------------------------------------------------------------

def _build_team_lookup(conn: sqlite3.Connection) -> dict[str, int]:
    """Build a mapping from team name variants to squad_id.

    Includes squad_name, squad_nickname, and squad_code for flexible matching.
    Also applies TEAM_NAME_OVERRIDES from config.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    dict[str, int]
        Lower-case team name → squad_id.
    """
    df = pd.read_sql_query(
        "SELECT squad_id, squad_name, squad_nickname, squad_code FROM teams", conn
    )
    lookup: dict[str, int] = {}
    for _, row in df.iterrows():
        sid = int(row["squad_id"])
        for col in ("squad_name", "squad_nickname", "squad_code"):
            val = str(row[col]).strip()
            if val:
                lookup[val.lower()] = sid

    # Apply config overrides (handles abbreviated scraped names)
    try:
        from src.config import TEAM_NICKNAME_OVERRIDES  # type: ignore[attr-defined]
        for alias, canonical in TEAM_NICKNAME_OVERRIDES.items():
            canonical_lower = canonical.lower()
            if canonical_lower in lookup:
                lookup[alias.lower()] = lookup[canonical_lower]
    except ImportError:
        pass

    return lookup


def _resolve_squad_id(
    team_name: str,
    lookup: dict[str, int],
) -> Optional[int]:
    """Resolve a scraped team name to a squad_id.

    Parameters
    ----------
    team_name : str
        Team name as scraped from the source.
    lookup : dict[str, int]
        Lower-case team name → squad_id mapping.

    Returns
    -------
    int | None
        squad_id or None if not found.
    """
    key = team_name.strip().lower()
    if key in lookup:
        return lookup[key]
    # Partial match: check if key is contained in a lookup key (or vice-versa)
    for name, sid in lookup.items():
        if key in name or name in key:
            return sid
    return None


# ------------------------------------------------------------------
# match_id resolution
# ------------------------------------------------------------------

def resolve_match_id(
    conn: sqlite3.Connection,
    round_number: int,
    squad_id: int,
    year: int,
) -> Optional[int]:
    """Look up match_id for a given round + squad in matches_{year}.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    round_number : int
        Round number.
    squad_id : int
        Team squad_id.
    year : int
        Season year.

    Returns
    -------
    int | None
        match_id or None if matches_{year} is not yet populated.
    """
    table = get_table("matches", year)
    if not table_exists(conn, table):
        LOGGER.warning("Table %s not found — match_id will be NULL", table)
        return None

    cursor = conn.execute(
        f"""
        SELECT match_id FROM {table}
        WHERE round_number = ?
          AND (home_squad_id = ? OR away_squad_id = ?)
        LIMIT 1
        """,
        (round_number, squad_id, squad_id),
    )
    row = cursor.fetchone()
    if row is None:
        LOGGER.warning(
            "No match_id found for round=%d squad_id=%d in %s", round_number, squad_id, table
        )
        return None
    return int(row[0])


# ------------------------------------------------------------------
# Upsert
# ------------------------------------------------------------------

def _upsert_team_lists(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
    year: int,
) -> int:
    """Upsert team list records into team_lists_{year}.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    records : list[dict]
        Enriched records with all required fields.
    year : int
        Season year.

    Returns
    -------
    int
        Number of rows upserted.
    """
    table = get_table("team_lists", year)
    rows = [
        (
            r.get("match_id"),
            r["round_number"],
            r["squad_id"],
            r.get("squad_name"),
            r["player_name"],
            r.get("player_id"),
            r["jersey_number"],
            r.get("position"),
            r.get("scraped_position"),
        )
        for r in records
    ]

    conn.executemany(
        f"""
        INSERT INTO {table}
            (match_id, round_number, squad_id, squad_name, player_name,
             player_id, jersey_number, position, scraped_position)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(round_number, squad_id, jersey_number) DO UPDATE SET
            match_id = excluded.match_id,
            player_name = excluded.player_name,
            player_id = excluded.player_id,
            squad_name = excluded.squad_name,
            position = excluded.position,
            scraped_position = excluded.scraped_position
        """,
        rows,
    )
    conn.commit()
    LOGGER.info("Upserted %d rows into %s", len(rows), table)
    return len(rows)


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def print_unmatched_report(
    conn: sqlite3.Connection,
    year: int,
    round_number: int,
) -> None:
    """Print players from team_lists_{year} with no player_id match.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    year : int
        Season year.
    round_number : int
        Round to report on.
    """
    table = get_table("team_lists", year)
    if not table_exists(conn, table):
        LOGGER.warning("Table %s not found", table)
        return

    df = pd.read_sql_query(
        f"""
        SELECT squad_name, jersey_number, player_name
        FROM {table}
        WHERE round_number = ?
          AND player_id IS NULL
        ORDER BY squad_name, jersey_number
        """,
        conn,
        params=(round_number,),
    )

    if df.empty:
        print(f"All players matched for {year} Round {round_number}.")
        return

    print(f"\nUnmatched players — {year} Round {round_number} ({len(df)} total):")
    print(df.to_string(index=False))
    print(
        "\nTo fix: add rows to nrl_player_mapping or players_{year} table, "
        "then re-run ingest_round_team_lists()."
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def ingest_round_team_lists(
    round_number: int,
    year: int,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Ingest team lists for a given round into the database.

    Steps:
        1. Scrape team lists from LeagueUnlimited (or nrl.com fallback)
        2. Map team names → squad_ids
        3. Match player names → player_ids (4-tier matching)
        4. Resolve match_ids from matches_{year}
        5. Upsert into team_lists_{year}
        6. Upsert player mappings into nrl_player_mapping
        7. Log unmatched players

    Parameters
    ----------
    round_number : int
        NRL round number (1-based).
    year : int
        Season year (e.g. 2026).
    conn : sqlite3.Connection, optional
        Existing DB connection. If None, a new connection is opened and closed
        at the end of this call.

    Returns
    -------
    dict[str, Any]
        Summary with keys: n_scraped, n_matched, n_unmatched, n_inserted,
        n_teams, unmatched_players (list of names).
    """
    year = normalize_year(year)
    LOGGER.info("Ingesting team lists: %d Round %d", year, round_number)

    _own_conn = conn is None
    if _own_conn:
        conn = get_connection(DB_PATH)

    try:
        # Ensure tables exist
        create_team_lists_table(conn, year)
        create_nrl_player_mapping_table(conn)

        # Step 1: Scrape
        raw_records = fetch_team_lists(round_number, year)
        n_scraped = len(raw_records)
        LOGGER.info("Scraped %d player records", n_scraped)

        if not raw_records:
            return {
                "n_scraped": 0, "n_matched": 0, "n_unmatched": 0,
                "n_inserted": 0, "n_teams": 0, "unmatched_players": [],
            }

        # Step 2: Map team names → squad_ids
        team_lookup = _build_team_lookup(conn)
        for record in raw_records:
            squad_id = _resolve_squad_id(record["team_name"], team_lookup)
            if squad_id is None:
                LOGGER.warning("Cannot resolve team '%s' to squad_id", record["team_name"])
            record["squad_id"] = squad_id
            # Find canonical squad_name from teams table
            for name, sid in team_lookup.items():
                if sid == squad_id:
                    record["squad_name"] = name.title()
                    break

        # Fetch canonical squad names from DB for cleaner display
        squad_name_df = pd.read_sql_query("SELECT squad_id, squad_name FROM teams", conn)
        squad_name_map = dict(zip(squad_name_df["squad_id"], squad_name_df["squad_name"]))
        for record in raw_records:
            if record.get("squad_id") is not None:
                record["squad_name"] = squad_name_map.get(record["squad_id"], record.get("squad_name"))

        # Drop records with no squad_id (unknown teams)
        unknown = [r for r in raw_records if r.get("squad_id") is None]
        if unknown:
            LOGGER.warning(
                "Dropping %d records with unresolved team names: %s",
                len(unknown),
                list({r["team_name"] for r in unknown}),
            )
        valid_records = [r for r in raw_records if r.get("squad_id") is not None]

        # Step 3: Match player names → player_ids
        enriched_records = build_player_mapping(valid_records, conn)

        # Step 4: Resolve match_ids
        match_id_cache: dict[tuple[int, int], Optional[int]] = {}
        for record in enriched_records:
            squad_id = record.get("squad_id")
            if squad_id is None:
                record["match_id"] = None
                continue
            cache_key = (int(squad_id), round_number)
            if cache_key not in match_id_cache:
                match_id_cache[cache_key] = resolve_match_id(
                    conn, round_number, int(squad_id), year
                )
            record["match_id"] = match_id_cache[cache_key]

        # Add round_number to each record (in case not already set)
        for record in enriched_records:
            record["round_number"] = round_number
            # scraped_position = position from source (nrl.com has position; LU does not)
            record["scraped_position"] = record.get("position")

        # Step 5: Upsert into team_lists_{year}
        n_inserted = _upsert_team_lists(conn, enriched_records, year)

        # Step 6: Upsert player mappings
        save_player_mapping(enriched_records, conn)

        # Step 7: Summarise unmatched
        unmatched = [r for r in enriched_records if r.get("player_id") is None]
        unmatched_names = [r["player_name"] for r in unmatched]
        if unmatched_names:
            LOGGER.warning(
                "Unmatched players (%d): %s", len(unmatched_names), unmatched_names
            )

        n_teams = len({r["squad_id"] for r in enriched_records if r.get("squad_id") is not None})
        n_matched = n_scraped - len(unknown) - len(unmatched)
        summary = {
            "n_scraped": n_scraped,
            "n_matched": n_matched,
            "n_unmatched": len(unmatched),
            "n_inserted": n_inserted,
            "n_teams": n_teams,
            "unmatched_players": unmatched_names,
        }
        LOGGER.info(
            "Ingestion complete: %d scraped, %d matched, %d unmatched, %d inserted",
            n_scraped, n_matched, len(unmatched), n_inserted,
        )
        return summary

    finally:
        if _own_conn and conn is not None:
            conn.close()
