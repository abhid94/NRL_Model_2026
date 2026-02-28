"""Bookmaker odds storage, player matching, and retrieval.

Stores odds fetched from The Odds API into year-suffixed
``bookmaker_odds_{year}`` tables. Reuses the proven 4-tier player
matcher from ``src.ingestion.player_matcher``.

Usage:
    from src.odds.bookmaker import ingest_round_odds
    summary = ingest_round_odds(round_number=1, season=2026, conn=conn)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import BOOKMAKER_DISPLAY_NAMES, DB_PATH
from src.db import get_connection
from src.ingestion.player_matcher import get_known_players, match_player_name
from src.odds.odds_api import fetch_round_odds

LOGGER = logging.getLogger(__name__)


def create_bookmaker_odds_table(conn: sqlite3.Connection, year: int) -> None:
    """Create ``bookmaker_odds_{year}`` table if it doesn't exist.

    Schema mirrors ``bookmaker_odds_2025`` with extra columns for
    Odds API provenance and raw player name.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    year : int
        Season year.
    """
    table = f"bookmaker_odds_{year}"
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            odds_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        INTEGER NOT NULL,
            player_id       INTEGER,
            bookmaker       TEXT NOT NULL,
            market_type     TEXT NOT NULL DEFAULT 'anytime_tryscorer',
            decimal_odds    REAL CHECK(decimal_odds >= 1.01 AND decimal_odds <= 1000.0),
            implied_probability REAL,
            odds_timestamp  TEXT,
            snapshot_type   TEXT DEFAULT 'closing',
            is_available    BOOLEAN DEFAULT 1,
            is_verified     BOOLEAN DEFAULT 0,
            odds_api_event_id TEXT,
            player_name_raw TEXT,
            UNIQUE(match_id, player_id, bookmaker, snapshot_type)
        )
    """)
    conn.commit()
    LOGGER.info("Ensured table %s exists", table)


def build_odds_player_mapping(
    raw_records: list[dict[str, Any]],
    conn: sqlite3.Connection,
    year: int,
    round_number: int,
) -> list[dict[str, Any]]:
    """Match Odds API player names to internal player_ids.

    Uses ``player_matcher.match_player_name()`` constrained to
    ``team_lists_{year}`` rosters for better disambiguation.

    Parameters
    ----------
    raw_records : list[dict]
        Records from ``odds_api.fetch_round_odds()``.
    conn : sqlite3.Connection
        Database connection.
    year : int
        Season year.
    round_number : int
        Round number (used to constrain to team lists).

    Returns
    -------
    list[dict]
        Records enriched with ``player_id``, ``match_type``,
        ``confidence``.
    """
    known_players = get_known_players(conn)

    # Try to load team lists for the round to constrain squad membership
    squad_for_match: dict[tuple[int, str], int] = {}
    try:
        tl_df = pd.read_sql_query(
            f"""
            SELECT match_id, squad_id, player_name
            FROM team_lists_{year}
            WHERE round_number = ?
            """,
            conn,
            params=(round_number,),
        )
        # Build match_id -> {home_squad_id, away_squad_id} lookup
        for _, row in tl_df.iterrows():
            squad_for_match[(int(row["match_id"]), "any")] = int(row["squad_id"])
    except Exception:  # noqa: BLE001
        LOGGER.info("No team_lists_%d table — using all known players", year)

    # Build match_id -> (home_squad_id, away_squad_id) from matches table
    match_squads: dict[int, tuple[int, int]] = {}
    try:
        matches = pd.read_sql_query(
            f"SELECT match_id, home_squad_id, away_squad_id FROM matches_{year}",
            conn,
        )
        for _, row in matches.iterrows():
            match_squads[int(row["match_id"])] = (
                int(row["home_squad_id"]),
                int(row["away_squad_id"]),
            )
    except Exception:  # noqa: BLE001
        pass

    cache: dict[tuple[str, int], tuple[int | None, str, float]] = {}
    enriched: list[dict[str, Any]] = []

    for record in raw_records:
        name = record.get("player_name_raw", "").strip()
        match_id = record.get("match_id")

        if not name or match_id is None:
            enriched.append({**record, "player_id": None, "match_type": "no_match", "confidence": 0.0})
            continue

        # Determine squad_ids for this match to guide matching
        squads = match_squads.get(match_id, ())
        best_result: tuple[int | None, str, float] = (None, "no_match", 0.0)

        for squad_id in squads:
            pid, mtype, conf = match_player_name(name, squad_id, known_players, cache)
            if pid is not None and conf > best_result[2]:
                best_result = (pid, mtype, conf)

        # Fall back to matching across all squads
        if best_result[0] is None:
            pid, mtype, conf = match_player_name(name, 0, known_players, cache)
            if pid is not None:
                best_result = (pid, mtype, conf)

        enriched.append({
            **record,
            "player_id": best_result[0],
            "match_type": best_result[1],
            "confidence": best_result[2],
        })

    n_matched = sum(1 for r in enriched if r.get("player_id") is not None)
    n_total = len(enriched)
    LOGGER.info(
        "Odds player matching: %d/%d matched (%.1f%%)",
        n_matched, n_total, 100 * n_matched / n_total if n_total else 0.0,
    )
    return enriched


def upsert_bookmaker_odds(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
    year: int,
    snapshot_type: str = "closing",
) -> int:
    """Insert or update bookmaker odds records.

    UNIQUE constraint on ``(match_id, player_id, bookmaker, snapshot_type)``
    ensures idempotent upserts.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    records : list[dict]
        Enriched records from ``build_odds_player_mapping()``.
    year : int
        Season year.
    snapshot_type : str
        One of "opening", "closing".

    Returns
    -------
    int
        Number of records upserted.
    """
    table = f"bookmaker_odds_{year}"
    timestamp = datetime.now(timezone.utc).isoformat()

    n_upserted = 0
    for rec in records:
        pid = rec.get("player_id")
        if pid is None:
            continue  # Skip unmatched players

        try:
            conn.execute(
                f"""
                INSERT INTO {table}
                    (match_id, player_id, bookmaker, market_type, decimal_odds,
                     implied_probability, odds_timestamp, snapshot_type,
                     is_available, is_verified, odds_api_event_id, player_name_raw)
                VALUES (?, ?, ?, 'anytime_tryscorer', ?, ?, ?, ?, 1, 0, ?, ?)
                ON CONFLICT(match_id, player_id, bookmaker, snapshot_type)
                DO UPDATE SET
                    decimal_odds = excluded.decimal_odds,
                    implied_probability = excluded.implied_probability,
                    odds_timestamp = excluded.odds_timestamp,
                    odds_api_event_id = excluded.odds_api_event_id,
                    player_name_raw = excluded.player_name_raw
                """,
                (
                    rec["match_id"],
                    pid,
                    rec["bookmaker"],
                    rec["decimal_odds"],
                    rec["implied_probability"],
                    timestamp,
                    snapshot_type,
                    rec.get("odds_api_event_id", ""),
                    rec.get("player_name_raw", ""),
                ),
            )
            n_upserted += 1
        except sqlite3.Error as exc:
            LOGGER.warning(
                "Failed to upsert odds for player_id=%s, bookmaker=%s: %s",
                pid, rec.get("bookmaker"), exc,
            )

    conn.commit()
    LOGGER.info("Upserted %d odds records into %s", n_upserted, table)
    return n_upserted


def ingest_round_odds(
    round_number: int,
    season: int,
    snapshot_type: str = "closing",
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Top-level: fetch odds from API, match players, store in DB.

    Parameters
    ----------
    round_number : int
        Round to fetch odds for.
    season : int
        Season year.
    snapshot_type : str
        "opening" or "closing".
    conn : sqlite3.Connection, optional
        Database connection. If None, opens a new one.

    Returns
    -------
    dict[str, Any]
        Summary with keys: n_raw, n_matched, n_upserted, n_unmatched,
        bookmakers, unmatched_players.
    """
    close_conn = False
    if conn is None:
        conn = get_connection(DB_PATH)
        close_conn = True

    try:
        # Step 1: Fetch raw odds from API
        raw_records = fetch_round_odds(round_number, season, conn)
        if not raw_records:
            return {
                "n_raw": 0, "n_matched": 0, "n_upserted": 0,
                "n_unmatched": 0, "bookmakers": [], "unmatched_players": [],
            }

        # Step 2: Match player names to player_ids
        enriched = build_odds_player_mapping(raw_records, conn, season, round_number)

        # Step 3: Create table and upsert
        create_bookmaker_odds_table(conn, season)
        n_upserted = upsert_bookmaker_odds(conn, enriched, season, snapshot_type)

        # Summary
        n_matched = sum(1 for r in enriched if r.get("player_id") is not None)
        n_unmatched = len(enriched) - n_matched
        bookmakers = sorted(set(r.get("bookmaker", "") for r in enriched))
        unmatched = sorted(set(
            r.get("player_name_raw", "")
            for r in enriched
            if r.get("player_id") is None
        ))

        summary = {
            "n_raw": len(raw_records),
            "n_matched": n_matched,
            "n_upserted": n_upserted,
            "n_unmatched": n_unmatched,
            "bookmakers": bookmakers,
            "unmatched_players": unmatched,
        }
        LOGGER.info(
            "Odds ingestion complete: %d raw, %d matched, %d upserted, %d unmatched",
            summary["n_raw"], summary["n_matched"],
            summary["n_upserted"], summary["n_unmatched"],
        )
        return summary

    finally:
        if close_conn:
            conn.close()


def get_best_bookmaker_odds(
    conn: sqlite3.Connection,
    match_id: int,
    player_id: int,
    year: int,
    snapshot_type: str = "closing",
) -> pd.DataFrame:
    """Get all bookmaker prices for a player-match, sorted by best price.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    match_id : int
        Match identifier.
    player_id : int
        Player identifier.
    year : int
        Season year.
    snapshot_type : str
        Snapshot type filter.

    Returns
    -------
    pd.DataFrame
        Columns: bookmaker, decimal_odds, implied_probability.
        Sorted by decimal_odds descending (best price first).
    """
    table = f"bookmaker_odds_{year}"
    try:
        df = pd.read_sql_query(
            f"""
            SELECT bookmaker, decimal_odds, implied_probability
            FROM {table}
            WHERE match_id = ? AND player_id = ? AND snapshot_type = ?
            ORDER BY decimal_odds DESC
            """,
            conn,
            params=(match_id, player_id, snapshot_type),
        )
    except Exception:  # noqa: BLE001
        df = pd.DataFrame(columns=["bookmaker", "decimal_odds", "implied_probability"])
    return df


def get_round_bookmaker_odds(
    conn: sqlite3.Connection,
    round_number: int,
    year: int,
    snapshot_type: str = "closing",
) -> pd.DataFrame:
    """Get all bookmaker odds for a round with best price per player.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    round_number : int
        Round number.
    year : int
        Season year.
    snapshot_type : str
        Snapshot type filter.

    Returns
    -------
    pd.DataFrame
        One row per (match_id, player_id) with columns:
        match_id, player_id, best_odds, best_bookmaker,
        best_implied_prob, plus per-bookmaker odds columns.
    """
    table = f"bookmaker_odds_{year}"
    try:
        df = pd.read_sql_query(
            f"""
            SELECT bo.match_id, bo.player_id, bo.bookmaker,
                   bo.decimal_odds, bo.implied_probability
            FROM {table} bo
            JOIN matches_{year} m ON bo.match_id = m.match_id
            WHERE m.round_number = ? AND bo.snapshot_type = ?
            """,
            conn,
            params=(round_number, snapshot_type),
        )
    except Exception:  # noqa: BLE001
        LOGGER.warning("Could not query %s for round %d", table, round_number)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Find best odds per (match_id, player_id) — highest decimal_odds
    best = (
        df.sort_values("decimal_odds", ascending=False)
        .drop_duplicates(subset=["match_id", "player_id"], keep="first")
        [["match_id", "player_id", "decimal_odds", "bookmaker", "implied_probability"]]
        .rename(columns={
            "decimal_odds": "best_odds",
            "bookmaker": "best_bookmaker",
            "implied_probability": "best_implied_prob",
        })
    )

    # Pivot per-bookmaker odds
    pivot = df.pivot_table(
        index=["match_id", "player_id"],
        columns="bookmaker",
        values="decimal_odds",
        aggfunc="first",
    ).reset_index()
    # Rename bookmaker columns to odds_<bookmaker>
    pivot.columns = [
        f"odds_{c}" if c not in ("match_id", "player_id") else c
        for c in pivot.columns
    ]

    result = best.merge(pivot, on=["match_id", "player_id"], how="left")

    # Map bookmaker keys to display names
    result["best_bookmaker_display"] = result["best_bookmaker"].map(
        BOOKMAKER_DISPLAY_NAMES
    ).fillna(result["best_bookmaker"])

    return result
