"""Ingest completed match data from the Champion Data API.

Fetches player stats, team stats, score flow, player info, and match reports
for all completed but not-yet-ingested matches in a given season.

Main entry point: fetch_and_ingest_completed_matches(year, conn, delay)

Usage::

    from src.ingestion.ingest_match_data import fetch_and_ingest_completed_matches

    summary = fetch_and_ingest_completed_matches(2026)
    # summary: {n_ingested, n_skipped, n_failed, errors}
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from src.config import (
    CHAMPION_DATA_BASE_URL,
    CHAMPION_DATA_COMP_IDS,
    DB_PATH,
    JERSEY_DEFAULT_SIDE,
    JERSEY_TO_SIDE,
)
from src.db import get_connection, get_table, normalize_year, table_exists

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — schemas match existing 2024/2025 tables exactly
# ---------------------------------------------------------------------------

_PLAYER_STATS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    match_id INT,
    player_id INT,
    squad_id INT,
    position TEXT,
    jumper_number INT,
    tries INT,
    try_assists INT,
    line_breaks INT,
    line_break_assists INT,
    tackle_breaks INT,
    run_metres INT,
    post_contact_metres INT,
    metres_gained INT,
    kicks_general_play INT,
    kick_metres INT,
    tackles INT,
    missed_tackles INT,
    errors INT,
    passes INT,
    possessions INT,
    penalties_conceded INT,
    conversions INT,
    conversion_attempts INT,
    sin_bins INT,
    on_reports INT,
    sent_offs INT,
    bomb_kicks_caught INT,
    runs_kick_return INT,
    runs_hitup INT,
    penalty_goal_attempts INT,
    penalty_goals_unsuccessful INT,
    field_goals_unsuccessful INT,
    kicks_caught INT,
    try_debits INT,
    try_saves INT,
    runs_dummy_half INT,
    runs_dummy_half_metres INT,
    offloads INT,
    goal_line_dropouts INT,
    forty_twenty INT,
    field_goals INT,
    penalty_goals INT,
    field_goal_attempts INT,
    side TEXT,
    opponent_squad_id INT,
    PRIMARY KEY (match_id, player_id)
)
"""

_TEAM_STATS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    match_id INT,
    squad_id INT,
    score INT,
    completion_rate_percentage REAL,
    line_breaks INT,
    possession_percentage REAL,
    run_metres INT,
    tackles INT,
    errors INT,
    missed_tackles INT,
    post_contact_metres INT,
    metres_gained INT,
    try_assists INT,
    line_break_assists INT,
    try_saves INT,
    tries INT,
    tackle_breaks INT,
    passes INT,
    bomb_kicks_caught INT,
    kicks_caught INT,
    kick_metres INT,
    kicks_general_play INT,
    field_goal_attempts INT,
    field_goals INT,
    conversion_attempts INT,
    conversions INT,
    conversions_unsuccessful INT,
    penalty_goal_attempts INT,
    penalty_goals INT,
    penalty_goals_unsuccessful INT,
    penalties_conceded INT,
    goal_line_dropouts INT,
    forty_twenty INT,
    scrum_wins INT,
    offloads INT,
    runs INT,
    runs_normal INT,
    runs_normal_metres INT,
    runs_hitup INT,
    runs_hitup_metres INT,
    runs_dummy_half INT,
    runs_dummy_half_metres INT,
    runs_kick_return INT,
    runs_kick_return_metres INT,
    time_in_own_half INT,
    time_in_opp_half INT,
    time_in_own20 INT,
    time_in_opp20 INT,
    complete_sets INT,
    incomplete_sets INT,
    handling_errors INT,
    set_restarts INT,
    set_restarts_ruck INT,
    set_restarts_10m INT,
    sin_bins INT,
    on_reports INT,
    sent_offs INT,
    possessions INT,
    tackleds INT,
    ineffective_tackles INT,
    tackles_ineffective INT,
    PRIMARY KEY (match_id, squad_id)
)
"""

_SCORE_FLOW_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    match_id INT,
    period INT,
    squad_id INT,
    player_id INT,
    score_name TEXT,
    score_points INT,
    period_seconds INT
)
"""

_PLAYERS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    player_id INTEGER PRIMARY KEY,
    firstname TEXT,
    surname TEXT,
    display_name TEXT,
    short_display_name TEXT
)
"""

_MATCH_REPORTS_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    match_id INTEGER,
    player_id INTEGER,
    squad_id INTEGER,
    period INTEGER,
    period_seconds INTEGER,
    type TEXT
)
"""

_INGESTED_MATCHES_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    match_id INTEGER PRIMARY KEY,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


def _ensure_tables(conn: sqlite3.Connection, year: int) -> None:
    """Create all year-suffixed tables if they do not exist."""
    ddl_map = {
        "player_stats": _PLAYER_STATS_DDL,
        "team_stats": _TEAM_STATS_DDL,
        "score_flow": _SCORE_FLOW_DDL,
        "players": _PLAYERS_DDL,
        "match_reports": _MATCH_REPORTS_DDL,
        "ingested_matches": _INGESTED_MATCHES_DDL,
    }
    for base_name, ddl in ddl_map.items():
        table = get_table(base_name, year)
        conn.execute(ddl.format(table=table))
    conn.commit()
    LOGGER.info("All tables ready for season %d", year)


# ---------------------------------------------------------------------------
# Fetch match JSON
# ---------------------------------------------------------------------------


def _fetch_match_json(comp_id: int, match_id: int) -> dict[str, Any]:
    """Fetch match data JSON from Champion Data API.

    Parameters
    ----------
    comp_id : int
        Champion Data competition ID.
    match_id : int
        Match identifier.

    Returns
    -------
    dict[str, Any]
        Parsed ``matchStats`` dict.

    Raises
    ------
    RuntimeError
        On HTTP or JSON errors.
    """
    url = f"{CHAMPION_DATA_BASE_URL}/{comp_id}/{match_id}.json"
    LOGGER.debug("Fetching %s", url)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} fetching match {match_id}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error fetching match {match_id}: {exc.reason}"
        ) from exc

    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON for match {match_id}: {exc}"
        ) from exc

    try:
        return payload["matchStats"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected JSON structure for match {match_id}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _side_from_jersey(jersey_number: int | None) -> str:
    """Derive the ``side`` column from jersey number."""
    if jersey_number is None:
        return JERSEY_DEFAULT_SIDE
    return JERSEY_TO_SIDE.get(jersey_number, JERSEY_DEFAULT_SIDE)


def _parse_player_stats(
    match_stats: dict[str, Any],
    match_id: int,
    home_squad_id: int,
    away_squad_id: int,
) -> list[tuple]:
    """Parse playerStats into rows matching the player_stats schema."""
    players = match_stats.get("playerStats", {}).get("player", [])
    rows: list[tuple] = []
    for p in players:
        squad_id = p.get("squadId")
        opponent_id = away_squad_id if squad_id == home_squad_id else home_squad_id
        jersey = p.get("jumperNumber")
        rows.append((
            match_id,
            p.get("playerId"),
            squad_id,
            p.get("position"),
            jersey,
            p.get("tries", 0),
            p.get("tryAssists", 0),
            p.get("lineBreaks", 0),
            p.get("lineBreakAssists", 0),
            p.get("tackleBreaks", 0),
            p.get("runMetres", 0),
            p.get("postContactMetres", 0),
            p.get("metresGained", 0),
            p.get("kicksGeneralPlay", 0),
            p.get("kickMetres", 0),
            p.get("tackles", 0),
            p.get("missedTackles", 0),
            p.get("errors", 0),
            p.get("passes", 0),
            p.get("possessions", 0),
            p.get("penaltiesConceded", 0),
            p.get("conversions", 0),
            p.get("conversionAttempts", 0),
            p.get("sinBins", 0),
            p.get("onReports", 0),
            p.get("sentOffs", 0),
            p.get("bombKicksCaught", 0),
            p.get("runsKickReturn", 0),
            p.get("runsHitup", 0),
            p.get("penaltyGoalAttempts", 0),
            p.get("penaltyGoalsUnsuccessful", 0),
            p.get("fieldGoalsUnsuccessful", 0),
            p.get("kicksCaught", 0),
            p.get("tryDebits", 0),
            p.get("trySaves", 0),
            p.get("runsDummyHalf", 0),
            p.get("runsDummyHalfMetres", 0),
            p.get("offloads", 0),
            p.get("goalLineDropouts", 0),
            p.get("fortyTwenty", 0),
            p.get("fieldGoals", 0),
            p.get("penaltyGoals", 0),
            p.get("fieldGoalAttempts", 0),
            _side_from_jersey(jersey),
            opponent_id,
        ))
    return rows


def _parse_team_stats(
    match_stats: dict[str, Any],
    match_id: int,
) -> list[tuple]:
    """Parse teamStats into rows matching the team_stats schema."""
    teams = match_stats.get("teamStats", {}).get("team", [])
    rows: list[tuple] = []
    for t in teams:
        rows.append((
            match_id,
            t.get("squadId"),
            t.get("score", 0),
            t.get("completionRatePercentage", 0),
            t.get("lineBreaks", 0),
            t.get("possessionPercentage", 0),
            t.get("runMetres", 0),
            t.get("tackles", 0),
            t.get("errors", 0),
            t.get("missedTackles", 0),
            t.get("postContactMetres", 0),
            t.get("metresGained", 0),
            t.get("tryAssists", 0),
            t.get("lineBreakAssists", 0),
            t.get("trySaves", 0),
            t.get("tries", 0),
            t.get("tackleBreaks", 0),
            t.get("passes", 0),
            t.get("bombKicksCaught", 0),
            t.get("kicksCaught", 0),
            t.get("kickMetres", 0),
            t.get("kicksGeneralPlay", 0),
            t.get("fieldGoalAttempts", 0),
            t.get("fieldGoals", 0),
            t.get("conversionAttempts", 0),
            t.get("conversions", 0),
            t.get("conversionsUnsuccessful", 0),
            t.get("penaltyGoalAttempts", 0),
            t.get("penaltyGoals", 0),
            t.get("penaltyGoalsUnsuccessful", 0),
            t.get("penaltiesConceded", 0),
            t.get("goalLineDropouts", 0),
            t.get("fortyTwenty", 0),
            t.get("scrumWins", 0),
            t.get("offloads", 0),
            t.get("runs", 0),
            t.get("runsNormal", 0),
            t.get("runsNormalMetres", 0),
            t.get("runsHitup", 0),
            t.get("runsHitupMetres", 0),
            t.get("runsDummyHalf", 0),
            t.get("runsDummyHalfMetres", 0),
            t.get("runsKickReturn", 0),
            t.get("runsKickReturnMetres", 0),
            t.get("timeInOwnHalf", 0),
            t.get("timeInOppHalf", 0),
            t.get("timeInOwn20", 0),
            t.get("timeInOpp20", 0),
            t.get("completeSets", 0),
            t.get("incompleteSets", 0),
            t.get("handlingErrors", 0),
            t.get("setRestarts", 0),
            t.get("setRestartsRuck", 0),
            t.get("setRestarts10m", 0),
            t.get("sinBins", 0),
            t.get("onReports", 0),
            t.get("sentOffs", 0),
            t.get("possessions", 0),
            t.get("tackleds", 0),
            t.get("ineffectiveTackles", 0),
            t.get("tacklesIneffective", 0),
        ))
    return rows


def _parse_score_flow(
    match_stats: dict[str, Any],
    match_id: int,
) -> list[tuple]:
    """Parse scoreFlow into rows matching the score_flow schema.

    Score names are stored lowercase to match existing data convention.
    """
    scores = match_stats.get("scoreFlow", {}).get("score", [])
    rows: list[tuple] = []
    for s in scores:
        score_name = s.get("scoreName", "")
        # Store lowercase to match existing convention in score_flow_2024/2025
        if isinstance(score_name, str):
            score_name = score_name.lower()
        rows.append((
            match_id,
            s.get("period"),
            s.get("squadId"),
            s.get("playerId"),
            score_name,
            s.get("scorepoints", 0),
            s.get("periodSeconds"),
        ))
    return rows


def _parse_players(
    match_stats: dict[str, Any],
) -> list[tuple]:
    """Parse playerInfo into rows for the players table (deduplicated via INSERT OR IGNORE)."""
    players = match_stats.get("playerInfo", {}).get("player", [])
    rows: list[tuple] = []
    for p in players:
        rows.append((
            p.get("playerId"),
            p.get("firstname"),
            p.get("surname"),
            p.get("displayName"),
            p.get("shortDisplayName"),
        ))
    return rows


def _parse_match_reports(
    match_stats: dict[str, Any],
    match_id: int,
) -> list[tuple]:
    """Parse reports (on_report events) into rows for the match_reports table."""
    reports = match_stats.get("reports", {})
    rows: list[tuple] = []
    on_reports = reports.get("onReport", [])
    for r in on_reports:
        rows.append((
            match_id,
            r.get("playerId"),
            r.get("squadId"),
            r.get("period"),
            r.get("periodSeconds"),
            "on_report",
        ))
    return rows


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------


def _insert_player_stats(
    conn: sqlite3.Connection, table: str, rows: list[tuple]
) -> int:
    """Insert player stats rows, replacing on conflict."""
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO {table} (
            match_id, player_id, squad_id, position, jumper_number,
            tries, try_assists, line_breaks, line_break_assists, tackle_breaks,
            run_metres, post_contact_metres, metres_gained, kicks_general_play, kick_metres,
            tackles, missed_tackles, errors, passes, possessions,
            penalties_conceded, conversions, conversion_attempts, sin_bins, on_reports,
            sent_offs, bomb_kicks_caught, runs_kick_return, runs_hitup,
            penalty_goal_attempts, penalty_goals_unsuccessful, field_goals_unsuccessful,
            kicks_caught, try_debits, try_saves, runs_dummy_half, runs_dummy_half_metres,
            offloads, goal_line_dropouts, forty_twenty, field_goals, penalty_goals,
            field_goal_attempts, side, opponent_squad_id
        ) VALUES ({','.join(['?'] * 45)})
        """,
        rows,
    )
    return len(rows)


def _insert_team_stats(
    conn: sqlite3.Connection, table: str, rows: list[tuple]
) -> int:
    """Insert team stats rows, replacing on conflict."""
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO {table} (
            match_id, squad_id, score, completion_rate_percentage, line_breaks,
            possession_percentage, run_metres, tackles, errors, missed_tackles,
            post_contact_metres, metres_gained, try_assists, line_break_assists, try_saves,
            tries, tackle_breaks, passes, bomb_kicks_caught, kicks_caught,
            kick_metres, kicks_general_play, field_goal_attempts, field_goals,
            conversion_attempts, conversions, conversions_unsuccessful,
            penalty_goal_attempts, penalty_goals, penalty_goals_unsuccessful,
            penalties_conceded, goal_line_dropouts, forty_twenty, scrum_wins, offloads,
            runs, runs_normal, runs_normal_metres, runs_hitup, runs_hitup_metres,
            runs_dummy_half, runs_dummy_half_metres, runs_kick_return, runs_kick_return_metres,
            time_in_own_half, time_in_opp_half, time_in_own20, time_in_opp20,
            complete_sets, incomplete_sets, handling_errors,
            set_restarts, set_restarts_ruck, set_restarts_10m,
            sin_bins, on_reports, sent_offs, possessions, tackleds,
            ineffective_tackles, tackles_ineffective
        ) VALUES ({','.join(['?'] * 61)})
        """,
        rows,
    )
    return len(rows)


def _insert_score_flow(
    conn: sqlite3.Connection, table: str, rows: list[tuple]
) -> int:
    """Insert score flow rows."""
    # Delete existing rows for this match first (no PK on score_flow)
    if rows:
        match_id = rows[0][0]
        conn.execute(f"DELETE FROM {table} WHERE match_id = ?", (match_id,))
    conn.executemany(
        f"""
        INSERT INTO {table} (
            match_id, period, squad_id, player_id, score_name, score_points, period_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _insert_players(
    conn: sqlite3.Connection, table: str, rows: list[tuple]
) -> int:
    """Insert player info rows (INSERT OR IGNORE for deduplication)."""
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO {table} (
            player_id, firstname, surname, display_name, short_display_name
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _insert_match_reports(
    conn: sqlite3.Connection, table: str, rows: list[tuple]
) -> int:
    """Insert match report rows."""
    if rows:
        match_id = rows[0][0]
        conn.execute(f"DELETE FROM {table} WHERE match_id = ?", (match_id,))
    conn.executemany(
        f"""
        INSERT INTO {table} (
            match_id, player_id, squad_id, period, period_seconds, type
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Pending match discovery
# ---------------------------------------------------------------------------


def _get_pending_matches(
    conn: sqlite3.Connection, year: int
) -> list[int]:
    """Find completed matches that have not yet been ingested.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    year : int
        Season year.

    Returns
    -------
    list[int]
        match_ids ready for ingestion.
    """
    matches_table = get_table("matches", year)
    ingested_table = get_table("ingested_matches", year)

    if not table_exists(conn, matches_table):
        LOGGER.warning("Table %s not found — no matches to ingest", matches_table)
        return []

    query = f"""
    SELECT m.match_id
    FROM {matches_table} m
    LEFT JOIN {ingested_table} i ON m.match_id = i.match_id
    WHERE m.match_status = 'complete'
      AND i.match_id IS NULL
    ORDER BY m.round_number, m.match_id
    """
    rows = conn.execute(query).fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Update matches table with completed match info
# ---------------------------------------------------------------------------


def _update_match_info(
    conn: sqlite3.Connection,
    year: int,
    match_id: int,
    match_info: dict[str, Any],
) -> None:
    """Update the matches row with period/status info from the match JSON."""
    matches_table = get_table("matches", year)
    conn.execute(
        f"""
        UPDATE {matches_table}
        SET period_completed = ?,
            period_seconds = ?,
            match_status = ?
        WHERE match_id = ?
        """,
        (
            match_info.get("periodCompleted"),
            match_info.get("periodSeconds"),
            match_info.get("matchStatus"),
            match_id,
        ),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def refresh_fixture(year: int, conn: sqlite3.Connection) -> dict[str, Any]:
    """Refresh the fixture from Champion Data to pick up newly completed matches.

    Parameters
    ----------
    year : int
        Season year.
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    dict[str, Any]
        Fixture ingestion summary.
    """
    from src.ingestion.ingest_champion_data import ingest_fixture

    comp_id = CHAMPION_DATA_COMP_IDS.get(year)
    if comp_id is None:
        raise ValueError(
            f"No Champion Data competition ID configured for {year}. "
            f"Known: {list(CHAMPION_DATA_COMP_IDS.keys())}"
        )
    url = f"{CHAMPION_DATA_BASE_URL}/{comp_id}/fixture.json"
    return ingest_fixture(url=url, year=year, conn=conn)


def fetch_and_ingest_completed_matches(
    year: int,
    conn: Optional[sqlite3.Connection] = None,
    delay: float = 2.0,
    refresh: bool = True,
) -> dict[str, Any]:
    """Fetch and ingest all completed but un-ingested matches for a season.

    Parameters
    ----------
    year : int
        Season year (e.g. 2026).
    conn : sqlite3.Connection, optional
        Existing DB connection.  If None, a new connection is opened and
        closed at the end of this call.
    delay : float
        Seconds to sleep between API calls (rate limiting).
    refresh : bool
        If True, refresh the fixture first to discover newly completed matches.

    Returns
    -------
    dict[str, Any]
        Summary with keys: n_pending, n_ingested, n_skipped, n_failed, errors.
    """
    year = normalize_year(year)
    LOGGER.info("Starting match data ingestion for season %d", year)

    _own_conn = conn is None
    if _own_conn:
        conn = get_connection(DB_PATH)

    try:
        comp_id = CHAMPION_DATA_COMP_IDS.get(year)
        if comp_id is None:
            raise ValueError(
                f"No Champion Data competition ID configured for {year}. "
                f"Known: {list(CHAMPION_DATA_COMP_IDS.keys())}"
            )

        # Ensure all target tables exist
        _ensure_tables(conn, year)

        # Step 1: Refresh fixture to update match statuses
        if refresh:
            LOGGER.info("Step 1: Refreshing fixture for %d", year)
            try:
                fixture_summary = refresh_fixture(year, conn)
                LOGGER.info(
                    "Fixture refreshed: %d matches", fixture_summary.get("n_matches", 0)
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Fixture refresh failed (non-fatal): %s", exc)

        # Step 2: Find pending matches
        pending = _get_pending_matches(conn, year)
        LOGGER.info("Step 2: Found %d pending matches", len(pending))

        if not pending:
            return {
                "n_pending": 0,
                "n_ingested": 0,
                "n_skipped": 0,
                "n_failed": 0,
                "errors": [],
            }

        # Table names
        ps_table = get_table("player_stats", year)
        ts_table = get_table("team_stats", year)
        sf_table = get_table("score_flow", year)
        pl_table = get_table("players", year)
        mr_table = get_table("match_reports", year)
        ig_table = get_table("ingested_matches", year)

        n_ingested = 0
        n_failed = 0
        errors: list[str] = []

        for i, match_id in enumerate(pending):
            LOGGER.info(
                "Ingesting match %d (%d/%d)", match_id, i + 1, len(pending)
            )
            try:
                # Fetch
                match_stats = _fetch_match_json(comp_id, match_id)
                match_info = match_stats.get("matchInfo", {})
                home_squad_id = match_info.get("homeSquadId")
                away_squad_id = match_info.get("awaySquadId")

                if home_squad_id is None or away_squad_id is None:
                    raise RuntimeError(
                        f"Missing homeSquadId/awaySquadId in match {match_id}"
                    )

                # Parse
                ps_rows = _parse_player_stats(
                    match_stats, match_id, home_squad_id, away_squad_id
                )
                ts_rows = _parse_team_stats(match_stats, match_id)
                sf_rows = _parse_score_flow(match_stats, match_id)
                pl_rows = _parse_players(match_stats)
                mr_rows = _parse_match_reports(match_stats, match_id)

                # Insert
                _insert_players(conn, pl_table, pl_rows)
                _insert_player_stats(conn, ps_table, ps_rows)
                _insert_team_stats(conn, ts_table, ts_rows)
                _insert_score_flow(conn, sf_table, sf_rows)
                _insert_match_reports(conn, mr_table, mr_rows)

                # Update matches table with period info
                _update_match_info(conn, year, match_id, match_info)

                # Mark as ingested
                conn.execute(
                    f"INSERT OR IGNORE INTO {ig_table} (match_id) VALUES (?)",
                    (match_id,),
                )
                conn.commit()

                n_ingested += 1
                LOGGER.info(
                    "  Match %d: %d players, %d team rows, %d score events",
                    match_id, len(ps_rows), len(ts_rows), len(sf_rows),
                )

            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to ingest match %d: %s", match_id, exc)
                errors.append(f"match {match_id}: {exc}")
                n_failed += 1
                # Rollback any partial writes for this match
                conn.rollback()

            # Rate limiting
            if i < len(pending) - 1:
                time.sleep(delay)

        summary = {
            "n_pending": len(pending),
            "n_ingested": n_ingested,
            "n_skipped": 0,
            "n_failed": n_failed,
            "errors": errors,
        }
        LOGGER.info(
            "Match ingestion complete: %d ingested, %d failed out of %d pending",
            n_ingested, n_failed, len(pending),
        )
        return summary

    finally:
        if _own_conn and conn is not None:
            conn.close()
