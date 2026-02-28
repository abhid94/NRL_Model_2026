"""The Odds API client for fetching NRL player try scorer odds.

Fetches anytime try scorer odds from Australian bookmakers via
https://the-odds-api.com. Designed for the weekly pipeline to pull
odds for upcoming rounds.

Usage:
    from src.odds.odds_api import fetch_round_odds
    records = fetch_round_odds(round_number=1, season=2026, conn=conn)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

import pandas as pd
import requests

from src.config import (
    ODDS_API_BOOKMAKERS,
    ODDS_API_MARKET,
    ODDS_API_REGIONS,
    ODDS_API_SPORT_KEY,
    TEAM_NICKNAME_OVERRIDES,
)

LOGGER = logging.getLogger(__name__)

_BASE_URL = "https://api.the-odds-api.com"
_REQUEST_TIMEOUT = 10


class OddsAPIError(RuntimeError):
    """Raised when The Odds API returns a non-200 response."""


def get_api_key() -> str:
    """Read API key from ``.env`` file or ``ODDS_API_KEY`` env var.

    Loads ``.env`` from the project root (if present) via python-dotenv,
    then reads ``ODDS_API_KEY``.

    Returns
    -------
    str
        API key.

    Raises
    ------
    OddsAPIError
        If the key is not found.
    """
    try:
        from dotenv import load_dotenv
        from src.config import PROJECT_ROOT
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        raise OddsAPIError(
            "ODDS_API_KEY not found. Add it to .env or set the environment variable. "
            "Get a key at https://the-odds-api.com"
        )
    return key


def _log_remaining_credits(response: requests.Response) -> None:
    """Log remaining API credits from response headers."""
    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    if remaining is not None:
        LOGGER.info(
            "Odds API credits — remaining: %s, used: %s", remaining, used
        )


def fetch_events(api_key: str) -> list[dict[str, Any]]:
    """Fetch upcoming NRL events (free, 0 credits).

    Parameters
    ----------
    api_key : str
        The Odds API key.

    Returns
    -------
    list[dict]
        List of event dicts with keys: id, sport_key, commence_time,
        home_team, away_team.

    Raises
    ------
    OddsAPIError
        On non-200 response.
    """
    url = f"{_BASE_URL}/v4/sports/{ODDS_API_SPORT_KEY}/events/"
    params = {"apiKey": api_key}

    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    _log_remaining_credits(resp)

    if resp.status_code != 200:
        raise OddsAPIError(
            f"Failed to fetch events: HTTP {resp.status_code} — {resp.text}"
        )

    events = resp.json()
    LOGGER.info("Fetched %d NRL events from Odds API", len(events))
    return events


def fetch_player_try_odds(
    api_key: str,
    event_id: str,
) -> dict[str, Any]:
    """Fetch anytime try scorer odds for a single event (1 credit).

    Parameters
    ----------
    api_key : str
        The Odds API key.
    event_id : str
        Odds API event ID.

    Returns
    -------
    dict
        Full API response for the event including bookmakers and markets.

    Raises
    ------
    OddsAPIError
        On non-200 response.
    """
    url = (
        f"{_BASE_URL}/v4/sports/{ODDS_API_SPORT_KEY}"
        f"/events/{event_id}/odds/"
    )
    params = {
        "apiKey": api_key,
        "markets": ODDS_API_MARKET,
        "regions": ODDS_API_REGIONS,
        "oddsFormat": "decimal",
    }

    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    _log_remaining_credits(resp)

    if resp.status_code != 200:
        raise OddsAPIError(
            f"Failed to fetch odds for event {event_id}: "
            f"HTTP {resp.status_code} — {resp.text}"
        )

    return resp.json()


def parse_player_odds_response(
    response: dict[str, Any],
    match_id: int | None,
) -> list[dict[str, Any]]:
    """Flatten an event odds response to per-player records.

    Extracts only the "Yes" outcome from the ATS market for each
    bookmaker.

    Parameters
    ----------
    response : dict
        Raw API response from fetch_player_try_odds().
    match_id : int or None
        Internal match_id (from match_events_to_matches).

    Returns
    -------
    list[dict]
        Flat records with keys: match_id, odds_api_event_id,
        home_team, away_team, bookmaker, player_name_raw,
        decimal_odds, implied_probability.
    """
    records: list[dict[str, Any]] = []
    event_id = response.get("id", "")
    home_team = response.get("home_team", "")
    away_team = response.get("away_team", "")

    for bookmaker in response.get("bookmakers", []):
        bk_key = bookmaker.get("key", "")
        # Only process bookmakers we care about
        if bk_key not in ODDS_API_BOOKMAKERS:
            continue

        for market in bookmaker.get("markets", []):
            if market.get("key") != ODDS_API_MARKET:
                continue

            for outcome in market.get("outcomes", []):
                # ATS market structure: name="Yes"/"No", description=player name.
                # We only want the "Yes" outcome (will score a try).
                outcome_name = outcome.get("name", "")
                if outcome_name.lower() != "yes":
                    continue

                price = outcome.get("price")
                if price is None or price <= 1.0:
                    continue

                player_name = outcome.get("description", "").strip()
                if not player_name:
                    continue

                records.append({
                    "match_id": match_id,
                    "odds_api_event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bk_key,
                    "player_name_raw": player_name,
                    "decimal_odds": round(float(price), 2),
                    "implied_probability": round(1.0 / float(price), 4),
                })

    return records


def _build_team_lookup(conn: sqlite3.Connection) -> dict[str, int]:
    """Build a mapping from team name variants to squad_id.

    Combines official squad_name / squad_nickname from the ``teams``
    table with TEAM_NICKNAME_OVERRIDES from config.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    dict[str, int]
        Lowercase team name -> squad_id.
    """
    teams = pd.read_sql_query(
        "SELECT squad_id, squad_name, squad_nickname FROM teams", conn
    )
    lookup: dict[str, int] = {}
    for _, row in teams.iterrows():
        sid = int(row["squad_id"])
        lookup[row["squad_name"].lower()] = sid
        lookup[row["squad_nickname"].lower()] = sid

    # Add overrides (value is canonical squad_name)
    name_to_sid = {row["squad_name"].lower(): int(row["squad_id"]) for _, row in teams.iterrows()}
    for alias, canonical in TEAM_NICKNAME_OVERRIDES.items():
        canonical_lower = canonical.lower()
        if canonical_lower in name_to_sid:
            lookup[alias.lower()] = name_to_sid[canonical_lower]

    return lookup


def match_events_to_matches(
    events: list[dict[str, Any]],
    conn: sqlite3.Connection,
    season: int,
    round_number: int | None = None,
) -> dict[str, int | None]:
    """Match Odds API events to internal match_ids.

    Parameters
    ----------
    events : list[dict]
        Events from fetch_events().
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    round_number : int, optional
        If given, only match against this round's matches.

    Returns
    -------
    dict[str, int | None]
        Mapping of Odds API event_id -> internal match_id (or None).
    """
    team_lookup = _build_team_lookup(conn)

    # Load matches for this season/round
    round_filter = f"AND round_number = {round_number}" if round_number else ""
    matches = pd.read_sql_query(
        f"""
        SELECT match_id, home_squad_id, away_squad_id, round_number
        FROM matches_{season}
        WHERE match_type = 'H'
        {round_filter}
        """,
        conn,
    )

    # Build (home_sid, away_sid) -> match_id lookup
    match_lookup: dict[tuple[int, int], int] = {}
    for _, row in matches.iterrows():
        match_lookup[(int(row["home_squad_id"]), int(row["away_squad_id"]))] = int(
            row["match_id"]
        )

    result: dict[str, int | None] = {}
    for event in events:
        eid = event.get("id", "")
        home = event.get("home_team", "").lower()
        away = event.get("away_team", "").lower()

        home_sid = team_lookup.get(home)
        away_sid = team_lookup.get(away)

        if home_sid is not None and away_sid is not None:
            mid = match_lookup.get((home_sid, away_sid))
            if mid is not None:
                result[eid] = mid
                LOGGER.debug(
                    "Matched event '%s vs %s' -> match_id %d",
                    event.get("home_team"), event.get("away_team"), mid,
                )
            else:
                result[eid] = None
                LOGGER.warning(
                    "No match_id for event '%s vs %s' (squad_ids %d vs %d)",
                    event.get("home_team"), event.get("away_team"),
                    home_sid, away_sid,
                )
        else:
            result[eid] = None
            LOGGER.warning(
                "Unknown team in event: home='%s' (sid=%s), away='%s' (sid=%s)",
                event.get("home_team"), home_sid,
                event.get("away_team"), away_sid,
            )

    n_matched = sum(1 for v in result.values() if v is not None)
    LOGGER.info(
        "Event matching: %d/%d events matched to match_ids",
        n_matched, len(result),
    )
    return result


def fetch_round_odds(
    round_number: int,
    season: int,
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Fetch ATS odds for all matches in a round.

    High-level orchestrator: fetch events -> match to match_ids ->
    fetch player odds per matched event -> return flat records.

    Parameters
    ----------
    round_number : int
        Round number to fetch odds for.
    season : int
        Season year.
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    list[dict]
        Flat per-player-bookmaker records ready for player matching
        and storage.
    """
    api_key = get_api_key()

    # Fetch upcoming events
    events = fetch_events(api_key)
    if not events:
        LOGGER.warning("No NRL events returned from Odds API")
        return []

    # Match events to our match_ids
    event_map = match_events_to_matches(events, conn, season, round_number)

    # Fetch player odds for each matched event
    all_records: list[dict[str, Any]] = []
    for event in events:
        eid = event.get("id", "")
        match_id = event_map.get(eid)
        if match_id is None:
            continue

        try:
            odds_response = fetch_player_try_odds(api_key, eid)
            records = parse_player_odds_response(odds_response, match_id)
            all_records.extend(records)
            LOGGER.info(
                "Fetched %d player odds for %s vs %s (match_id=%d)",
                len(records),
                event.get("home_team"), event.get("away_team"),
                match_id,
            )
        except OddsAPIError as exc:
            LOGGER.warning(
                "Failed to fetch odds for event %s: %s", eid, exc
            )

    LOGGER.info(
        "Total: %d player-bookmaker odds records for round %d",
        len(all_records), round_number,
    )
    return all_records
