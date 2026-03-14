"""Odds-API.io client for fetching Bet365 NRL player try scorer odds.

Secondary odds source to complement The Odds API. Designed to fetch
Bet365 prices which are unavailable through the-odds-api.com for NRL
player props.

Free tier: 2 bookmakers selected, 100 requests/hour.

API docs: https://docs.odds-api.io

Usage:
    # Discovery (run once to find NRL league and confirm market availability):
    from src.odds.odds_api_io import discover_nrl_coverage
    discover_nrl_coverage()

    # Fetch odds (once NRL league slug is confirmed):
    from src.odds.odds_api_io import fetch_nrl_ats_odds
    records = fetch_nrl_ats_odds(conn=conn, season=2026, round_number=1)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

import requests

from src.config import (
    ODDS_API_IO_BASE_URL,
    ODDS_API_IO_BOOKMAKERS,
    ODDS_API_IO_REQUEST_TIMEOUT,
    ODDS_API_IO_SPORT,
    TEAM_NICKNAME_OVERRIDES,
)

LOGGER = logging.getLogger(__name__)

# Confirmed via live discovery (2026-03-05):
#   - NRL events: 59 pending under "rugby-league-nrl-premiership"
#   - Bet365: match-level markets ONLY (Game Betting 2-Way, Handicap 2-Way)
#   - NO player props / ATS markets available for NRL
NRL_LEAGUE_SLUG: str = "rugby-league-nrl-premiership"

# ATS market not available on odds-api.io for NRL (as of 2026-03-05).
# Only match-level markets are available.
ATS_MARKET_NAME: str | None = None

# Match-level markets that ARE available from Bet365 on this API
MATCH_MARKET_NAMES: tuple[str, ...] = ("Game Betting 2-Way", "Handicap 2-Way")


class OddsAPIIOError(RuntimeError):
    """Raised when odds-api.io returns a non-200 response."""


def get_api_key() -> str:
    """Read API key from ``.env`` or ``ODDS_API_IO_KEY`` env var.

    Returns
    -------
    str
        API key.

    Raises
    ------
    OddsAPIIOError
        If the key is not found.
    """
    try:
        from dotenv import load_dotenv
        from src.config import PROJECT_ROOT
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    key = os.environ.get("ODDS_API_IO_KEY", "").strip()
    if not key:
        raise OddsAPIIOError(
            "ODDS_API_IO_KEY not found. Add it to .env or set the environment variable. "
            "Sign up free at https://odds-api.io"
        )
    return key


def _get(endpoint: str, params: dict[str, str] | None = None) -> Any:
    """Make authenticated GET request to odds-api.io.

    Parameters
    ----------
    endpoint : str
        API endpoint path (e.g., "/leagues").
    params : dict, optional
        Query parameters (apiKey is added automatically).

    Returns
    -------
    Any
        Parsed JSON response.

    Raises
    ------
    OddsAPIIOError
        On non-200 response.
    """
    url = f"{ODDS_API_IO_BASE_URL}{endpoint}"
    if params is None:
        params = {}
    params["apiKey"] = get_api_key()

    resp = requests.get(url, params=params, timeout=ODDS_API_IO_REQUEST_TIMEOUT)

    if resp.status_code != 200:
        raise OddsAPIIOError(
            f"odds-api.io {endpoint}: HTTP {resp.status_code} — {resp.text}"
        )

    return resp.json()


def _get_public(endpoint: str) -> Any:
    """Make unauthenticated GET request (for /sports, /bookmakers).

    Parameters
    ----------
    endpoint : str
        API endpoint path.

    Returns
    -------
    Any
        Parsed JSON response.
    """
    url = f"{ODDS_API_IO_BASE_URL}{endpoint}"
    resp = requests.get(url, timeout=ODDS_API_IO_REQUEST_TIMEOUT)

    if resp.status_code != 200:
        raise OddsAPIIOError(
            f"odds-api.io {endpoint}: HTTP {resp.status_code} — {resp.text}"
        )

    return resp.json()


# ---------------------------------------------------------------------------
# Discovery functions — run once to confirm NRL + ATS availability
# ---------------------------------------------------------------------------


def list_sports() -> list[dict[str, Any]]:
    """List all supported sports (no auth required).

    Returns
    -------
    list[dict]
        Each dict has at least 'slug' and 'name'.
    """
    data = _get_public("/sports")
    sports = data if isinstance(data, list) else data.get("data", data)
    LOGGER.info("odds-api.io: %d sports available", len(sports))
    return sports


def list_bookmakers() -> list[dict[str, Any]]:
    """List all supported bookmakers (no auth required).

    Returns
    -------
    list[dict]
        Bookmaker objects with name/slug.
    """
    data = _get_public("/bookmakers")
    bookmakers = data if isinstance(data, list) else data.get("data", data)
    LOGGER.info("odds-api.io: %d bookmakers available", len(bookmakers))
    return bookmakers


def list_leagues(sport: str = ODDS_API_IO_SPORT) -> list[dict[str, Any]]:
    """List leagues for a sport (requires auth).

    Parameters
    ----------
    sport : str
        Sport slug (default: "rugby").

    Returns
    -------
    list[dict]
        League objects with name, slug, eventsCount.
    """
    data = _get("/leagues", {"sport": sport})
    leagues = data if isinstance(data, list) else data.get("data", data)
    LOGGER.info(
        "odds-api.io: %d leagues for sport '%s'", len(leagues), sport
    )
    for lg in leagues:
        LOGGER.info(
            "  League: %s (slug=%s, events=%s)",
            lg.get("name"), lg.get("slug"), lg.get("eventsCount"),
        )
    return leagues


def list_events(
    sport: str = ODDS_API_IO_SPORT,
    league: str | None = None,
    bookmaker: str | None = None,
) -> list[dict[str, Any]]:
    """List events for a sport/league (requires auth).

    Parameters
    ----------
    sport : str
        Sport slug.
    league : str, optional
        League slug to filter by.
    bookmaker : str, optional
        Bookmaker name to filter by (only events with this bookmaker).

    Returns
    -------
    list[dict]
        Event objects with id, home, away, date, league, etc.
    """
    params: dict[str, str] = {"sport": sport}
    if league:
        params["league"] = league
    if bookmaker:
        params["bookmaker"] = bookmaker

    data = _get("/events", params)
    events = data if isinstance(data, list) else data.get("data", data)
    LOGGER.info(
        "odds-api.io: %d events for sport=%s, league=%s", len(events), sport, league
    )
    return events


def get_event_odds(
    event_id: str,
    bookmakers: tuple[str, ...] = ODDS_API_IO_BOOKMAKERS,
) -> dict[str, Any]:
    """Fetch odds for a single event (requires auth).

    Parameters
    ----------
    event_id : str
        Event ID from list_events().
    bookmakers : tuple[str, ...]
        Bookmaker names to fetch (comma-joined).

    Returns
    -------
    dict
        Event data with bookmakers and markets.
    """
    params: dict[str, str] = {
        "eventId": event_id,
        "bookmakers": ",".join(bookmakers),
    }
    data = _get("/odds", params)
    return data


def discover_nrl_coverage() -> dict[str, Any]:
    """Run full discovery to check NRL + Bet365 + ATS availability.

    Prints findings to logger and returns a summary dict.
    Should be run once after signing up for an API key.

    Returns
    -------
    dict
        Keys: has_nrl_league, nrl_league_slug, nrl_events,
        has_bet365, markets_found, has_ats_market.
    """
    result: dict[str, Any] = {
        "has_nrl_league": False,
        "nrl_league_slug": None,
        "nrl_events": 0,
        "has_bet365": False,
        "markets_found": [],
        "has_ats_market": False,
    }

    # Step 1: Find NRL league under "rugby" sport
    LOGGER.info("=== Step 1: Checking rugby leagues ===")
    try:
        leagues = list_leagues("rugby")
    except OddsAPIIOError as exc:
        LOGGER.error("Failed to list leagues: %s", exc)
        return result

    # Prioritize exact NRL match over generic "rugby league"
    nrl_candidates = [
        lg for lg in leagues
        if "nrl" in lg.get("name", "").lower() or "nrl" in lg.get("slug", "").lower()
    ]
    if not nrl_candidates:
        nrl_candidates = [
            lg for lg in leagues
            if "national rugby league" in lg.get("name", "").lower()
        ]

    if nrl_candidates:
        nrl_league = nrl_candidates[0]
        result["has_nrl_league"] = True
        result["nrl_league_slug"] = nrl_league.get("slug")
        LOGGER.info("Found NRL league: %s (slug=%s)", nrl_league.get("name"), nrl_league.get("slug"))
    else:
        LOGGER.warning(
            "No NRL league found. Available leagues: %s",
            [lg.get("name") for lg in leagues],
        )
        return result

    # Step 2: List NRL events with Bet365
    LOGGER.info("=== Step 2: Checking NRL events with Bet365 ===")
    try:
        events = list_events(
            sport="rugby",
            league=result["nrl_league_slug"],
            bookmaker="bet365",
        )
    except OddsAPIIOError:
        # Try without bookmaker filter
        events = list_events(sport="rugby", league=result["nrl_league_slug"])

    result["nrl_events"] = len(events)
    if events:
        LOGGER.info("Found %d NRL events", len(events))
        for ev in events[:3]:
            LOGGER.info(
                "  %s vs %s (%s)", ev.get("home"), ev.get("away"), ev.get("date")
            )
    else:
        LOGGER.warning("No NRL events found")
        return result

    # Step 3: Check markets on first event
    LOGGER.info("=== Step 3: Checking available markets ===")
    first_event_id = events[0].get("id")
    if first_event_id:
        try:
            odds_data = get_event_odds(str(first_event_id))
            # Extract all unique market names
            markets: set[str] = set()
            bookmakers_data = (
                odds_data.get("bookmakers", {})
                if isinstance(odds_data, dict)
                else {}
            )
            # Handle both dict and list bookmaker formats
            if isinstance(bookmakers_data, dict):
                for bk_name, bk_markets in bookmakers_data.items():
                    if bk_name.lower() == "bet365":
                        result["has_bet365"] = True
                    if isinstance(bk_markets, list):
                        for mkt in bk_markets:
                            markets.add(mkt.get("name", "unknown"))
                    elif isinstance(bk_markets, dict):
                        for mkt_name in bk_markets:
                            markets.add(mkt_name)
            elif isinstance(bookmakers_data, list):
                for bk in bookmakers_data:
                    bk_name = bk.get("name", bk.get("key", ""))
                    if "bet365" in bk_name.lower():
                        result["has_bet365"] = True
                    for mkt in bk.get("markets", bk.get("odds", [])):
                        if isinstance(mkt, dict):
                            markets.add(mkt.get("name", mkt.get("key", "unknown")))

            result["markets_found"] = sorted(markets)
            LOGGER.info("Available markets: %s", result["markets_found"])

            # Check for ATS-like markets
            ats_keywords = ("try", "tryscorer", "try scorer", "anytime")
            for mkt in markets:
                if any(kw in mkt.lower() for kw in ats_keywords):
                    result["has_ats_market"] = True
                    LOGGER.info("Found ATS-like market: %s", mkt)

        except OddsAPIIOError as exc:
            LOGGER.warning("Failed to fetch odds for event %s: %s", first_event_id, exc)

    # Summary
    LOGGER.info("=== Discovery Summary ===")
    LOGGER.info("NRL league found: %s (%s)", result["has_nrl_league"], result["nrl_league_slug"])
    LOGGER.info("NRL events: %d", result["nrl_events"])
    LOGGER.info("Bet365 available: %s", result["has_bet365"])
    LOGGER.info("ATS market available: %s", result["has_ats_market"])
    LOGGER.info("All markets: %s", result["markets_found"])

    return result


# ---------------------------------------------------------------------------
# Odds fetching — use after discovery confirms NRL + ATS
# ---------------------------------------------------------------------------


def parse_event_odds(
    odds_data: dict[str, Any],
    match_id: int | None,
    ats_market_name: str | None = None,
) -> list[dict[str, Any]]:
    """Parse odds-api.io event odds into flat per-player records.

    Adapts to the actual API response structure, trying both dict-style
    and list-style bookmaker formats.

    Parameters
    ----------
    odds_data : dict
        Raw response from get_event_odds().
    match_id : int or None
        Internal match_id.
    ats_market_name : str, optional
        Market name for ATS (discovered via discover_nrl_coverage).
        If None, searches for any try-scorer-like market.

    Returns
    -------
    list[dict]
        Flat records with: match_id, bookmaker, player_name_raw,
        decimal_odds, implied_probability, odds_source.
    """
    records: list[dict[str, Any]] = []
    ats_keywords = ("try", "tryscorer", "try scorer", "anytime")

    home_team = odds_data.get("home", "")
    away_team = odds_data.get("away", "")
    event_id = odds_data.get("id", "")

    bookmakers_data = odds_data.get("bookmakers", {})

    def _is_ats_market(name: str) -> bool:
        if ats_market_name:
            return name.lower() == ats_market_name.lower()
        return any(kw in name.lower() for kw in ats_keywords)

    def _process_market(bk_name: str, market: dict[str, Any]) -> None:
        """Extract player odds from a single market."""
        mkt_name = market.get("name", market.get("key", ""))
        if not _is_ats_market(mkt_name):
            return

        outcomes = market.get("odds", market.get("outcomes", []))
        if isinstance(outcomes, list):
            for outcome in outcomes:
                _process_outcome(bk_name, outcome)
        elif isinstance(outcomes, dict):
            for player_name, price_data in outcomes.items():
                price = (
                    price_data if isinstance(price_data, (int, float))
                    else price_data.get("price", price_data.get("odds"))
                )
                if price and float(price) > 1.0:
                    _add_record(bk_name, player_name, float(price))

    def _process_outcome(bk_name: str, outcome: dict[str, Any]) -> None:
        """Extract a single player outcome."""
        # Try multiple field names for player name
        player_name = (
            outcome.get("description")
            or outcome.get("name")
            or outcome.get("participant")
            or ""
        ).strip()

        # Some APIs use "Yes"/"No" structure like The Odds API
        if outcome.get("name", "").lower() in ("yes", "over"):
            player_name = outcome.get("description", "").strip()
        elif outcome.get("name", "").lower() in ("no", "under"):
            return  # Skip "No" outcomes

        price = outcome.get("price", outcome.get("odds"))
        if price is None or float(price) <= 1.0:
            return
        if not player_name:
            return

        _add_record(bk_name, player_name, float(price))

    def _add_record(bk_name: str, player_name: str, price: float) -> None:
        records.append({
            "match_id": match_id,
            "odds_api_io_event_id": str(event_id),
            "home_team": home_team,
            "away_team": away_team,
            "bookmaker": f"bet365" if "bet365" in bk_name.lower() else bk_name.lower(),
            "player_name_raw": player_name,
            "decimal_odds": round(price, 2),
            "implied_probability": round(1.0 / price, 4),
            "odds_source": "odds_api_io",
        })

    # Handle dict-style: {"bookmakers": {"Bet365": [...markets], ...}}
    if isinstance(bookmakers_data, dict):
        for bk_name, bk_content in bookmakers_data.items():
            if isinstance(bk_content, list):
                for market in bk_content:
                    if isinstance(market, dict):
                        _process_market(bk_name, market)
            elif isinstance(bk_content, dict):
                for mkt_name, mkt_data in bk_content.items():
                    if isinstance(mkt_data, dict):
                        mkt_data["name"] = mkt_name
                        _process_market(bk_name, mkt_data)

    # Handle list-style: {"bookmakers": [{"name": "Bet365", "markets": [...]}]}
    elif isinstance(bookmakers_data, list):
        for bk in bookmakers_data:
            bk_name = bk.get("name", bk.get("key", ""))
            for market in bk.get("markets", bk.get("odds", [])):
                if isinstance(market, dict):
                    _process_market(bk_name, market)

    return records


def _build_team_lookup(conn: sqlite3.Connection) -> dict[str, int]:
    """Build team name → squad_id lookup (reuses logic from odds_api.py).

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    dict[str, int]
        Lowercase team name -> squad_id.
    """
    import pandas as pd

    teams = pd.read_sql_query(
        "SELECT squad_id, squad_name, squad_nickname FROM teams", conn
    )
    lookup: dict[str, int] = {}
    for _, row in teams.iterrows():
        sid = int(row["squad_id"])
        lookup[row["squad_name"].lower()] = sid
        lookup[row["squad_nickname"].lower()] = sid

    name_to_sid = {
        row["squad_name"].lower(): int(row["squad_id"])
        for _, row in teams.iterrows()
    }
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
    """Match odds-api.io events to internal match_ids.

    Parameters
    ----------
    events : list[dict]
        Events from list_events().
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    round_number : int, optional
        If given, only match against this round.

    Returns
    -------
    dict[str, int | None]
        Event ID -> match_id (or None).
    """
    import pandas as pd

    team_lookup = _build_team_lookup(conn)

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

    match_lookup: dict[tuple[int, int], int] = {}
    for _, row in matches.iterrows():
        match_lookup[(int(row["home_squad_id"]), int(row["away_squad_id"]))] = int(
            row["match_id"]
        )

    result: dict[str, int | None] = {}
    for event in events:
        eid = str(event.get("id", ""))
        home = (event.get("home", {}) if isinstance(event.get("home"), dict)
                else {"name": event.get("home", "")})
        away = (event.get("away", {}) if isinstance(event.get("away"), dict)
                else {"name": event.get("away", "")})

        home_name = (home.get("name", "") if isinstance(home, dict) else str(home)).lower()
        away_name = (away.get("name", "") if isinstance(away, dict) else str(away)).lower()

        home_sid = team_lookup.get(home_name)
        away_sid = team_lookup.get(away_name)

        if home_sid is not None and away_sid is not None:
            mid = match_lookup.get((home_sid, away_sid))
            result[eid] = mid
            if mid:
                LOGGER.debug("Matched event '%s vs %s' -> match_id %d", home_name, away_name, mid)
            else:
                LOGGER.warning("No match_id for '%s vs %s'", home_name, away_name)
        else:
            result[eid] = None
            LOGGER.warning(
                "Unknown team: home='%s' (sid=%s), away='%s' (sid=%s)",
                home_name, home_sid, away_name, away_sid,
            )

    n_matched = sum(1 for v in result.values() if v is not None)
    LOGGER.info("Event matching: %d/%d events matched", n_matched, len(result))
    return result


def parse_match_odds(
    odds_data: dict[str, Any],
    match_id: int | None,
) -> list[dict[str, Any]]:
    """Parse Bet365 match-level odds (h2h, handicaps, totals).

    Parameters
    ----------
    odds_data : dict
        Raw response from get_event_odds().
    match_id : int or None
        Internal match_id.

    Returns
    -------
    list[dict]
        Records with: match_id, bookmaker, market, label, home_odds,
        away_odds, handicap, odds_source.
    """
    records: list[dict[str, Any]] = []
    home_team = odds_data.get("home", "")
    away_team = odds_data.get("away", "")
    event_id = odds_data.get("id", "")

    bookmakers_data = odds_data.get("bookmakers", {})
    if not isinstance(bookmakers_data, dict):
        return records

    for bk_name, bk_markets in bookmakers_data.items():
        if not isinstance(bk_markets, list):
            continue
        for market in bk_markets:
            mkt_name = market.get("name", "")
            updated_at = market.get("updatedAt", "")
            for outcome in market.get("odds", []):
                label = outcome.get("label", "")
                record = {
                    "match_id": match_id,
                    "event_id": str(event_id),
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": "bet365",
                    "market": mkt_name,
                    "label": label,
                    "home_odds": _parse_price(outcome.get("home")),
                    "away_odds": _parse_price(outcome.get("away")),
                    "handicap": outcome.get("hdp"),
                    "updated_at": updated_at,
                    "odds_source": "odds_api_io",
                }
                records.append(record)

    return records


def _parse_price(val: Any) -> float | None:
    """Parse price string to float, or return None."""
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None


def parse_h2h_odds(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Extract h2h implied win probability and handicap from match odds.

    Parameters
    ----------
    records : list[dict]
        Output from ``parse_match_odds()`` or ``fetch_nrl_match_odds()``.

    Returns
    -------
    pd.DataFrame
        One row per match with: match_id, bet365_home_odds,
        bet365_away_odds, bet365_implied_home_win_prob,
        bet365_handicap_line.
    """
    import pandas as pd

    if not records:
        return pd.DataFrame(columns=[
            "match_id", "bet365_home_odds", "bet365_away_odds",
            "bet365_implied_home_win_prob", "bet365_handicap_line",
        ])

    df = pd.DataFrame(records)
    results = []

    for mid, group in df.groupby("match_id"):
        row: dict[str, Any] = {"match_id": mid}

        # H2H
        h2h = group[group["market"] == "Game Betting 2-Way"]
        if not h2h.empty:
            home_odds = h2h["home_odds"].dropna()
            away_odds = h2h["away_odds"].dropna()
            if not home_odds.empty and not away_odds.empty:
                ho = home_odds.iloc[0]
                ao = away_odds.iloc[0]
                row["bet365_home_odds"] = ho
                row["bet365_away_odds"] = ao
                if ho > 1 and ao > 1:
                    overround = (1.0 / ho) + (1.0 / ao)
                    row["bet365_implied_home_win_prob"] = (1.0 / ho) / overround

        # Handicap — take the first line (usually the main line)
        hcap = group[group["market"] == "Handicap 2-Way"]
        if not hcap.empty and "handicap" in hcap.columns:
            hdp = hcap["handicap"].dropna()
            if not hdp.empty:
                row["bet365_handicap_line"] = hdp.iloc[0]

        results.append(row)

    return pd.DataFrame(results)


def fetch_nrl_match_odds(
    conn: sqlite3.Connection,
    season: int,
    round_number: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch Bet365 match-level odds (h2h, handicaps) for NRL.

    This is the primary use case for odds-api.io since ATS/player props
    are NOT available for NRL on this API.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    round_number : int, optional
        Filter to specific round.

    Returns
    -------
    list[dict]
        Match odds records with h2h and handicap lines.
    """
    events = list_events(
        sport=ODDS_API_IO_SPORT, league=NRL_LEAGUE_SLUG
    )
    if not events:
        LOGGER.warning("No NRL events from odds-api.io")
        return []

    event_map = match_events_to_matches(events, conn, season, round_number)

    all_records: list[dict[str, Any]] = []
    for event in events:
        eid = str(event.get("id", ""))
        mid = event_map.get(eid)
        if mid is None:
            continue

        try:
            odds_data = get_event_odds(eid)
            records = parse_match_odds(odds_data, mid)
            all_records.extend(records)
            LOGGER.info(
                "Fetched %d Bet365 match odds for event %s (match_id=%d)",
                len(records), eid, mid,
            )
        except OddsAPIIOError as exc:
            LOGGER.warning("Failed to fetch odds for event %s: %s", eid, exc)

    LOGGER.info("Total: %d Bet365 match odds records", len(all_records))
    return all_records


def fetch_nrl_ats_odds(
    conn: sqlite3.Connection,
    season: int,
    round_number: int | None = None,
    nrl_league_slug: str | None = None,
    ats_market_name: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch Bet365 ATS odds for NRL matches from odds-api.io.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    round_number : int, optional
        Round number to filter matches.
    nrl_league_slug : str, optional
        NRL league slug (from discovery). Falls back to module-level
        NRL_LEAGUE_SLUG.
    ats_market_name : str, optional
        ATS market name (from discovery). Falls back to module-level
        ATS_MARKET_NAME.

    Returns
    -------
    list[dict]
        Flat per-player records ready for player matching and storage.
        Same schema as odds_api.py records plus 'odds_source' field.
    """
    league = nrl_league_slug or NRL_LEAGUE_SLUG
    market = ats_market_name or ATS_MARKET_NAME

    if league is None:
        raise OddsAPIIOError(
            "NRL league slug not set. Run discover_nrl_coverage() first, then set "
            "NRL_LEAGUE_SLUG or pass nrl_league_slug parameter."
        )

    # Fetch NRL events
    events = list_events(sport=ODDS_API_IO_SPORT, league=league, bookmaker="bet365")
    if not events:
        LOGGER.warning("No NRL events from odds-api.io")
        return []

    # Match to internal match_ids
    event_map = match_events_to_matches(events, conn, season, round_number)

    # Fetch odds per matched event
    all_records: list[dict[str, Any]] = []
    for event in events:
        eid = str(event.get("id", ""))
        match_id = event_map.get(eid)
        if match_id is None:
            continue

        try:
            odds_data = get_event_odds(eid)
            records = parse_event_odds(odds_data, match_id, market)
            all_records.extend(records)
            LOGGER.info(
                "Fetched %d Bet365 player odds for event %s (match_id=%d)",
                len(records), eid, match_id,
            )
        except OddsAPIIOError as exc:
            LOGGER.warning("Failed to fetch odds for event %s: %s", eid, exc)

    LOGGER.info(
        "Total: %d Bet365 player odds records from odds-api.io", len(all_records)
    )
    return all_records
