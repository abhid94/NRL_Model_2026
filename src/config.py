"""Project configuration and constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "nrl_data.db"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
MODEL_ARTIFACTS_DIR = DATA_DIR / "model_artifacts"
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"

DEFAULT_PLAYER_FEATURE_WINDOWS = (3, 5, 10)
DEFAULT_TEAM_FEATURE_WINDOWS = (3, 5, 10)

# Edge mappings for try scoring analysis
# Based on validated team edge attack patterns (see docs/plans/ats_strategy.md)
JERSEY_TO_EDGE: Mapping[int, str] = {
    2: "left",  # Left Wing
    3: "left",  # Left Centre
    11: "left",  # Left Second Row
    4: "right",  # Right Centre
    5: "right",  # Right Wing
    12: "right",  # Right Second Row
    8: "middle",  # Prop
    9: "middle",  # Hooker
    10: "middle",  # Prop
    13: "middle",  # Lock
}


@dataclass(frozen=True)
class PositionInfo:
    """Metadata about a jersey position."""

    code: str
    label: str


JERSEY_NUMBER_POSITION: Mapping[int, PositionInfo] = {
    1: PositionInfo("FB", "Fullback"),
    2: PositionInfo("WG", "Wing"),
    3: PositionInfo("CE", "Centre"),
    4: PositionInfo("CE", "Centre"),
    5: PositionInfo("WG", "Wing"),
    6: PositionInfo("FE", "Five-eighth"),
    7: PositionInfo("HB", "Halfback"),
    8: PositionInfo("PR", "Prop"),
    9: PositionInfo("HK", "Hooker"),
    10: PositionInfo("PR", "Prop"),
    11: PositionInfo("SR", "Second Row"),
    12: PositionInfo("SR", "Second Row"),
    13: PositionInfo("LK", "Lock"),
    14: PositionInfo("INT", "Interchange"),
    15: PositionInfo("INT", "Interchange"),
    16: PositionInfo("INT", "Interchange"),
    17: PositionInfo("INT", "Interchange"),
}

DEFAULT_RESERVE_POSITION = PositionInfo("RES", "Reserve")

# ---------------------------------------------------------------------------
# Staking & Risk Parameters (CLAUDE.md Section 11)
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_BANKROLL: float = 10_000.0
DEFAULT_KELLY_FRACTION: float = 0.25
MAX_STAKE_PCT: float = 0.05
MAX_ROUND_EXPOSURE_PCT: float = 0.20
MAX_BETS_PER_ROUND: int = 15
MAX_BETS_PER_MATCH: int = 4
MIN_STAKE: float = 5.0
MIN_EDGE_THRESHOLD: float = 0.05

# Position eligibility for ATS betting
# Only bet: backs + halves + back-rowers. Never bet: props, hookers, interchange, reserves.
ELIGIBLE_POSITION_CODES: frozenset[str] = frozenset({
    "FB", "WG", "CE", "FE", "HB", "SR", "LK",
})


# ---------------------------------------------------------------------------
# Team name aliases for scraping (maps scraped short names → canonical squad_name)
# Used by src/ingestion/ingest_team_lists._build_team_lookup()
# ---------------------------------------------------------------------------
TEAM_NICKNAME_OVERRIDES: dict[str, str] = {
    # LeagueUnlimited commonly uses these short forms
    "Knights": "Newcastle Knights",
    "Cowboys": "North Queensland Cowboys",
    "Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Dragons": "St George-Illawarra Dragons",
    "Roosters": "Sydney Roosters",
    "Broncos": "Brisbane Broncos",
    "Raiders": "Canberra Raiders",
    "Storm": "Melbourne Storm",
    "Eels": "Parramatta Eels",
    "Panthers": "Penrith Panthers",
    "Sharks": "Cronulla-Sutherland Sharks",
    "Wests Tigers": "Wests Tigers",
    "Tigers": "Wests Tigers",
    "Rabbitohs": "South Sydney Rabbitohs",
    "Sea Eagles": "Manly-Warringah Sea Eagles",
    "Titans": "Gold Coast Titans",
    "Warriors": "Warriors",
    "Dolphins": "Dolphins",
    # The Odds API uses slightly different names (no hyphens)
    "Canterbury Bulldogs": "Canterbury-Bankstown Bulldogs",
    "St George Illawarra Dragons": "St George-Illawarra Dragons",
    "New Zealand Warriors": "Warriors",
    "Cronulla Sutherland Sharks": "Cronulla-Sutherland Sharks",
    "Manly Warringah Sea Eagles": "Manly-Warringah Sea Eagles",
}


# ---------------------------------------------------------------------------
# Odds API Configuration (The Odds API — https://the-odds-api.com)
# ---------------------------------------------------------------------------
ODDS_API_SPORT_KEY: str = "rugbyleague_nrl"
ODDS_API_MARKET: str = "player_try_scorer_anytime"
ODDS_API_REGIONS: str = "au"
ODDS_API_BOOKMAKERS: tuple[str, ...] = (
    "sportsbet", "tab", "ladbrokes_au", "pointsbetau", "betr_au",
    "tabtouch", "unibet", "betright",
)
BOOKMAKER_DISPLAY_NAMES: dict[str, str] = {
    "sportsbet": "Sportsbet",
    "tab": "TAB",
    "ladbrokes_au": "Ladbrokes",
    "pointsbetau": "Pointsbet",
    "betr_au": "Betr",
    "tabtouch": "TABtouch",
    "unibet": "Unibet",
    "betright": "BetRight",
}
# Margin correction: bookmaker implied probs are ~12-18% inflated vs Betfair's 4-8%
# Multiply bookmaker implied_prob by this factor to approximate Betfair scale.
BOOKMAKER_MARGIN_CORRECTION: float = 0.88


def position_from_jersey(jersey_number: int | None) -> PositionInfo:
    """Infer position metadata from a jersey number.

    Parameters
    ----------
    jersey_number : int | None
        Jersey number for a player appearance.

    Returns
    -------
    PositionInfo
        Position metadata inferred from the jersey number.
    """
    if jersey_number is None:
        return DEFAULT_RESERVE_POSITION
    return JERSEY_NUMBER_POSITION.get(jersey_number, DEFAULT_RESERVE_POSITION)
