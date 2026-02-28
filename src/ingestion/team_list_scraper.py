"""Scrape NRL team lists from LeagueUnlimited (primary) or nrl.com (fallback).

Primary source URL pattern (fully predictable, no discovery needed):
    https://leagueunlimited.com/news/leagueunlimited-nrl-teams-{year}-round-{N}/

Fallback source:
    https://www.nrl.com/news/topic/team-lists/  →  latest article link

Each team list record:
    {
        "team_name": str,       # e.g. "Knights"
        "jersey_number": int,   # 1–21+
        "player_name": str,     # e.g. "Kalyn Ponga"
        "position": str | None, # populated from nrl.com fallback; None from LeagueUnlimited
        "section": str,         # raw section heading from source
    }
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

# LeagueUnlimited
_LU_BASE = "https://leagueunlimited.com/news/leagueunlimited-nrl-teams-{year}-round-{round_number}/"

# nrl.com topic page for fallback discovery
_NRL_TOPIC_URL = "https://www.nrl.com/news/topic/team-lists/"

_REQUEST_TIMEOUT = 15  # seconds
_MIN_TEAMS = 2
_MIN_PLAYERS_PER_TEAM = 13  # at least 13 starters expected

# Regex: "1. Player Name" or "1 Player Name"
_NUMBERED_PLAYER_RE = re.compile(r"^(\d{1,2})[.\s]+(.+)$")

# Regex for nrl.com article format: "Fullback for Knights is number 1 Kalyn Ponga"
_NRL_ARTICLE_RE = re.compile(
    r"^(?P<position>\w[\w\s-]*)\s+for\s+(?P<team>[\w\s]+)\s+is\s+number\s+(?P<jersey>\d+)\s+(?P<name>.+)$",
    re.IGNORECASE,
)


def _get_html(url: str) -> str | None:
    """Fetch a URL and return its HTML text, or None on failure."""
    try:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; NRL2026Model/1.0; +https://github.com)"
            )
        })
        if response.status_code == 200:
            return response.text
        LOGGER.warning("HTTP %d for %s", response.status_code, url)
        return None
    except requests.RequestException as exc:
        LOGGER.warning("Request failed for %s: %s", url, exc)
        return None


def _parse_league_unlimited(html: str) -> list[dict[str, Any]]:
    """Parse team list entries from LeagueUnlimited article HTML.

    LeagueUnlimited structure (verified 2026):
        <figure><img alt="Newcastle Knights" ...></figure>
        <p><!--SQUAD-XXXXX-Home-START-->
          <strong>1.</strong> Kalyn Ponga <strong>2.</strong> Dominic Young ...
        <!--SQUAD-XXXXX-Home-END--></p>

    Team name comes from the <img alt="..."> in the preceding <figure> tag.
    Players are packed into a single <p> tag with jersey numbers in <strong> tags.

    Parameters
    ----------
    html : str
        Raw HTML content.

    Returns
    -------
    list[dict]
        List of player entries with keys: team_name, jersey_number, player_name,
        position, section.
    """
    from bs4 import Comment

    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []

    article = (
        soup.find("article")
        or soup.find(class_=re.compile(r"(entry|post)-content"))
        or soup.find("div", class_=re.compile(r"content"))
        or soup.body
    )
    if article is None:
        LOGGER.warning("Could not find article body in LeagueUnlimited HTML")
        return records

    # Walk through elements; track team name from <figure><img alt="..."> preceding each squad block
    current_team: str | None = None
    elements = list(article.find_all(True))  # all tags

    for element in elements:
        # Team name: <figure> containing <img alt="TEAM NAME">
        if element.name == "figure":
            img = element.find("img")
            if img and img.get("alt"):
                alt = img["alt"].strip()
                # Filter out non-team images (logos, ads, etc.) — team names are title-case multi-word
                if 3 < len(alt) < 60 and not alt[0].isdigit():
                    current_team = alt
                    LOGGER.debug("Team from figure img: %s", current_team)

        # Player list: <p> with SQUAD comment inside
        elif element.name == "p" and current_team is not None:
            # Check for SQUAD comment inside this <p>
            has_squad_comment = any(
                isinstance(child, Comment) and "SQUAD" in child
                for child in element.children
            )
            if not has_squad_comment:
                continue

            # Extract jersey numbers (in <strong> tags) and names (text nodes between them)
            # Build alternating sequence: strong_text, following_text, strong_text, ...
            parts: list[tuple[str, str]] = []  # (jersey_text, player_name)
            pending_jersey: int | None = None

            for child in element.children:
                if hasattr(child, "name") and child.name == "strong":
                    strong_text = child.get_text(strip=True).rstrip(".")
                    if strong_text.isdigit():
                        pending_jersey = int(strong_text)
                elif isinstance(child, str) and pending_jersey is not None:
                    name = child.strip().strip(".")
                    if name:
                        parts.append((pending_jersey, name))
                        pending_jersey = None

            for jersey, name in parts:
                if jersey <= 25 and name:
                    records.append({
                        "team_name": current_team,
                        "jersey_number": jersey,
                        "player_name": name,
                        "position": None,
                        "section": current_team,
                    })
            LOGGER.debug("Parsed %d players for %s", len(parts), current_team)

    return records


def _parse_nrl_article(html: str) -> list[dict[str, Any]]:
    """Parse team list entries from an nrl.com article HTML.

    Parameters
    ----------
    html : str
        Raw HTML content.

    Returns
    -------
    list[dict]
        List of player entries.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []

    article = soup.find("article") or soup.find(class_=re.compile(r"article")) or soup.body
    if article is None:
        return records

    for element in article.find_all(["li", "p"]):
        text = element.get_text(strip=True)
        match = _NRL_ARTICLE_RE.match(text)
        if match:
            records.append({
                "team_name": match.group("team").strip(),
                "jersey_number": int(match.group("jersey")),
                "player_name": match.group("name").strip(),
                "position": match.group("position").strip(),
                "section": match.group("team").strip(),
            })

    return records


def _discover_nrl_topic_article_url(year: int, round_number: int) -> str | None:
    """Find the team lists article URL from nrl.com topic page.

    Parameters
    ----------
    year : int
        Season year.
    round_number : int
        Round number to find.

    Returns
    -------
    str | None
        Article URL if found.
    """
    html = _get_html(_NRL_TOPIC_URL)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    # Look for links containing "round" and the round number
    round_str = str(round_number)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if ("team" in text or "team" in href.lower()) and (
            f"round-{round_str}" in href.lower() or f"round {round_str}" in text
        ):
            url = href if href.startswith("http") else f"https://www.nrl.com{href}"
            LOGGER.info("Found nrl.com team lists article: %s", url)
            return url

    LOGGER.warning("Could not find nrl.com team lists article for round %d", round_number)
    return None


def _validate_records(records: list[dict[str, Any]], source: str) -> bool:
    """Validate scraped records meet minimum coverage expectations.

    Parameters
    ----------
    records : list[dict]
        Scraped player records.
    source : str
        Source name for logging.

    Returns
    -------
    bool
        True if records meet minimum expectations.
    """
    if not records:
        LOGGER.warning("%s: No records returned", source)
        return False

    teams: dict[str, int] = {}
    for r in records:
        teams[r["team_name"]] = teams.get(r["team_name"], 0) + 1

    n_teams = len(teams)
    if n_teams < _MIN_TEAMS:
        LOGGER.warning(
            "%s: Only %d teams found (expected >= %d)", source, n_teams, _MIN_TEAMS
        )
        return False

    thin_teams = [t for t, count in teams.items() if count < _MIN_PLAYERS_PER_TEAM]
    if len(thin_teams) > n_teams // 2:
        LOGGER.warning(
            "%s: Many teams with < %d players: %s", source, _MIN_PLAYERS_PER_TEAM, thin_teams
        )
        return False

    LOGGER.info(
        "%s: %d players across %d teams (valid)",
        source, len(records), n_teams,
    )
    return True


def scrape_league_unlimited(round_number: int, year: int) -> list[dict[str, Any]]:
    """Scrape team lists from LeagueUnlimited for a given round.

    Parameters
    ----------
    round_number : int
        NRL round number (1-based).
    year : int
        Season year.

    Returns
    -------
    list[dict]
        Player records. Empty list if scraping fails or no data found.
    """
    url = _LU_BASE.format(year=year, round_number=round_number)
    LOGGER.info("Fetching LeagueUnlimited: %s", url)
    html = _get_html(url)
    if html is None:
        return []

    records = _parse_league_unlimited(html)
    if not _validate_records(records, "LeagueUnlimited"):
        return []
    return records


def scrape_nrl_topic_article(round_number: int, year: int) -> list[dict[str, Any]]:
    """Scrape team lists from nrl.com (fallback).

    Parameters
    ----------
    round_number : int
        NRL round number.
    year : int
        Season year.

    Returns
    -------
    list[dict]
        Player records. Empty list if scraping fails.
    """
    article_url = _discover_nrl_topic_article_url(year, round_number)
    if article_url is None:
        return []

    LOGGER.info("Fetching nrl.com article: %s", article_url)
    html = _get_html(article_url)
    if html is None:
        return []

    records = _parse_nrl_article(html)
    if not _validate_records(records, "nrl.com"):
        return []
    return records


def fetch_team_lists(round_number: int, year: int) -> list[dict[str, Any]]:
    """Fetch team lists for a given round and year.

    Tries LeagueUnlimited first (predictable URL). Falls back to nrl.com
    topic article if LeagueUnlimited returns no data.

    Parameters
    ----------
    round_number : int
        NRL round number (1-based).
    year : int
        Season year.

    Returns
    -------
    list[dict]
        Player records with keys: team_name, jersey_number, player_name,
        position, section.

    Raises
    ------
    ValueError
        If both sources fail to return valid data.
    """
    LOGGER.info("Fetching team lists for %d Round %d", year, round_number)

    records = scrape_league_unlimited(round_number, year)
    if records:
        LOGGER.info("LeagueUnlimited: %d player records", len(records))
        return records

    LOGGER.info("LeagueUnlimited failed, trying nrl.com fallback")
    records = scrape_nrl_topic_article(round_number, year)
    if records:
        LOGGER.info("nrl.com fallback: %d player records", len(records))
        return records

    raise ValueError(
        f"Both LeagueUnlimited and nrl.com failed to return team lists for "
        f"{year} Round {round_number}. Check that the round has been published."
    )
