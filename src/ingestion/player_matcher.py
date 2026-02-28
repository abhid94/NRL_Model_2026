"""Map scraped player names to internal player_ids.

Four-tier matching strategy (in order of precedence):
    Tier 1 — Exact display_name match (case-sensitive)
    Tier 2 — Normalised match (lowercase, strip punctuation/accents)
    Tier 3 — Surname match within squad (unique surname only)
    Tier 4 — difflib fuzzy match (SequenceMatcher ratio >= 0.85) on display_name

Any player that cannot be matched at Tier 4 is returned as (None, 'no_match', 0.0)
and logged as a warning for manual review.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Minimum fuzzy match ratio to accept (Tier 4)
_FUZZY_THRESHOLD = 0.85

# Punctuation stripper: keep letters, digits, spaces
_NON_ALPHANUM_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalise_name(name: str) -> str:
    """Normalise a name for comparison: lowercase, strip punctuation, collapse spaces.

    Parameters
    ----------
    name : str
        Raw player name.

    Returns
    -------
    str
        Normalised name.
    """
    # Decompose unicode (e.g. é → e + ́) then drop non-ASCII combining chars
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    lower = ascii_only.lower()
    stripped = _NON_ALPHANUM_RE.sub(" ", lower)
    return " ".join(stripped.split())


def get_known_players(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all known players from the database.

    Combines players_2024 and players_2025, deduplicating by player_id.
    Adds a normalised_display_name and surname_lower column for matching.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, display_name, firstname, surname,
                 normalised_display_name, surname_lower, squad_ids (list).
    """
    queries: list[str] = []
    for year in (2024, 2025):
        try:
            conn.execute(f"SELECT 1 FROM players_{year} LIMIT 1")
            queries.append(
                f"SELECT DISTINCT player_id, display_name, firstname, surname "
                f"FROM players_{year}"
            )
        except Exception:  # noqa: BLE001
            pass

    if not queries:
        raise RuntimeError("No players_2024 or players_2025 table found in database")

    union_sql = " UNION " .join(queries)
    df = pd.read_sql_query(union_sql, conn)

    # Deduplicate: keep first occurrence per player_id
    df = df.drop_duplicates(subset=["player_id"], keep="first").copy()

    df["normalised_display_name"] = df["display_name"].apply(_normalise_name)
    df["surname_lower"] = df["surname"].str.lower().str.strip()

    # Load squad-level appearances to know which teams each player has played for
    squad_queries: list[str] = []
    for year in (2024, 2025):
        try:
            conn.execute(f"SELECT 1 FROM player_stats_{year} LIMIT 1")
            squad_queries.append(
                f"SELECT DISTINCT player_id, squad_id FROM player_stats_{year}"
            )
        except Exception:  # noqa: BLE001
            pass

    if squad_queries:
        squad_sql = " UNION ".join(squad_queries)
        squad_df = pd.read_sql_query(squad_sql, conn)
        squad_map = squad_df.groupby("player_id")["squad_id"].apply(list).reset_index()
        squad_map.columns = ["player_id", "squad_ids"]
        df = df.merge(squad_map, on="player_id", how="left")
        df["squad_ids"] = df["squad_ids"].apply(lambda x: x if isinstance(x, list) else [])
    else:
        df["squad_ids"] = [[] for _ in range(len(df))]

    LOGGER.info("Loaded %d known players from DB", len(df))
    return df.reset_index(drop=True)


def match_player_name(
    name: str,
    team_squad_id: int,
    known_players: pd.DataFrame,
    mapping_cache: dict[tuple[str, int], tuple[Optional[int], str, float]],
) -> tuple[Optional[int], str, float]:
    """Match a scraped player name to an internal player_id.

    Parameters
    ----------
    name : str
        Scraped player name.
    team_squad_id : int
        squad_id of the team this player is listed for.
    known_players : pd.DataFrame
        DataFrame from get_known_players().
    mapping_cache : dict
        Cache keyed by (name, squad_id) to avoid re-matching. Updated in-place.

    Returns
    -------
    tuple[Optional[int], str, float]
        (player_id, match_type, confidence) where match_type is one of:
        'exact', 'normalised', 'surname', 'fuzzy', 'no_match'.
    """
    cache_key = (name, team_squad_id)
    if cache_key in mapping_cache:
        return mapping_cache[cache_key]

    # Prefer candidates who played for this team; fall back to all players
    squad_mask = known_players["squad_ids"].apply(lambda ids: team_squad_id in ids)
    squad_candidates = known_players[squad_mask]
    all_candidates = known_players

    # Parse name parts for initial-based matching
    name_parts = name.strip().split()
    first_initial = name_parts[0][0].lower() if name_parts else ""
    scraped_surname = name_parts[-1].lower() if len(name_parts) > 1 else name_parts[0].lower() if name_parts else ""

    def _search(candidates: pd.DataFrame) -> tuple[Optional[int], str, float] | None:
        # Tier 1: Exact display_name match
        exact = candidates[candidates["display_name"] == name]
        if len(exact) == 1:
            return int(exact.iloc[0]["player_id"]), "exact", 1.0
        if len(exact) > 1:
            # Multiple exact matches (shouldn't happen for same squad)
            pid = int(exact.iloc[0]["player_id"])
            LOGGER.warning(
                "Multiple exact matches for '%s' in squad %d — using first: %d",
                name, team_squad_id, pid,
            )
            return pid, "exact", 1.0

        # Tier 2: Normalised name
        norm_name = _normalise_name(name)
        norm_match = candidates[candidates["normalised_display_name"] == norm_name]
        if len(norm_match) == 1:
            return int(norm_match.iloc[0]["player_id"]), "normalised", 0.95
        if len(norm_match) > 1:
            pid = int(norm_match.iloc[0]["player_id"])
            LOGGER.warning(
                "Multiple normalised matches for '%s' — using first: %d", name, pid
            )
            return pid, "normalised", 0.95

        # Tier 2.5: First initial + surname match (handles "Kalyn Ponga" → "K.Ponga")
        # The DB stores names as "F.Surname" — match on initial + surname
        if first_initial and scraped_surname:
            init_surname_match = candidates[
                (candidates["surname_lower"] == scraped_surname)
                & (candidates["firstname"].str[:1].str.lower() == first_initial)
            ]
            if len(init_surname_match) == 1:
                return int(init_surname_match.iloc[0]["player_id"]), "normalised", 0.92
            if len(init_surname_match) > 1:
                # Multiple players with same initial+surname (e.g. brothers) — fall through
                LOGGER.debug(
                    "Ambiguous initial+surname match for '%s': %d candidates",
                    name, len(init_surname_match),
                )

        # Tier 3: Surname match within squad (unique only)
        surname_match = candidates[candidates["surname_lower"] == scraped_surname]
        if len(surname_match) == 1:
            return int(surname_match.iloc[0]["player_id"]), "surname", 0.80

        # Tier 4: Fuzzy match (difflib) on firstname+surname against display_name within squad
        if not candidates.empty:
            best_ratio = 0.0
            best_pid: Optional[int] = None
            for _, row in candidates.iterrows():
                # Try fuzzy on full name vs "Firstname Surname" (reconstructed from DB)
                db_full = f"{row['firstname']} {row['surname']}".lower()
                ratio_full = SequenceMatcher(None, name.lower(), db_full).ratio()
                # Also try against the short display_name (e.g. "K.Ponga")
                ratio_short = SequenceMatcher(None, name.lower(), row["display_name"].lower()).ratio()
                ratio = max(ratio_full, ratio_short)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pid = int(row["player_id"])
            if best_ratio >= _FUZZY_THRESHOLD and best_pid is not None:
                return best_pid, "fuzzy", best_ratio

        return None

    # Try squad-specific candidates first, then fall back to all players
    result = _search(squad_candidates) or _search(all_candidates)

    if result is None:
        LOGGER.warning("No match for player '%s' (squad_id=%d)", name, team_squad_id)
        result = (None, "no_match", 0.0)

    mapping_cache[cache_key] = result
    return result


def build_player_mapping(
    team_lists: list[dict[str, Any]],
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Match all scraped players to internal player_ids.

    Parameters
    ----------
    team_lists : list[dict]
        Scraped records from team_list_scraper (includes squad_id field).
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    list[dict]
        Enriched records with player_id, match_type, confidence added.
    """
    known_players = get_known_players(conn)
    cache: dict[tuple[str, int], tuple[Optional[int], str, float]] = {}
    enriched: list[dict[str, Any]] = []

    for record in team_lists:
        squad_id = record.get("squad_id")
        name = record.get("player_name", "")
        if not name or squad_id is None:
            enriched.append({**record, "player_id": None, "match_type": "no_match", "confidence": 0.0})
            continue

        player_id, match_type, confidence = match_player_name(
            name, int(squad_id), known_players, cache
        )
        enriched.append({**record, "player_id": player_id, "match_type": match_type, "confidence": confidence})

    n_matched = sum(1 for r in enriched if r.get("player_id") is not None)
    n_total = len(enriched)
    LOGGER.info(
        "Player matching: %d/%d matched (%.1f%%)",
        n_matched, n_total, 100 * n_matched / n_total if n_total else 0.0,
    )
    return enriched


def save_player_mapping(
    mappings: list[dict[str, Any]],
    conn: sqlite3.Connection,
) -> None:
    """Upsert player name mappings into nrl_player_mapping table.

    Parameters
    ----------
    mappings : list[dict]
        Enriched records from build_player_mapping() with squad_id, player_id,
        match_type, confidence.
    conn : sqlite3.Connection
        Database connection (nrl_player_mapping must already exist).
    """
    rows: list[tuple] = []
    seen: set[tuple[str, int | None]] = set()

    for record in mappings:
        name = record.get("player_name", "")
        squad_id = record.get("squad_id")
        key = (name, squad_id)
        if not name or key in seen:
            continue
        seen.add(key)
        rows.append((
            name,
            squad_id,
            record.get("player_id"),
            record.get("match_type", "no_match"),
            record.get("confidence", 0.0),
        ))

    if not rows:
        return

    conn.executemany(
        """
        INSERT INTO nrl_player_mapping
            (scraped_name, squad_id, official_player_id, match_type, confidence_score)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(scraped_name, squad_id) DO UPDATE SET
            official_player_id = excluded.official_player_id,
            match_type = excluded.match_type,
            confidence_score = excluded.confidence_score
        """,
        rows,
    )
    conn.commit()
    LOGGER.info("Upserted %d rows into nrl_player_mapping", len(rows))
