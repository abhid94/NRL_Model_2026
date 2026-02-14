"""Database access helpers for year-suffixed tables.

This module centralizes SQLite access and year-suffix table resolution
so that downstream code does not hardcode table names.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .config import DEFAULT_RESERVE_POSITION, JERSEY_NUMBER_POSITION

LOGGER = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data") / "nrl_data.db"
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class TableNotFoundError(ValueError):
    """Raised when an expected table is missing."""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection with row access by name.

    Parameters
    ----------
    db_path : Path | str
        Path to the SQLite database.

    Returns
    -------
    sqlite3.Connection
        Open connection to the database.
    """
    resolved = Path(db_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Database not found at {resolved}")

    connection = sqlite3.connect(str(resolved))
    connection.row_factory = sqlite3.Row
    return connection


def validate_identifier(name: str) -> None:
    """Ensure SQL identifiers only contain safe characters.

    Parameters
    ----------
    name : str
        Identifier to validate.

    Raises
    ------
    ValueError
        If the identifier contains unexpected characters.
    """
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"Invalid SQL identifier: {name}")


def normalize_year(year: int | str) -> int:
    """Validate and normalize a season year.

    Parameters
    ----------
    year : int | str
        Season year.

    Returns
    -------
    int
        Normalized year.
    """
    try:
        normalized = int(year)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid year value: {year}") from exc

    if normalized < 1900 or normalized > 2100:
        raise ValueError(f"Year out of expected range: {normalized}")

    return normalized


def get_table(base_name: str, year: Optional[int | str] = None) -> str:
    """Resolve a table name given a base name and optional year.

    Parameters
    ----------
    base_name : str
        Base table name (e.g., ``player_stats``).
    year : int | str | None
        Season year to suffix the table with. If None, returns ``base_name``.

    Returns
    -------
    str
        Resolved table name.
    """
    validate_identifier(base_name)
    if year is None:
        return base_name
    normalized = normalize_year(year)
    return f"{base_name}_{normalized}"


def list_tables(connection: sqlite3.Connection) -> list[str]:
    """List all tables in the SQLite database."""
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check whether a table exists."""
    validate_identifier(table_name)
    cursor = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def resolve_table(
    connection: sqlite3.Connection,
    base_name: str,
    year: Optional[int | str] = None,
    *,
    require_exists: bool = True,
) -> str:
    """Resolve a table name and optionally verify it exists.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    base_name : str
        Base table name.
    year : int | str | None
        Season year to suffix the table with.
    require_exists : bool
        When True, raise if the table is missing.

    Returns
    -------
    str
        Resolved table name.
    """
    table_name = get_table(base_name, year)
    if require_exists and not table_exists(connection, table_name):
        raise TableNotFoundError(f"Table not found: {table_name}")
    return table_name


def create_union_view(
    connection: sqlite3.Connection,
    base_name: str,
    years: Sequence[int | str],
    *,
    view_name: Optional[str] = None,
) -> str:
    """Create a TEMP view that unions seasonal tables with a season column.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    base_name : str
        Base table name to union.
    years : Sequence[int | str]
        Years to include.
    view_name : str | None
        Optional explicit view name. Defaults to ``base_name``.

    Returns
    -------
    str
        Name of the created view.
    """
    validate_identifier(base_name)
    view = view_name or base_name
    validate_identifier(view)

    if not years:
        raise ValueError("At least one year is required to build a union view")

    selects: list[str] = []
    for year in years:
        table_name = resolve_table(connection, base_name, year, require_exists=True)
        selects.append(f"SELECT *, {int(year)} AS season FROM {table_name}")

    union_sql = " UNION ALL ".join(selects)
    connection.execute(f"DROP VIEW IF EXISTS {view}")
    connection.execute(f"CREATE TEMP VIEW {view} AS {union_sql}")
    LOGGER.info("Created TEMP view %s for %s", view, base_name)
    return view


def fetch_df(
    connection: sqlite3.Connection,
    query: str,
    params: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Execute a query and return a pandas DataFrame."""
    return pd.read_sql_query(query, connection, params=params)


def execute(
    connection: sqlite3.Connection,
    query: str,
    params: Optional[Sequence[Any]] = None,
) -> None:
    """Execute a parameterized SQL statement."""
    connection.execute(query, params or [])
    connection.commit()


def _case_expression(column: str, mapping: dict[str, set[int]], default: str) -> str:
    """Build a CASE expression for a numeric column.

    Parameters
    ----------
    column : str
        Column name to compare.
    mapping : dict[str, set[int]]
        Mapping of return values to numeric sets.
    default : str
        Default value when no match is found.

    Returns
    -------
    str
        SQL CASE expression.
    """
    clauses = []
    for value, numbers in mapping.items():
        if not numbers:
            continue
        if len(numbers) == 1:
            num = next(iter(numbers))
            clauses.append(f"WHEN {column} = {num} THEN '{value}'")
        else:
            nums = ", ".join(str(num) for num in sorted(numbers))
            clauses.append(f"WHEN {column} IN ({nums}) THEN '{value}'")
    clauses.append(f"ELSE '{default}'")
    return "CASE " + " ".join(clauses) + " END"


def _position_mappings() -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Group jersey numbers by position code and label."""
    code_mapping: dict[str, set[int]] = {}
    label_mapping: dict[str, set[int]] = {}
    for jersey, info in JERSEY_NUMBER_POSITION.items():
        code_mapping.setdefault(info.code, set()).add(jersey)
        label_mapping.setdefault(info.label, set()).add(jersey)
    return code_mapping, label_mapping


def create_cleansed_views(
    connection: sqlite3.Connection,
    year: int | str,
) -> dict[str, str]:
    """Create TEMP views with cleaned, analysis-ready base tables.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to cleanse.

    Returns
    -------
    dict[str, str]
        Mapping of logical table names to created view names.
    """
    normalized_year = normalize_year(year)
    created: dict[str, str] = {}

    players_table = get_table("players", normalized_year)
    if table_exists(connection, players_table):
        view_name = f"players_cleaned_{normalized_year}"
        validate_identifier(view_name)
        connection.execute(f"DROP VIEW IF EXISTS {view_name}")
        connection.execute(
            f"""
            CREATE TEMP VIEW {view_name} AS
            SELECT DISTINCT player_id,
                firstname,
                surname,
                display_name,
                short_display_name
            FROM {players_table}
            """
        )
        created["players"] = view_name

    team_lists_table = get_table("team_lists", normalized_year)
    if table_exists(connection, team_lists_table):
        view_name = f"team_lists_cleaned_{normalized_year}"
        validate_identifier(view_name)
        code_mapping, label_mapping = _position_mappings()
        position_code_case = _case_expression(
            "jersey_number", code_mapping, DEFAULT_RESERVE_POSITION.code
        )
        position_label_case = _case_expression(
            "jersey_number", label_mapping, DEFAULT_RESERVE_POSITION.label
        )
        connection.execute(f"DROP VIEW IF EXISTS {view_name}")
        connection.execute(
            f"""
            CREATE TEMP VIEW {view_name} AS
            SELECT
                match_id,
                round_number,
                squad_id,
                squad_name,
                player_name,
                player_id,
                jersey_number,
                COALESCE(NULLIF(position, ''), {position_code_case}) AS position_code,
                {position_label_case} AS position_label
            FROM {team_lists_table}
            """
        )
        created["team_lists"] = view_name

    betfair_table = get_table("betfair_markets", normalized_year)
    if table_exists(connection, betfair_table):
        view_name = f"betfair_markets_cleaned_{normalized_year}"
        validate_identifier(view_name)
        connection.execute(f"DROP VIEW IF EXISTS {view_name}")
        connection.execute(
            f"""
            CREATE TEMP VIEW {view_name} AS
            SELECT
                *,
                CAST(
                    COALESCE(
                        NULLIF(last_preplay_price, ''),
                        NULLIF(best_back_price_1_min_prior, ''),
                        NULLIF(best_back_price_30_min_prior, '')
                    ) AS REAL
                ) AS last_preplay_price_cleaned
            FROM {betfair_table}
            """
        )
        created["betfair_markets"] = view_name

    odds_table = get_table("bookmaker_odds", normalized_year)
    if table_exists(connection, odds_table):
        view_name = f"bookmaker_odds_cleaned_{normalized_year}"
        validate_identifier(view_name)
        connection.execute(f"DROP VIEW IF EXISTS {view_name}")
        connection.execute(
            f"""
            CREATE TEMP VIEW {view_name} AS
            SELECT
                *,
                CASE WHEN bookmaker IS NULL THEN NULL ELSE TRIM(bookmaker) END AS bookmaker_cleaned
            FROM {odds_table}
            """
        )
        created["bookmaker_odds"] = view_name

    return created


def data_quality_report(connection: sqlite3.Connection, year: int | str) -> dict[str, Any]:
    """Generate a data quality summary for a season.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to profile.

    Returns
    -------
    dict[str, Any]
        Summary metrics for validation against expectations.
    """
    normalized_year = normalize_year(year)
    report: dict[str, Any] = {}

    players_table = resolve_table(connection, "players", normalized_year)
    report["players_total"] = connection.execute(
        f"SELECT COUNT(*) FROM {players_table}"
    ).fetchone()[0]
    report["players_unique"] = connection.execute(
        f"SELECT COUNT(DISTINCT player_id) FROM {players_table}"
    ).fetchone()[0]

    team_lists_table = get_table("team_lists", normalized_year)
    if table_exists(connection, team_lists_table):
        report["team_lists_position_empty"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {team_lists_table}
            WHERE position IS NULL OR position = ''
            """
        ).fetchone()[0]
    else:
        report["team_lists_position_empty"] = None

    odds_table = get_table("bookmaker_odds", normalized_year)
    if table_exists(connection, odds_table):
        rows = connection.execute(
            f"SELECT bookmaker, COUNT(*) FROM {odds_table} GROUP BY bookmaker"
        ).fetchall()
        report["bookmaker_counts"] = {row[0]: row[1] for row in rows}
    else:
        report["bookmaker_counts"] = {}

    betfair_table = get_table("betfair_markets", normalized_year)
    if table_exists(connection, betfair_table):
        report["betfair_last_preplay_empty"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {betfair_table}
            WHERE market_type='TO_SCORE'
              AND (last_preplay_price IS NULL OR last_preplay_price = '')
            """
        ).fetchone()[0]
        report["betfair_last_preplay_fallback_missing"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {betfair_table}
            WHERE market_type='TO_SCORE'
              AND COALESCE(
                NULLIF(last_preplay_price, ''),
                NULLIF(best_back_price_1_min_prior, ''),
                NULLIF(best_back_price_30_min_prior, '')
              ) IS NULL
            """
        ).fetchone()[0]
        report["betfair_unmapped_runners"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {betfair_table}
            WHERE market_type='TO_SCORE'
              AND (AD_player_id IS NULL OR AD_player_id = '')
            """
        ).fetchone()[0]
    else:
        report["betfair_last_preplay_empty"] = None
        report["betfair_last_preplay_fallback_missing"] = None
        report["betfair_unmapped_runners"] = None

    player_stats_table = get_table("player_stats", normalized_year)
    if table_exists(connection, player_stats_table):
        rows = connection.execute(
            f"SELECT tries, COUNT(*) FROM {player_stats_table} GROUP BY tries"
        ).fetchall()
        report["try_distribution"] = {row[0]: row[1] for row in rows}
        total = sum(report["try_distribution"].values())
        positives = sum(count for tries, count in report["try_distribution"].items() if tries > 0)
        report["try_rate"] = positives / total if total else 0.0

        position_rows = connection.execute(
            f"""
            SELECT position,
                   SUM(CASE WHEN tries > 0 THEN 1 ELSE 0 END) AS positives,
                   COUNT(*) AS total
            FROM {player_stats_table}
            GROUP BY position
            """
        ).fetchall()
        report["position_try_rates"] = {
            row[0]: (row[1] / row[2] if row[2] else 0.0) for row in position_rows
        }
    else:
        report["try_distribution"] = {}
        report["try_rate"] = None
        report["position_try_rates"] = {}

    matches_table = get_table("matches", normalized_year)
    if table_exists(connection, matches_table):
        rounds = connection.execute(
            f"""
            SELECT round_number, COUNT(*) AS match_count
            FROM {matches_table}
            GROUP BY round_number
            ORDER BY round_number
            """
        ).fetchall()
        report["rounds_with_fewer_matches"] = [row[0] for row in rounds if row[1] < 8]

        home_away = connection.execute(
            f"""
            SELECT
              CASE WHEN ps.squad_id = m.home_squad_id THEN 'home'
                   WHEN ps.squad_id = m.away_squad_id THEN 'away'
                   ELSE 'unknown' END AS loc,
              SUM(CASE WHEN ps.tries > 0 THEN 1 ELSE 0 END) AS positives,
              COUNT(*) AS total
            FROM {player_stats_table} ps
            JOIN {matches_table} m ON ps.match_id = m.match_id
            GROUP BY loc
            """
        ).fetchall()
        totals = {row[0]: (row[1], row[2]) for row in home_away}
        home_pos, home_total = totals.get("home", (0, 0))
        away_pos, away_total = totals.get("away", (0, 0))
        report["home_try_rate"] = home_pos / home_total if home_total else 0.0
        report["away_try_rate"] = away_pos / away_total if away_total else 0.0
    else:
        report["rounds_with_fewer_matches"] = []
        report["home_try_rate"] = None
        report["away_try_rate"] = None

    return report


def referential_integrity_report(
    connection: sqlite3.Connection, year: int | str
) -> dict[str, int]:
    """Check for missing foreign key references.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to validate.

    Returns
    -------
    dict[str, int]
        Counts of missing references by relationship.
    """
    normalized_year = normalize_year(year)
    report: dict[str, int] = {}

    players_table = get_table("players", normalized_year)
    player_stats_table = get_table("player_stats", normalized_year)
    matches_table = get_table("matches", normalized_year)

    if table_exists(connection, players_table) and table_exists(connection, player_stats_table):
        report["player_stats_players"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {player_stats_table} ps
            LEFT JOIN {players_table} p ON ps.player_id = p.player_id
            WHERE p.player_id IS NULL
            """
        ).fetchone()[0]
    if table_exists(connection, matches_table) and table_exists(connection, player_stats_table):
        report["player_stats_matches"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {player_stats_table} ps
            LEFT JOIN {matches_table} m ON ps.match_id = m.match_id
            WHERE m.match_id IS NULL
            """
        ).fetchone()[0]

    team_lists_table = get_table("team_lists", normalized_year)
    if table_exists(connection, team_lists_table) and table_exists(connection, players_table):
        report["team_lists_players"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {team_lists_table} tl
            LEFT JOIN {players_table} p ON tl.player_id = p.player_id
            WHERE p.player_id IS NULL
            """
        ).fetchone()[0]
    if table_exists(connection, team_lists_table) and table_exists(connection, matches_table):
        report["team_lists_matches"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {team_lists_table} tl
            LEFT JOIN {matches_table} m ON tl.match_id = m.match_id
            WHERE m.match_id IS NULL
            """
        ).fetchone()[0]

    betfair_table = get_table("betfair_markets", normalized_year)
    if table_exists(connection, betfair_table) and table_exists(connection, players_table):
        report["betfair_ad_player"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {betfair_table} bm
            LEFT JOIN {players_table} p ON bm.AD_player_id = p.player_id
            WHERE bm.market_type='TO_SCORE'
              AND bm.AD_player_id IS NOT NULL
              AND p.player_id IS NULL
            """
        ).fetchone()[0]
    if table_exists(connection, betfair_table) and table_exists(connection, matches_table):
        report["betfair_ad_match"] = connection.execute(
            f"""
            SELECT COUNT(*) FROM {betfair_table} bm
            LEFT JOIN {matches_table} m ON bm.AD_match_id = m.match_id
            WHERE bm.market_type='TO_SCORE'
              AND bm.AD_match_id IS NOT NULL
              AND m.match_id IS NULL
            """
        ).fetchone()[0]

    return report
