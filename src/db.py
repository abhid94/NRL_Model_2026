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
