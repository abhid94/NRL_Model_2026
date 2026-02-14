"""Tests for database helper utilities."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import db


@pytest.fixture()
def connection() -> sqlite3.Connection:
    """Open a connection to the test database."""
    connection = db.get_connection()
    yield connection
    connection.close()


@pytest.fixture()
def year() -> int:
    """Return a season year present in the test database."""
    return 2024


def test_create_cleansed_views(connection: sqlite3.Connection, year: int) -> None:
    """Cleansed views should exist and populate expected columns."""
    views = db.create_cleansed_views(connection, year)

    assert "players" in views
    assert "betfair_markets" in views

    if "team_lists" in views:
        team_lists_view = views["team_lists"]
        cursor = connection.execute(
            f"SELECT position_code, position_label FROM {team_lists_view} LIMIT 5"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[1] is not None

    betfair_view = views["betfair_markets"]
    cursor = connection.execute(
        f"""
        SELECT last_preplay_price_cleaned
        FROM {betfair_view}
        WHERE market_type='TO_SCORE'
        LIMIT 10
        """
    )
    assert cursor.fetchone() is not None


def test_data_quality_report(connection: sqlite3.Connection, year: int) -> None:
    """Data quality report returns expected keys and ranges."""
    report = db.data_quality_report(connection, year)

    assert report["players_total"] >= report["players_unique"]
    assert report["betfair_last_preplay_empty"] is not None
    assert report["betfair_unmapped_runners"] is not None
    assert report["try_rate"] is not None
    assert 0 <= report["try_rate"] <= 1
    assert report["rounds_with_fewer_matches"]


def test_referential_integrity_report(connection: sqlite3.Connection, year: int) -> None:
    """Referential integrity report should include expected keys."""
    report = db.referential_integrity_report(connection, year)

    assert "player_stats_players" in report
    assert "player_stats_matches" in report
    assert "betfair_ad_player" in report
    assert "betfair_ad_match" in report

    assert report["player_stats_players"] == 0
    assert report["player_stats_matches"] == 0
