"""Tests for odds-api.io client module."""

from __future__ import annotations

import pytest

from src.odds.odds_api_io import (
    NRL_LEAGUE_SLUG,
    _parse_price,
    parse_event_odds,
    parse_match_odds,
)


class TestParsePrice:
    """Tests for _parse_price helper."""

    def test_float_string(self) -> None:
        assert _parse_price("2.55") == 2.55

    def test_int_string(self) -> None:
        assert _parse_price("10") == 10.0

    def test_float_value(self) -> None:
        assert _parse_price(1.91) == 1.91

    def test_none(self) -> None:
        assert _parse_price(None) is None

    def test_empty_string(self) -> None:
        assert _parse_price("") is None

    def test_invalid_string(self) -> None:
        assert _parse_price("N/A") is None


class TestNRLLeagueSlug:
    """Verify the discovered NRL league slug is set."""

    def test_league_slug_set(self) -> None:
        assert NRL_LEAGUE_SLUG == "rugby-league-nrl-premiership"


class TestParseMatchOdds:
    """Tests for parse_match_odds (Bet365 match-level markets)."""

    @pytest.fixture()
    def sample_response(self) -> dict:
        """Realistic odds-api.io response for an NRL match."""
        return {
            "id": 65687314,
            "home": "New Zealand Warriors",
            "away": "Sydney Roosters",
            "date": "2026-03-06T07:00:00Z",
            "status": "pending",
            "bookmakers": {
                "Bet365": [
                    {
                        "name": "Game Betting 2-Way",
                        "updatedAt": "2026-03-05T11:14:34.628Z",
                        "odds": [
                            {"label": "To Win (1)", "home": "2.55"},
                            {"label": "To Win (2)", "away": "1.52"},
                            {"label": "Handicap (1) (5.5)", "hdp": 5.5, "home": "1.91"},
                            {"label": "Total (1) (40.5)", "hdp": 40.5, "home": "1.93", "away": "1.89"},
                        ],
                    },
                    {
                        "name": "Handicap 2-Way",
                        "updatedAt": "2026-03-05T11:14:34.628Z",
                        "odds": [
                            {"label": "1 (-4.5)", "hdp": -4.5, "home": "3.50", "away": "1.85"},
                            {"label": "1 (4.5)", "hdp": 4.5, "home": "1.91", "away": "1.28"},
                        ],
                    },
                ],
            },
        }

    def test_parse_count(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        assert len(records) == 6

    def test_match_id_set(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        assert all(r["match_id"] == 100 for r in records)

    def test_bookmaker_normalized(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        assert all(r["bookmaker"] == "bet365" for r in records)

    def test_h2h_home_odds(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        h2h_home = [r for r in records if r["label"] == "To Win (1)"]
        assert len(h2h_home) == 1
        assert h2h_home[0]["home_odds"] == 2.55
        assert h2h_home[0]["away_odds"] is None

    def test_h2h_away_odds(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        h2h_away = [r for r in records if r["label"] == "To Win (2)"]
        assert len(h2h_away) == 1
        assert h2h_away[0]["away_odds"] == 1.52

    def test_handicap_included(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        handicap = [r for r in records if r["market"] == "Handicap 2-Way"]
        assert len(handicap) == 2
        assert handicap[0]["handicap"] == -4.5

    def test_total_line(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        total = [r for r in records if "Total" in r["label"]]
        assert len(total) == 1
        assert total[0]["handicap"] == 40.5
        assert total[0]["home_odds"] == 1.93
        assert total[0]["away_odds"] == 1.89

    def test_odds_source_tag(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=100)
        assert all(r["odds_source"] == "odds_api_io" for r in records)

    def test_empty_bookmakers(self) -> None:
        response = {"id": 1, "home": "A", "away": "B", "bookmakers": {}}
        records = parse_match_odds(response, match_id=100)
        assert records == []

    def test_none_match_id(self, sample_response: dict) -> None:
        records = parse_match_odds(sample_response, match_id=None)
        assert all(r["match_id"] is None for r in records)


class TestParseEventOdds:
    """Tests for parse_event_odds (ATS market parser)."""

    @pytest.fixture()
    def ats_response_yes_no(self) -> dict:
        """Hypothetical ATS response with Yes/No structure."""
        return {
            "id": 99999,
            "home": "Brisbane Broncos",
            "away": "Penrith Panthers",
            "bookmakers": [
                {
                    "name": "Bet365",
                    "markets": [
                        {
                            "name": "Anytime Tryscorer",
                            "outcomes": [
                                {"name": "Yes", "description": "Reece Walsh", "price": 2.50},
                                {"name": "No", "description": "Reece Walsh", "price": 1.55},
                                {"name": "Yes", "description": "Nathan Cleary", "price": 5.00},
                            ],
                        },
                    ],
                },
            ],
        }

    @pytest.fixture()
    def ats_response_dict_style(self) -> dict:
        """Hypothetical ATS response with dict-style bookmakers."""
        return {
            "id": 99998,
            "home": "Melbourne Storm",
            "away": "Sydney Roosters",
            "bookmakers": {
                "Bet365": [
                    {
                        "name": "Anytime Tryscorer",
                        "odds": [
                            {"name": "Yes", "description": "Ryan Papenhuyzen", "price": 1.80},
                            {"name": "No", "description": "Ryan Papenhuyzen", "price": 2.10},
                        ],
                    },
                ],
            },
        }

    def test_yes_no_structure(self, ats_response_yes_no: dict) -> None:
        records = parse_event_odds(ats_response_yes_no, match_id=200)
        # Should only get "Yes" outcomes
        assert len(records) == 2
        names = {r["player_name_raw"] for r in records}
        assert names == {"Reece Walsh", "Nathan Cleary"}

    def test_implied_prob(self, ats_response_yes_no: dict) -> None:
        records = parse_event_odds(ats_response_yes_no, match_id=200)
        walsh = [r for r in records if r["player_name_raw"] == "Reece Walsh"][0]
        assert walsh["decimal_odds"] == 2.50
        assert walsh["implied_probability"] == 0.4

    def test_dict_style_bookmakers(self, ats_response_dict_style: dict) -> None:
        records = parse_event_odds(ats_response_dict_style, match_id=300)
        assert len(records) == 1
        assert records[0]["player_name_raw"] == "Ryan Papenhuyzen"
        assert records[0]["decimal_odds"] == 1.80

    def test_no_ats_market(self) -> None:
        response = {
            "id": 1,
            "home": "A",
            "away": "B",
            "bookmakers": [
                {"name": "Bet365", "markets": [{"name": "Match Winner", "outcomes": []}]},
            ],
        }
        records = parse_event_odds(response, match_id=100)
        assert records == []

    def test_bookmaker_normalized_to_bet365(self, ats_response_yes_no: dict) -> None:
        records = parse_event_odds(ats_response_yes_no, match_id=200)
        assert all(r["bookmaker"] == "bet365" for r in records)

    def test_odds_source_tagged(self, ats_response_yes_no: dict) -> None:
        records = parse_event_odds(ats_response_yes_no, match_id=200)
        assert all(r["odds_source"] == "odds_api_io" for r in records)
