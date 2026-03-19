"""Betfair trading integration via flumine framework.

Provides a scaffold for automated Betfair exchange trading of ATS
markets. Supports paper trading (simulation) and live execution.

This bypasses bookmaker account limitations by using the Betfair
exchange directly. Requires Betfair API credentials.

Usage:
    # Paper trading (simulation)
    from src.odds.betfair_trading import run_paper_trade
    results = run_paper_trade(predictions, season=2026, round_number=3)

    # Live trading (requires Betfair credentials)
    from src.odds.betfair_trading import run_live_trade
    results = run_live_trade(predictions, season=2026, round_number=3)

Setup:
    1. Create a Betfair account and get API app key
    2. Set environment variables:
       BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY, BETFAIR_CERT_PATH
    3. Install: pip install flumine betfairlightweight
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a trading session."""

    n_orders_placed: int
    n_orders_matched: int
    n_orders_failed: int
    total_staked: float
    orders: list[dict[str, Any]]


def get_betfair_client():
    """Create an authenticated Betfair API client.

    Reads credentials from environment variables:
    - BETFAIR_USERNAME
    - BETFAIR_PASSWORD
    - BETFAIR_APP_KEY
    - BETFAIR_CERT_PATH (directory containing client-2048.crt and client-2048.key)

    Returns
    -------
    betfairlightweight.APIClient
        Authenticated client.

    Raises
    ------
    EnvironmentError
        If credentials are not set.
    """
    import betfairlightweight

    username = os.environ.get("BETFAIR_USERNAME")
    password = os.environ.get("BETFAIR_PASSWORD")
    app_key = os.environ.get("BETFAIR_APP_KEY")
    cert_path = os.environ.get("BETFAIR_CERT_PATH")

    if not all([username, password, app_key]):
        raise EnvironmentError(
            "Betfair credentials not set. Required env vars: "
            "BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY"
        )

    client = betfairlightweight.APIClient(
        username=username,
        password=password,
        app_key=app_key,
        certs=cert_path,
    )
    client.login()
    LOGGER.info("Betfair API client authenticated as %s", username)
    return client


def find_ats_markets(
    client,
    event_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find Betfair ATS (TO_SCORE) markets for NRL matches.

    Parameters
    ----------
    client : betfairlightweight.APIClient
        Authenticated Betfair client.
    event_ids : list[str], optional
        Specific event IDs to filter. If None, finds all NRL events.

    Returns
    -------
    list[dict]
        Market info dicts with keys: market_id, event_name, n_runners.
    """
    import betfairlightweight.filters as filters

    # NRL event type ID on Betfair
    NRL_EVENT_TYPE_ID = "1477"

    market_filter = filters.market_filter(
        event_type_ids=[NRL_EVENT_TYPE_ID],
        market_type_codes=["ANYTIME_SCORE"],
    )

    if event_ids:
        market_filter["event_ids"] = event_ids

    catalogues = client.betting.list_market_catalogue(
        filter=market_filter,
        max_results=100,
        market_projection=["RUNNER_DESCRIPTION", "EVENT"],
    )

    markets = []
    for cat in catalogues:
        markets.append({
            "market_id": cat.market_id,
            "event_name": cat.event.name if cat.event else "",
            "market_name": cat.market_name,
            "n_runners": len(cat.runners) if cat.runners else 0,
            "runners": [
                {"selection_id": r.selection_id, "runner_name": r.runner_name}
                for r in (cat.runners or [])
            ],
        })

    LOGGER.info("Found %d ATS markets on Betfair", len(markets))
    return markets


def run_paper_trade(
    predictions: pd.DataFrame,
    season: int,
    round_number: int,
    bankroll: float = 10000.0,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.25,
) -> TradeResult:
    """Simulate trading on Betfair ATS markets (paper trade).

    Uses flumine's simulation mode to test order placement logic
    without risking real money.

    Parameters
    ----------
    predictions : pd.DataFrame
        From predict_round(). Must have: player_id, model_prob,
        betfair_closing_odds, edge, is_eligible.
    season : int
        Season year.
    round_number : int
        Round number.
    bankroll : float
        Simulated bankroll.
    min_edge : float
        Minimum edge to place a bet.
    kelly_fraction : float
        Kelly fraction for stake sizing.

    Returns
    -------
    TradeResult
        Paper trading results.
    """
    # Filter to eligible bets with edge
    bets = predictions[
        predictions["is_eligible"]
        & (predictions["edge"] >= min_edge)
    ].copy()

    if bets.empty:
        LOGGER.info("No eligible bets for paper trade")
        return TradeResult(0, 0, 0, 0.0, [])

    orders = []
    for _, row in bets.iterrows():
        edge = float(row["edge"])
        odds = float(row["betfair_closing_odds"])
        if odds <= 1 or edge <= 0:
            continue

        stake = kelly_fraction * edge / (odds - 1) * bankroll
        stake = min(stake, 0.05 * bankroll)  # Cap at 5%
        stake = max(stake, 5.0)  # Min $5

        orders.append({
            "player_id": int(row["player_id"]),
            "match_id": int(row["match_id"]),
            "side": "BACK",
            "odds": round(odds, 2),
            "stake": round(stake, 2),
            "model_prob": round(float(row["model_prob"]), 4),
            "edge": round(edge, 4),
            "status": "SIMULATED",
        })

    total_staked = sum(o["stake"] for o in orders)
    LOGGER.info(
        "Paper trade: %d orders, $%.0f total staked (%.1f%% of bankroll)",
        len(orders), total_staked, total_staked / bankroll * 100,
    )

    return TradeResult(
        n_orders_placed=len(orders),
        n_orders_matched=len(orders),  # All "matched" in simulation
        n_orders_failed=0,
        total_staked=total_staked,
        orders=orders,
    )


def run_live_trade(
    predictions: pd.DataFrame,
    season: int,
    round_number: int,
    bankroll: float = 10000.0,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.25,
    dry_run: bool = True,
) -> TradeResult:
    """Place live orders on Betfair ATS markets.

    CAUTION: This places real money bets on Betfair. Use dry_run=True
    (default) to verify orders before executing.

    Parameters
    ----------
    predictions : pd.DataFrame
        From predict_round().
    season : int
        Season year.
    round_number : int
        Round number.
    bankroll : float
        Current bankroll.
    min_edge : float
        Minimum edge.
    kelly_fraction : float
        Kelly fraction.
    dry_run : bool
        If True (default), log orders but don't place them.

    Returns
    -------
    TradeResult
        Trading results.
    """
    if dry_run:
        LOGGER.info("DRY RUN — orders will be logged but not placed")
        return run_paper_trade(
            predictions, season, round_number, bankroll, min_edge, kelly_fraction,
        )

    # Live trading requires credentials
    client = get_betfair_client()

    # Find ATS markets
    markets = find_ats_markets(client)
    if not markets:
        LOGGER.warning("No ATS markets found on Betfair")
        return TradeResult(0, 0, 0, 0.0, [])

    # TODO: Match predictions to Betfair runner selection_ids
    # TODO: Place back orders via client.betting.place_orders()
    # This requires mapping player_id -> Betfair selection_id
    # which is available via betfair_player_mapping table

    LOGGER.warning(
        "Live trading not fully implemented. Use paper trading for now. "
        "Found %d ATS markets to trade.", len(markets),
    )
    return run_paper_trade(
        predictions, season, round_number, bankroll, min_edge, kelly_fraction,
    )
