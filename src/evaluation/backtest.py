"""Walk-forward backtest engine for ATS betting strategies.

Strategy-agnostic: runs any BaseStrategy through historical data,
applying realistic staking constraints from CLAUDE.md Section 11.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_INITIAL_BANKROLL,
    DEFAULT_KELLY_FRACTION,
    ELIGIBLE_POSITION_CODES,
    MAX_BETS_PER_MATCH,
    MAX_BETS_PER_ROUND,
    MAX_ROUND_EXPOSURE_PCT,
    MAX_STAKE_PCT,
    MIN_EDGE_THRESHOLD,
    MIN_STAKE,
)
from src.models.baseline import BaseModel, BaseStrategy, BetRecommendation

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """All staking parameters. Defaults match CLAUDE.md Section 11."""

    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL
    kelly_fraction: float = DEFAULT_KELLY_FRACTION
    max_stake_pct: float = MAX_STAKE_PCT
    max_round_exposure_pct: float = MAX_ROUND_EXPOSURE_PCT
    max_bets_per_round: int = MAX_BETS_PER_ROUND
    max_bets_per_match: int = MAX_BETS_PER_MATCH
    min_stake: float = MIN_STAKE
    min_edge: float = MIN_EDGE_THRESHOLD
    flat_stake: float | None = None  # If set, use fixed stake instead of Kelly


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    """Results for a single round."""

    season: int
    round_number: int
    bets: list[dict[str, Any]]
    n_bets: int
    total_staked: float
    total_payout: float
    profit: float
    bankroll_after: float
    n_eligible: int = 0
    n_with_odds: int = 0


@dataclass
class BacktestResult:
    """Complete backtest result across all rounds."""

    strategy_name: str
    model_name: str
    config: BacktestConfig
    round_results: list[RoundResult] = field(default_factory=list)

    @property
    def total_staked(self) -> float:
        return sum(r.total_staked for r in self.round_results)

    @property
    def total_payout(self) -> float:
        return sum(r.total_payout for r in self.round_results)

    @property
    def total_profit(self) -> float:
        return self.total_payout - self.total_staked

    @property
    def roi(self) -> float:
        if self.total_staked == 0:
            return 0.0
        return self.total_profit / self.total_staked

    @property
    def n_bets(self) -> int:
        return sum(r.n_bets for r in self.round_results)

    def to_bet_dataframe(self) -> pd.DataFrame:
        """Flatten all bets into a DataFrame."""
        rows = []
        for rr in self.round_results:
            for bet in rr.bets:
                rows.append({
                    "season": rr.season,
                    "round_number": rr.round_number,
                    **bet,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def to_round_dataframe(self) -> pd.DataFrame:
        """One row per round with summary stats."""
        rows = []
        for rr in self.round_results:
            rows.append({
                "season": rr.season,
                "round_number": rr.round_number,
                "n_bets": rr.n_bets,
                "total_staked": rr.total_staked,
                "total_payout": rr.total_payout,
                "profit": rr.profit,
                "bankroll_after": rr.bankroll_after,
                "n_eligible": rr.n_eligible,
                "n_with_odds": rr.n_with_odds,
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """Return summary metrics."""
        round_df = self.to_round_dataframe()
        bet_df = self.to_bet_dataframe()
        result: dict[str, Any] = {
            "strategy": self.strategy_name,
            "model": self.model_name,
            "n_rounds": len(self.round_results),
            "n_bets": self.n_bets,
            "total_staked": self.total_staked,
            "total_payout": self.total_payout,
            "profit": self.total_profit,
            "roi": self.roi,
            "final_bankroll": (
                self.round_results[-1].bankroll_after
                if self.round_results else self.config.initial_bankroll
            ),
        }
        if not bet_df.empty and "won" in bet_df.columns:
            result["hit_rate"] = float(bet_df["won"].mean())
            result["avg_odds"] = float(bet_df["odds"].mean())
            result["avg_edge"] = float(bet_df["edge"].mean())
        if not round_df.empty:
            pnl = round_df["profit"].cumsum()
            running_max = pnl.cummax()
            drawdown = (running_max - pnl).max()
            result["max_drawdown"] = float(drawdown)
        return result


# ---------------------------------------------------------------------------
# Staking engine
# ---------------------------------------------------------------------------

def apply_staking(
    bets: list[BetRecommendation],
    bankroll: float,
    config: BacktestConfig,
) -> list[BetRecommendation]:
    """Apply all staking constraints in order.

    Constraint order:
    1. Kelly sizing
    2. Per-bet cap (max_stake_pct * bankroll)
    3. Drop bets below min_stake
    4. Per-match cap (keep highest edge, max max_bets_per_match per match)
    5. Per-round exposure cap (scale down if total > max_round_exposure_pct * bankroll)
    6. Bet count cap (keep highest edge, max max_bets_per_round)

    Parameters
    ----------
    bets : list[BetRecommendation]
        Raw bet recommendations (stake not yet set).
    bankroll : float
        Current bankroll.
    config : BacktestConfig
        Staking parameters.

    Returns
    -------
    list[BetRecommendation]
        Bets with stakes set, filtered by constraints.
    """
    if not bets or bankroll <= 0:
        return []

    max_round = config.max_round_exposure_pct * bankroll

    if config.flat_stake is not None:
        # Flat-stake mode: fixed amount per bet, skip Kelly and per-bet cap
        for bet in bets:
            if bet.edge <= 0 or bet.odds <= 1:
                bet.stake = 0.0
            else:
                bet.stake = config.flat_stake
    else:
        # Kelly sizing + per-bet cap
        max_bet = config.max_stake_pct * bankroll
        # 1. Kelly sizing
        for bet in bets:
            if bet.edge <= 0 or bet.odds <= 1:
                bet.stake = 0.0
                continue
            kelly = config.kelly_fraction * bet.edge / (bet.odds - 1)
            bet.stake = kelly * bankroll

        # 2. Per-bet cap
        for bet in bets:
            bet.stake = min(bet.stake, max_bet)

    # 3. Drop below min_stake
    bets = [b for b in bets if b.stake >= config.min_stake]

    # 4. Per-match cap: keep highest edge per match
    bets.sort(key=lambda b: b.edge, reverse=True)
    match_counts: dict[int, int] = {}
    filtered = []
    for bet in bets:
        count = match_counts.get(bet.match_id, 0)
        if count < config.max_bets_per_match:
            filtered.append(bet)
            match_counts[bet.match_id] = count + 1
    bets = filtered

    # 5. Per-round exposure cap
    total_stake = sum(b.stake for b in bets)
    if total_stake > max_round:
        scale = max_round / total_stake
        for bet in bets:
            bet.stake *= scale
        # Re-filter below min_stake after scaling
        bets = [b for b in bets if b.stake >= config.min_stake]

    # 6. Bet count cap
    if len(bets) > config.max_bets_per_round:
        bets.sort(key=lambda b: b.edge, reverse=True)
        bets = bets[: config.max_bets_per_round]

    return bets


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    feature_store: pd.DataFrame,
    strategy: BaseStrategy,
    model: BaseModel | None = None,
    config: BacktestConfig | None = None,
    seasons: list[int] | None = None,
    min_round: int = 3,
) -> BacktestResult:
    """Run walk-forward backtest on the feature store.

    For each round R in each season:
    1. Training data = all rows where (season < current) OR (season == current AND round < R)
    2. Prediction data = rows where season == current AND round == R
    3. Fit model on training, predict on round, strategy selects bets
    4. Apply staking, resolve outcomes, update bankroll

    Parameters
    ----------
    feature_store : pd.DataFrame
        Combined feature store (must have season, round_number, scored_try columns).
    strategy : BaseStrategy
        Betting strategy.
    model : BaseModel, optional
        Model to fit/predict. None for rule-based strategies.
    config : BacktestConfig, optional
        Staking parameters. Defaults used if None.
    seasons : list[int], optional
        Seasons to backtest. Defaults to all seasons in data.
    min_round : int
        Skip rounds below this (need history for rolling features).

    Returns
    -------
    BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    if seasons is None:
        seasons = sorted(feature_store["season"].unique())

    model_name = type(model).__name__ if model else "None"
    result = BacktestResult(
        strategy_name=strategy.name,
        model_name=model_name,
        config=config,
    )
    bankroll = config.initial_bankroll

    for season in seasons:
        season_data = feature_store[feature_store["season"] == season]
        rounds = sorted(season_data["round_number"].unique())

        for rnd in rounds:
            if rnd < min_round:
                continue

            # Training data: all prior data
            train_mask = (
                (feature_store["season"] < season)
                | ((feature_store["season"] == season) & (feature_store["round_number"] < rnd))
            )
            train_df = feature_store[train_mask]

            # Prediction data: current round
            pred_mask = (feature_store["season"] == season) & (feature_store["round_number"] == rnd)
            pred_df = feature_store[pred_mask].copy()

            if pred_df.empty or train_df.empty:
                continue

            # Fit model if provided
            if model is not None and "scored_try" in train_df.columns:
                try:
                    model.fit(train_df, train_df["scored_try"].values)
                except Exception as e:
                    LOGGER.warning("Model fit failed round %d: %s", rnd, e)
                    continue

            # Strategy selects bets
            try:
                bets = strategy.select_bets(pred_df, model=model)
            except Exception as e:
                LOGGER.warning("Strategy failed round %d: %s", rnd, e)
                bets = []

            # Apply staking constraints
            bets = apply_staking(bets, bankroll, config)

            # Resolve outcomes
            bet_records = []
            total_staked = 0.0
            total_payout = 0.0

            for bet in bets:
                actual_row = pred_df[
                    (pred_df["match_id"] == bet.match_id)
                    & (pred_df["player_id"] == bet.player_id)
                ]
                if actual_row.empty:
                    continue

                scored = int(actual_row.iloc[0]["scored_try"])
                payout = bet.stake * bet.odds if scored else 0.0

                total_staked += bet.stake
                total_payout += payout

                bet_records.append({
                    "match_id": bet.match_id,
                    "player_id": bet.player_id,
                    "position_code": bet.position_code,
                    "model_prob": bet.model_prob,
                    "implied_prob": bet.implied_prob,
                    "odds": bet.odds,
                    "edge": bet.edge,
                    "stake": bet.stake,
                    "payout": payout,
                    "won": scored,
                })

            profit = total_payout - total_staked
            bankroll += profit

            # Count eligible players
            n_eligible = pred_df["position_code"].isin(ELIGIBLE_POSITION_CODES).sum()
            n_with_odds = (
                pred_df["betfair_implied_prob"].notna()
                & (pred_df["betfair_implied_prob"] > 0)
                & pred_df["position_code"].isin(ELIGIBLE_POSITION_CODES)
            ).sum()

            round_result = RoundResult(
                season=season,
                round_number=rnd,
                bets=bet_records,
                n_bets=len(bet_records),
                total_staked=total_staked,
                total_payout=total_payout,
                profit=profit,
                bankroll_after=bankroll,
                n_eligible=int(n_eligible),
                n_with_odds=int(n_with_odds),
            )
            result.round_results.append(round_result)

    LOGGER.info(
        "Backtest complete: %s/%s — %d rounds, %d bets, ROI=%.2f%%",
        strategy.name, model_name,
        len(result.round_results), result.n_bets, result.roi * 100,
    )
    return result


def compare_backtests(results: list[BacktestResult]) -> pd.DataFrame:
    """Compare multiple backtest results side-by-side.

    Parameters
    ----------
    results : list[BacktestResult]
        Backtest results to compare.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per strategy.
    """
    rows = []
    for r in results:
        s = r.summary()
        rows.append(s)
    return pd.DataFrame(rows)
