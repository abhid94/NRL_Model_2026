"""Generate bet recommendations from predictions.

Applies strategy selection, Kelly staking, and all risk constraints
from CLAUDE.md Section 11. Outputs a human-readable bet card.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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

LOGGER = logging.getLogger(__name__)


@dataclass
class BetCard:
    """A complete set of bet recommendations for a round."""

    season: int
    round_number: int
    bankroll: float
    bets: list[dict[str, Any]]
    total_staked: float
    exposure_pct: float
    n_matches_bet: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert bets to DataFrame."""
        if not self.bets:
            return pd.DataFrame()
        return pd.DataFrame(self.bets)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Bet Card: Season {self.season} Round {self.round_number} ===",
            f"Bankroll: ${self.bankroll:,.0f}",
            f"Bets: {len(self.bets)} across {self.n_matches_bet} matches",
            f"Total staked: ${self.total_staked:,.0f} ({self.exposure_pct:.1f}% of bankroll)",
            "",
        ]
        if self.bets:
            lines.append(
                f"{'Player':>8} {'Pos':>4} {'Odds':>6} {'Edge':>6} "
                f"{'Model%':>7} {'Mkt%':>6} {'Stake':>8}"
            )
            lines.append("-" * 55)
            for b in self.bets:
                lines.append(
                    f"{b['player_id']:>8} {b['position_code']:>4} "
                    f"{b['odds']:>6.2f} {b['edge']*100:>5.1f}% "
                    f"{b['model_prob']*100:>6.1f}% {b['implied_prob']*100:>5.1f}% "
                    f"${b['stake']:>7.0f}"
                )
        else:
            lines.append("  No bets recommended.")
        return "\n".join(lines)


def generate_bet_card(
    predictions: pd.DataFrame,
    bankroll: float = DEFAULT_INITIAL_BANKROLL,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    min_edge: float = MIN_EDGE_THRESHOLD,
    max_stake_pct: float = MAX_STAKE_PCT,
    max_round_exposure_pct: float = MAX_ROUND_EXPOSURE_PCT,
    max_bets_per_match: int = MAX_BETS_PER_MATCH,
    max_bets_per_round: int = MAX_BETS_PER_ROUND,
    min_stake: float = MIN_STAKE,
    flat_stake: float | None = None,
) -> BetCard:
    """Generate a bet card from predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output from predict_round(). Must contain: match_id, player_id,
        position_code, model_prob, betfair_implied_prob, betfair_closing_odds,
        edge, is_eligible.
    bankroll : float
        Current bankroll.
    kelly_fraction : float
        Fraction of Kelly criterion to use.
    min_edge : float
        Minimum edge to bet.
    max_stake_pct : float
        Max stake as fraction of bankroll.
    max_round_exposure_pct : float
        Max total exposure as fraction of bankroll.
    max_bets_per_match : int
        Max bets per match.
    max_bets_per_round : int
        Max bets per round.
    min_stake : float
        Minimum stake amount.
    flat_stake : float | None
        If set, use fixed stake instead of Kelly.

    Returns
    -------
    BetCard
        Complete bet recommendations.
    """
    if predictions.empty:
        return BetCard(
            season=0, round_number=0, bankroll=bankroll,
            bets=[], total_staked=0.0, exposure_pct=0.0, n_matches_bet=0,
        )

    season = int(predictions["season"].iloc[0]) if "season" in predictions.columns else 0
    round_number = int(predictions["round_number"].iloc[0]) if "round_number" in predictions.columns else 0

    # Filter to eligible bets with positive edge
    df = predictions[
        predictions["is_eligible"]
        & (predictions["edge"] >= min_edge)
    ].copy()

    if df.empty:
        return BetCard(
            season=season, round_number=round_number, bankroll=bankroll,
            bets=[], total_staked=0.0, exposure_pct=0.0, n_matches_bet=0,
        )

    # Calculate stakes
    max_bet = max_stake_pct * bankroll
    max_round = max_round_exposure_pct * bankroll

    if flat_stake is not None:
        df["stake"] = flat_stake
    else:
        # Kelly staking
        df["stake"] = df.apply(
            lambda row: _kelly_stake(
                row["edge"], row["betfair_closing_odds"],
                bankroll, kelly_fraction,
            ),
            axis=1,
        )
        # Per-bet cap
        df["stake"] = df["stake"].clip(upper=max_bet)

    # Drop below min_stake
    df = df[df["stake"] >= min_stake]

    # Per-match cap: keep highest edge per match
    df = df.sort_values("edge", ascending=False)
    match_counts: dict[int, int] = {}
    keep = []
    for idx, row in df.iterrows():
        mid = int(row["match_id"])
        count = match_counts.get(mid, 0)
        if count < max_bets_per_match:
            keep.append(idx)
            match_counts[mid] = count + 1
    df = df.loc[keep]

    # Per-round exposure cap
    total_stake = df["stake"].sum()
    if total_stake > max_round:
        scale = max_round / total_stake
        df["stake"] = df["stake"] * scale
        df = df[df["stake"] >= min_stake]

    # Bet count cap
    if len(df) > max_bets_per_round:
        df = df.head(max_bets_per_round)

    # Build bet records
    bets = []
    for _, row in df.iterrows():
        bets.append({
            "match_id": int(row["match_id"]),
            "player_id": int(row["player_id"]),
            "position_code": str(row.get("position_code", "")),
            "model_prob": round(float(row["model_prob"]), 4),
            "implied_prob": round(float(row["betfair_implied_prob"]), 4),
            "odds": round(float(row["betfair_closing_odds"]), 2),
            "edge": round(float(row["edge"]), 4),
            "stake": round(float(row["stake"]), 0),
        })

    total_staked = sum(b["stake"] for b in bets)
    exposure_pct = (total_staked / bankroll * 100) if bankroll > 0 else 0.0
    n_matches = len(set(b["match_id"] for b in bets))

    return BetCard(
        season=season,
        round_number=round_number,
        bankroll=bankroll,
        bets=bets,
        total_staked=total_staked,
        exposure_pct=exposure_pct,
        n_matches_bet=n_matches,
    )


def _kelly_stake(
    edge: float,
    odds: float,
    bankroll: float,
    kelly_fraction: float,
) -> float:
    """Compute fractional Kelly stake.

    Parameters
    ----------
    edge : float
        Model probability - implied probability.
    odds : float
        Decimal odds.
    bankroll : float
        Current bankroll.
    kelly_fraction : float
        Kelly fraction (e.g. 0.25 for quarter Kelly).

    Returns
    -------
    float
        Stake amount.
    """
    if edge <= 0 or odds <= 1:
        return 0.0
    kelly = kelly_fraction * edge / (odds - 1)
    return kelly * bankroll
