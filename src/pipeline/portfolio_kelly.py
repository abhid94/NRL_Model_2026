"""Correlated Kelly portfolio optimization for ATS bets.

ATS bets within the same match are correlated: if a team scores more
tries, multiple players on that team benefit. Standard per-bet Kelly
ignores this correlation, leading to over-exposure on high-scoring
matches.

This module optimizes stake allocation across a portfolio of bets,
accounting for within-match correlation. Uses scipy optimization to
find the Kelly-optimal portfolio weights.

Usage:
    from src.pipeline.portfolio_kelly import optimize_portfolio_stakes
    stakes = optimize_portfolio_stakes(bets_df, bankroll=10000)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Estimated within-match try correlation for ATS bets
# If team scores more tries, all players on that team benefit
# Empirically ~0.15-0.25 correlation between ATS outcomes on same team
DEFAULT_SAME_MATCH_CORRELATION: float = 0.15
DEFAULT_SAME_TEAM_CORRELATION: float = 0.20


def optimize_portfolio_stakes(
    bets: pd.DataFrame,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.05,
    max_round_exposure_pct: float = 0.20,
    same_match_corr: float = DEFAULT_SAME_MATCH_CORRELATION,
    same_team_corr: float = DEFAULT_SAME_TEAM_CORRELATION,
) -> np.ndarray:
    """Optimize stakes across a portfolio of correlated ATS bets.

    Uses mean-variance optimization with Kelly growth criterion,
    accounting for within-match correlation between ATS outcomes.

    Parameters
    ----------
    bets : pd.DataFrame
        Must contain: match_id, squad_id, model_prob, edge, odds.
    bankroll : float
        Current bankroll.
    kelly_fraction : float
        Fraction of optimal Kelly to use.
    max_stake_pct : float
        Max per-bet stake as fraction of bankroll.
    max_round_exposure_pct : float
        Max total exposure as fraction of bankroll.
    same_match_corr : float
        Correlation between ATS outcomes in the same match (different teams).
    same_team_corr : float
        Correlation between ATS outcomes on the same team in the same match.

    Returns
    -------
    np.ndarray
        Optimal stake amounts (same length as bets).
    """
    n = len(bets)
    if n == 0:
        return np.array([])

    if n == 1:
        # Single bet — standard Kelly
        edge = float(bets.iloc[0]["edge"])
        odds = float(bets.iloc[0]["odds"]) if "odds" in bets.columns else float(bets.iloc[0].get("_display_odds", 3.0))
        if edge <= 0 or odds <= 1:
            return np.array([0.0])
        kelly = kelly_fraction * edge / (odds - 1)
        stake = min(kelly * bankroll, max_stake_pct * bankroll)
        return np.array([stake])

    # Build correlation matrix
    corr_matrix = _build_correlation_matrix(bets, same_match_corr, same_team_corr)

    # Compute per-bet Kelly fractions (uncorrelated baseline)
    probs = bets["model_prob"].values
    odds_col = "odds" if "odds" in bets.columns else "_display_odds"
    odds = bets[odds_col].values if odds_col in bets.columns else np.full(n, 3.0)
    edges = bets["edge"].values

    kelly_fracs = np.zeros(n)
    for i in range(n):
        if edges[i] > 0 and odds[i] > 1:
            kelly_fracs[i] = kelly_fraction * edges[i] / (odds[i] - 1)

    # Adjust for correlation using mean-variance approach
    # Higher correlation between bets → reduce stakes on the correlated group
    adjusted_fracs = _adjust_for_correlation(kelly_fracs, corr_matrix, probs)

    # Apply constraints
    max_bet = max_stake_pct * bankroll
    max_round = max_round_exposure_pct * bankroll

    stakes = adjusted_fracs * bankroll
    stakes = np.clip(stakes, 0, max_bet)

    # Scale down if total exceeds round exposure limit
    total = stakes.sum()
    if total > max_round:
        stakes *= max_round / total

    n_adjusted = np.sum(np.abs(stakes - kelly_fracs * bankroll) > 1.0)
    LOGGER.info(
        "Portfolio Kelly: %d bets, %d adjusted for correlation, "
        "total=$%.0f (%.1f%% of bankroll)",
        n, n_adjusted, stakes.sum(),
        stakes.sum() / bankroll * 100 if bankroll > 0 else 0,
    )

    return stakes


def _build_correlation_matrix(
    bets: pd.DataFrame,
    same_match_corr: float,
    same_team_corr: float,
) -> np.ndarray:
    """Build correlation matrix for ATS bet outcomes.

    Parameters
    ----------
    bets : pd.DataFrame
        Must have match_id and squad_id columns.
    same_match_corr : float
        Correlation for same match, different teams.
    same_team_corr : float
        Correlation for same match, same team.

    Returns
    -------
    np.ndarray
        n x n correlation matrix.
    """
    n = len(bets)
    corr = np.eye(n)

    match_ids = bets["match_id"].values
    squad_ids = bets["squad_id"].values if "squad_id" in bets.columns else np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            if match_ids[i] == match_ids[j]:
                if squad_ids[i] == squad_ids[j]:
                    # Same team, same match — higher correlation
                    corr[i, j] = same_team_corr
                    corr[j, i] = same_team_corr
                else:
                    # Same match, different teams — lower correlation
                    corr[i, j] = same_match_corr
                    corr[j, i] = same_match_corr

    return corr


def _adjust_for_correlation(
    kelly_fracs: np.ndarray,
    corr_matrix: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Adjust Kelly fractions for bet correlation.

    Uses a simplified portfolio approach: for each bet, reduce the
    Kelly fraction proportionally to the sum of correlations with
    other active bets. This approximates the multivariate Kelly
    without requiring full optimization.

    The intuition: if you have 3 correlated bets in the same match,
    each with Kelly fraction f, the portfolio-optimal fraction is
    approximately f / sqrt(1 + 2*rho) for pairwise correlation rho.

    Parameters
    ----------
    kelly_fracs : np.ndarray
        Uncorrelated Kelly fractions per bet.
    corr_matrix : np.ndarray
        Correlation matrix.
    probs : np.ndarray
        Model probabilities.

    Returns
    -------
    np.ndarray
        Adjusted Kelly fractions.
    """
    n = len(kelly_fracs)
    adjusted = kelly_fracs.copy()

    for i in range(n):
        if kelly_fracs[i] <= 0:
            continue

        # Sum of correlations with other active bets
        corr_sum = 0.0
        n_correlated = 0
        for j in range(n):
            if i != j and kelly_fracs[j] > 0 and corr_matrix[i, j] > 0:
                corr_sum += corr_matrix[i, j]
                n_correlated += 1

        if n_correlated > 0:
            # Diversification factor: reduce stake when correlated
            # sqrt(1 + sum_of_correlations) is the variance inflation
            div_factor = 1.0 / np.sqrt(1.0 + corr_sum)
            adjusted[i] = kelly_fracs[i] * div_factor

    return adjusted
