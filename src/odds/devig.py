"""Devigging methods for extracting true probabilities from bookmaker odds.

IMPORTANT: ATS (Anytime Try Scorer) is a non-mutually-exclusive market.
Multiple players can score tries in the same match. The standard approach
of normalizing to sum=1 (Shin, multiplicative) is WRONG for ATS.

For ATS, we use:
1. Binary devigging (Betfair) — back/lay midpoint per player
2. Per-bookmaker margin correction — empirical correction factor
3. Per-market overround correction — actual overround → correction factor

For mutually exclusive markets (match winner, first try scorer), Shin's
method IS appropriate and available.

Usage:
    from src.odds.devig import devig_binary, devig_bookmaker_ats
    true_prob = devig_binary(back_odds=3.0, lay_odds=3.2)
    true_prob = devig_bookmaker_ats(3.0, market_overround=7.8, n_players=34)
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

# Historical average: ~6.9 unique try scorers per NRL match (2024-2025)
DEFAULT_EXPECTED_TRY_SCORERS: float = 6.9


def devig_binary(
    back_odds: float,
    lay_odds: float | None = None,
) -> float:
    """Devig a binary market (player scores / doesn't score).

    Best method for Betfair ATS where we have both back and lay prices.
    Uses the midpoint of back and lay implied probabilities.

    Parameters
    ----------
    back_odds : float
        Decimal back odds (e.g., 3.0 → 33.3% implied).
    lay_odds : float, optional
        Decimal lay odds (e.g., 3.2 → lay implies 31.25% chance of scoring).
        If None, applies ~2% Betfair commission estimate.

    Returns
    -------
    float
        True implied probability for the selection.
    """
    if back_odds <= 1.0 or not np.isfinite(back_odds):
        return np.nan

    back_implied = 1.0 / back_odds

    if lay_odds is not None and lay_odds > 1.0 and np.isfinite(lay_odds):
        # Lay implied = probability from the layer's perspective
        # If you lay at 3.2, you're saying there's a 1/3.2 = 31.25% chance
        lay_implied = 1.0 / lay_odds
        # Midpoint: average of the two
        true_prob = (back_implied + lay_implied) / 2.0
    else:
        # No lay odds — assume ~2% per side (typical Betfair binary)
        true_prob = back_implied * 0.98

    return float(np.clip(true_prob, 0.0, 1.0))


def devig_bookmaker_ats(
    decimal_odds: float,
    market_overround: float | None = None,
    n_players: int = 34,
    expected_try_scorers: float = DEFAULT_EXPECTED_TRY_SCORERS,
    bookmaker_correction: float | None = None,
) -> float:
    """Devig a single bookmaker ATS selection.

    For bookmaker odds (where we don't have lay prices), estimates true
    probability by removing margin. Two approaches:

    1. If market_overround is known, use multiplicative correction:
       correction = expected / overround → true_prob = naive × correction
    2. If bookmaker_correction is provided, use empirical factor:
       true_prob = naive × correction

    Parameters
    ----------
    decimal_odds : float
        Bookmaker decimal odds for the player.
    market_overround : float, optional
        Sum of 1/odds for all players in this match at this bookmaker.
    n_players : int
        Number of players in the bookmaker market.
    expected_try_scorers : float
        Expected try scorers per match.
    bookmaker_correction : float, optional
        Empirical margin correction factor (e.g., 0.88).

    Returns
    -------
    float
        True implied probability.
    """
    if decimal_odds <= 1.0 or not np.isfinite(decimal_odds):
        return np.nan

    naive_implied = 1.0 / decimal_odds

    if bookmaker_correction is not None:
        return float(np.clip(naive_implied * bookmaker_correction, 0.0, 1.0))

    if market_overround is not None and market_overround > 0:
        # Margin = (overround - expected) / expected
        # Per-selection correction = expected / overround
        correction = expected_try_scorers / market_overround
        return float(np.clip(naive_implied * correction, 0.0, 1.0))

    # Fallback: assume ~12% average margin for AU bookmakers
    return float(np.clip(naive_implied * 0.88, 0.0, 1.0))


def devig_betfair_market(
    back_odds: np.ndarray,
    lay_odds: np.ndarray | None = None,
) -> np.ndarray:
    """Devig all players in a Betfair ATS market using binary devigging.

    Parameters
    ----------
    back_odds : np.ndarray
        Back odds for each player.
    lay_odds : np.ndarray, optional
        Lay odds for each player. If None, uses ~2% estimate.

    Returns
    -------
    np.ndarray
        True probabilities. NaN where odds are invalid.
    """
    result = np.full(len(back_odds), np.nan)
    for i in range(len(back_odds)):
        lay = lay_odds[i] if lay_odds is not None else None
        result[i] = devig_binary(float(back_odds[i]), float(lay) if lay is not None else None)
    return result


def devig_shin(
    decimal_odds: list[float] | np.ndarray,
) -> np.ndarray:
    """Shin's method for mutually exclusive markets ONLY.

    ONLY use for markets where exactly one outcome wins (match winner,
    first try scorer). Do NOT use for ATS (anytime try scorer).

    Parameters
    ----------
    decimal_odds : list[float] or np.ndarray
        Decimal odds for all selections (must be mutually exclusive).

    Returns
    -------
    np.ndarray
        True probabilities (sum to 1.0). NaN for invalid odds.
    """
    import shin as shin_pkg

    odds = np.asarray(decimal_odds, dtype=float)
    valid_mask = (odds > 1.0) & np.isfinite(odds)
    valid_odds = odds[valid_mask]

    if len(valid_odds) < 2:
        return np.full(len(odds), np.nan)

    try:
        true_probs = shin_pkg.calculate_implied_probabilities(valid_odds.tolist())
        result = np.full(len(odds), np.nan)
        result[valid_mask] = np.array(true_probs)
        return result
    except Exception as exc:
        LOGGER.warning("Shin's method failed (%s), falling back to multiplicative", exc)
        return devig_multiplicative(decimal_odds)


def devig_multiplicative(
    decimal_odds: list[float] | np.ndarray,
) -> np.ndarray:
    """Multiplicative devigging for mutually exclusive markets.

    Divides each implied probability by the total overround so they
    sum to 1.0. Only correct for mutually exclusive outcomes.

    Parameters
    ----------
    decimal_odds : list[float] or np.ndarray
        Decimal odds for all selections.

    Returns
    -------
    np.ndarray
        True probabilities (sum to 1.0). NaN for invalid odds.
    """
    odds = np.asarray(decimal_odds, dtype=float)
    valid_mask = (odds > 1.0) & np.isfinite(odds)
    valid_odds = odds[valid_mask]

    if len(valid_odds) == 0:
        return np.full(len(odds), np.nan)

    implied = 1.0 / valid_odds
    overround = implied.sum()

    if overround <= 0:
        result = np.full(len(odds), np.nan)
        result[valid_mask] = implied
        return result

    result = np.full(len(odds), np.nan)
    result[valid_mask] = implied / overround
    return result


def compute_overround(decimal_odds: list[float] | np.ndarray) -> float:
    """Compute the overround (sum of implied probs) for a market.

    Parameters
    ----------
    decimal_odds : list[float] or np.ndarray
        Decimal odds for all selections.

    Returns
    -------
    float
        Overround. For ATS markets, values of 4-6 are typical (Betfair)
        and 7-9 for traditional bookmakers.
    """
    odds = np.asarray(decimal_odds, dtype=float)
    valid = odds[(odds > 1.0) & np.isfinite(odds)]
    if len(valid) == 0:
        return 0.0
    return float(np.sum(1.0 / valid))
