"""Evaluation metrics for ATS model assessment.

Pure, stateless functions: arrays in, numbers out.
Covers discrimination, calibration, and economic metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC AUC.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        ROC AUC score, or NaN if undefined.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Precision-Recall AUC.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_prob : array-like
        Predicted probabilities.

    Returns
    -------
    float
        PR AUC score, or NaN if undefined.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower is better).

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_prob : array-like
        Predicted probabilities.

    Returns
    -------
    float
    """
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob)))


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute log loss.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_prob : array-like
        Predicted probabilities (clipped to [1e-15, 1-1e-15]).

    Returns
    -------
    float
    """
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    return float(log_loss(np.asarray(y_true), y_prob))


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        Weighted average absolute calibration error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += (n / total) * abs(avg_true - avg_pred)
    return float(ece)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration curve data.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    pd.DataFrame
        Columns: bin_mid, avg_predicted, avg_actual, count
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin_mid": (lo + hi) / 2,
            "avg_predicted": float(y_prob[mask].mean()),
            "avg_actual": float(y_true[mask].mean()),
            "count": n,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Economic metrics
# ---------------------------------------------------------------------------

def compute_roi(stakes: np.ndarray, payouts: np.ndarray) -> float:
    """Compute Return on Investment.

    Parameters
    ----------
    stakes : array-like
        Amounts wagered.
    payouts : array-like
        Amounts received (0 for losses, stake * odds for wins).

    Returns
    -------
    float
        ROI as a decimal (0.05 = 5%).
    """
    stakes = np.asarray(stakes, dtype=float)
    payouts = np.asarray(payouts, dtype=float)
    total_staked = stakes.sum()
    if total_staked == 0:
        return 0.0
    return float((payouts.sum() - total_staked) / total_staked)


def compute_clv(
    model_prob: np.ndarray,
    implied_prob: np.ndarray,
) -> float:
    """Compute average Closing Line Value.

    CLV = mean(model_prob - implied_prob) for selected bets.

    Parameters
    ----------
    model_prob : array-like
        Model predicted probabilities for selected bets.
    implied_prob : array-like
        Bookmaker implied probabilities for selected bets.

    Returns
    -------
    float
        Average CLV in probability points.
    """
    model_prob = np.asarray(model_prob, dtype=float)
    implied_prob = np.asarray(implied_prob, dtype=float)
    if len(model_prob) == 0:
        return 0.0
    return float((model_prob - implied_prob).mean())


def compute_pnl_series(stakes: np.ndarray, payouts: np.ndarray) -> np.ndarray:
    """Compute cumulative P&L series.

    Parameters
    ----------
    stakes : array-like
        Amounts wagered per bet.
    payouts : array-like
        Amounts received per bet.

    Returns
    -------
    np.ndarray
        Cumulative profit/loss after each bet.
    """
    stakes = np.asarray(stakes, dtype=float)
    payouts = np.asarray(payouts, dtype=float)
    return np.cumsum(payouts - stakes)


def compute_max_drawdown(pnl_series: np.ndarray) -> float:
    """Compute maximum drawdown from a cumulative P&L series.

    Parameters
    ----------
    pnl_series : array-like
        Cumulative P&L.

    Returns
    -------
    float
        Maximum drawdown (non-negative value, 0 if no drawdown).
    """
    pnl = np.asarray(pnl_series, dtype=float)
    if len(pnl) == 0:
        return 0.0
    running_max = np.maximum.accumulate(pnl)
    drawdowns = running_max - pnl
    return float(drawdowns.max())


def compute_sharpe_ratio(
    stakes: np.ndarray,
    payouts: np.ndarray,
) -> float:
    """Compute Sharpe-like ratio of bet returns.

    Parameters
    ----------
    stakes : array-like
        Amounts wagered.
    payouts : array-like
        Amounts received.

    Returns
    -------
    float
        Mean return / std of returns, or 0 if no variance.
    """
    stakes = np.asarray(stakes, dtype=float)
    payouts = np.asarray(payouts, dtype=float)
    if len(stakes) == 0 or stakes.sum() == 0:
        return 0.0
    returns = (payouts - stakes) / np.where(stakes > 0, stakes, 1.0)
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std)


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def compute_segment_metrics(
    bet_results: pd.DataFrame,
    segment_col: str,
) -> pd.DataFrame:
    """Compute ROI, hit rate, and bet count by segment.

    Parameters
    ----------
    bet_results : pd.DataFrame
        Must contain columns: ``stake``, ``payout``, ``won``, and *segment_col*.
    segment_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Columns: segment, n_bets, total_staked, total_payout, roi, hit_rate
    """
    if segment_col not in bet_results.columns:
        raise ValueError(f"Segment column '{segment_col}' not found")

    grouped = bet_results.groupby(segment_col).agg(
        n_bets=("stake", "count"),
        total_staked=("stake", "sum"),
        total_payout=("payout", "sum"),
        wins=("won", "sum"),
    ).reset_index()
    grouped.rename(columns={segment_col: "segment"}, inplace=True)
    grouped["roi"] = np.where(
        grouped["total_staked"] > 0,
        (grouped["total_payout"] - grouped["total_staked"]) / grouped["total_staked"],
        0.0,
    )
    grouped["hit_rate"] = np.where(
        grouped["n_bets"] > 0,
        grouped["wins"] / grouped["n_bets"],
        0.0,
    )
    return grouped


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def build_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    stakes: np.ndarray | None = None,
    payouts: np.ndarray | None = None,
    model_prob: np.ndarray | None = None,
    implied_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a comprehensive evaluation report.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth for all predictions.
    y_prob : array-like
        Predicted probabilities for all predictions.
    stakes : array-like, optional
        Amounts wagered (for economic metrics).
    payouts : array-like, optional
        Amounts received (for economic metrics).
    model_prob : array-like, optional
        Model probabilities for *selected bets only* (for CLV).
    implied_prob : array-like, optional
        Implied probabilities for *selected bets only* (for CLV).

    Returns
    -------
    dict[str, Any]
        All metrics in one dictionary.
    """
    report: dict[str, Any] = {
        "auc": compute_auc(y_true, y_prob),
        "pr_auc": compute_pr_auc(y_true, y_prob),
        "brier_score": compute_brier_score(y_true, y_prob),
        "log_loss": compute_log_loss(y_true, y_prob),
        "calibration_error": compute_calibration_error(y_true, y_prob),
        "n_predictions": len(y_true),
    }
    if stakes is not None and payouts is not None:
        stakes = np.asarray(stakes)
        payouts = np.asarray(payouts)
        pnl = compute_pnl_series(stakes, payouts)
        report["roi"] = compute_roi(stakes, payouts)
        report["max_drawdown"] = compute_max_drawdown(pnl)
        report["sharpe_ratio"] = compute_sharpe_ratio(stakes, payouts)
        report["total_staked"] = float(stakes.sum())
        report["total_payout"] = float(payouts.sum())
        report["n_bets"] = int((stakes > 0).sum())
    if model_prob is not None and implied_prob is not None:
        report["clv"] = compute_clv(model_prob, implied_prob)
    return report
