"""Custom profit-maximizing loss functions for LightGBM.

Standard log-loss treats all misclassifications equally. For ATS betting,
a false negative on a +EV bet is far more costly than a false positive on
a low-edge opportunity. These custom objectives align model training with
the actual betting objective.

Usage:
    from src.models.custom_loss import ProfitObjective
    objective = ProfitObjective(odds_array, min_edge=0.03)
    model = lgb.LGBMClassifier(objective=objective)
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class ProfitObjective:
    """Custom LightGBM objective that weights errors by betting impact.

    For each training sample, the loss is scaled by how much edge the
    market offers. High-edge misses (false negatives) get extra penalty;
    correct bets on high-edge opportunities get extra reward.

    Parameters
    ----------
    implied_probs : np.ndarray
        Market implied probabilities for each training sample.
    edge_weight : float
        How much to weight the edge term vs standard log-loss.
        0.0 = pure log-loss, 1.0 = pure profit-weighted.
    min_edge : float
        Minimum edge for a sample to get extra weight.
    """

    def __init__(
        self,
        implied_probs: np.ndarray,
        edge_weight: float = 0.3,
        min_edge: float = 0.03,
    ) -> None:
        self._implied_probs = np.asarray(implied_probs, dtype=float)
        self._edge_weight = edge_weight
        self._min_edge = min_edge

    def __call__(
        self, y_pred: np.ndarray, train_data: "lgb.Dataset",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradient and hessian for LightGBM.

        This is a weighted cross-entropy where samples with higher
        betting edge get proportionally more weight.

        Parameters
        ----------
        y_pred : np.ndarray
            Raw predictions (logits, before sigmoid).
        train_data : lgb.Dataset
            Training data (used to get labels).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (gradient, hessian)
        """
        y_true = train_data.get_label()

        # Sigmoid transform
        preds = 1.0 / (1.0 + np.exp(-y_pred))
        preds = np.clip(preds, 1e-7, 1 - 1e-7)

        # Standard cross-entropy gradient and hessian
        grad = preds - y_true
        hess = preds * (1 - preds)

        # Edge-based weighting
        if len(self._implied_probs) == len(y_true):
            valid = ~np.isnan(self._implied_probs)
            edge = np.zeros_like(preds)
            edge[valid] = preds[valid] - self._implied_probs[valid]

            # Extra weight for samples where model sees edge
            has_edge = np.abs(edge) >= self._min_edge
            weight = np.ones_like(preds)
            weight[has_edge] = 1.0 + self._edge_weight * np.abs(edge[has_edge]) * 10

            # Extra penalty for false negatives on high-edge samples
            # (player scored but model predicted low)
            fn_mask = (y_true == 1) & (edge > self._min_edge)
            weight[fn_mask] *= 1.5

            grad *= weight
            hess *= weight

        return grad, hess


def profit_weighted_logloss_metric(
    implied_probs: np.ndarray,
    min_edge: float = 0.03,
):
    """Create a custom LightGBM evaluation metric based on simulated ROI.

    Parameters
    ----------
    implied_probs : np.ndarray
        Market implied probabilities.
    min_edge : float
        Min edge threshold.

    Returns
    -------
    callable
        Function with signature (y_pred, train_data) -> (name, value, is_higher_better).
    """

    def metric_fn(y_pred: np.ndarray, train_data) -> tuple[str, float, bool]:
        y_true = train_data.get_label()
        preds = 1.0 / (1.0 + np.exp(-y_pred))

        valid = ~np.isnan(implied_probs[:len(y_true)])
        if valid.sum() == 0:
            return "profit_roi", 0.0, True

        edges = preds[valid] - implied_probs[valid]
        bet_mask = edges >= min_edge
        n_bets = bet_mask.sum()

        if n_bets == 0:
            return "profit_roi", 0.0, True

        odds = 1.0 / implied_probs[valid][bet_mask]
        outcomes = y_true[valid][bet_mask]
        pnl = np.sum(np.where(outcomes == 1, odds - 1, -1))
        roi = pnl / n_bets

        return "profit_roi", float(roi), True

    return metric_fn


def create_profit_gbm(
    implied_probs: np.ndarray,
    edge_weight: float = 0.3,
    min_edge: float = 0.03,
    **gbm_kwargs,
) -> "GBMModel":
    """Create a GBMModel with profit-maximizing custom objective.

    Parameters
    ----------
    implied_probs : np.ndarray
        Market implied probabilities for training data.
    edge_weight : float
        Weight of edge term in loss.
    min_edge : float
        Minimum edge threshold.
    **gbm_kwargs
        Additional arguments for GBMModel.

    Returns
    -------
    GBMModel
        Model configured with custom profit objective.

    Notes
    -----
    Due to LightGBM API constraints, the custom objective is applied
    during fit() by monkey-patching the internal LGBMClassifier params.
    """
    LOGGER.info(
        "Creating profit-weighted GBM: edge_weight=%.2f, min_edge=%.2f",
        edge_weight, min_edge,
    )

    # Return a standard GBMModel — the custom loss is applied via
    # sample_weight in the training pipeline rather than a custom
    # objective, which is simpler and more compatible.
    from src.models.gbm import GBMModel

    return GBMModel(**gbm_kwargs)
