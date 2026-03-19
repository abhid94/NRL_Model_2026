"""Conformal prediction intervals for ATS probability estimates.

Uses MAPIE to produce calibrated prediction intervals with guaranteed
coverage. Bets where the interval is wide (high uncertainty) can be
skipped to improve ROI.

Usage:
    from src.models.conformal import ConformalModel
    model = ConformalModel(base_model, alpha=0.1)
    model.fit(X_train, y_train)
    probs, intervals = model.predict_with_intervals(X_test)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.baseline import BaseModel

LOGGER = logging.getLogger(__name__)


class ConformalModel(BaseModel):
    """Wraps a BaseModel with MAPIE conformal prediction intervals.

    Provides calibrated prediction intervals alongside point predictions.
    The interval width is a measure of prediction uncertainty — narrower
    intervals indicate higher confidence.

    Parameters
    ----------
    base_model : BaseModel
        The underlying ATS model to wrap.
    alpha : float
        Significance level (e.g., 0.1 for 90% prediction intervals).
    cal_rounds : int
        Number of most-recent training rounds held out for conformal
        calibration.
    max_interval_width : float
        Maximum allowed interval width. Predictions wider than this
        are flagged as low-confidence.
    """

    def __init__(
        self,
        base_model: BaseModel,
        alpha: float = 0.1,
        cal_rounds: int = 5,
        max_interval_width: float = 0.30,
    ) -> None:
        self._base_model = base_model
        self._alpha = alpha
        self._cal_rounds = cal_rounds
        self._max_interval_width = max_interval_width
        self._conformal_scores: np.ndarray | None = None
        self._quantile: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> None:
        """Fit base model and compute conformal calibration scores.

        Uses split conformal prediction: fit the model on all but the
        last cal_rounds, then compute nonconformity scores on held-out
        rounds.

        Parameters
        ----------
        X : pd.DataFrame
            Training data with ``round_number`` column.
        y : array-like
            Binary target (0/1).
        **kwargs
            Forwarded to base model fit (e.g., sample_weight).
        """
        y = np.asarray(y, dtype=float)

        # Temporal split
        if "round_number" in X.columns:
            rounds = sorted(X["round_number"].unique())
            if len(rounds) > self._cal_rounds:
                cal_cutoff = rounds[-self._cal_rounds]
                train_mask = X["round_number"] < cal_cutoff
                cal_mask = X["round_number"] >= cal_cutoff
            else:
                n_train = int(len(X) * 0.8)
                train_mask = pd.Series(
                    [True] * n_train + [False] * (len(X) - n_train),
                    index=X.index,
                )
                cal_mask = ~train_mask
        else:
            n_train = int(len(X) * 0.8)
            train_mask = pd.Series(
                [True] * n_train + [False] * (len(X) - n_train),
                index=X.index,
            )
            cal_mask = ~train_mask

        X_train, y_train = X[train_mask], y[train_mask.values]
        X_cal, y_cal = X[cal_mask], y[cal_mask.values]

        # Split sample_weight if provided
        train_kwargs = dict(kwargs)
        sample_weight = kwargs.get("sample_weight")
        if sample_weight is not None:
            train_kwargs["sample_weight"] = np.asarray(sample_weight)[train_mask.values]

        if len(X_train) == 0 or len(X_cal) == 0:
            LOGGER.warning("Not enough data for conformal split; fitting without intervals")
            self._base_model.fit(X, y, **kwargs)
            return

        # 1. Fit base model on training portion
        self._base_model.fit(X_train, y_train, **train_kwargs)

        # 2. Compute nonconformity scores on calibration set
        cal_probs = self._base_model.predict_proba(X_cal)
        # Score = |predicted_prob - actual_outcome|
        self._conformal_scores = np.abs(cal_probs - y_cal)

        # 3. Compute the conformal quantile
        n_cal = len(self._conformal_scores)
        q_level = np.ceil((1 - self._alpha) * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)
        self._quantile = float(np.quantile(self._conformal_scores, q_level))

        LOGGER.info(
            "ConformalModel fitted: %d train, %d cal, alpha=%.2f, "
            "quantile=%.4f, max_interval=%.2f",
            len(X_train), len(X_cal), self._alpha,
            self._quantile, self._max_interval_width,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return point predictions (same as base model).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return self._base_model.predict_proba(X)

    def predict_with_intervals(
        self, X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return predictions with conformal intervals and confidence flags.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        tuple of (probs, lower, upper, is_confident)
            probs : np.ndarray — point predictions
            lower : np.ndarray — lower bound of prediction interval
            upper : np.ndarray — upper bound of prediction interval
            is_confident : np.ndarray (bool) — True if interval width
                <= max_interval_width
        """
        probs = self._base_model.predict_proba(X)

        if self._quantile is None:
            # No conformal calibration — return wide intervals
            lower = np.zeros_like(probs)
            upper = np.ones_like(probs)
            is_confident = np.ones(len(probs), dtype=bool)
            return probs, lower, upper, is_confident

        lower = np.clip(probs - self._quantile, 0.0, 1.0)
        upper = np.clip(probs + self._quantile, 0.0, 1.0)
        interval_width = upper - lower
        is_confident = interval_width <= self._max_interval_width

        n_confident = is_confident.sum()
        LOGGER.info(
            "Conformal prediction: %d/%d confident (%.1f%%), "
            "mean interval=%.3f",
            n_confident, len(probs),
            100 * n_confident / len(probs) if len(probs) > 0 else 0,
            float(interval_width.mean()),
        )

        return probs, lower, upper, is_confident

    def feature_names(self) -> list[str]:
        """Return feature names from the base model."""
        return self._base_model.feature_names()

    @property
    def base_model(self) -> BaseModel:
        """Access the underlying base model."""
        return self._base_model

    @property
    def quantile(self) -> float | None:
        """The conformal quantile (interval half-width)."""
        return self._quantile
