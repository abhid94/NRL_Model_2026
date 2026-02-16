"""Post-hoc probability calibration for ATS models.

Wraps any BaseModel with Platt scaling (sigmoid) or isotonic regression
to produce well-calibrated probabilities. Uses temporal-safe splitting:
the calibrator is fit on the last N rounds of training data, not the
same data used for model fitting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.baseline import BaseModel

LOGGER = logging.getLogger(__name__)


class CalibratedModel(BaseModel):
    """Calibration wrapper for any BaseModel.

    Splits training data temporally: fits the base model on all but the
    last ``cal_rounds`` rounds, then fits a calibrator on those held-out
    rounds using the base model's raw predictions.

    Parameters
    ----------
    base_model : BaseModel
        The underlying model to calibrate.
    method : str
        ``"isotonic"`` for non-parametric or ``"sigmoid"`` for Platt scaling.
    cal_rounds : int
        Number of most-recent training rounds to hold out for calibration.
    """

    def __init__(
        self,
        base_model: BaseModel,
        method: str = "isotonic",
        cal_rounds: int = 5,
    ) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'")
        self._base_model = base_model
        self._method = method
        self._cal_rounds = cal_rounds
        self._calibrator: IsotonicRegression | LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit base model + calibrator with temporal split.

        Parameters
        ----------
        X : pd.DataFrame
            Training data. Must contain ``round_number`` column.
        y : array-like
            Binary target (0/1).
        """
        y = np.asarray(y, dtype=float)

        # Temporal split: hold out last cal_rounds for calibration
        if "round_number" in X.columns:
            rounds = sorted(X["round_number"].unique())
            if len(rounds) > self._cal_rounds:
                cal_round_cutoff = rounds[-self._cal_rounds]
                train_mask = X["round_number"] < cal_round_cutoff
                cal_mask = X["round_number"] >= cal_round_cutoff
            else:
                # Not enough rounds — use 80/20 split on rows
                n_train = int(len(X) * 0.8)
                train_mask = pd.Series([True] * n_train + [False] * (len(X) - n_train), index=X.index)
                cal_mask = ~train_mask
        else:
            # No round info — use 80/20 split
            n_train = int(len(X) * 0.8)
            train_mask = pd.Series([True] * n_train + [False] * (len(X) - n_train), index=X.index)
            cal_mask = ~train_mask

        X_train, y_train = X[train_mask], y[train_mask.values]
        X_cal, y_cal = X[cal_mask], y[cal_mask.values]

        if len(X_train) == 0 or len(X_cal) == 0:
            # Fallback: fit on all data, no calibration
            LOGGER.warning("Not enough data for calibration split; fitting without calibration")
            self._base_model.fit(X, y)
            self._calibrator = None
            return

        # 1. Fit base model on training portion
        self._base_model.fit(X_train, y_train)

        # 2. Get raw predictions on calibration set
        raw_probs = self._base_model.predict_proba(X_cal)

        # 3. Fit calibrator
        if self._method == "isotonic":
            self._calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip",
            )
            self._calibrator.fit(raw_probs, y_cal)
        else:
            # Platt scaling: logistic regression on log-odds
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self._calibrator.fit(raw_probs.reshape(-1, 1), y_cal)

        LOGGER.info(
            "CalibratedModel fitted: %d train, %d cal, method=%s",
            len(X_train), len(X_cal), self._method,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(ATS).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Calibrated probabilities.
        """
        raw_probs = self._base_model.predict_proba(X)

        if self._calibrator is None:
            return raw_probs

        if self._method == "isotonic":
            return self._calibrator.predict(raw_probs)
        else:
            return self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    def feature_names(self) -> list[str]:
        """Return feature names from the base model."""
        return self._base_model.feature_names()
