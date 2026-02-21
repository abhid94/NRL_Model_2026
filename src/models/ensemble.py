"""Model ensemble and stacking for ATS prediction.

Two approaches:
1. WeightedEnsemble: weighted average of base model probabilities
2. StackedEnsemble: meta-learner trained on base model out-of-fold predictions

Both conform to BaseModel interface for seamless backtest integration.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.baseline import BaseModel

LOGGER = logging.getLogger(__name__)


class WeightedEnsemble(BaseModel):
    """Weighted average ensemble of base models.

    Weights can be equal (default) or learned by optimizing Brier score
    on a temporal holdout set.

    Parameters
    ----------
    base_models : list[BaseModel]
        Base models to ensemble.
    weights : list[float] | None
        Fixed weights. If None, uses equal weights or learns them.
    learn_weights : bool
        If True and weights is None, learn weights via temporal holdout.
    holdout_rounds : int
        Number of most-recent training rounds for weight learning.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        weights: list[float] | None = None,
        learn_weights: bool = False,
        holdout_rounds: int = 5,
    ) -> None:
        if len(base_models) < 2:
            raise ValueError("Need at least 2 base models")
        self._base_models = base_models
        self._n_models = len(base_models)
        self._learn_weights = learn_weights
        self._holdout_rounds = holdout_rounds

        if weights is not None:
            if len(weights) != self._n_models:
                raise ValueError(
                    f"weights length ({len(weights)}) != models ({self._n_models})"
                )
            total = sum(weights)
            self._weights = np.array([w / total for w in weights])
        else:
            self._weights = np.ones(self._n_models) / self._n_models

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit all base models and optionally learn weights.

        Parameters
        ----------
        X : pd.DataFrame
            Training data. Must contain ``round_number`` for weight learning.
        y : array-like
            Binary target (0/1).
        """
        y = np.asarray(y, dtype=float)

        if self._learn_weights and "round_number" in X.columns:
            self._fit_with_weight_learning(X, y)
        else:
            for model in self._base_models:
                model.fit(X, y)

        LOGGER.info(
            "WeightedEnsemble fitted %d models, weights=%s",
            self._n_models,
            [f"{w:.3f}" for w in self._weights],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted average P(ATS).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Ensemble probabilities.
        """
        preds = np.column_stack([
            model.predict_proba(X) for model in self._base_models
        ])
        return preds @ self._weights

    def feature_names(self) -> list[str]:
        """Return feature names from all base models (deduplicated)."""
        seen = set()
        names = []
        for model in self._base_models:
            for name in model.feature_names():
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    @property
    def weights(self) -> np.ndarray:
        """Current ensemble weights."""
        return self._weights.copy()

    def _fit_with_weight_learning(
        self, X: pd.DataFrame, y: np.ndarray,
    ) -> None:
        """Fit models and learn weights via temporal holdout."""
        rounds = sorted(X["round_number"].unique())
        if len(rounds) <= self._holdout_rounds:
            # Not enough data — fit all on full data, equal weights
            for model in self._base_models:
                model.fit(X, y)
            return

        cutoff = rounds[-self._holdout_rounds]
        train_mask = X["round_number"] < cutoff
        holdout_mask = X["round_number"] >= cutoff

        X_train, y_train = X[train_mask], y[train_mask.values]
        X_hold, y_hold = X[holdout_mask], y[holdout_mask.values]

        # Fit models on training portion
        for model in self._base_models:
            model.fit(X_train, y_train)

        # Get holdout predictions
        holdout_preds = np.column_stack([
            model.predict_proba(X_hold) for model in self._base_models
        ])

        # Grid search for best weights (Brier score)
        best_brier = float("inf")
        best_weights = self._weights.copy()

        # Simple grid over weight combinations (resolution 0.1)
        from itertools import product
        steps = 11
        for combo in product(range(steps), repeat=self._n_models):
            if sum(combo) == 0:
                continue
            w = np.array(combo, dtype=float)
            w /= w.sum()
            blended = holdout_preds @ w
            brier = float(np.mean((blended - y_hold) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_weights = w

        self._weights = best_weights

        # Refit all models on full data for final predictions
        for model in self._base_models:
            model.fit(X, y)


class StackedEnsemble(BaseModel):
    """Two-level stacking ensemble.

    Level-0: base models produce out-of-fold predictions via temporal CV.
    Level-1: logistic regression meta-learner on base predictions.

    Parameters
    ----------
    base_models : list[BaseModel]
        Level-0 base models.
    n_folds : int
        Number of temporal CV folds for generating OOF predictions.
    include_market : bool
        If True, include betfair_implied_prob as meta-feature.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        n_folds: int = 5,
        include_market: bool = True,
    ) -> None:
        if len(base_models) < 2:
            raise ValueError("Need at least 2 base models")
        self._base_models = base_models
        self._n_models = len(base_models)
        self._n_folds = n_folds
        self._include_market = include_market
        self._meta_learner: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit stacking ensemble with temporal CV.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``round_number`` column for temporal splitting.
        y : array-like
            Binary target (0/1).
        """
        y = np.asarray(y, dtype=float)

        if "round_number" not in X.columns:
            raise ValueError("StackedEnsemble requires round_number for temporal CV")

        rounds = sorted(X["round_number"].unique())
        n_rounds = len(rounds)

        if n_rounds < self._n_folds + 1:
            # Not enough rounds — fall back to simple average
            LOGGER.warning(
                "Only %d rounds, need %d folds + 1. Falling back to simple average.",
                n_rounds, self._n_folds,
            )
            for model in self._base_models:
                model.fit(X, y)
            self._meta_learner = None
            return

        # Generate out-of-fold predictions via temporal CV
        oof_preds = np.full((len(X), self._n_models), np.nan)

        # Split rounds into folds for walk-forward
        fold_size = max(1, n_rounds // (self._n_folds + 1))

        for fold_idx in range(self._n_folds):
            # Training rounds: first (fold_idx + 1) * fold_size rounds
            train_end_idx = (fold_idx + 1) * fold_size
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + fold_size, n_rounds)

            if val_start_idx >= n_rounds:
                break

            train_rounds = set(rounds[:train_end_idx])
            val_rounds = set(rounds[val_start_idx:val_end_idx])

            train_mask = X["round_number"].isin(train_rounds)
            val_mask = X["round_number"].isin(val_rounds)

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue

            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask.values]
            X_val_fold = X[val_mask]

            for m_idx, model in enumerate(self._base_models):
                try:
                    model.fit(X_train_fold, y_train_fold)
                    preds = model.predict_proba(X_val_fold)
                    val_indices = np.where(val_mask.values)[0]
                    oof_preds[val_indices, m_idx] = preds
                except Exception as e:
                    LOGGER.warning(
                        "Model %d failed on fold %d: %s",
                        m_idx, fold_idx, e,
                    )

        # Fit meta-learner on rows that have all OOF predictions
        valid_mask = ~np.isnan(oof_preds).any(axis=1)
        if valid_mask.sum() < 20:
            LOGGER.warning(
                "Only %d valid OOF rows; falling back to simple average.",
                valid_mask.sum(),
            )
            for model in self._base_models:
                model.fit(X, y)
            self._meta_learner = None
            return

        meta_X = oof_preds[valid_mask]
        meta_y = y[valid_mask]

        # Optionally add market probability as meta-feature
        if self._include_market and "betfair_implied_prob" in X.columns:
            market = X.loc[valid_mask, "betfair_implied_prob"].values.reshape(-1, 1)
            market = np.where(np.isnan(market), 0.0, market)
            meta_X = np.hstack([meta_X, market])

        self._meta_learner = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
        )
        self._meta_learner.fit(meta_X, meta_y)

        # Refit all base models on full training data
        for model in self._base_models:
            model.fit(X, y)

        LOGGER.info(
            "StackedEnsemble fitted: %d base models, %d OOF rows, include_market=%s",
            self._n_models, valid_mask.sum(), self._include_market,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return stacked ensemble P(ATS).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Ensemble probabilities.
        """
        base_preds = np.column_stack([
            model.predict_proba(X) for model in self._base_models
        ])

        if self._meta_learner is None:
            # Fallback: simple average
            return base_preds.mean(axis=1)

        meta_X = base_preds
        if self._include_market and "betfair_implied_prob" in X.columns:
            market = X["betfair_implied_prob"].values.reshape(-1, 1)
            market = np.where(np.isnan(market), 0.0, market)
            meta_X = np.hstack([meta_X, market])

        return self._meta_learner.predict_proba(meta_X)[:, 1]

    def feature_names(self) -> list[str]:
        """Return meta-feature names."""
        names = [f"base_model_{i}" for i in range(self._n_models)]
        if self._include_market:
            names.append("betfair_implied_prob")
        return names


def prediction_diversity(
    models: list[BaseModel],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise prediction correlation between models.

    Parameters
    ----------
    models : list[BaseModel]
        Fitted models.
    X : pd.DataFrame
        Data to predict on.

    Returns
    -------
    pd.DataFrame
        Correlation matrix (model_names x model_names).
    """
    preds = {}
    for i, model in enumerate(models):
        name = type(model).__name__
        if name in preds:
            name = f"{name}_{i}"
        preds[name] = model.predict_proba(X)

    pred_df = pd.DataFrame(preds)
    return pred_df.corr()
