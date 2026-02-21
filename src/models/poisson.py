"""Poisson regression model for ATS prediction.

Predicts lambda (expected tries per match) using statsmodels GLM
with Poisson family. P(ATS) = 1 - exp(-lambda).

Provides a diverse signal for ensemble with GBM and logistic models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

from src.models.baseline import BaseModel

LOGGER = logging.getLogger(__name__)

# Columns that are IDs, metadata, or targets — never features
EXCLUDE_COLS = frozenset({
    "match_id", "player_id", "squad_id", "opponent_squad_id",
    "round_number", "season", "tries", "scored_try",
    "round_number_team_own", "round_number_context", "round_number_matchup",
    "squad_id_matchup", "squad_id_context",
    "opponent_squad_id_matchup", "opponent_squad_id_context",
    "is_home_team_own", "is_home_context",
    "position_label",
    "betfair_odds_source",
})

CATEGORICAL_COLS = {"position_group", "position_code", "player_edge"}


def _detect_numeric_features(
    df: pd.DataFrame,
    exclude_betfair: bool = False,
) -> list[str]:
    """Auto-detect numeric feature columns."""
    features = []
    for col in df.columns:
        if col in EXCLUDE_COLS or col in CATEGORICAL_COLS:
            continue
        if exclude_betfair and col.startswith("betfair_"):
            continue
        if df[col].dtype in ("float64", "float32", "int64", "int32", "bool"):
            features.append(col)
    return sorted(features)


class PoissonModel(BaseModel):
    """Poisson GLM for try count prediction.

    Predicts lambda (expected tries) then converts to P(ATS) = 1 - exp(-lambda).
    Uses L2 regularization via fit_regularized.

    Parameters
    ----------
    reg_alpha : float
        L2 regularization strength.
    exclude_betfair : bool
        If True, exclude Betfair features.
    """

    def __init__(
        self,
        reg_alpha: float = 1.0,
        exclude_betfair: bool = False,
    ) -> None:
        self._reg_alpha = reg_alpha
        self._exclude_betfair = exclude_betfair
        self._model: sm.GLM | None = None
        self._result = None
        self._numeric_features: list[str] = []
        self._cat_features: list[str] = []
        self._encoder: OneHotEncoder | None = None
        self._train_mean: np.ndarray | None = None
        self._all_feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit Poisson GLM on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature store DataFrame. If 'tries' column exists,
            uses raw counts; otherwise treats y as binary (0/1 counts).
        y : array-like
            Binary target (0/1). Used as count if 'tries' not in X.
        """
        y = np.asarray(y)

        # Use raw try counts if available, otherwise use binary target
        if "tries" in X.columns:
            y_count = X["tries"].values.astype(float)
        else:
            y_count = y.astype(float)

        self._numeric_features = _detect_numeric_features(
            X, exclude_betfair=self._exclude_betfair,
        )
        self._cat_features = sorted(
            col for col in CATEGORICAL_COLS if col in X.columns
        )

        X_mat, feature_names = self._prepare(X, fit_encoder=True)

        # Impute NaN with column means (statsmodels can't handle NaN)
        self._train_mean = np.nanmean(X_mat, axis=0)
        self._train_mean = np.where(
            np.isnan(self._train_mean), 0.0, self._train_mean,
        )
        X_imp = np.where(np.isnan(X_mat), self._train_mean, X_mat)

        # Add constant for intercept
        X_with_const = sm.add_constant(X_imp, has_constant="add")
        self._all_feature_names = ["const"] + feature_names

        # Fit Poisson GLM with L2 regularization
        glm = sm.GLM(
            y_count,
            X_with_const,
            family=sm.families.Poisson(),
        )
        try:
            self._result = glm.fit_regularized(
                alpha=self._reg_alpha,
                L1_wt=0.0,  # Pure L2
            )
        except Exception:
            # Fallback: fit without regularization if it fails
            LOGGER.warning("Regularized fit failed, falling back to unregularized")
            self._result = glm.fit()

        LOGGER.info(
            "PoissonModel fitted on %d rows × %d features (exclude_betfair=%s)",
            len(y), len(feature_names), self._exclude_betfair,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(ATS) = 1 - exp(-lambda) for each row.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        lambdas = self.predict_lambda(X)
        probs = 1.0 - np.exp(-lambdas)
        return np.clip(probs, 0.0, 1.0)

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted lambda (expected tries) for each row.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Expected try counts (>= 0).
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_mat, _ = self._prepare(X, fit_encoder=False)

        # Impute with training means
        if self._train_mean is not None:
            X_mat = np.where(np.isnan(X_mat), self._train_mean, X_mat)

        X_with_const = sm.add_constant(X_mat, has_constant="add")
        return self._result.predict(X_with_const)

    def feature_names(self) -> list[str]:
        """Return list of feature names the model uses."""
        return list(self._all_feature_names)

    def _prepare(
        self,
        X: pd.DataFrame,
        fit_encoder: bool,
    ) -> tuple[np.ndarray, list[str]]:
        """Build feature matrix with one-hot encoded categoricals.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            Feature matrix and feature names.
        """
        # Numeric features
        available_numeric = [f for f in self._numeric_features if f in X.columns]
        numeric = X[available_numeric].values.astype(float)
        feature_names = list(available_numeric)

        # Categorical features: one-hot encode
        if self._cat_features:
            cat_data = X[self._cat_features].fillna("__missing__")
            if fit_encoder:
                self._encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                )
                cat_encoded = self._encoder.fit_transform(cat_data)
            else:
                if self._encoder is None:
                    raise RuntimeError("Encoder not fitted. Call fit() first.")
                cat_encoded = self._encoder.transform(cat_data)

            cat_names = list(self._encoder.get_feature_names_out(self._cat_features))
            numeric = np.hstack([numeric, cat_encoded])
            feature_names.extend(cat_names)

        return numeric, feature_names
