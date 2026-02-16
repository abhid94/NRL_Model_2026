"""LightGBM models for ATS prediction.

Two variants:
- GBMModel: uses all features including Betfair odds
- GBMModelNoBetfair: excludes Betfair columns (ablation study)
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

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

BETFAIR_COLS_PREFIX = "betfair_"

# Categorical columns to encode as LightGBM native categoricals
CATEGORICAL_COLS = {"position_group", "position_code", "player_edge"}


def _detect_features(
    df: pd.DataFrame,
    exclude_betfair: bool = False,
) -> list[str]:
    """Auto-detect feature columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature store DataFrame.
    exclude_betfair : bool
        If True, exclude all columns starting with 'betfair_'.

    Returns
    -------
    list[str]
        Sorted list of feature column names.
    """
    features = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if exclude_betfair and col.startswith(BETFAIR_COLS_PREFIX):
            continue
        # Only keep numeric or known categorical columns
        if df[col].dtype in ("float64", "float32", "int64", "int32", "bool"):
            features.append(col)
        elif col in CATEGORICAL_COLS:
            features.append(col)
    return sorted(features)


class GBMModel(BaseModel):
    """LightGBM model using all available features.

    Handles NaN natively. Encodes categorical columns as LightGBM
    native categoricals for proper split handling.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Boosting learning rate.
    min_child_samples : int
        Minimum samples per leaf.
    subsample : float
        Row subsampling ratio.
    colsample_bytree : float
        Column subsampling ratio.
    reg_alpha : float
        L1 regularization.
    reg_lambda : float
        L2 regularization.
    exclude_betfair : bool
        If True, exclude Betfair features (for ablation).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 1.0,
        reg_lambda: float = 1.0,
        exclude_betfair: bool = False,
    ) -> None:
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        self._exclude_betfair = exclude_betfair
        self._model: lgb.LGBMClassifier | None = None
        self._features: list[str] = []
        self._cat_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit LightGBM on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature store DataFrame.
        y : array-like
            Binary target (0/1).
        """
        y = np.asarray(y, dtype=int)
        self._features = _detect_features(X, exclude_betfair=self._exclude_betfair)
        self._cat_features = [c for c in self._features if c in CATEGORICAL_COLS]

        X_train = self._prepare(X)

        # Auto-calculate scale_pos_weight
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self._model = lgb.LGBMClassifier(
            **self._params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            importance_type="gain",
        )
        self._model.fit(
            X_train, y,
            categorical_feature=self._cat_features if self._cat_features else "auto",
        )
        LOGGER.info(
            "GBM fitted on %d rows × %d features (exclude_betfair=%s)",
            len(y), len(self._features), self._exclude_betfair,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(ATS) for each row.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_pred = self._prepare(X)
        return self._model.predict_proba(X_pred)[:, 1]

    def feature_names(self) -> list[str]:
        """Return list of feature names the model uses."""
        return list(self._features)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance sorted by gain.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        importances = self._model.feature_importances_
        return (
            pd.DataFrame({
                "feature": self._features,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and encode features.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature store.

        Returns
        -------
        pd.DataFrame
            Ready for LightGBM.
        """
        df = X[self._features].copy()
        for col in self._cat_features:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df


class GBMModelNoBetfair(GBMModel):
    """LightGBM model excluding all Betfair features (ablation study)."""

    def __init__(self, **kwargs) -> None:
        kwargs["exclude_betfair"] = True
        super().__init__(**kwargs)
