"""Baseline models and betting strategies for ATS prediction.

Models predict P(ATS). Strategies decide which bets to place.
All strategies conform to a common interface for the backtest engine.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from src.config import ELIGIBLE_POSITION_CODES, MIN_EDGE_THRESHOLD

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bet recommendation dataclass
# ---------------------------------------------------------------------------

@dataclass
class BetRecommendation:
    """A single bet recommendation from a strategy."""

    match_id: int
    player_id: int
    model_prob: float
    implied_prob: float
    odds: float
    edge: float
    stake: float = 0.0  # Filled by staking engine
    position_code: str = ""
    player_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base model protocol
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Abstract base for ATS probability models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(ATS) for each row."""

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names the model uses."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PositionBaselineModel(BaseModel):
    """Predicts P(ATS) as the historical try rate for each position group.

    This is the absolute floor — any useful model must beat this.
    """

    def __init__(self) -> None:
        self._rates: dict[str, float] = {}
        self._global_rate: float = 0.19  # Fallback

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Learn position-group try rates from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``position_group`` column.
        y : array-like
            Binary target (0/1).
        """
        y = np.asarray(y)
        df = X[["position_group"]].copy()
        df["target"] = y
        self._global_rate = float(y.mean())
        rates = df.groupby("position_group")["target"].mean()
        self._rates = rates.to_dict()
        LOGGER.info("PositionBaseline rates: %s", self._rates)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return position-group try rate for each row.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``position_group`` column.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return np.array([
            self._rates.get(pg, self._global_rate)
            for pg in X["position_group"]
        ])

    def feature_names(self) -> list[str]:
        return ["position_group"]


class LogisticBaselineModel(BaseModel):
    """Logistic regression on 6 MVP features.

    Features: position_group (one-hot), expected_team_tries_5,
    player_try_share_5, is_home, is_starter, opponent_rolling_defence_tries_conceded_5.
    """

    NUMERIC_FEATURES = [
        "expected_team_tries_5",
        "player_try_share_5",
        "is_home",
        "is_starter",
        "opponent_rolling_defence_tries_conceded_5",
    ]

    def __init__(self, C: float = 1.0) -> None:
        self._model = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs", random_state=42,
        )
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit logistic regression on MVP features.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain position_group and the 5 numeric features.
        y : array-like
            Binary target.
        """
        X_mat = self._prepare_features(X, fit_encoder=True)
        y = np.asarray(y)
        # Drop rows with NaN
        mask = ~np.isnan(X_mat).any(axis=1)
        self._model.fit(X_mat[mask], y[mask])
        self._fitted = True
        LOGGER.info(
            "LogisticBaseline fitted on %d rows (%d dropped for NaN)",
            mask.sum(), (~mask).sum(),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict P(ATS).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Predicted probabilities. NaN rows get global mean.
        """
        X_mat = self._prepare_features(X, fit_encoder=False)
        result = np.full(len(X), 0.19)  # Fallback for NaN rows
        mask = ~np.isnan(X_mat).any(axis=1)
        if mask.any():
            result[mask] = self._model.predict_proba(X_mat[mask])[:, 1]
        return result

    def feature_names(self) -> list[str]:
        return ["position_group"] + self.NUMERIC_FEATURES

    def _prepare_features(self, X: pd.DataFrame, fit_encoder: bool) -> np.ndarray:
        """Build feature matrix from DataFrame."""
        if fit_encoder:
            cat_encoded = self._encoder.fit_transform(X[["position_group"]])
        else:
            cat_encoded = self._encoder.transform(X[["position_group"]])
        numeric = X[self.NUMERIC_FEATURES].values.astype(float)
        return np.hstack([cat_encoded, numeric])


class EnrichedLogisticModel(BaseModel):
    """Logistic regression with ~15-20 features including edge, lineup, context, odds.

    Tests whether additional features add incremental value over the MVP set.
    """

    NUMERIC_FEATURES = [
        "expected_team_tries_5",
        "player_try_share_5",
        "is_home",
        "is_starter",
        "opponent_rolling_defence_tries_conceded_5",
        "rolling_try_rate_5",
        "rolling_line_breaks_5",
        "rolling_attack_tackle_breaks_3",
        "rolling_attack_tries_5",
        "opponent_rolling_defence_missed_tackles_5",
        "edge_matchup_score_rolling_5",
        "team_edge_try_share_rolling_5",
        "opponent_edge_conceded_rolling_5",
        "betfair_implied_prob",
    ]

    def __init__(self, C: float = 1.0) -> None:
        self._model = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs", random_state=42,
        )
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._fitted = False
        self._train_mean: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        X_mat = self._prepare_features(X, fit_encoder=True)
        y = np.asarray(y)
        # Impute NaN with column means for richer feature set
        self._train_mean = np.nanmean(X_mat, axis=0)
        # Replace any remaining NaN (all-NaN columns) with 0
        self._train_mean = np.where(np.isnan(self._train_mean), 0.0, self._train_mean)
        X_imp = np.where(np.isnan(X_mat), self._train_mean, X_mat)
        self._model.fit(X_imp, y)
        self._fitted = True
        LOGGER.info("EnrichedLogistic fitted on %d rows", len(y))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = self._prepare_features(X, fit_encoder=False)
        if self._train_mean is not None:
            X_mat = np.where(np.isnan(X_mat), self._train_mean, X_mat)
        return self._model.predict_proba(X_mat)[:, 1]

    def feature_names(self) -> list[str]:
        return ["position_group"] + self.NUMERIC_FEATURES

    def _prepare_features(self, X: pd.DataFrame, fit_encoder: bool) -> np.ndarray:
        if fit_encoder:
            cat_encoded = self._encoder.fit_transform(X[["position_group"]])
        else:
            cat_encoded = self._encoder.transform(X[["position_group"]])
        # Use available numeric features only
        available = [f for f in self.NUMERIC_FEATURES if f in X.columns]
        numeric = X[available].values.astype(float)
        # Pad missing features with NaN
        missing_count = len(self.NUMERIC_FEATURES) - len(available)
        if missing_count > 0:
            numeric = np.hstack([numeric, np.full((len(X), missing_count), np.nan)])
        return np.hstack([cat_encoded, numeric])


# ---------------------------------------------------------------------------
# Base strategy protocol
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Abstract base for betting strategies."""

    @abstractmethod
    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        """Select bets from a round's predictions.

        Parameters
        ----------
        predictions : pd.DataFrame
            One row per eligible player in the round. Must contain at least:
            match_id, player_id, position_code, betfair_implied_prob,
            betfair_closing_odds, scored_try.
        model : BaseModel, optional
            Fitted model (only needed for model-based strategies).

        Returns
        -------
        list[BetRecommendation]
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""


def _eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for position-eligible players with Betfair odds."""
    has_odds = df["betfair_implied_prob"].notna() & (df["betfair_implied_prob"] > 0)
    eligible_pos = df["position_code"].isin(ELIGIBLE_POSITION_CODES)
    return has_odds & eligible_pos


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class ModelEdgeStrategy(BaseStrategy):
    """Bet when model_prob > implied_prob + min_edge, only eligible positions."""

    def __init__(self, min_edge: float = MIN_EDGE_THRESHOLD) -> None:
        self._min_edge = min_edge

    @property
    def name(self) -> str:
        return "ModelEdge"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        if model is None:
            raise ValueError("ModelEdgeStrategy requires a model")

        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        model_probs = model.predict_proba(df)
        df = df.copy()
        df["model_prob"] = model_probs
        df["edge"] = df["model_prob"] - df["betfair_implied_prob"]

        bets = []
        for _, row in df[df["edge"] >= self._min_edge].iterrows():
            bets.append(BetRecommendation(
                match_id=int(row["match_id"]),
                player_id=int(row["player_id"]),
                model_prob=float(row["model_prob"]),
                implied_prob=float(row["betfair_implied_prob"]),
                odds=float(row["betfair_closing_odds"]),
                edge=float(row["edge"]),
                position_code=str(row["position_code"]),
            ))
        return bets


class SegmentPlayStrategy(BaseStrategy):
    """Rule-based: backs vs weak defence at odds 2.00-4.00.

    'Weak defence' = top quartile of opponent_rolling_defence_tries_conceded_5.
    """

    BACK_POSITIONS = {"FB", "WG", "CE"}

    def __init__(
        self,
        min_odds: float = 2.0,
        max_odds: float = 4.0,
        weak_defence_quantile: float = 0.75,
    ) -> None:
        self._min_odds = min_odds
        self._max_odds = max_odds
        self._quantile = weak_defence_quantile

    @property
    def name(self) -> str:
        return "SegmentPlay"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        col = "opponent_rolling_defence_tries_conceded_5"
        if col not in df.columns or df[col].isna().all():
            return []

        weak_threshold = df[col].quantile(self._quantile)
        selected = df[
            (df["position_code"].isin(self.BACK_POSITIONS))
            & (df["betfair_closing_odds"] >= self._min_odds)
            & (df["betfair_closing_odds"] <= self._max_odds)
            & (df[col] >= weak_threshold)
        ]

        bets = []
        for _, row in selected.iterrows():
            implied = float(row["betfair_implied_prob"])
            # Use position-average as crude model_prob estimate
            model_p = implied + 0.05  # Assume 5pp edge for rule-based
            bets.append(BetRecommendation(
                match_id=int(row["match_id"]),
                player_id=int(row["player_id"]),
                model_prob=model_p,
                implied_prob=implied,
                odds=float(row["betfair_closing_odds"]),
                edge=model_p - implied,
                position_code=str(row["position_code"]),
            ))
        return bets


class EdgeMatchupStrategy(BaseStrategy):
    """Rule-based: edge players with top-quartile edge_matchup_score."""

    EDGE_POSITIONS = {"WG", "CE", "SR"}

    def __init__(self, matchup_quantile: float = 0.75) -> None:
        self._quantile = matchup_quantile

    @property
    def name(self) -> str:
        return "EdgeMatchup"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        col = "edge_matchup_score_rolling_5"
        if col not in df.columns or df[col].isna().all():
            return []

        threshold = df[col].quantile(self._quantile)
        selected = df[
            (df["position_code"].isin(self.EDGE_POSITIONS))
            & (df[col] >= threshold)
            & (df[col] > 0)  # Must have positive matchup score
        ]

        bets = []
        for _, row in selected.iterrows():
            implied = float(row["betfair_implied_prob"])
            matchup_score = float(row[col])
            # Conservative edge: small fixed boost + capped matchup component
            model_p = implied + 0.03 + min(matchup_score * 0.05, 0.05)
            model_p = min(model_p, 0.95)
            bets.append(BetRecommendation(
                match_id=int(row["match_id"]),
                player_id=int(row["player_id"]),
                model_prob=model_p,
                implied_prob=implied,
                odds=float(row["betfair_closing_odds"]),
                edge=model_p - implied,
                position_code=str(row["position_code"]),
            ))
        return bets


class FadeHotStreakStrategy(BaseStrategy):
    """Contrarian: avoid players whose recent try rate is well above position average.

    Instead bets on 'cold' eligible players at good odds — regression to mean.
    """

    def __init__(self, streak_threshold: float = 1.5, min_odds: float = 2.0) -> None:
        self._streak_threshold = streak_threshold  # Multiple of position avg
        self._min_odds = min_odds

    @property
    def name(self) -> str:
        return "FadeHotStreak"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        rate_col = "rolling_try_rate_3"
        if rate_col not in df.columns or df[rate_col].isna().all():
            return []

        # Compute position average try rate
        pos_avg = df.groupby("position_code")[rate_col].transform("mean")
        df = df.copy()
        df["_pos_avg_rate"] = pos_avg

        # Select players NOT on hot streaks, at decent odds
        selected = df[
            (df[rate_col] <= df["_pos_avg_rate"] * self._streak_threshold)
            & (df["betfair_closing_odds"] >= self._min_odds)
            & (df["betfair_closing_odds"] <= 5.0)  # Not too long
        ]

        bets = []
        for _, row in selected.iterrows():
            implied = float(row["betfair_implied_prob"])
            model_p = implied + 0.04  # Assume regression-to-mean edge
            bets.append(BetRecommendation(
                match_id=int(row["match_id"]),
                player_id=int(row["player_id"]),
                model_prob=model_p,
                implied_prob=implied,
                odds=float(row["betfair_closing_odds"]),
                edge=model_p - implied,
                position_code=str(row["position_code"]),
            ))
        return bets


class MarketImpliedStrategy(BaseStrategy):
    """Control: bet where Betfair implied prob is high, in eligible positions.

    Tests if the market itself identifies profitable segments.
    """

    def __init__(self, min_implied_prob: float = 0.30) -> None:
        self._min_implied = min_implied_prob

    @property
    def name(self) -> str:
        return "MarketImplied"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        selected = df[df["betfair_implied_prob"] >= self._min_implied]

        bets = []
        for _, row in selected.iterrows():
            implied = float(row["betfair_implied_prob"])
            # Use implied as both model_prob and implied for control
            bets.append(BetRecommendation(
                match_id=int(row["match_id"]),
                player_id=int(row["player_id"]),
                model_prob=implied + 0.02,  # Small assumed edge
                implied_prob=implied,
                odds=float(row["betfair_closing_odds"]),
                edge=0.02,
                position_code=str(row["position_code"]),
            ))
        return bets


class RefinedEdgeStrategy(BaseStrategy):
    """Multi-condition bet selection combining model edge with segment filters.

    Requires ALL conditions to be met:
    - Model edge >= min_edge
    - Position in allowed set
    - Odds in specified range
    - Expected team tries above threshold (if available)

    Parameters
    ----------
    min_edge : float
        Minimum model-vs-market edge.
    positions : frozenset[str]
        Allowed position codes.
    min_odds : float
        Minimum closing odds.
    max_odds : float
        Maximum closing odds.
    min_team_tries : float
        Minimum expected team tries (context feature).
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        positions: frozenset[str] | None = None,
        min_odds: float = 2.0,
        max_odds: float = 5.0,
        min_team_tries: float = 4.0,
    ) -> None:
        self._min_edge = min_edge
        self._positions = positions or frozenset({"FB", "WG", "CE", "FE", "HB"})
        self._min_odds = min_odds
        self._max_odds = max_odds
        self._min_team_tries = min_team_tries

    @property
    def name(self) -> str:
        return "RefinedEdge"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        if model is None:
            raise ValueError("RefinedEdgeStrategy requires a model")

        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        model_probs = model.predict_proba(df)
        df = df.copy()
        df["model_prob"] = model_probs
        df["edge"] = df["model_prob"] - df["betfair_implied_prob"]

        # Apply all filters
        selected = df[
            (df["edge"] >= self._min_edge)
            & (df["position_code"].isin(self._positions))
            & (df["betfair_closing_odds"] >= self._min_odds)
            & (df["betfair_closing_odds"] <= self._max_odds)
        ]

        # Team tries filter (if feature available)
        team_tries_col = "expected_team_tries_5"
        if team_tries_col in selected.columns and selected[team_tries_col].notna().any():
            # Only apply to rows that have the feature
            has_feat = selected[team_tries_col].notna()
            selected = selected[
                (~has_feat) | (selected[team_tries_col] >= self._min_team_tries)
            ]

        bets = []
        for _, row in selected.iterrows():
            bets.append(BetRecommendation(
                match_id=int(row["match_id"]),
                player_id=int(row["player_id"]),
                model_prob=float(row["model_prob"]),
                implied_prob=float(row["betfair_implied_prob"]),
                odds=float(row["betfair_closing_odds"]),
                edge=float(row["edge"]),
                position_code=str(row["position_code"]),
            ))
        return bets


class CompositeStrategy(BaseStrategy):
    """Combines multiple strategies, deduplicates by (match_id, player_id), keeps highest edge."""

    def __init__(self, strategies: list[BaseStrategy]) -> None:
        self._strategies = strategies

    @property
    def name(self) -> str:
        names = "+".join(s.name for s in self._strategies)
        return f"Composite({names})"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        all_bets: dict[tuple[int, int], BetRecommendation] = {}
        for strategy in self._strategies:
            try:
                bets = strategy.select_bets(predictions, model=model)
            except ValueError:
                continue
            for bet in bets:
                key = (bet.match_id, bet.player_id)
                if key not in all_bets or bet.edge > all_bets[key].edge:
                    all_bets[key] = bet
        return list(all_bets.values())


class MarketBlendedStrategy(BaseStrategy):
    """Blend model predictions with market probabilities, then bet on edge.

    Uses ``blended_prob = alpha * model_prob + (1 - alpha) * market_prob``.
    Edge = blended_prob - market_prob. This produces smaller but more
    reliable edges by anchoring to the market.

    Parameters
    ----------
    alpha : float
        Weight on model (0.0 = pure market, 1.0 = pure model).
    min_edge : float
        Minimum blended edge to bet.
    positions : frozenset[str] | None
        Allowed position codes.
    min_odds : float
        Minimum closing odds.
    max_odds : float
        Maximum closing odds.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        min_edge: float = 0.03,
        positions: frozenset[str] | None = None,
        min_odds: float = 1.5,
        max_odds: float = 6.0,
    ) -> None:
        self._alpha = alpha
        self._min_edge = min_edge
        self._positions = positions or frozenset({"FB", "WG", "CE", "FE", "HB", "SR", "LK"})
        self._min_odds = min_odds
        self._max_odds = max_odds

    @property
    def name(self) -> str:
        return f"MarketBlend(a={self._alpha})"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        if model is None:
            raise ValueError("MarketBlendedStrategy requires a model")

        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        model_probs = model.predict_proba(df)
        df["model_prob"] = model_probs
        df["blended_prob"] = (
            self._alpha * df["model_prob"]
            + (1 - self._alpha) * df["betfair_implied_prob"]
        )
        df["edge"] = df["blended_prob"] - df["betfair_implied_prob"]

        selected = df[
            (df["edge"] >= self._min_edge)
            & (df["position_code"].isin(self._positions))
            & (df["betfair_closing_odds"] >= self._min_odds)
            & (df["betfair_closing_odds"] <= self._max_odds)
        ]

        bets = []
        for _, row in selected.iterrows():
            bets.append(BetRecommendation(
                match_id=int(row["match_id"]),
                player_id=int(row["player_id"]),
                model_prob=float(row["blended_prob"]),
                implied_prob=float(row["betfair_implied_prob"]),
                odds=float(row["betfair_closing_odds"]),
                edge=float(row["edge"]),
                position_code=str(row["position_code"]),
            ))
        return bets


class DataDrivenStrategy(BaseStrategy):
    """Bet only in segments empirically shown profitable in both seasons.

    Based on discovery analysis: Backs and Halfbacks at mid-short odds
    with specific feature thresholds. Uses dynamic edge thresholds
    by odds band.

    Parameters
    ----------
    alpha : float
        Market blending weight (0.0 = pure market, 1.0 = pure model).
    """

    # Odds bands with dynamic edge thresholds (from segment mining)
    ODDS_BANDS = {
        "short": (1.5, 2.5, 0.02),     # Short: small edge threshold (market accurate)
        "mid_short": (2.5, 4.0, 0.03), # Mid-short: sweet spot from discovery
        "mid_long": (4.0, 6.0, 0.05),  # Mid-long: need larger edge
        "long": (6.0, 15.0, 0.07),     # Long: highest noise, highest threshold
    }

    # Eligible positions (forwards excluded from long shots)
    ELIGIBLE_BY_BAND = {
        "short": frozenset({"FB", "WG", "CE", "FE", "HB", "SR", "LK", "PR", "HK"}),
        "mid_short": frozenset({"FB", "WG", "CE", "FE", "HB", "SR", "LK"}),
        "mid_long": frozenset({"FB", "WG", "CE", "FE", "HB"}),
        "long": frozenset({"FB", "WG", "CE"}),
    }

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha

    @property
    def name(self) -> str:
        return "DataDriven"

    def select_bets(
        self,
        predictions: pd.DataFrame,
        model: BaseModel | None = None,
    ) -> list[BetRecommendation]:
        if model is None:
            raise ValueError("DataDrivenStrategy requires a model")

        mask = _eligible_mask(predictions)
        df = predictions[mask].copy()
        if df.empty:
            return []

        model_probs = model.predict_proba(df)
        df["model_prob"] = model_probs
        df["blended_prob"] = (
            self._alpha * df["model_prob"]
            + (1 - self._alpha) * df["betfair_implied_prob"]
        )
        df["edge"] = df["blended_prob"] - df["betfair_implied_prob"]

        bets = []
        for _, row in df.iterrows():
            odds = row["betfair_closing_odds"]
            pos = row["position_code"]

            # Find which odds band this falls in
            for band_name, (lo, hi, min_edge) in self.ODDS_BANDS.items():
                if lo <= odds < hi:
                    # Check position eligibility for this band
                    if pos not in self.ELIGIBLE_BY_BAND[band_name]:
                        break
                    # Check dynamic edge threshold
                    if row["edge"] >= min_edge:
                        bets.append(BetRecommendation(
                            match_id=int(row["match_id"]),
                            player_id=int(row["player_id"]),
                            model_prob=float(row["blended_prob"]),
                            implied_prob=float(row["betfair_implied_prob"]),
                            odds=float(odds),
                            edge=float(row["edge"]),
                            position_code=str(pos),
                            metadata={"odds_band": band_name},
                        ))
                    break

        return bets
