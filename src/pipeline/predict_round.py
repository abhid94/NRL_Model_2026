"""Generate predictions for an upcoming round.

Takes a fitted model, the feature store, and a target round, then
produces per-player P(ATS) predictions with model-vs-market edge.
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from src.config import ELIGIBLE_POSITION_CODES
from src.features.feature_store import build_feature_store
from src.models.baseline import BaseModel

LOGGER = logging.getLogger(__name__)


def predict_round(
    model: BaseModel,
    feature_store: pd.DataFrame,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    """Generate predictions for a single round.

    Parameters
    ----------
    model : BaseModel
        Fitted model.
    feature_store : pd.DataFrame
        Full feature store (used to extract round data).
    season : int
        Season year.
    round_number : int
        Round to predict.

    Returns
    -------
    pd.DataFrame
        Predictions with columns: match_id, player_id, squad_id,
        position_code, position_group, model_prob, betfair_implied_prob,
        betfair_closing_odds, edge, is_eligible.
    """
    mask = (
        (feature_store["season"] == season)
        & (feature_store["round_number"] == round_number)
    )
    round_df = feature_store[mask].copy()

    if round_df.empty:
        LOGGER.warning("No data for season %d round %d", season, round_number)
        return pd.DataFrame()

    # Predict
    model_probs = model.predict_proba(round_df)
    round_df["model_prob"] = model_probs

    # Compute edge vs market
    round_df["edge"] = np.where(
        round_df["betfair_implied_prob"].notna(),
        round_df["model_prob"] - round_df["betfair_implied_prob"],
        np.nan,
    )

    # Mark eligibility
    round_df["is_eligible"] = (
        round_df["position_code"].isin(ELIGIBLE_POSITION_CODES)
        & round_df["betfair_implied_prob"].notna()
        & (round_df["betfair_implied_prob"] > 0)
    )

    # Select output columns
    output_cols = [
        "match_id", "player_id", "squad_id", "opponent_squad_id",
        "round_number", "season",
        "position_code", "position_group", "is_starter", "is_home",
        "model_prob", "betfair_implied_prob", "betfair_closing_odds",
        "edge", "is_eligible",
    ]
    available = [c for c in output_cols if c in round_df.columns]
    result = round_df[available].copy()

    # Sort by edge descending
    result = result.sort_values("edge", ascending=False, na_position="last")

    LOGGER.info(
        "Predictions for season %d round %d: %d players, %d eligible, %d with positive edge",
        season, round_number, len(result),
        result["is_eligible"].sum(),
        (result["edge"] > 0).sum() if "edge" in result.columns else 0,
    )

    return result.reset_index(drop=True)


def build_and_predict_round(
    conn: sqlite3.Connection,
    model: BaseModel,
    training_store: pd.DataFrame,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    """Build features for a round and generate predictions.

    This is the main entry point for weekly pipeline predictions.
    It rebuilds features for the target round using latest data,
    fits the model on all prior data, and predicts.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    model : BaseModel
        Model to fit and predict with.
    training_store : pd.DataFrame
        Historical feature store for training (all prior rounds).
    season : int
        Target season.
    round_number : int
        Target round.

    Returns
    -------
    pd.DataFrame
        Predictions (same format as predict_round).
    """
    # Build feature store for the target round
    LOGGER.info("Building features for season %d round %d", season, round_number)
    round_store = build_feature_store(
        conn, season, as_of_round=round_number, include_target=False,
    )

    if round_store.empty:
        LOGGER.warning("No feature store data for round %d", round_number)
        return pd.DataFrame()

    # Fit model on training data
    if "scored_try" in training_store.columns:
        LOGGER.info("Fitting model on %d training rows", len(training_store))
        model.fit(training_store, training_store["scored_try"].values)
    else:
        LOGGER.warning("No scored_try in training store; model must be pre-fitted")

    # Predict
    return predict_round(model, round_store, season, round_number)
