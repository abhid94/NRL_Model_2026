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
    conn: sqlite3.Connection | None = None,
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
    conn : sqlite3.Connection, optional
        Database connection. If provided and season >= 2026, multi-bookmaker
        odds are loaded and used for edge calculation.

    Returns
    -------
    pd.DataFrame
        Predictions with columns: match_id, player_id, squad_id,
        position_code, position_group, model_prob, betfair_implied_prob,
        betfair_closing_odds, edge, is_eligible, and (if bookmaker odds
        available) best_odds, best_bookmaker, best_implied_prob.
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

    # Enrich with multi-bookmaker odds for 2026+ seasons
    if conn is not None and season >= 2026:
        round_df = _add_bookmaker_odds(round_df, conn, round_number, season)

    # Mark eligibility — use best_implied_prob when available (bookmaker we'd bet with)
    implied_col = (
        "best_implied_prob"
        if "best_implied_prob" in round_df.columns
        else "betfair_implied_prob"
    )
    round_df["is_eligible"] = (
        round_df["position_code"].isin(ELIGIBLE_POSITION_CODES)
        & round_df[implied_col].notna()
        & (round_df[implied_col] > 0)
    )

    # Add player name from DB
    if conn is not None:
        round_df = _add_player_names(round_df, conn, season)

    # Add per-match rank by model probability (1 = most likely to score)
    round_df["match_rank"] = (
        round_df.groupby("match_id")["model_prob"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Select output columns — include per-bookmaker odds columns
    output_cols = [
        "match_id", "player_id", "player_name",
        "squad_id", "opponent_squad_id",
        "round_number", "season",
        "position_code", "position_group", "is_starter", "is_home",
        "model_prob", "match_rank",
        "betfair_implied_prob", "betfair_closing_odds",
        "edge", "is_eligible",
        "best_odds", "best_bookmaker", "best_implied_prob",
    ]
    # Append any odds_<bookmaker> columns
    odds_cols = sorted([c for c in round_df.columns if c.startswith("odds_")])
    output_cols.extend(odds_cols)

    available = [c for c in output_cols if c in round_df.columns]
    result = round_df[available].copy()

    # Sort by match then rank
    result = result.sort_values(
        ["match_id", "match_rank"], ascending=[True, True],
    )

    LOGGER.info(
        "Predictions for season %d round %d: %d players, %d eligible, %d with positive edge",
        season, round_number, len(result),
        result["is_eligible"].sum(),
        (result["edge"] > 0).sum() if "edge" in result.columns else 0,
    )

    return result.reset_index(drop=True)


def _add_player_names(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    season: int,
) -> pd.DataFrame:
    """Add player_name column by looking up team_lists or players table.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain player_id column.
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.

    Returns
    -------
    pd.DataFrame
        With player_name column added.
    """
    from src.db import table_exists

    name_df = pd.DataFrame()

    # Try team_lists first (has player_name directly)
    tl_table = f"team_lists_{season}"
    if table_exists(conn, tl_table):
        name_df = pd.read_sql_query(
            f"SELECT DISTINCT player_id, player_name FROM {tl_table} WHERE player_id IS NOT NULL",
            conn,
        )

    # Fallback to players table
    if name_df.empty:
        for yr in [season, season - 1]:
            p_table = f"players_{yr}"
            if table_exists(conn, p_table):
                name_df = pd.read_sql_query(
                    f"SELECT DISTINCT player_id, display_name AS player_name FROM {p_table}",
                    conn,
                )
                break

    if not name_df.empty:
        df = df.merge(name_df, on="player_id", how="left")
    else:
        df["player_name"] = None

    return df


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


def _add_bookmaker_odds(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    round_number: int,
    season: int,
) -> pd.DataFrame:
    """Enrich predictions with multi-bookmaker odds.

    Adds best_odds, best_bookmaker, best_implied_prob, and per-bookmaker
    price columns. Recomputes edge against the best bookmaker price.

    Parameters
    ----------
    df : pd.DataFrame
        Round predictions (must have match_id, player_id, model_prob).
    conn : sqlite3.Connection
        Database connection.
    round_number : int
        Round number.
    season : int
        Season year.

    Returns
    -------
    pd.DataFrame
        Enriched predictions.
    """
    try:
        from src.odds.bookmaker import get_round_bookmaker_odds

        bk_df = get_round_bookmaker_odds(conn, round_number, season)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not load bookmaker odds: %s", exc)
        return df

    if bk_df.empty:
        LOGGER.info("No bookmaker odds available for round %d", round_number)
        return df

    # Merge bookmaker odds onto predictions
    merge_cols = [c for c in bk_df.columns if c not in ("match_id", "player_id")]
    df = df.merge(
        bk_df,
        on=["match_id", "player_id"],
        how="left",
    )

    # Recompute edge against best bookmaker implied prob
    if "best_implied_prob" in df.columns:
        df["edge"] = np.where(
            df["best_implied_prob"].notna(),
            df["model_prob"] - df["best_implied_prob"],
            df["edge"],
        )

    n_with_bk = df["best_odds"].notna().sum() if "best_odds" in df.columns else 0
    LOGGER.info(
        "Bookmaker odds: %d/%d players have bookmaker prices", n_with_bk, len(df)
    )

    return df
