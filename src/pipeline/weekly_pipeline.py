"""End-to-end weekly pipeline for 2026 ATS predictions.

Orchestrates the full workflow from CLAUDE.md Section 8:
1. Rebuild feature store with new round data
2. Retrain model on all available data
3. Generate predictions for upcoming round
4. Produce bet recommendations with stake sizing
5. Log predictions for ongoing evaluation
6. Monitor drawdown and risk controls

Usage:
    python -m src.pipeline.weekly_pipeline --season 2026 --round 5
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    BACKTEST_RESULTS_DIR,
    DB_PATH,
    DEFAULT_INITIAL_BANKROLL,
    DEFAULT_KELLY_FRACTION,
    FEATURE_STORE_DIR,
    MODEL_ARTIFACTS_DIR,
)
from src.db import get_connection
from src.features.feature_store import build_feature_store, save_feature_store
from src.models.baseline import BaseModel
from src.models.calibration import CalibratedModel
from src.models.gbm import GBMModel
from src.pipeline.bet_recommendations import BetCard, generate_bet_card
from src.pipeline.predict_round import predict_round

LOGGER = logging.getLogger(__name__)

# Drawdown thresholds from CLAUDE.md Section 11
DRAWDOWN_WARNING = 0.15
DRAWDOWN_HALT = 0.25
DRAWDOWN_STOP = 0.40


def get_default_model() -> BaseModel:
    """Return the default production model.

    Returns
    -------
    BaseModel
        CalibratedGBM with regularization (Phase 5B best config).
        n=150, depth=4, reg=3.0, min_child=80 — tuned for cross-season stability.
    """
    return CalibratedModel(
        GBMModel(
            n_estimators=150,
            max_depth=4,
            reg_alpha=3.0,
            reg_lambda=3.0,
            min_child_samples=80,
        ),
        method="isotonic",
        cal_rounds=5,
    )


def run_weekly_pipeline(
    season: int,
    round_number: int,
    bankroll: float = DEFAULT_INITIAL_BANKROLL,
    model: BaseModel | None = None,
    training_seasons: list[int] | None = None,
    rebuild_features: bool = True,
    flat_stake: float | None = None,
) -> dict[str, Any]:
    """Run the full weekly pipeline.

    Parameters
    ----------
    season : int
        Current season (e.g. 2026).
    round_number : int
        Round to predict.
    bankroll : float
        Current bankroll.
    model : BaseModel, optional
        Model to use. Defaults to CalibratedGBM.
    training_seasons : list[int], optional
        Seasons to include in training. Defaults to [2024, 2025] + current.
    rebuild_features : bool
        If True, rebuild feature stores from DB. If False, load from disk.
    flat_stake : float | None
        If set, use flat staking instead of Kelly.

    Returns
    -------
    dict[str, Any]
        Pipeline results with predictions, bet card, and metadata.
    """
    start_time = datetime.now()
    LOGGER.info("=" * 60)
    LOGGER.info("Weekly Pipeline: Season %d Round %d", season, round_number)
    LOGGER.info("=" * 60)

    if model is None:
        model = get_default_model()

    if training_seasons is None:
        training_seasons = [2024, 2025]
        if season not in training_seasons:
            training_seasons.append(season)

    # Step 1: Build/load feature stores
    LOGGER.info("Step 1: Loading feature stores")
    conn = get_connection(DB_PATH)

    training_dfs = []
    for train_season in training_seasons:
        if rebuild_features:
            LOGGER.info("  Rebuilding features for %d", train_season)
            max_round = round_number - 1 if train_season == season else None
            df = build_feature_store(conn, train_season, as_of_round=max_round)
            save_feature_store(
                df,
                str(FEATURE_STORE_DIR / f"feature_store_{train_season}.parquet"),
            )
        else:
            path = FEATURE_STORE_DIR / f"feature_store_{train_season}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                # Filter training data for current season
                if train_season == season:
                    df = df[df["round_number"] < round_number]
                LOGGER.info("  Loaded %d rows from %s", len(df), path)
            else:
                LOGGER.warning("  Feature store not found: %s, rebuilding", path)
                max_round = round_number - 1 if train_season == season else None
                df = build_feature_store(conn, train_season, as_of_round=max_round)
        training_dfs.append(df)

    training_store = pd.concat(training_dfs, ignore_index=True)
    LOGGER.info("Training data: %d rows across seasons %s", len(training_store), training_seasons)

    # Step 2: Fit model
    LOGGER.info("Step 2: Fitting model on training data")
    y_train = training_store["scored_try"].values
    model.fit(training_store, y_train)
    LOGGER.info("  Model fitted: %s", type(model).__name__)

    # Step 3: Build features and predict for target round
    LOGGER.info("Step 3: Generating predictions for round %d", round_number)
    round_store = build_feature_store(
        conn, season, as_of_round=round_number, include_target=False,
    )
    # Add season column if missing
    if "season" not in round_store.columns:
        round_store["season"] = season

    predictions = predict_round(model, round_store, season, round_number)
    LOGGER.info("  Predictions: %d players", len(predictions))

    conn.close()

    # Step 4: Generate bet recommendations
    LOGGER.info("Step 4: Generating bet recommendations")
    bet_card = generate_bet_card(
        predictions,
        bankroll=bankroll,
        flat_stake=flat_stake,
    )
    LOGGER.info("  Bet card: %d bets, $%.0f staked", len(bet_card.bets), bet_card.total_staked)

    # Step 5: Drawdown check
    drawdown_status = check_drawdown(bankroll)
    LOGGER.info("Step 5: Drawdown status: %s", drawdown_status["status"])

    # Step 6: Log predictions
    LOGGER.info("Step 6: Logging predictions")
    log_entry = log_predictions(
        season, round_number, predictions, bet_card, bankroll,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    LOGGER.info("Pipeline complete in %.1f seconds", elapsed)

    return {
        "predictions": predictions,
        "bet_card": bet_card,
        "training_rows": len(training_store),
        "drawdown_status": drawdown_status,
        "log_entry": log_entry,
        "elapsed_seconds": elapsed,
    }


def check_drawdown(
    current_bankroll: float,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
) -> dict[str, Any]:
    """Check drawdown level and return risk status.

    Parameters
    ----------
    current_bankroll : float
        Current bankroll value.
    initial_bankroll : float
        Starting bankroll value.

    Returns
    -------
    dict[str, Any]
        Keys: drawdown_pct, status, kelly_adjustment, message.
    """
    if initial_bankroll <= 0:
        return {"drawdown_pct": 0.0, "status": "ok", "kelly_adjustment": 1.0, "message": ""}

    drawdown_pct = (initial_bankroll - current_bankroll) / initial_bankroll

    if drawdown_pct >= DRAWDOWN_STOP:
        return {
            "drawdown_pct": drawdown_pct,
            "status": "STOP",
            "kelly_adjustment": 0.0,
            "message": f"STOP: Drawdown {drawdown_pct*100:.1f}% exceeds {DRAWDOWN_STOP*100:.0f}% threshold. "
                       f"Halt all betting. Conduct fundamental review.",
        }
    elif drawdown_pct >= DRAWDOWN_HALT:
        return {
            "drawdown_pct": drawdown_pct,
            "status": "HALT",
            "kelly_adjustment": 0.0,
            "message": f"HALT: Drawdown {drawdown_pct*100:.1f}% exceeds {DRAWDOWN_HALT*100:.0f}% threshold. "
                       f"Pause for 2 rounds and audit.",
        }
    elif drawdown_pct >= DRAWDOWN_WARNING:
        return {
            "drawdown_pct": drawdown_pct,
            "status": "WARNING",
            "kelly_adjustment": 0.6,  # Reduce to 15% Kelly (from 25%)
            "message": f"WARNING: Drawdown {drawdown_pct*100:.1f}% exceeds {DRAWDOWN_WARNING*100:.0f}% threshold. "
                       f"Reducing Kelly fraction.",
        }
    else:
        return {
            "drawdown_pct": drawdown_pct,
            "status": "OK",
            "kelly_adjustment": 1.0,
            "message": f"OK: Drawdown {drawdown_pct*100:.1f}%. Within normal range.",
        }


def log_predictions(
    season: int,
    round_number: int,
    predictions: pd.DataFrame,
    bet_card: BetCard,
    bankroll: float,
) -> dict[str, Any]:
    """Log predictions and recommendations for future evaluation.

    Parameters
    ----------
    season : int
        Season year.
    round_number : int
        Round number.
    predictions : pd.DataFrame
        Full predictions.
    bet_card : BetCard
        Bet recommendations.
    bankroll : float
        Current bankroll.

    Returns
    -------
    dict[str, Any]
        Log entry metadata.
    """
    log_dir = BACKTEST_RESULTS_DIR / "prediction_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    log_id = f"{season}_R{round_number:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Save predictions
    if not predictions.empty:
        pred_path = log_dir / f"predictions_{log_id}.csv"
        predictions.to_csv(pred_path, index=False)
        LOGGER.info("  Saved predictions to %s", pred_path)

    # Save bet card
    bet_df = bet_card.to_dataframe()
    if not bet_df.empty:
        bets_path = log_dir / f"bets_{log_id}.csv"
        bet_df.to_csv(bets_path, index=False)
        LOGGER.info("  Saved bets to %s", bets_path)

    # Save metadata
    meta = {
        "timestamp": timestamp,
        "season": season,
        "round_number": round_number,
        "bankroll": bankroll,
        "n_predictions": len(predictions),
        "n_bets": len(bet_card.bets),
        "total_staked": bet_card.total_staked,
        "exposure_pct": bet_card.exposure_pct,
    }
    meta_path = log_dir / f"meta_{log_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_prediction_log(
    season: int,
    round_number: int | None = None,
) -> pd.DataFrame:
    """Load historical prediction logs.

    Parameters
    ----------
    season : int
        Season year.
    round_number : int, optional
        If provided, load only this round.

    Returns
    -------
    pd.DataFrame
        Prediction log entries.
    """
    log_dir = BACKTEST_RESULTS_DIR / "prediction_logs"
    if not log_dir.exists():
        return pd.DataFrame()

    pattern = f"predictions_{season}_R*.csv"
    if round_number is not None:
        pattern = f"predictions_{season}_R{round_number:02d}_*.csv"

    files = sorted(log_dir.glob(pattern))
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)
