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

import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_RESULTS_DIR,
    CLV_NEGATIVE_MULTIPLIER,
    CLV_NEGATIVE_ROUNDS_THRESHOLD,
    CLV_TABLE_NAME,
    DB_PATH,
    DEFAULT_INITIAL_BANKROLL,
    DEFAULT_KELLY_FRACTION,
    EARLY_SEASON_KELLY_MULTIPLIER,
    EARLY_SEASON_ROUNDS,
    FEATURE_STORE_DIR,
    MODEL_ARTIFACTS_DIR,
)
from src.db import get_connection, get_table, table_exists
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


def discover_training_seasons() -> list[int]:
    """Query the database for seasons that have player_stats data.

    Looks for ``player_stats_{year}`` tables with at least one row and
    returns the year suffixes sorted ascending.

    Returns
    -------
    list[int]
        Sorted list of season years with available training data.
    """
    conn = get_connection(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'player_stats_%'"
        ).fetchall()
        seasons: list[int] = []
        for (name,) in rows:
            suffix = name.replace("player_stats_", "")
            try:
                year = int(suffix)
            except ValueError:
                continue
            # Verify the table actually has data
            count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            if count > 0:
                seasons.append(year)
        return sorted(seasons)
    finally:
        conn.close()


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
        method="sigmoid",
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
    pull_odds: bool = True,
    exclude_player_ids: set[int] | None = None,
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
    pull_odds : bool
        If True, ingest latest team lists and odds from external APIs.
        If False, skip ingestion and use whatever is already in the DB.
    exclude_player_ids : set[int], optional
        Player IDs to exclude from predictions (e.g. late withdrawals).

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
        training_seasons = discover_training_seasons()
        if season not in training_seasons:
            training_seasons.append(season)

    # Step 0: Ingest team lists (idempotent — safe to re-run)
    tl_summary: dict = {}
    odds_summary: dict = {}
    if pull_odds:
        LOGGER.info("Step 0: Ingesting team lists for %d Round %d", season, round_number)
        try:
            from src.ingestion.ingest_team_lists import ingest_round_team_lists
            tl_summary = ingest_round_team_lists(round_number=round_number, year=season)
            LOGGER.info(
                "Team lists: %d/%d players matched (%d unmatched)",
                tl_summary["n_matched"],
                tl_summary["n_scraped"],
                tl_summary["n_unmatched"],
            )
            if tl_summary["unmatched_players"]:
                LOGGER.warning("Unmatched players: %s", tl_summary["unmatched_players"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Team list ingestion failed (non-fatal): %s", exc)
            tl_summary = {}

        # Step 0.5: Ingest bookmaker odds from The Odds API (non-fatal)
        LOGGER.info("Step 0.5: Ingesting bookmaker odds for %d Round %d", season, round_number)
        try:
            from src.odds.bookmaker import ingest_round_odds as _ingest_odds
            odds_summary = _ingest_odds(
                round_number=round_number, season=season, snapshot_type="closing",
            )
            LOGGER.info(
                "Bookmaker odds: %d raw, %d matched, %d upserted (%s)",
                odds_summary.get("n_raw", 0),
                odds_summary.get("n_matched", 0),
                odds_summary.get("n_upserted", 0),
                odds_summary.get("bookmakers", []),
            )
            if odds_summary.get("unmatched_players"):
                LOGGER.warning(
                    "Unmatched odds players: %s", odds_summary["unmatched_players"][:10]
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Odds API ingestion failed (non-fatal): %s", exc)
    else:
        LOGGER.info("Steps 0/0.5: Skipping odds ingestion (pull_odds=False)")

    # Step 0.75: Compute per-bookmaker margin corrections from actual data
    LOGGER.info("Step 0.75: Computing bookmaker margin corrections")
    try:
        from src.odds.bookmaker import compute_bookmaker_margins
        margin_conn = get_connection(DB_PATH)
        margins = compute_bookmaker_margins(margin_conn, season)
        margin_conn.close()
        LOGGER.info("  Margin corrections: %s", margins)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Margin computation failed (non-fatal, using defaults): %s", exc)

    # Step 1: Build/load feature stores
    LOGGER.info("Step 1: Loading feature stores")
    conn = get_connection(DB_PATH)

    training_dfs = []
    for train_season in training_seasons:
        # Skip current season from training when no prior rounds exist
        if train_season == season and round_number <= 1:
            LOGGER.info("  Skipping %d (no prior rounds for training)", train_season)
            continue

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

    # Step 2: Fit model (with recency weights for multi-season training)
    LOGGER.info("Step 2: Fitting model on training data")
    y_train = training_store["scored_try"].values
    sample_weight = compute_recency_weights(training_store, season)
    model.fit(training_store, y_train, sample_weight=sample_weight)
    LOGGER.info("  Model fitted: %s", type(model).__name__)

    # Step 3: Build features and predict for target round
    LOGGER.info("Step 3: Generating predictions for round %d", round_number)
    round_store = build_feature_store(
        conn, season, as_of_round=round_number, include_target=False,
    )
    # Add season column if missing
    if "season" not in round_store.columns:
        round_store["season"] = season

    predictions = predict_round(
        model, round_store, season, round_number, conn=conn,
        exclude_player_ids=exclude_player_ids,
    )
    LOGGER.info("  Predictions: %d players", len(predictions))

    conn.close()

    # Step 4: Drawdown check (BEFORE bet generation — fixes wiring bug)
    drawdown_status = check_drawdown(bankroll)
    LOGGER.info("Step 4: Drawdown status: %s", drawdown_status["status"])

    if drawdown_status["status"] in ("HALT", "STOP"):
        LOGGER.warning(drawdown_status["message"])
        bet_card = BetCard(
            season=season, round_number=round_number, bankroll=bankroll,
            bets=[], total_staked=0.0, exposure_pct=0.0, n_matches_bet=0,
        )
    else:
        # Step 5: Compute adaptive Kelly fraction and generate bets
        LOGGER.info("Step 5: Generating bet recommendations")
        adaptive_fraction = compute_adaptive_kelly(
            round_number=round_number,
            drawdown_adjustment=drawdown_status["kelly_adjustment"],
            bankroll=bankroll,
            season=season,
        )
        LOGGER.info("  Adaptive Kelly fraction: %.3f (base=%.2f)", adaptive_fraction, DEFAULT_KELLY_FRACTION)
        bet_card = generate_bet_card(
            predictions,
            bankroll=bankroll,
            kelly_fraction=adaptive_fraction,
            flat_stake=flat_stake,
        )
    LOGGER.info("  Bet card: %d bets, $%.0f staked", len(bet_card.bets), bet_card.total_staked)

    # Step 6: Log predictions
    LOGGER.info("Step 6: Logging predictions and CLV data")
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


def compute_recency_weights(
    training_store: pd.DataFrame,
    current_season: int,
) -> "np.ndarray":
    """Compute per-sample recency weights for training data.

    More recent seasons get higher weights:
    - Current season: 1.0
    - Previous season: 0.8
    - Two seasons ago: 0.5
    - Older: 0.3

    Parameters
    ----------
    training_store : pd.DataFrame
        Must contain a ``season`` column.
    current_season : int
        The season being predicted.

    Returns
    -------
    np.ndarray
        Weight per training sample.
    """
    weight_map = {
        0: 1.0,   # current season
        1: 0.8,   # previous season
        2: 0.5,   # two seasons ago
    }
    default_weight = 0.3

    seasons = training_store["season"].values
    age = current_season - seasons
    weights = np.array([weight_map.get(int(a), default_weight) for a in age])

    unique_weights = {int(s): float(w) for s, w in zip(
        np.unique(seasons), [weight_map.get(current_season - int(s), default_weight) for s in np.unique(seasons)]
    )}
    LOGGER.info("Recency weights: %s", unique_weights)

    return weights


def compute_adaptive_kelly(
    round_number: int,
    drawdown_adjustment: float = 1.0,
    bankroll: float = DEFAULT_INITIAL_BANKROLL,
    season: int = 2026,
) -> float:
    """Compute adaptive Kelly fraction based on context.

    Combines multiple adjustment factors:
    - Early-season (rounds 1-4): 0.5x (thin data)
    - Drawdown: from check_drawdown() kelly_adjustment
    - CLV trend: if negative CLV for 3+ consecutive rounds, 0.6x

    Parameters
    ----------
    round_number : int
        Current round number.
    drawdown_adjustment : float
        Kelly multiplier from drawdown check (1.0 = normal, 0.6 = warning).
    bankroll : float
        Current bankroll.
    season : int
        Current season.

    Returns
    -------
    float
        Adjusted Kelly fraction.
    """
    fraction = DEFAULT_KELLY_FRACTION

    # Early-season reduction
    if round_number <= EARLY_SEASON_ROUNDS:
        fraction *= EARLY_SEASON_KELLY_MULTIPLIER
        LOGGER.info("  Early season (round %d <= %d): Kelly *= %.1f",
                     round_number, EARLY_SEASON_ROUNDS, EARLY_SEASON_KELLY_MULTIPLIER)

    # Drawdown adjustment
    if drawdown_adjustment < 1.0:
        fraction *= drawdown_adjustment
        LOGGER.info("  Drawdown adjustment: Kelly *= %.1f", drawdown_adjustment)

    # CLV trend adjustment
    clv_multiplier = _get_clv_kelly_multiplier(season)
    if clv_multiplier < 1.0:
        fraction *= clv_multiplier
        LOGGER.info("  CLV trend negative: Kelly *= %.1f", clv_multiplier)

    return fraction


def _get_clv_kelly_multiplier(season: int) -> float:
    """Check recent CLV trend and return Kelly multiplier.

    If CLV has been negative for CLV_NEGATIVE_ROUNDS_THRESHOLD or more
    consecutive rounds, return CLV_NEGATIVE_MULTIPLIER. Otherwise 1.0.

    Parameters
    ----------
    season : int
        Current season.

    Returns
    -------
    float
        1.0 (normal) or CLV_NEGATIVE_MULTIPLIER.
    """
    try:
        conn = get_connection(DB_PATH)
        if not table_exists(conn, CLV_TABLE_NAME):
            conn.close()
            return 1.0

        clv_df = pd.read_sql_query(
            f"""
            SELECT round_number, AVG(clv) AS avg_clv
            FROM {CLV_TABLE_NAME}
            WHERE season = ?
            GROUP BY round_number
            ORDER BY round_number DESC
            LIMIT ?
            """,
            conn,
            params=(season, CLV_NEGATIVE_ROUNDS_THRESHOLD),
        )
        conn.close()

        if len(clv_df) < CLV_NEGATIVE_ROUNDS_THRESHOLD:
            return 1.0

        # Check if all recent rounds have negative CLV
        if (clv_df["avg_clv"] < 0).all():
            return CLV_NEGATIVE_MULTIPLIER

        return 1.0
    except Exception:
        return 1.0


def record_clv(
    season: int,
    round_number: int,
    bet_records: list[dict],
    conn: "sqlite3.Connection | None" = None,
) -> int:
    """Record CLV data after a round completes.

    Compares model probability at bet time vs closing line probability.

    Parameters
    ----------
    season : int
        Season year.
    round_number : int
        Round number.
    bet_records : list[dict]
        Each dict must have: player_id, match_id, model_prob, closing_prob.
        Optionally: bookmaker.
    conn : sqlite3.Connection, optional
        Database connection. Opens new one if None.

    Returns
    -------
    int
        Number of CLV records inserted.
    """
    import sqlite3 as _sqlite3

    close_conn = False
    if conn is None:
        conn = get_connection(DB_PATH)
        close_conn = True

    try:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {CLV_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER NOT NULL,
                round_number INTEGER NOT NULL,
                match_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                model_prob REAL,
                closing_prob REAL,
                clv REAL,
                bookmaker TEXT,
                timestamp TEXT,
                UNIQUE(season, round_number, match_id, player_id)
            )
        """)
        conn.commit()

        timestamp = datetime.now().isoformat()
        n_inserted = 0
        for rec in bet_records:
            model_prob = rec.get("model_prob")
            closing_prob = rec.get("closing_prob")
            clv = (model_prob - closing_prob) if (model_prob is not None and closing_prob is not None) else None

            try:
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {CLV_TABLE_NAME}
                        (season, round_number, match_id, player_id,
                         model_prob, closing_prob, clv, bookmaker, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        season, round_number, rec["match_id"], rec["player_id"],
                        model_prob, closing_prob, clv,
                        rec.get("bookmaker", ""), timestamp,
                    ),
                )
                n_inserted += 1
            except _sqlite3.Error as exc:
                LOGGER.warning("Failed to insert CLV record: %s", exc)

        conn.commit()
        LOGGER.info("Recorded %d CLV entries for season %d round %d", n_inserted, season, round_number)
        return n_inserted

    finally:
        if close_conn:
            conn.close()


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
