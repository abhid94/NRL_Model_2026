"""Optuna-based hyperparameter optimization for ATS models.

Uses Bayesian optimization with pruning to find optimal GBM
hyperparameters. Objective is walk-forward ROI on training data,
not log-loss — we optimize for the actual goal.

Usage:
    from src.models.hyperopt import optimize_gbm_params
    best_params = optimize_gbm_params(feature_store, n_trials=100)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def optimize_gbm_params(
    feature_store: pd.DataFrame,
    n_trials: int = 100,
    n_walk_forward_folds: int = 3,
    metric: str = "roi",
    min_bets_per_fold: int = 20,
    min_edge: float = 0.03,
    timeout: int | None = 600,
) -> dict[str, Any]:
    """Find optimal GBM hyperparameters using Optuna.

    Runs walk-forward cross-validation where each fold trains on all
    prior rounds and evaluates on the next chunk of rounds. The
    objective can be ROI, log-loss, or Brier score.

    Parameters
    ----------
    feature_store : pd.DataFrame
        Combined feature store with ``round_number``, ``season``,
        ``scored_try``, and ``betfair_implied_prob`` columns.
    n_trials : int
        Number of Optuna trials.
    n_walk_forward_folds : int
        Number of temporal folds for walk-forward validation.
    metric : str
        Optimization metric: "roi", "log_loss", or "brier".
    min_bets_per_fold : int
        Minimum number of bets per fold to count it.
    min_edge : float
        Minimum edge threshold for simulated bets (ROI metric only).
    timeout : int, optional
        Maximum seconds for the entire study.

    Returns
    -------
    dict[str, Any]
        Best parameters with keys: params, value, n_trials_completed,
        all_trials_summary.
    """
    import optuna
    from optuna.pruners import MedianPruner

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-sort data temporally
    df = feature_store.sort_values(["season", "round_number"]).reset_index(drop=True)
    rounds = df[["season", "round_number"]].drop_duplicates().sort_values(
        ["season", "round_number"]
    )
    round_list = list(rounds.itertuples(index=False, name=None))

    if len(round_list) < n_walk_forward_folds + 3:
        raise ValueError(
            f"Need at least {n_walk_forward_folds + 3} rounds, got {len(round_list)}"
        )

    # Define fold boundaries
    fold_size = max(1, len(round_list) // (n_walk_forward_folds + 1))
    folds = []
    for i in range(n_walk_forward_folds):
        train_end_idx = len(round_list) - (n_walk_forward_folds - i) * fold_size
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + fold_size

        if train_end_idx < 3 or val_end_idx > len(round_list):
            continue

        train_rounds = set(round_list[:train_end_idx])
        val_rounds = set(round_list[val_start_idx:val_end_idx])
        folds.append((train_rounds, val_rounds))

    LOGGER.info(
        "Optuna study: %d trials, %d folds, %d total rounds, metric=%s",
        n_trials, len(folds), len(round_list), metric,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        }

        fold_scores = []
        for fold_idx, (train_rounds, val_rounds) in enumerate(folds):
            score = _evaluate_fold(
                df, params, train_rounds, val_rounds,
                metric=metric, min_edge=min_edge,
                min_bets=min_bets_per_fold,
            )
            if score is None:
                continue
            fold_scores.append(score)

            # Report intermediate for pruning
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_scores:
            return float("-inf") if metric == "roi" else float("inf")

        return float(np.mean(fold_scores))

    # Direction depends on metric
    direction = "maximize" if metric == "roi" else "minimize"

    study = optuna.create_study(
        direction=direction,
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name="nrl_ats_gbm",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_trial
    LOGGER.info(
        "Optuna complete: best %s=%.4f, params=%s",
        metric, best.value, best.params,
    )

    # Summary of top trials
    trials_df = study.trials_dataframe()
    top_trials = (
        trials_df.nsmallest(10, "value")
        if direction == "minimize"
        else trials_df.nlargest(10, "value")
    )

    return {
        "params": best.params,
        "value": best.value,
        "n_trials_completed": len(study.trials),
        "top_trials": top_trials.to_dict("records") if not top_trials.empty else [],
    }


def _evaluate_fold(
    df: pd.DataFrame,
    params: dict[str, Any],
    train_rounds: set[tuple[int, int]],
    val_rounds: set[tuple[int, int]],
    metric: str = "roi",
    min_edge: float = 0.03,
    min_bets: int = 20,
) -> float | None:
    """Evaluate a single fold with given params.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature store.
    params : dict
        GBM hyperparameters.
    train_rounds : set
        (season, round_number) tuples for training.
    val_rounds : set
        (season, round_number) tuples for validation.
    metric : str
        "roi", "log_loss", or "brier".
    min_edge : float
        Min edge for ROI calculation.
    min_bets : int
        Min bets to count this fold.

    Returns
    -------
    float or None
        Score for this fold, or None if insufficient data.
    """
    from src.models.gbm import GBMModel

    train_key = df.apply(
        lambda r: (int(r["season"]), int(r["round_number"])), axis=1,
    )
    train_mask = train_key.isin(train_rounds)
    val_mask = train_key.isin(val_rounds)

    X_train, y_train = df[train_mask], df.loc[train_mask, "scored_try"].values
    X_val, y_val = df[val_mask], df.loc[val_mask, "scored_try"].values

    if len(X_train) < 100 or len(X_val) < 20:
        return None

    model = GBMModel(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)

    if metric == "log_loss":
        from sklearn.metrics import log_loss
        return log_loss(y_val, probs)

    if metric == "brier":
        return float(np.mean((probs - y_val) ** 2))

    # ROI metric — simulate flat-stake betting
    if "betfair_implied_prob" not in X_val.columns:
        return None

    implied = X_val["betfair_implied_prob"].values
    valid = (~np.isnan(implied)) & (~np.isnan(probs))
    edges = probs[valid] - implied[valid]
    outcomes = y_val[valid]
    odds = 1.0 / implied[valid]

    bet_mask = edges >= min_edge
    if bet_mask.sum() < min_bets:
        return None

    bet_outcomes = outcomes[bet_mask]
    bet_odds = odds[bet_mask]
    pnl = np.sum(np.where(bet_outcomes == 1, bet_odds - 1, -1))
    roi = pnl / bet_mask.sum()

    return float(roi)


def get_optimized_model(
    best_params: dict[str, Any],
    calibrated: bool = True,
    cal_rounds: int = 5,
) -> "BaseModel":
    """Create a model instance from optimized parameters.

    Parameters
    ----------
    best_params : dict
        Parameters from ``optimize_gbm_params()``.
    calibrated : bool
        If True, wrap in CalibratedModel.
    cal_rounds : int
        Calibration rounds.

    Returns
    -------
    BaseModel
        Ready-to-fit model.
    """
    from src.models.calibration import CalibratedModel
    from src.models.gbm import GBMModel

    gbm = GBMModel(**best_params)

    if calibrated:
        return CalibratedModel(gbm, method="sigmoid", cal_rounds=cal_rounds)
    return gbm
