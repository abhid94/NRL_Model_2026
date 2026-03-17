"""Hierarchical Bayesian model for ATS prediction using PyMC.

Implements a structured Bayesian logistic regression with:
- Team attack strength (how many tries the team scores)
- Team defence weakness (how many tries the team concedes)
- Position-specific try rate priors (wings ~48%, props ~9%)
- Player-specific random effects (shrunk toward position mean)
- Home advantage effect

The model uses NUTS sampling and stores posterior means for fast
prediction at inference time. New players/teams not seen during
training are handled gracefully via position-group priors and zero
effects respectively.

PyMC is imported lazily inside methods so the module loads even
without pymc installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid / logistic function

from src.config import POSITION_TRY_RATES
from src.models.baseline import BaseModel

LOGGER = logging.getLogger(__name__)

# Columns required by the Bayesian model
REQUIRED_COLUMNS = ("squad_id", "opponent_squad_id", "position_code", "player_id", "is_home")

# Position codes ordered for consistent indexing
_DEFAULT_POSITION_ORDER = (
    "FB", "WG", "CE", "FE", "HB", "SR", "LK", "HK", "PR", "INT", "RES",
)


def _position_logit_prior(position_code: str) -> float:
    """Convert a position try rate to logit scale for use as a prior mean.

    Parameters
    ----------
    position_code : str
        Position code (e.g. "WG", "FB", "PR").

    Returns
    -------
    float
        Log-odds of the position try rate.
    """
    rate = POSITION_TRY_RATES.get(position_code, 0.19)
    # Clip to avoid log(0) or log(inf)
    rate = np.clip(rate, 0.01, 0.99)
    return float(np.log(rate / (1.0 - rate)))


class HierarchicalBayesianModel(BaseModel):
    """Hierarchical Bayesian logistic regression for ATS prediction.

    The generative model on the logit scale is::

        logit(p_i) = intercept
                   + attack[squad_i]
                   + defence[opponent_i]
                   + position[pos_i]
                   + player[player_i]
                   + home_advantage * is_home_i

    where:
    - ``attack[t] ~ Normal(0, sigma_attack)`` captures team-level
      attacking strength (positive = scores more tries).
    - ``defence[t] ~ Normal(0, sigma_defence)`` captures team-level
      defensive weakness (positive = concedes more tries).
    - ``position[p]`` has an informative prior centered on the empirical
      position try rate (logit scale) from 2024-2025 data.
    - ``player[j] ~ Normal(0, sigma_player)`` is a player-level random
      effect shrunk toward zero (and therefore toward the position mean).
    - ``home_advantage ~ Normal(0.1, 0.1)`` captures the ~1.8pp home
      try rate advantage observed in the data.

    Posterior means are stored after sampling for fast prediction.

    Parameters
    ----------
    draws : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning (warmup) steps per chain.
    chains : int
        Number of MCMC chains.
    target_accept : float
        Target acceptance rate for NUTS.
    random_seed : int
        Random seed for reproducibility.

    Examples
    --------
    >>> model = HierarchicalBayesianModel(draws=500, tune=250, chains=2)
    >>> model.fit(X_train, y_train)
    >>> probs = model.predict_proba(X_test)
    """

    def __init__(
        self,
        draws: int = 1000,
        tune: int = 500,
        chains: int = 2,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> None:
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._target_accept = target_accept
        self._random_seed = random_seed

        # Populated during fit()
        self._fitted: bool = False
        self._intercept: float = 0.0
        self._home_advantage: float = 0.0
        self._attack: dict[int, float] = {}       # squad_id -> posterior mean
        self._defence: dict[int, float] = {}       # squad_id -> posterior mean
        self._position: dict[str, float] = {}      # position_code -> posterior mean
        self._player: dict[int, float] = {}        # player_id -> posterior mean

        # Index mappings built during fit (entity -> integer index)
        self._squad_idx: dict[int, int] = {}
        self._position_idx: dict[str, int] = {}
        self._player_idx: dict[int, int] = {}
        self._player_position: dict[int, str] = {} # player_id -> position_code

        # Trace stored for diagnostics
        self._trace: Any = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> None:
        """Fit the hierarchical Bayesian model via NUTS sampling.

        Parameters
        ----------
        X : pd.DataFrame
            Training data. Must contain columns: ``squad_id``,
            ``opponent_squad_id``, ``position_code``, ``player_id``,
            ``is_home``.
        y : array-like
            Binary target (1 = scored a try, 0 = did not).
        **kwargs
            Ignored (accepted for interface compatibility).

        Raises
        ------
        ImportError
            If PyMC is not installed.
        ValueError
            If required columns are missing from X.
        """
        import pymc as pm

        y = np.asarray(y, dtype=int)
        self._validate_columns(X)

        # Build integer index mappings for categorical entities
        squads = sorted(X["squad_id"].unique())
        positions = sorted(X["position_code"].unique())
        players = sorted(X["player_id"].unique())

        self._squad_idx = {sid: i for i, sid in enumerate(squads)}
        self._position_idx = {pc: i for i, pc in enumerate(positions)}
        self._player_idx = {pid: i for i, pid in enumerate(players)}

        # Map each player to their most frequent position (for priors on new data)
        player_pos = (
            X.groupby("player_id")["position_code"]
            .agg(lambda s: s.mode().iloc[0])
        )
        self._player_position = player_pos.to_dict()

        n_squads = len(squads)
        n_positions = len(positions)
        n_players = len(players)

        # Convert data to integer indices
        squad_idx_arr = X["squad_id"].map(self._squad_idx).values.astype(int)
        opp_idx_arr = X["opponent_squad_id"].map(self._squad_idx).values.astype(int)
        pos_idx_arr = X["position_code"].map(self._position_idx).values.astype(int)
        player_idx_arr = X["player_id"].map(self._player_idx).values.astype(int)
        is_home_arr = X["is_home"].values.astype(float)

        # Build informative position priors on logit scale
        position_prior_means = np.array([
            _position_logit_prior(pc) for pc in positions
        ])

        LOGGER.info(
            "Building PyMC model: %d observations, %d squads, %d positions, %d players",
            len(y), n_squads, n_positions, n_players,
        )

        with pm.Model() as model:
            # ---- Hyperpriors ----
            sigma_attack = pm.HalfNormal("sigma_attack", sigma=0.5)
            sigma_defence = pm.HalfNormal("sigma_defence", sigma=0.5)
            sigma_player = pm.HalfNormal("sigma_player", sigma=0.3)

            # ---- Global intercept ----
            # Prior centered near overall logit(0.19) = -1.45
            intercept = pm.Normal("intercept", mu=-1.45, sigma=0.5)

            # ---- Home advantage ----
            # Prior centered on logit-scale effect of ~1.8pp home advantage
            # At baseline 19%, going to 19.9% is about +0.05 on logit scale
            home_advantage = pm.Normal("home_advantage", mu=0.1, sigma=0.1)

            # ---- Team attack strength ----
            attack = pm.Normal("attack", mu=0.0, sigma=sigma_attack, shape=n_squads)

            # ---- Team defence weakness ----
            defence = pm.Normal("defence", mu=0.0, sigma=sigma_defence, shape=n_squads)

            # ---- Position effects ----
            # Informative priors from empirical position try rates
            # The intercept absorbs the global mean, so position effects
            # are deviations from that mean
            position_effect = pm.Normal(
                "position_effect",
                mu=position_prior_means - (-1.45),  # deviation from intercept prior
                sigma=0.3,
                shape=n_positions,
            )

            # ---- Player random effects ----
            # Shrunk toward zero (toward position mean via the position effect)
            player_effect = pm.Normal(
                "player_effect", mu=0.0, sigma=sigma_player, shape=n_players,
            )

            # ---- Linear predictor ----
            mu = (
                intercept
                + attack[squad_idx_arr]
                + defence[opp_idx_arr]
                + position_effect[pos_idx_arr]
                + player_effect[player_idx_arr]
                + home_advantage * is_home_arr
            )

            # ---- Likelihood ----
            pm.Bernoulli("y_obs", logit_p=mu, observed=y)

        LOGGER.info(
            "Sampling: %d draws, %d tune, %d chains (target_accept=%.2f)",
            self._draws, self._tune, self._chains, self._target_accept,
        )

        with model:
            self._trace = pm.sample(
                draws=self._draws,
                tune=self._tune,
                chains=self._chains,
                target_accept=self._target_accept,
                random_seed=self._random_seed,
                progressbar=True,
                return_inferencedata=True,
            )

        # Extract posterior means
        self._intercept = float(self._trace.posterior["intercept"].mean().values)
        self._home_advantage = float(self._trace.posterior["home_advantage"].mean().values)

        attack_means = self._trace.posterior["attack"].mean(dim=("chain", "draw")).values
        defence_means = self._trace.posterior["defence"].mean(dim=("chain", "draw")).values
        position_means = self._trace.posterior["position_effect"].mean(dim=("chain", "draw")).values
        player_means = self._trace.posterior["player_effect"].mean(dim=("chain", "draw")).values

        self._attack = {sid: float(attack_means[i]) for sid, i in self._squad_idx.items()}
        self._defence = {sid: float(defence_means[i]) for sid, i in self._squad_idx.items()}
        self._position = {pc: float(position_means[i]) for pc, i in self._position_idx.items()}
        self._player = {pid: float(player_means[i]) for pid, i in self._player_idx.items()}

        self._fitted = True

        # Log diagnostics
        sigma_att_post = float(self._trace.posterior["sigma_attack"].mean().values)
        sigma_def_post = float(self._trace.posterior["sigma_defence"].mean().values)
        sigma_pl_post = float(self._trace.posterior["sigma_player"].mean().values)
        LOGGER.info(
            "Posterior means: intercept=%.3f, home=%.3f, "
            "sigma_attack=%.3f, sigma_defence=%.3f, sigma_player=%.3f",
            self._intercept, self._home_advantage,
            sigma_att_post, sigma_def_post, sigma_pl_post,
        )
        LOGGER.info(
            "Attack range: [%.3f, %.3f], Defence range: [%.3f, %.3f]",
            min(self._attack.values()), max(self._attack.values()),
            min(self._defence.values()), max(self._defence.values()),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(ATS) for each row using posterior means.

        For players not seen during training, the player effect defaults
        to zero (i.e., the prediction falls back to the position-group
        prior). For teams not seen during training, attack and defence
        effects default to zero (league average).

        Parameters
        ----------
        X : pd.DataFrame
            Prediction data. Must contain columns: ``squad_id``,
            ``opponent_squad_id``, ``position_code``, ``player_id``,
            ``is_home``.

        Returns
        -------
        np.ndarray
            Predicted probabilities of scoring a try (shape ``(n,)``).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If required columns are missing from X.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._validate_columns(X)

        n = len(X)
        logits = np.full(n, self._intercept)

        # Vectorised lookups with fallback to 0.0 for unseen entities
        attack_vals = X["squad_id"].map(self._attack).fillna(0.0).values
        defence_vals = X["opponent_squad_id"].map(self._defence).fillna(0.0).values
        is_home_vals = X["is_home"].values.astype(float)

        # Position effect: for unseen positions, compute from POSITION_TRY_RATES
        position_vals = np.empty(n)
        for i, pc in enumerate(X["position_code"]):
            if pc in self._position:
                position_vals[i] = self._position[pc]
            else:
                # Unseen position: use prior (deviation from intercept)
                position_vals[i] = _position_logit_prior(pc) - self._intercept

        # Player effect: unseen players get 0.0 (shrunk to position mean)
        player_vals = np.empty(n)
        n_unseen = 0
        for i, pid in enumerate(X["player_id"]):
            if pid in self._player:
                player_vals[i] = self._player[pid]
            else:
                player_vals[i] = 0.0
                n_unseen += 1
        if n_unseen > 0:
            LOGGER.info(
                "%d/%d players unseen in training — using position-group prior",
                n_unseen, n,
            )

        logits += attack_vals + defence_vals + position_vals + player_vals
        logits += self._home_advantage * is_home_vals

        return expit(logits)

    def feature_names(self) -> list[str]:
        """Return the column names required by this model.

        Returns
        -------
        list[str]
            The five required columns.
        """
        return list(REQUIRED_COLUMNS)

    # ------------------------------------------------------------------
    # Diagnostic / inspection methods
    # ------------------------------------------------------------------

    def get_team_effects(self) -> pd.DataFrame:
        """Return a DataFrame of team attack and defence posterior means.

        Returns
        -------
        pd.DataFrame
            Columns: ``squad_id``, ``attack``, ``defence``, ``net``
            (attack - defence, higher = more tries scored relative to
            conceded). Sorted by ``net`` descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        rows = []
        for sid in self._attack:
            rows.append({
                "squad_id": sid,
                "attack": self._attack[sid],
                "defence": self._defence.get(sid, 0.0),
                "net": self._attack[sid] - self._defence.get(sid, 0.0),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("net", ascending=False)
            .reset_index(drop=True)
        )

    def get_position_effects(self) -> pd.DataFrame:
        """Return a DataFrame of position effects on the logit scale.

        Returns
        -------
        pd.DataFrame
            Columns: ``position_code``, ``logit_effect``, ``implied_prob``
            (intercept + position effect transformed to probability).
            Sorted by ``implied_prob`` descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        rows = []
        for pc, eff in self._position.items():
            logit_total = self._intercept + eff
            rows.append({
                "position_code": pc,
                "logit_effect": eff,
                "implied_prob": float(expit(logit_total)),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("implied_prob", ascending=False)
            .reset_index(drop=True)
        )

    def get_player_effects(self, top_n: int = 20) -> pd.DataFrame:
        """Return the top player random effects by magnitude.

        Parameters
        ----------
        top_n : int
            Number of top players to return (by absolute effect size).

        Returns
        -------
        pd.DataFrame
            Columns: ``player_id``, ``player_effect``, ``position_code``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        rows = []
        for pid, eff in self._player.items():
            rows.append({
                "player_id": pid,
                "player_effect": eff,
                "position_code": self._player_position.get(pid, "UNK"),
            })
        return (
            pd.DataFrame(rows)
            .assign(abs_effect=lambda d: d["player_effect"].abs())
            .sort_values("abs_effect", ascending=False)
            .head(top_n)
            .drop(columns="abs_effect")
            .reset_index(drop=True)
        )

    @property
    def trace(self) -> Any:
        """Access the raw ArviZ InferenceData trace for diagnostics.

        Returns
        -------
        arviz.InferenceData or None
            The posterior trace, or None if not fitted.
        """
        return self._trace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        """Check that all required columns are present.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to validate.

        Raises
        ------
        ValueError
            If any required column is missing.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Expected: {list(REQUIRED_COLUMNS)}"
            )
