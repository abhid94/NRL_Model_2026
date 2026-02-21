"""Edge discovery and segment analysis for ATS betting strategies.

Pure functions that take a bet DataFrame (from BacktestResult.to_bet_dataframe())
and/or the feature store and compute segment-level profitability breakdowns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_calibration_error,
    compute_clv,
    compute_max_drawdown,
    compute_pnl_series,
    compute_roi,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default binning parameters
# ---------------------------------------------------------------------------

ODDS_BANDS = [1.01, 2.0, 3.0, 4.0, 6.0, 100.0]
ODDS_LABELS = ["$1.01-2.00", "$2.00-3.00", "$3.00-4.00", "$4.00-6.00", "$6.00+"]

TEAM_TRIES_BINS = [0.0, 3.5, 4.5, float("inf")]
TEAM_TRIES_LABELS = ["low (<3.5)", "mid (3.5-4.5)", "high (>4.5)"]


# ---------------------------------------------------------------------------
# Core segment helpers
# ---------------------------------------------------------------------------

def _compute_segment_stats(group: pd.DataFrame) -> dict[str, Any]:
    """Compute standard stats for a segment group."""
    n_bets = len(group)
    total_staked = group["stake"].sum()
    total_payout = group["payout"].sum()
    wins = group["won"].sum()
    roi = (total_payout - total_staked) / total_staked if total_staked > 0 else 0.0
    hit_rate = wins / n_bets if n_bets > 0 else 0.0
    avg_odds = group["odds"].mean() if n_bets > 0 else 0.0
    avg_edge = group["edge"].mean() if n_bets > 0 else 0.0
    profit = total_payout - total_staked
    return {
        "n_bets": n_bets,
        "wins": int(wins),
        "total_staked": round(total_staked, 2),
        "total_payout": round(total_payout, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 4),
        "hit_rate": round(hit_rate, 4),
        "avg_odds": round(avg_odds, 2),
        "avg_edge": round(avg_edge, 4),
    }


def _enrich_bets(
    bet_df: pd.DataFrame,
    feature_store: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Join feature store columns onto bet DataFrame.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Bet records from BacktestResult.to_bet_dataframe().
    feature_store : pd.DataFrame
        Full feature store with match_id, player_id.
    columns : list[str]
        Columns to join from the feature store.

    Returns
    -------
    pd.DataFrame
        Bet DataFrame with additional columns.
    """
    if bet_df.empty:
        return bet_df

    # Only join columns that exist in the feature store
    available = [c for c in columns if c in feature_store.columns]
    if not available:
        return bet_df

    join_cols = ["match_id", "player_id"] + available
    fs_subset = feature_store[join_cols].drop_duplicates(subset=["match_id", "player_id"])
    merged = bet_df.merge(fs_subset, on=["match_id", "player_id"], how="left")
    return merged


# ---------------------------------------------------------------------------
# Segment ROI functions
# ---------------------------------------------------------------------------

def segment_roi(
    bet_df: pd.DataFrame,
    segment_col: str,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute ROI and stats by segment.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: stake, payout, won, odds, edge, and *segment_col*.
    segment_col : str
        Column to group by.
    bins : list[float], optional
        If provided, bin the segment_col into these edges before grouping.
    labels : list[str], optional
        Labels for the bins.

    Returns
    -------
    pd.DataFrame
        One row per segment with stats.
    """
    if bet_df.empty:
        return pd.DataFrame()

    df = bet_df.copy()
    group_col = segment_col

    if bins is not None:
        group_col = f"{segment_col}_bin"
        df[group_col] = pd.cut(
            df[segment_col],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        # Drop rows where binning produced NaN
        df = df.dropna(subset=[group_col])

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in bet DataFrame")

    rows = []
    for seg_val, group in df.groupby(group_col, observed=True):
        stats = _compute_segment_stats(group)
        stats["segment"] = str(seg_val)
        rows.append(stats)

    result = pd.DataFrame(rows)
    if not result.empty:
        # Reorder columns
        col_order = ["segment"] + [c for c in result.columns if c != "segment"]
        result = result[col_order]
    return result


def odds_band_roi(bet_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ROI by odds band.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: stake, payout, won, odds, edge.

    Returns
    -------
    pd.DataFrame
        One row per odds band.
    """
    return segment_roi(
        bet_df,
        segment_col="odds",
        bins=ODDS_BANDS,
        labels=ODDS_LABELS,
    )


def season_breakdown(bet_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ROI by season.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: stake, payout, won, odds, edge, season.

    Returns
    -------
    pd.DataFrame
        One row per season.
    """
    return segment_roi(bet_df, segment_col="season")


def clv_analysis(bet_df: pd.DataFrame) -> pd.DataFrame:
    """Compute CLV breakdown for winners vs losers.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: model_prob, implied_prob, won.

    Returns
    -------
    pd.DataFrame
        CLV stats for winners, losers, and overall.
    """
    if bet_df.empty:
        return pd.DataFrame()

    df = bet_df.dropna(subset=["model_prob", "implied_prob"])
    if df.empty:
        return pd.DataFrame()

    rows = []
    for label, mask in [
        ("winners", df["won"] == 1),
        ("losers", df["won"] == 0),
        ("overall", pd.Series(True, index=df.index)),
    ]:
        subset = df[mask]
        if subset.empty:
            rows.append({
                "group": label,
                "n_bets": 0,
                "avg_clv": 0.0,
                "avg_model_prob": 0.0,
                "avg_implied_prob": 0.0,
            })
        else:
            clv = compute_clv(
                subset["model_prob"].values,
                subset["implied_prob"].values,
            )
            rows.append({
                "group": label,
                "n_bets": len(subset),
                "avg_clv": round(clv, 4),
                "avg_model_prob": round(subset["model_prob"].mean(), 4),
                "avg_implied_prob": round(subset["implied_prob"].mean(), 4),
            })
    return pd.DataFrame(rows)


def calibration_by_position(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    positions: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration error by position group.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_prob : array-like
        Predicted probabilities.
    positions : array-like
        Position code for each observation.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    pd.DataFrame
        Columns: position, ece, n_obs, actual_rate, avg_predicted.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    positions = np.asarray(positions)

    rows = []
    for pos in sorted(set(positions)):
        mask = positions == pos
        if mask.sum() < 5:
            continue
        ece = compute_calibration_error(y_true[mask], y_prob[mask], n_bins=n_bins)
        rows.append({
            "position": pos,
            "ece": round(ece, 4),
            "n_obs": int(mask.sum()),
            "actual_rate": round(float(y_true[mask].mean()), 4),
            "avg_predicted": round(float(y_prob[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


def cumulative_pnl_by_round(bet_df: pd.DataFrame) -> pd.DataFrame:
    """Compute round-by-round cumulative P&L.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: season, round_number, stake, payout.

    Returns
    -------
    pd.DataFrame
        Columns: season, round_number, n_bets, round_profit, cumulative_pnl.
    """
    if bet_df.empty:
        return pd.DataFrame()

    round_pnl = (
        bet_df
        .groupby(["season", "round_number"])
        .agg(
            n_bets=("stake", "count"),
            total_staked=("stake", "sum"),
            total_payout=("payout", "sum"),
        )
        .reset_index()
        .sort_values(["season", "round_number"])
    )
    round_pnl["round_profit"] = round_pnl["total_payout"] - round_pnl["total_staked"]
    round_pnl["cumulative_pnl"] = round_pnl["round_profit"].cumsum()
    return round_pnl[["season", "round_number", "n_bets", "round_profit", "cumulative_pnl"]]


# ---------------------------------------------------------------------------
# Full edge report
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Deep edge analysis functions (Sprint 4C)
# ---------------------------------------------------------------------------

def model_vs_market_disagreement(
    bet_df: pd.DataFrame,
    n_quartiles: int = 4,
) -> pd.DataFrame:
    """Analyse where model and market disagree most.

    Bins bets by model-vs-market edge quartile and reports hit rates and ROI.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: model_prob, implied_prob, stake, payout, won, odds, edge.
    n_quartiles : int
        Number of quartile bins.

    Returns
    -------
    pd.DataFrame
        One row per quartile with hit rate, ROI, and average edge.
    """
    if bet_df.empty:
        return pd.DataFrame()

    df = bet_df.dropna(subset=["model_prob", "implied_prob"]).copy()
    if len(df) < n_quartiles:
        return pd.DataFrame()

    df["disagreement"] = df["model_prob"] - df["implied_prob"]
    try:
        df["disagreement_q"] = pd.qcut(
            df["disagreement"], n_quartiles,
            labels=[f"Q{i+1}" for i in range(n_quartiles)],
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    rows = []
    for q, group in df.groupby("disagreement_q", observed=True):
        stats = _compute_segment_stats(group)
        stats["quartile"] = str(q)
        stats["avg_disagreement"] = round(float(group["disagreement"].mean()), 4)
        rows.append(stats)
    return pd.DataFrame(rows)


def conditional_edge_analysis(
    bet_df: pd.DataFrame,
    feature_store: pd.DataFrame,
    position_filter: list[str] | None = None,
    min_odds: float | None = None,
    max_odds: float | None = None,
    min_team_tries: float | None = None,
) -> dict[str, Any]:
    """ROI under multi-condition filters.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Bet records.
    feature_store : pd.DataFrame
        For enrichment.
    position_filter : list[str], optional
        Position codes to include.
    min_odds, max_odds : float, optional
        Odds range filter.
    min_team_tries : float, optional
        Minimum expected team tries.

    Returns
    -------
    dict[str, Any]
        Segment stats for bets matching all conditions.
    """
    if bet_df.empty:
        return {}

    df = _enrich_bets(bet_df, feature_store, ["expected_team_tries_5"])

    if position_filter:
        df = df[df["position_code"].isin(position_filter)]
    if min_odds is not None:
        df = df[df["odds"] >= min_odds]
    if max_odds is not None:
        df = df[df["odds"] <= max_odds]
    if min_team_tries is not None and "expected_team_tries_5" in df.columns:
        df = df[df["expected_team_tries_5"] >= min_team_tries]

    if df.empty:
        return {"n_bets": 0, "roi": 0.0}

    return _compute_segment_stats(df)


def stability_analysis(
    bet_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Bootstrap confidence intervals for segment ROI.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: stake, payout.
    n_bootstrap : int
        Number of bootstrap samples.
    random_state : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Keys: roi, roi_ci_lower, roi_ci_upper, p_positive_roi, n_bets.
    """
    if bet_df.empty or len(bet_df) < 5:
        return {"roi": 0.0, "roi_ci_lower": 0.0, "roi_ci_upper": 0.0,
                "p_positive_roi": 0.0, "n_bets": 0}

    rng = np.random.RandomState(random_state)
    stakes = bet_df["stake"].values
    payouts = bet_df["payout"].values
    n = len(stakes)

    rois = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        total_s = stakes[idx].sum()
        total_p = payouts[idx].sum()
        rois[i] = (total_p - total_s) / total_s if total_s > 0 else 0.0

    actual_roi = (payouts.sum() - stakes.sum()) / stakes.sum() if stakes.sum() > 0 else 0.0

    return {
        "roi": round(float(actual_roi), 4),
        "roi_ci_lower": round(float(np.percentile(rois, 2.5)), 4),
        "roi_ci_upper": round(float(np.percentile(rois, 97.5)), 4),
        "p_positive_roi": round(float((rois > 0).mean()), 4),
        "n_bets": n,
    }


def cross_season_stability(
    bet_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Verify segment profitability in each season independently.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Must contain: season, stake, payout.
    n_bootstrap : int
        Bootstrap samples per season.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        One row per season with bootstrap CI and P(ROI > 0).
    """
    if bet_df.empty or "season" not in bet_df.columns:
        return pd.DataFrame()

    rows = []
    for season in sorted(bet_df["season"].unique()):
        s_df = bet_df[bet_df["season"] == season]
        stats = stability_analysis(s_df, n_bootstrap=n_bootstrap, random_state=random_state)
        stats["season"] = int(season)
        rows.append(stats)
    return pd.DataFrame(rows)


def two_way_segment_roi(
    bet_df: pd.DataFrame,
    row_col: str,
    col_col: str,
    row_bins: list[float] | None = None,
    row_labels: list[str] | None = None,
    col_bins: list[float] | None = None,
    col_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Two-way interaction ROI table.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Bet records.
    row_col, col_col : str
        Columns for the two dimensions.
    row_bins, col_bins : list[float], optional
        Bin edges for continuous columns.
    row_labels, col_labels : list[str], optional
        Labels for bins.

    Returns
    -------
    pd.DataFrame
        Pivot table with ROI values (rows = row_col, cols = col_col).
    """
    if bet_df.empty:
        return pd.DataFrame()

    df = bet_df.copy()

    # Bin row column if needed
    row_group = row_col
    if row_bins is not None:
        row_group = f"{row_col}_bin"
        df[row_group] = pd.cut(df[row_col], bins=row_bins, labels=row_labels, include_lowest=True)
        df = df.dropna(subset=[row_group])

    # Bin col column if needed
    col_group = col_col
    if col_bins is not None:
        col_group = f"{col_col}_bin"
        df[col_group] = pd.cut(df[col_col], bins=col_bins, labels=col_labels, include_lowest=True)
        df = df.dropna(subset=[col_group])

    if df.empty:
        return pd.DataFrame()

    # Compute ROI per cell
    rows = []
    for (rv, cv), group in df.groupby([row_group, col_group], observed=True):
        total_s = group["stake"].sum()
        total_p = group["payout"].sum()
        roi = (total_p - total_s) / total_s if total_s > 0 else 0.0
        rows.append({
            row_group: rv,
            col_group: cv,
            "roi": round(roi, 4),
            "n_bets": len(group),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Pivot to matrix form
    pivot = result.pivot_table(
        index=row_group, columns=col_group, values="roi", aggfunc="first",
    )
    return pivot


# ---------------------------------------------------------------------------
# Full edge report
# ---------------------------------------------------------------------------

def generate_edge_report(
    bet_df: pd.DataFrame,
    feature_store: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate comprehensive edge analysis report.

    Parameters
    ----------
    bet_df : pd.DataFrame
        Bet records from BacktestResult.to_bet_dataframe().
    feature_store : pd.DataFrame
        Full feature store for enrichment.

    Returns
    -------
    dict[str, pd.DataFrame]
        Named DataFrames for each analysis dimension.
    """
    if bet_df.empty:
        return {}

    # Enrich bets with feature store columns
    enrich_cols = [
        "edge_matchup_score_rolling_5",
        "expected_team_tries_5",
        "player_edge",
        "team_edge_try_share_rolling_5",
    ]
    enriched = _enrich_bets(bet_df, feature_store, enrich_cols)

    report: dict[str, pd.DataFrame] = {}

    # 1. ROI by position
    report["position_roi"] = segment_roi(enriched, "position_code")

    # 2. ROI by edge matchup quartile
    col = "edge_matchup_score_rolling_5"
    if col in enriched.columns and enriched[col].notna().any():
        valid = enriched.dropna(subset=[col])
        if len(valid) >= 4:
            try:
                valid = valid.copy()
                valid["matchup_quartile"] = pd.qcut(
                    valid[col], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                    duplicates="drop",
                )
                report["matchup_quartile_roi"] = segment_roi(valid, "matchup_quartile")
            except ValueError:
                LOGGER.warning("Could not compute matchup quartiles (insufficient unique values)")

    # 3. ROI by expected team tries bucket
    col = "expected_team_tries_5"
    if col in enriched.columns and enriched[col].notna().any():
        report["team_tries_roi"] = segment_roi(
            enriched.dropna(subset=[col]),
            segment_col=col,
            bins=TEAM_TRIES_BINS,
            labels=TEAM_TRIES_LABELS,
        )

    # 4. ROI by odds band
    report["odds_band_roi"] = odds_band_roi(enriched)

    # 5. ROI by season
    report["season_roi"] = season_breakdown(enriched)

    # 6. CLV analysis
    report["clv"] = clv_analysis(enriched)

    # 7. ROI by player edge zone
    if "player_edge" in enriched.columns and enriched["player_edge"].notna().any():
        report["edge_zone_roi"] = segment_roi(
            enriched.dropna(subset=["player_edge"]),
            "player_edge",
        )

    # 8. Round-by-round cumulative P&L
    report["cumulative_pnl"] = cumulative_pnl_by_round(enriched)

    return report
