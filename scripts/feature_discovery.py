"""Feature Discovery Script — Data-driven edge discovery for ATS prediction.

Analyses the feature store to find:
1. Univariate feature predictive power (per position group)
2. Pairwise interaction discovery
3. Segment profitability mining
4. Conditional probability analysis (model vs market)

Run: python scripts/feature_discovery.py
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr

from src.config import FEATURE_STORE_DIR

logging.basicConfig(level=logging.WARNING, format="%(message)s")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Columns that are NOT features (IDs, metadata, target, strings)
NON_FEATURE_COLS = {
    "match_id", "player_id", "squad_id", "opponent_squad_id", "round_number",
    "season", "scored_try", "tries", "position_code", "position_label",
    "position_group", "betfair_odds_source", "player_edge",
}
SUFFIX_EXCLUDE = ("_context", "_matchup", "_team_own", "_team_opp")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns, excluding IDs/metadata/duplicates."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not c.endswith(SUFFIX_EXCLUDE)
        and df[c].dtype in ("float64", "int64", "float32", "int32")
    ]


# ---------------------------------------------------------------------------
# 1. Univariate Feature Screen
# ---------------------------------------------------------------------------

def univariate_screen(
    df: pd.DataFrame,
    feature_cols: list[str],
    position_group: str | None = None,
) -> pd.DataFrame:
    """Screen features by correlation, AUC, and mutual information.

    Parameters
    ----------
    df : pd.DataFrame
        Feature store with scored_try target.
    feature_cols : list[str]
        Numeric feature columns.
    position_group : str | None
        If set, filter to this position group.

    Returns
    -------
    pd.DataFrame
        Ranked features with correlation, AUC, MI.
    """
    subset = df.copy()
    if position_group:
        subset = subset[subset["position_group"] == position_group]

    y = subset["scored_try"].values
    if len(np.unique(y)) < 2:
        return pd.DataFrame()

    results = []
    for col in feature_cols:
        x = subset[col].values.astype(float)
        valid = ~np.isnan(x)
        if valid.sum() < 50:
            continue

        xv, yv = x[valid], y[valid]
        if np.std(xv) < 1e-10:
            continue

        # Point-biserial correlation
        try:
            corr, pval = pointbiserialr(yv, xv)
        except Exception:
            corr, pval = np.nan, np.nan

        # Univariate AUC
        try:
            auc = roc_auc_score(yv, xv)
        except Exception:
            auc = np.nan

        results.append({
            "feature": col,
            "correlation": corr,
            "p_value": pval,
            "auc": auc,
            "n_valid": int(valid.sum()),
            "position_group": position_group or "All",
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Add mutual information (batch computation is faster)
    valid_cols = [r["feature"] for r in results]
    X_mi = subset[valid_cols].fillna(0).values
    try:
        mi_scores = mutual_info_classif(X_mi, y, random_state=42, n_neighbors=5)
        mi_map = dict(zip(valid_cols, mi_scores))
        result_df["mutual_info"] = result_df["feature"].map(mi_map)
    except Exception:
        result_df["mutual_info"] = np.nan

    result_df["abs_auc_from_05"] = (result_df["auc"] - 0.5).abs()
    result_df = result_df.sort_values("abs_auc_from_05", ascending=False)
    return result_df


# ---------------------------------------------------------------------------
# 2. Interaction Discovery
# ---------------------------------------------------------------------------

def discover_interactions(
    df: pd.DataFrame,
    top_features: list[str],
    max_pairs: int = 100,
) -> pd.DataFrame:
    """Test pairwise feature interactions against try scoring.

    For top features, compute products, ratios, and differences,
    then test each interaction's AUC vs scored_try.
    """
    y = df["scored_try"].values
    results = []

    pairs = []
    for i, f1 in enumerate(top_features):
        for f2 in top_features[i + 1:]:
            pairs.append((f1, f2))
    pairs = pairs[:max_pairs]

    for f1, f2 in pairs:
        x1 = df[f1].values.astype(float)
        x2 = df[f2].values.astype(float)
        valid = ~(np.isnan(x1) | np.isnan(x2))
        if valid.sum() < 100:
            continue

        x1v, x2v, yv = x1[valid], x2[valid], y[valid]

        # Product interaction
        product = x1v * x2v
        if np.std(product) > 1e-10:
            try:
                auc_prod = roc_auc_score(yv, product)
            except Exception:
                auc_prod = np.nan
            results.append({
                "f1": f1, "f2": f2, "type": "product",
                "auc": auc_prod, "n": int(valid.sum()),
            })

        # Ratio (with guard)
        denom = np.where(np.abs(x2v) > 1e-10, x2v, np.nan)
        ratio = x1v / denom
        ratio_valid = ~np.isnan(ratio)
        if ratio_valid.sum() > 100 and np.std(ratio[ratio_valid]) > 1e-10:
            try:
                auc_ratio = roc_auc_score(yv[ratio_valid], ratio[ratio_valid])
            except Exception:
                auc_ratio = np.nan
            results.append({
                "f1": f1, "f2": f2, "type": "ratio",
                "auc": auc_ratio, "n": int(ratio_valid.sum()),
            })

        # Difference
        diff = x1v - x2v
        if np.std(diff) > 1e-10:
            try:
                auc_diff = roc_auc_score(yv, diff)
            except Exception:
                auc_diff = np.nan
            results.append({
                "f1": f1, "f2": f2, "type": "difference",
                "auc": auc_diff, "n": int(valid.sum()),
            })

    if not results:
        return pd.DataFrame()

    idf = pd.DataFrame(results)
    idf["abs_auc_from_05"] = (idf["auc"] - 0.5).abs()
    return idf.sort_values("abs_auc_from_05", ascending=False)


# ---------------------------------------------------------------------------
# 3. Segment Profitability Mining
# ---------------------------------------------------------------------------

def mine_segments(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_quantiles: int = 4,
) -> pd.DataFrame:
    """Find profitable segments by position × odds band × feature threshold.

    Uses actual try outcomes and betfair implied probabilities to compute
    hypothetical flat-stake ROI for each segment.
    """
    if "betfair_implied_prob" not in df.columns:
        return pd.DataFrame()

    # Only rows with valid odds
    valid = df[df["betfair_implied_prob"].notna() & (df["betfair_implied_prob"] > 0)].copy()
    if len(valid) < 200:
        return pd.DataFrame()

    valid["betfair_odds"] = 1.0 / valid["betfair_implied_prob"]
    valid["payout"] = valid["scored_try"] * valid["betfair_odds"]

    # Odds bands
    valid["odds_band"] = pd.qcut(valid["betfair_odds"], q=4, labels=["short", "mid_short", "mid_long", "long"], duplicates="drop")

    results = []

    # 1. Position × Odds band
    for pos in valid["position_group"].unique():
        for band in valid["odds_band"].unique():
            seg = valid[(valid["position_group"] == pos) & (valid["odds_band"] == band)]
            if len(seg) < 30:
                continue
            roi = (seg["payout"].sum() - len(seg)) / len(seg)
            results.append({
                "segment": f"{pos} × {band}",
                "filter_type": "pos_x_odds",
                "position": pos,
                "odds_band": str(band),
                "n_bets": len(seg),
                "actual_rate": seg["scored_try"].mean(),
                "implied_rate": seg["betfair_implied_prob"].mean(),
                "edge": seg["scored_try"].mean() - seg["betfair_implied_prob"].mean(),
                "roi": roi,
            })

    # 2. Position × Odds band × feature quartile (top features only)
    top_feats = feature_cols[:20]  # Already sorted by importance
    for feat in top_feats:
        if feat in NON_FEATURE_COLS or valid[feat].isna().mean() > 0.5:
            continue
        try:
            valid[f"_{feat}_q"] = pd.qcut(valid[feat].fillna(valid[feat].median()), q=n_quantiles, labels=False, duplicates="drop")
        except Exception:
            continue

        for pos in ["Back", "Halfback"]:  # Focus on bettable positions
            for band in valid["odds_band"].unique():
                for q in range(n_quantiles):
                    seg = valid[
                        (valid["position_group"] == pos)
                        & (valid["odds_band"] == band)
                        & (valid[f"_{feat}_q"] == q)
                    ]
                    if len(seg) < 20:
                        continue
                    roi = (seg["payout"].sum() - len(seg)) / len(seg)
                    results.append({
                        "segment": f"{pos} × {band} × {feat}_Q{q}",
                        "filter_type": "pos_x_odds_x_feat",
                        "position": pos,
                        "odds_band": str(band),
                        "n_bets": len(seg),
                        "actual_rate": seg["scored_try"].mean(),
                        "implied_rate": seg["betfair_implied_prob"].mean(),
                        "edge": seg["scored_try"].mean() - seg["betfair_implied_prob"].mean(),
                        "roi": roi,
                        "feature": feat,
                        "quartile": q,
                    })

        valid.drop(columns=[f"_{feat}_q"], inplace=True)

    if not results:
        return pd.DataFrame()

    seg_df = pd.DataFrame(results)
    return seg_df.sort_values("roi", ascending=False)


def mine_cross_season_segments(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Find segments profitable in BOTH seasons."""
    segments_2024 = mine_segments(df[df["season"] == 2024], feature_cols)
    segments_2025 = mine_segments(df[df["season"] == 2025], feature_cols)

    if segments_2024.empty or segments_2025.empty:
        return pd.DataFrame()

    # Merge on segment name
    merged = segments_2024.merge(
        segments_2025,
        on="segment",
        suffixes=("_2024", "_2025"),
        how="inner",
    )

    # Both must be profitable
    merged["both_profitable"] = (merged["roi_2024"] > 0) & (merged["roi_2025"] > 0)
    merged["combined_roi"] = (merged["roi_2024"] + merged["roi_2025"]) / 2
    merged["min_n_bets"] = merged[["n_bets_2024", "n_bets_2025"]].min(axis=1)

    profitable = merged[merged["both_profitable"] & (merged["min_n_bets"] >= 20)]
    return profitable.sort_values("combined_roi", ascending=False)


# ---------------------------------------------------------------------------
# 4. Conditional Probability Analysis
# ---------------------------------------------------------------------------

def conditional_probability_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 4,
) -> pd.DataFrame:
    """Compare actual vs market probability conditioned on feature values.

    Finds where the market is miscalibrated relative to feature signals.
    """
    valid = df[df["betfair_implied_prob"].notna() & (df["betfair_implied_prob"] > 0)].copy()
    if len(valid) < 200:
        return pd.DataFrame()

    results = []
    for feat in feature_cols[:30]:  # Top 30 features
        if valid[feat].isna().mean() > 0.5:
            continue
        try:
            valid[f"_bin"] = pd.qcut(valid[feat].fillna(valid[feat].median()), q=n_bins, labels=False, duplicates="drop")
        except Exception:
            continue

        for b in range(n_bins):
            seg = valid[valid["_bin"] == b]
            if len(seg) < 50:
                continue

            actual_rate = seg["scored_try"].mean()
            market_rate = seg["betfair_implied_prob"].mean()
            edge = actual_rate - market_rate

            results.append({
                "feature": feat,
                "bin": b,
                "bin_label": f"Q{b}" if n_bins == 4 else f"bin_{b}",
                "n": len(seg),
                "actual_try_rate": actual_rate,
                "market_implied_rate": market_rate,
                "edge_vs_market": edge,
                "feat_mean": seg[feat].mean(),
                "feat_std": seg[feat].std(),
            })

        valid.drop(columns=["_bin"], inplace=True)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("edge_vs_market", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("FEATURE DISCOVERY: Data-driven edge analysis")
    print("=" * 80)

    path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    fs = pd.read_parquet(path)
    feature_cols = get_feature_columns(fs)
    print(f"Data: {len(fs)} rows, {len(feature_cols)} numeric features")

    # ===================================================================
    # 1. UNIVARIATE SCREEN
    # ===================================================================
    print("\n" + "=" * 80)
    print("1. UNIVARIATE FEATURE SCREEN")
    print("=" * 80)

    # Overall
    all_screen = univariate_screen(fs, feature_cols)
    print(f"\n--- All positions (top 20 by |AUC - 0.5|) ---")
    if not all_screen.empty:
        for _, r in all_screen.head(20).iterrows():
            print(f"  AUC={r['auc']:.3f}  corr={r['correlation']:+.3f}  MI={r.get('mutual_info', 0):.4f}  {r['feature']}")

    # Per position group
    for pg in ["Back", "Halfback", "Forward"]:
        pg_screen = univariate_screen(fs, feature_cols, position_group=pg)
        if pg_screen.empty:
            continue
        print(f"\n--- {pg} (top 10) ---")
        for _, r in pg_screen.head(10).iterrows():
            print(f"  AUC={r['auc']:.3f}  corr={r['correlation']:+.3f}  {r['feature']}")

    # ===================================================================
    # 2. INTERACTION DISCOVERY
    # ===================================================================
    print("\n" + "=" * 80)
    print("2. INTERACTION DISCOVERY")
    print("=" * 80)

    # Use top 15 univariate features for interactions
    if not all_screen.empty:
        top15 = all_screen.head(15)["feature"].tolist()
        interactions = discover_interactions(fs, top15)
        if not interactions.empty:
            print(f"\n--- Top 20 interactions (by |AUC - 0.5|) ---")
            for _, r in interactions.head(20).iterrows():
                print(f"  AUC={r['auc']:.3f}  {r['type']:>10}  {r['f1']} × {r['f2']}")

    # ===================================================================
    # 3. SEGMENT PROFITABILITY MINING
    # ===================================================================
    print("\n" + "=" * 80)
    print("3. SEGMENT PROFITABILITY MINING")
    print("=" * 80)

    # Overall segments
    segments = mine_segments(fs, feature_cols)
    if not segments.empty:
        pos_odds = segments[segments["filter_type"] == "pos_x_odds"].sort_values("roi", ascending=False)
        print(f"\n--- Position × Odds Band (all data) ---")
        for _, r in pos_odds.iterrows():
            marker = " <<<" if r["roi"] > 0 else ""
            print(f"  ROI={r['roi']*100:+6.1f}%  edge={r['edge']*100:+5.1f}pp  n={r['n_bets']:>4}  {r['segment']}{marker}")

    # Cross-season stable segments
    print(f"\n--- Cross-season profitable segments (profitable BOTH 2024 & 2025) ---")
    cross = mine_cross_season_segments(fs, feature_cols)
    if not cross.empty:
        for _, r in cross.head(20).iterrows():
            print(f"  ROI: 2024={r['roi_2024']*100:+6.1f}%, 2025={r['roi_2025']*100:+6.1f}%  "
                  f"n={r['n_bets_2024']:>3}+{r['n_bets_2025']:>3}  {r['segment']}")
    else:
        print("  No segments found profitable in BOTH seasons with n>=20.")

    # ===================================================================
    # 4. CONDITIONAL PROBABILITY ANALYSIS
    # ===================================================================
    print("\n" + "=" * 80)
    print("4. CONDITIONAL PROBABILITY ANALYSIS (market miscalibration)")
    print("=" * 80)

    cond = conditional_probability_analysis(fs, feature_cols)
    if not cond.empty:
        # Show features where at least one bin has significant edge
        big_edges = cond[cond["edge_vs_market"].abs() > 0.03]
        if not big_edges.empty:
            print(f"\n--- Feature bins with >3pp edge vs market ---")
            seen_features = set()
            for _, r in big_edges.head(30).iterrows():
                if r["feature"] not in seen_features:
                    seen_features.add(r["feature"])
                    # Show all bins for this feature
                    feat_rows = cond[cond["feature"] == r["feature"]]
                    print(f"\n  {r['feature']}:")
                    for _, fr in feat_rows.iterrows():
                        marker = " <<<" if abs(fr["edge_vs_market"]) > 0.03 else ""
                        print(f"    {fr['bin_label']}: actual={fr['actual_try_rate']:.1%} "
                              f"market={fr['market_implied_rate']:.1%} "
                              f"edge={fr['edge_vs_market']*100:+.1f}pp "
                              f"n={fr['n']:>4}{marker}")

    print("\n" + "=" * 80)
    print("DISCOVERY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
