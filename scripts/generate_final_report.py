"""Generate comprehensive final comparison report.

Combines results from:
1. Model improvements (Kelly, PositionCal, Ensemble) — model_improvements_report.xlsx
2. Hyperparameter search — hyperparameter_search.xlsx
3. New feature validation — re-runs best config on expanded feature store

Output: data/backtest_results/final_comparison_report.xlsx
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURE_STORE_DIR, BACKTEST_RESULTS_DIR
from src.features.feature_store import load_feature_store
from src.models.gbm import GBMModel
from src.models.calibration import CalibratedModel, PositionCalibratedModel
from src.models.baseline import MarketBlendedStrategy
from src.evaluation.backtest import run_backtest, BacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def bootstrap_roi(bet_df: pd.DataFrame, n_bootstrap: int = 5000) -> dict:
    """Bootstrap confidence interval for ROI."""
    if bet_df.empty:
        return {"p_positive": 0.0, "roi_ci_lo": 0.0, "roi_ci_hi": 0.0}

    profits = bet_df["payout"].values - bet_df["stake"].values
    stakes = bet_df["stake"].values
    rng = np.random.default_rng(42)
    rois = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(profits), size=len(profits), replace=True)
        total_stake = stakes[idx].sum()
        total_profit = profits[idx].sum()
        if total_stake > 0:
            rois.append(total_profit / total_stake)
    rois = np.array(rois)
    return {
        "p_positive": float((rois > 0).mean()),
        "roi_ci_lo": float(np.percentile(rois, 2.5)),
        "roi_ci_hi": float(np.percentile(rois, 97.5)),
    }


def run_config(fs, label, gbm_params, strategy_params, bt_config):
    """Run a single config and return summary dict."""
    gbm = GBMModel(
        n_estimators=gbm_params.get("n_estimators", 150),
        max_depth=gbm_params.get("max_depth", 4),
        learning_rate=0.05,
        min_child_samples=gbm_params.get("min_child_samples", 80),
        reg_alpha=gbm_params.get("reg_lambda", 3.0),
        reg_lambda=gbm_params.get("reg_lambda", 3.0),
    )
    model = CalibratedModel(base_model=gbm, method="isotonic", cal_rounds=5)
    strategy = MarketBlendedStrategy(
        alpha=strategy_params.get("alpha", 0.25),
        min_edge=strategy_params.get("min_edge", 0.03),
    )
    bt = run_backtest(fs, strategy, model, bt_config, seasons=[2024, 2025], min_round=3)
    s = bt.summary()
    s["config_label"] = label

    # Per-season
    for season in [2024, 2025]:
        season_bets = [r for r in bt.round_results if r.season == season]
        staked = sum(r.total_staked for r in season_bets)
        payout = sum(r.total_payout for r in season_bets)
        profit = payout - staked
        n = sum(r.n_bets for r in season_bets)
        s[f"roi_{season}"] = profit / staked if staked > 0 else 0.0
        s[f"profit_{season}"] = profit
        s[f"n_bets_{season}"] = n

    # Sharpe
    rnd_df = bt.to_round_dataframe()
    if len(rnd_df) > 1 and rnd_df["profit"].std() > 0:
        s["sharpe"] = rnd_df["profit"].mean() / rnd_df["profit"].std()
    else:
        s["sharpe"] = 0.0

    # Bootstrap
    bet_df = bt.to_bet_dataframe()
    boot = bootstrap_roi(bet_df)
    s.update(boot)

    return s, bt


def main():
    start = time.time()

    # Load expanded feature store
    fs_path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    logger.info("Loading feature store from %s", fs_path)
    fs = load_feature_store(str(fs_path))
    logger.info("Feature store: %d rows x %d cols", len(fs), len(fs.columns))

    all_results = []

    # ─────────────────────────────────────────────────────────
    # 1. Best config from Phase 5B (baseline) — expanded features
    # ─────────────────────────────────────────────────────────
    logger.info("Running baseline config on expanded feature store...")
    s, bt_baseline = run_config(
        fs, "Phase5B Baseline (expanded features)",
        {"n_estimators": 150, "max_depth": 4, "reg_lambda": 3.0, "min_child_samples": 80},
        {"alpha": 0.25, "min_edge": 0.03},
        BacktestConfig(flat_stake=100.0),
    )
    all_results.append(s)
    logger.info("  Baseline: %d bets, ROI=%.1f%%", s["n_bets"], s["roi"] * 100)

    # ─────────────────────────────────────────────────────────
    # 2. Kelly staking on expanded features
    # ─────────────────────────────────────────────────────────
    logger.info("Running Kelly staking configs...")
    for label, kf in [("Quarter Kelly", 0.25), ("Eighth Kelly", 0.125), ("Sixteenth Kelly", 0.0625)]:
        s, _ = run_config(
            fs, f"{label} (expanded)",
            {"n_estimators": 150, "max_depth": 4, "reg_lambda": 3.0, "min_child_samples": 80},
            {"alpha": 0.25, "min_edge": 0.03},
            BacktestConfig(kelly_fraction=kf),
        )
        all_results.append(s)
        logger.info("  %s: %d bets, ROI=%.1f%%", label, s["n_bets"], s["roi"] * 100)

    # ─────────────────────────────────────────────────────────
    # 3. Load top configs from hyperparameter search
    # ─────────────────────────────────────────────────────────
    hp_path = BACKTEST_RESULTS_DIR / "hyperparameter_search.xlsx"
    if hp_path.exists():
        logger.info("Loading hyperparameter search results...")
        hp_df = pd.read_excel(str(hp_path), sheet_name="Profitable Both Seasons")
        if not hp_df.empty:
            # Run top 5 from grid search on expanded feature store
            for i, (_, row) in enumerate(hp_df.head(5).iterrows(), 1):
                label = f"GridSearch #{i} (n={int(row['n_estimators'])},d={int(row['max_depth'])},reg={row['reg_lambda']},mcs={int(row['min_child_samples'])},a={row['alpha']})"
                s, _ = run_config(
                    fs, label,
                    {
                        "n_estimators": int(row["n_estimators"]),
                        "max_depth": int(row["max_depth"]),
                        "reg_lambda": row["reg_lambda"],
                        "min_child_samples": int(row["min_child_samples"]),
                    },
                    {"alpha": row["alpha"], "min_edge": 0.03},
                    BacktestConfig(flat_stake=100.0),
                )
                all_results.append(s)
                logger.info("  %s: %d bets, ROI=%.1f%%", label, s["n_bets"], s["roi"] * 100)

    # ─────────────────────────────────────────────────────────
    # 4. Additional edge thresholds on best config
    # ─────────────────────────────────────────────────────────
    logger.info("Testing edge thresholds...")
    for min_edge in [0.02, 0.04, 0.05]:
        s, _ = run_config(
            fs, f"min_edge={min_edge}",
            {"n_estimators": 150, "max_depth": 4, "reg_lambda": 3.0, "min_child_samples": 80},
            {"alpha": 0.25, "min_edge": min_edge},
            BacktestConfig(flat_stake=100.0),
        )
        all_results.append(s)
        logger.info("  min_edge=%.2f: %d bets, ROI=%.1f%%", min_edge, s["n_bets"], s["roi"] * 100)

    # ─────────────────────────────────────────────────────────
    # Build Excel report
    # ─────────────────────────────────────────────────────────
    logger.info("Building final report...")

    BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BACKTEST_RESULTS_DIR / "final_comparison_report.xlsx"

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("roi", ascending=False).reset_index(drop=True)

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        # Main comparison
        show_cols = [
            "config_label", "n_bets", "roi", "profit", "hit_rate", "avg_odds",
            "avg_edge", "max_drawdown", "sharpe",
            "roi_2024", "roi_2025", "profit_2024", "profit_2025",
            "n_bets_2024", "n_bets_2025",
            "p_positive", "roi_ci_lo", "roi_ci_hi",
        ]
        show = results_df[[c for c in show_cols if c in results_df.columns]].copy()
        for c in ["roi", "hit_rate", "avg_edge", "roi_2024", "roi_2025", "p_positive",
                   "roi_ci_lo", "roi_ci_hi"]:
            if c in show.columns:
                show[c] = (show[c] * 100).round(1)
        for c in ["profit", "max_drawdown", "profit_2024", "profit_2025"]:
            if c in show.columns:
                show[c] = show[c].round(2)
        if "sharpe" in show.columns:
            show["sharpe"] = show["sharpe"].round(3)

        show = show.rename(columns={
            "config_label": "Configuration",
            "n_bets": "Total Bets",
            "roi": "ROI %",
            "profit": "Profit $",
            "hit_rate": "Hit Rate %",
            "avg_odds": "Avg Odds",
            "avg_edge": "Avg Edge %",
            "max_drawdown": "Max DD $",
            "sharpe": "Sharpe",
            "roi_2024": "ROI 2024 %",
            "roi_2025": "ROI 2025 %",
            "profit_2024": "Profit 2024 $",
            "profit_2025": "Profit 2025 $",
            "n_bets_2024": "Bets 2024",
            "n_bets_2025": "Bets 2025",
            "p_positive": "P(ROI>0) %",
            "roi_ci_lo": "ROI 2.5% CI",
            "roi_ci_hi": "ROI 97.5% CI",
        })
        show.to_excel(writer, sheet_name="All Configs Ranked", index=False)

        # Recommendation sheet
        # Filter to configs profitable in BOTH seasons with P(ROI>0) > 90%
        recommended = results_df[
            (results_df.get("roi_2024", 0) > 0) &
            (results_df.get("roi_2025", 0) > 0) &
            (results_df.get("p_positive", 0) > 0.9)
        ].head(5)

        if not recommended.empty:
            rec_show = recommended[[c for c in show_cols if c in recommended.columns]].copy()
            for c in ["roi", "hit_rate", "avg_edge", "roi_2024", "roi_2025", "p_positive",
                       "roi_ci_lo", "roi_ci_hi"]:
                if c in rec_show.columns:
                    rec_show[c] = (rec_show[c] * 100).round(1)
            for c in ["profit", "max_drawdown", "profit_2024", "profit_2025"]:
                if c in rec_show.columns:
                    rec_show[c] = rec_show[c].round(2)
            if "sharpe" in rec_show.columns:
                rec_show["sharpe"] = rec_show["sharpe"].round(3)
            rec_show.to_excel(writer, sheet_name="Recommended for 2026", index=False)

        # Bet-level detail for best config
        bet_df_baseline = bt_baseline.to_bet_dataframe()
        if not bet_df_baseline.empty:
            import sqlite3
            from src.config import DB_PATH
            conn = sqlite3.connect(str(DB_PATH))

            # Enrich
            players = pd.read_sql_query(
                "SELECT DISTINCT player_id, display_name FROM players_2025", conn
            )
            bet_df_baseline = bet_df_baseline.merge(players, on="player_id", how="left")

            matches = pd.read_sql_query("""
                SELECT m.match_id, ht.squad_name AS home_team, at.squad_name AS away_team
                FROM matches_2025 m
                JOIN teams ht ON m.home_squad_id = ht.squad_id
                JOIN teams at ON m.away_squad_id = at.squad_id
            """, conn)
            # Also get 2024 matches
            matches2024 = pd.read_sql_query("""
                SELECT m.match_id, ht.squad_name AS home_team, at.squad_name AS away_team
                FROM matches_2024 m
                JOIN teams ht ON m.home_squad_id = ht.squad_id
                JOIN teams at ON m.away_squad_id = at.squad_id
            """, conn)
            players2024 = pd.read_sql_query(
                "SELECT DISTINCT player_id, display_name FROM players_2024", conn
            )
            bet_df_baseline = bet_df_baseline.merge(
                pd.concat([players, players2024]).drop_duplicates("player_id"),
                on="player_id", how="left", suffixes=("", "_dup")
            )
            all_matches = pd.concat([matches, matches2024]).drop_duplicates("match_id")
            bet_df_baseline = bet_df_baseline.merge(all_matches, on="match_id", how="left")

            conn.close()

            bet_df_baseline["match"] = bet_df_baseline["home_team"] + " v " + bet_df_baseline["away_team"]
            bet_df_baseline["result"] = bet_df_baseline["won"].map({1: "WON", 0: "LOST"})
            bet_df_baseline["profit"] = bet_df_baseline["payout"] - bet_df_baseline["stake"]

            bets_show = bet_df_baseline[[
                "season", "round_number", "match", "display_name", "position_code",
                "odds", "model_prob", "implied_prob", "edge", "stake", "payout",
                "profit", "result",
            ]].copy()
            bets_show = bets_show.sort_values(["season", "round_number"]).reset_index(drop=True)
            bets_show["cumulative_pnl"] = bets_show["profit"].cumsum().round(2)
            for c in ["odds", "model_prob", "implied_prob", "edge"]:
                bets_show[c] = bets_show[c].round(3)
            bets_show.to_excel(writer, sheet_name="Best Config All Bets", index=False)

    elapsed = time.time() - start
    logger.info("Final report saved to %s (%.0fs)", output_path, elapsed)

    # Print summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON REPORT")
    print(f"{'='*80}")
    for i, (_, r) in enumerate(results_df.head(10).iterrows(), 1):
        roi_24 = r.get("roi_2024", 0) * 100
        roi_25 = r.get("roi_2025", 0) * 100
        p_pos = r.get("p_positive", 0) * 100
        print(
            f"  {i:2d}. {r['config_label']:<50s} "
            f"ROI={r['roi']*100:+6.1f}%  Bets={r['n_bets']:3.0f}  "
            f"2024={roi_24:+5.1f}%  2025={roi_25:+5.1f}%  "
            f"P(+)={p_pos:4.0f}%  DD=${r.get('max_drawdown', 0):,.0f}"
        )
    print(f"{'='*80}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
