"""Top-2 per game ROI analysis for the 2025 season.

Instead of using an edge-threshold filter, this script places a flat $100
bet on the TOP 2 HIGHEST-RATED picks per game (by blended model probability),
regardless of edge, across all 2025 games.

Same model config as Phase 5B:
- Model: CalibratedGBM (n=150, depth=4, reg=3.0, min_child_samples=80)
- Blending: alpha=0.25 (model) + 0.75 (market)
- Selection: top 2 per game by blended_prob (eligible positions + valid odds only)
- Stake: $100 flat per bet

Output: data/backtest_results/top2_2025_roi.xlsx
"""

import sqlite3
import logging
import pandas as pd
import numpy as np

from src.config import DB_PATH, FEATURE_STORE_DIR, ELIGIBLE_POSITION_CODES
from src.features.feature_store import load_feature_store
from src.models.gbm import GBMModel
from src.models.calibration import CalibratedModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FLAT_STAKE = 100.0
ALPHA = 0.25          # model weight in blending
TOP_N_PER_GAME = 2    # pick top 2 per game
MIN_ROUND = 3         # skip first 2 rounds (no rolling history)
MIN_ODDS = 1.5        # ignore extreme shorteners
MAX_ODDS = 6.0        # ignore extreme longshots


def get_eligible(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to position-eligible players that have valid Betfair odds."""
    has_odds = (
        pred_df["betfair_implied_prob"].notna()
        & (pred_df["betfair_implied_prob"] > 0)
        & pred_df["betfair_closing_odds"].notna()
        & (pred_df["betfair_closing_odds"] >= MIN_ODDS)
        & (pred_df["betfair_closing_odds"] <= MAX_ODDS)
    )
    eligible_pos = pred_df["position_code"].isin(ELIGIBLE_POSITION_CODES)
    return pred_df[has_odds & eligible_pos].copy()


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Load combined feature store
    # -------------------------------------------------------------------------
    fs_path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    logger.info("Loading feature store from %s", fs_path)
    fs = load_feature_store(str(fs_path))
    logger.info("Feature store: %d rows x %d cols", len(fs), len(fs.columns))

    # -------------------------------------------------------------------------
    # 2. Build the Phase 5B model
    # -------------------------------------------------------------------------
    base_gbm = GBMModel(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=80,
        reg_alpha=3.0,
        reg_lambda=3.0,
    )
    model = CalibratedModel(base_model=base_gbm, method="isotonic", cal_rounds=5)

    # -------------------------------------------------------------------------
    # 3. Walk-forward through 2025 rounds
    # -------------------------------------------------------------------------
    season_2025 = fs[fs["season"] == 2025]
    all_rounds = sorted(season_2025["round_number"].unique())

    bet_records = []

    for rnd in all_rounds:
        if rnd < MIN_ROUND:
            continue

        # Training: all prior data
        train_mask = (
            (fs["season"] < 2025)
            | ((fs["season"] == 2025) & (fs["round_number"] < rnd))
        )
        train_df = fs[train_mask]

        # Prediction: this round
        pred_mask = (fs["season"] == 2025) & (fs["round_number"] == rnd)
        pred_df = fs[pred_mask].copy()

        if pred_df.empty or train_df.empty:
            logger.warning("Round %d: empty train or pred, skipping", rnd)
            continue

        # Fit model
        try:
            model.fit(train_df, train_df["scored_try"].values)
        except Exception as exc:
            logger.warning("Round %d: model fit failed — %s", rnd, exc)
            continue

        # Get eligible players
        eligible_df = get_eligible(pred_df)
        if eligible_df.empty:
            logger.info("Round %d: no eligible players with odds", rnd)
            continue

        # Compute model probs + blended probs
        try:
            model_probs = model.predict_proba(eligible_df)
        except Exception as exc:
            logger.warning("Round %d: predict_proba failed — %s", rnd, exc)
            continue

        eligible_df = eligible_df.copy()
        eligible_df["model_prob"] = model_probs
        eligible_df["blended_prob"] = (
            ALPHA * eligible_df["model_prob"]
            + (1 - ALPHA) * eligible_df["betfair_implied_prob"]
        )
        eligible_df["edge"] = (
            eligible_df["blended_prob"] - eligible_df["betfair_implied_prob"]
        )

        # Top 2 per game by blended_prob
        top2 = (
            eligible_df
            .sort_values("blended_prob", ascending=False)
            .groupby("match_id", sort=False)
            .head(TOP_N_PER_GAME)
        )

        # Resolve outcomes
        for _, row in top2.iterrows():
            actual_row = pred_df[
                (pred_df["match_id"] == row["match_id"])
                & (pred_df["player_id"] == row["player_id"])
            ]
            if actual_row.empty:
                continue

            scored = int(actual_row.iloc[0]["scored_try"])
            odds = float(row["betfair_closing_odds"])
            payout = FLAT_STAKE * odds if scored else 0.0

            bet_records.append({
                "round_number": rnd,
                "match_id": int(row["match_id"]),
                "player_id": int(row["player_id"]),
                "position_code": str(row["position_code"]),
                "odds": odds,
                "model_prob": float(row["model_prob"]),
                "blended_prob": float(row["blended_prob"]),
                "implied_prob": float(row["betfair_implied_prob"]),
                "edge": float(row["edge"]),
                "stake": FLAT_STAKE,
                "payout": payout,
                "won": scored,
            })

        n_games = eligible_df["match_id"].nunique()
        n_bets_rnd = len(top2)
        logger.info("Round %d: %d games → %d bets selected", rnd, n_games, n_bets_rnd)

    if not bet_records:
        logger.error("No bets recorded — check feature store and model config")
        return

    bet_df = pd.DataFrame(bet_records)
    total_staked = bet_df["stake"].sum()
    total_payout = bet_df["payout"].sum()
    total_profit = total_payout - total_staked
    total_bets = len(bet_df)
    total_wins = int(bet_df["won"].sum())
    roi_pct = total_profit / total_staked * 100

    logger.info(
        "Complete: %d bets, %d wins (%.1f%%), staked=$%.2f, profit=$%.2f, ROI=%.1f%%",
        total_bets, total_wins, total_wins / total_bets * 100,
        total_staked, total_profit, roi_pct,
    )

    # -------------------------------------------------------------------------
    # 4. Enrich with player/match names
    # -------------------------------------------------------------------------
    conn = sqlite3.connect(str(DB_PATH))

    players = pd.read_sql_query(
        "SELECT DISTINCT player_id, display_name FROM players_2025", conn
    )
    bet_df = bet_df.merge(players, on="player_id", how="left")

    matches = pd.read_sql_query("""
        SELECT m.match_id, m.venue_name,
               ht.squad_name AS home_team, at.squad_name AS away_team
        FROM matches_2025 m
        JOIN teams ht ON m.home_squad_id = ht.squad_id
        JOIN teams at ON m.away_squad_id = at.squad_id
    """, conn)
    bet_df = bet_df.merge(matches, on="match_id", how="left")

    player_squads = pd.read_sql_query("""
        SELECT DISTINCT ps.match_id, ps.player_id, t.squad_name AS player_team
        FROM player_stats_2025 ps
        JOIN teams t ON ps.squad_id = t.squad_id
    """, conn)
    bet_df = bet_df.merge(player_squads, on=["match_id", "player_id"], how="left")
    conn.close()

    # -------------------------------------------------------------------------
    # 5. Format report DataFrame
    # -------------------------------------------------------------------------
    bet_df["match"] = bet_df["home_team"] + " v " + bet_df["away_team"]
    bet_df["result"] = bet_df["won"].map({1: "WON", 0: "LOST"})
    bet_df["profit"] = (bet_df["payout"] - bet_df["stake"]).round(2)
    bet_df["model_prob_pct"] = (bet_df["model_prob"] * 100).round(1)
    bet_df["blended_prob_pct"] = (bet_df["blended_prob"] * 100).round(1)
    bet_df["implied_prob_pct"] = (bet_df["implied_prob"] * 100).round(1)
    bet_df["edge_pct"] = (bet_df["edge"] * 100).round(1)

    bet_df = bet_df.sort_values(
        ["round_number", "match", "blended_prob"], ascending=[True, True, False]
    ).reset_index(drop=True)
    bet_df["cumulative_pnl"] = bet_df["profit"].cumsum().round(2)

    report_cols = [
        "round_number", "match", "display_name", "player_team",
        "position_code", "odds", "model_prob_pct", "blended_prob_pct",
        "implied_prob_pct", "edge_pct", "stake", "payout", "profit",
        "result", "cumulative_pnl",
    ]
    report_df = bet_df[report_cols].copy()
    report_df = report_df.rename(columns={
        "round_number": "Round",
        "match": "Match",
        "display_name": "Player",
        "player_team": "Team",
        "position_code": "Position",
        "odds": "Odds",
        "model_prob_pct": "Model Prob %",
        "blended_prob_pct": "Blended Prob %",
        "implied_prob_pct": "Market Prob %",
        "edge_pct": "Edge %",
        "stake": "Stake",
        "payout": "Payout",
        "profit": "Profit",
        "result": "Result",
        "cumulative_pnl": "Cumulative P&L",
    })
    report_df["Odds"] = report_df["Odds"].round(2)
    report_df["Payout"] = report_df["Payout"].round(2)

    # -------------------------------------------------------------------------
    # 6. Round summary
    # -------------------------------------------------------------------------
    round_summary = (
        report_df.groupby("Round")
        .agg(
            Bets=("Player", "count"),
            Wins=("Result", lambda x: (x == "WON").sum()),
            Staked=("Stake", "sum"),
            Payout=("Payout", "sum"),
            Profit=("Profit", "sum"),
        )
        .reset_index()
    )
    round_summary["Hit Rate %"] = (round_summary["Wins"] / round_summary["Bets"] * 100).round(1)
    round_summary["ROI %"] = (round_summary["Profit"] / round_summary["Staked"] * 100).round(1)
    round_summary["Cumulative Profit"] = round_summary["Profit"].cumsum().round(2)
    round_summary["Staked"] = round_summary["Staked"].round(2)
    round_summary["Payout"] = round_summary["Payout"].round(2)
    round_summary["Profit"] = round_summary["Profit"].round(2)

    # -------------------------------------------------------------------------
    # 7. Overall summary
    # -------------------------------------------------------------------------
    rs = report_df
    t_bets = len(rs)
    t_wins = int((rs["Result"] == "WON").sum())
    t_staked = rs["Stake"].sum()
    t_payout = rs["Payout"].sum()
    t_profit = rs["Profit"].sum()

    overall = pd.DataFrame([
        {"Metric": "Strategy", "Value": f"Top {TOP_N_PER_GAME} per game (no edge filter)"},
        {"Metric": "Model", "Value": "CalibratedGBM(n=150, d=4, reg=3.0, mcs=80) + alpha=0.25 blending"},
        {"Metric": "Season", "Value": "2025 (walk-forward, train on 2024+prior rounds)"},
        {"Metric": "Total Bets", "Value": t_bets},
        {"Metric": "Total Wins", "Value": t_wins},
        {"Metric": "Hit Rate", "Value": f"{t_wins / t_bets * 100:.1f}%"},
        {"Metric": "Total Staked", "Value": f"${t_staked:,.2f}"},
        {"Metric": "Total Payout", "Value": f"${t_payout:,.2f}"},
        {"Metric": "Total Profit", "Value": f"${t_profit:,.2f}"},
        {"Metric": "ROI", "Value": f"{t_profit / t_staked * 100:.1f}%"},
        {"Metric": "Avg Odds", "Value": f"{rs['Odds'].mean():.2f}"},
        {"Metric": "Avg Edge %", "Value": f"{rs['Edge %'].mean():.1f}%"},
        {"Metric": "Avg Blended Prob %", "Value": f"{rs['Blended Prob %'].mean():.1f}%"},
        {"Metric": "Flat Stake", "Value": f"${FLAT_STAKE:.0f}"},
    ])

    # -------------------------------------------------------------------------
    # 8. Write Excel
    # -------------------------------------------------------------------------
    output_path = "data/backtest_results/top2_2025_roi.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="Summary", index=False)
        round_summary.to_excel(writer, sheet_name="By Round", index=False)
        report_df.to_excel(writer, sheet_name="All Bets", index=False)

        for rnd in sorted(report_df["Round"].unique()):
            rnd_df = report_df[report_df["Round"] == rnd].copy()
            sheet_name = f"Round {int(rnd)}"
            rnd_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Report saved to %s", output_path)
    logger.info(
        "RESULT: %d bets | %d wins (%.1f%%) | ROI=%.1f%% | Profit=$%.2f",
        t_bets, t_wins, t_wins / t_bets * 100, t_profit / t_staked * 100, t_profit,
    )


if __name__ == "__main__":
    main()
