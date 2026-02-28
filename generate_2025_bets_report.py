"""Generate Excel report of all 2025 bets placed by the best model config.

Best config from Phase 5B:
- Model: CalibratedGBM (n=150, depth=4, reg=3.0, min_child_samples=80)
- Strategy: MarketBlendedStrategy(alpha=0.25, min_edge=0.03)
- Flat stake: $100

Output: data/backtest_results/2025_bets_report.xlsx
"""

import sqlite3
import logging
import pandas as pd
import numpy as np

from src.config import DB_PATH, FEATURE_STORE_DIR
from src.features.feature_store import load_feature_store
from src.models.gbm import GBMModel
from src.models.calibration import CalibratedModel
from src.models.baseline import MarketBlendedStrategy
from src.evaluation.backtest import run_backtest, BacktestConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # 1. Load combined feature store
    fs_path = FEATURE_STORE_DIR / "feature_store_combined.parquet"
    logger.info("Loading feature store from %s", fs_path)
    fs = load_feature_store(str(fs_path))
    logger.info("Feature store: %d rows x %d cols", len(fs), len(fs.columns))

    # 2. Set up the best model + strategy from Phase 5B
    base_gbm = GBMModel(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=80,
        reg_alpha=3.0,
        reg_lambda=3.0,
    )
    model = CalibratedModel(base_model=base_gbm, method="isotonic", cal_rounds=5)

    strategy = MarketBlendedStrategy(
        alpha=0.25,
        min_edge=0.03,
    )

    config = BacktestConfig(
        initial_bankroll=10_000.0,
        flat_stake=100.0,
    )

    # 3. Run backtest on 2025 only (2024 used as training seed)
    logger.info("Running backtest on 2025 season...")
    result = run_backtest(
        feature_store=fs,
        strategy=strategy,
        model=model,
        config=config,
        seasons=[2025],
        min_round=3,
    )

    logger.info(
        "Backtest complete: %d bets, ROI=%.1f%%, profit=$%.2f",
        result.n_bets, result.roi * 100, result.total_profit,
    )

    # 4. Get bet-level DataFrame
    bet_df = result.to_bet_dataframe()
    if bet_df.empty:
        logger.warning("No bets placed!")
        return

    # 5. Enrich with player names and match details
    conn = sqlite3.connect(str(DB_PATH))

    # Player names
    players = pd.read_sql_query(
        "SELECT DISTINCT player_id, display_name FROM players_2025", conn
    )
    bet_df = bet_df.merge(players, on="player_id", how="left")

    # Match details (teams, venue)
    matches = pd.read_sql_query("""
        SELECT m.match_id, m.venue_name,
               ht.squad_name AS home_team, at.squad_name AS away_team
        FROM matches_2025 m
        JOIN teams ht ON m.home_squad_id = ht.squad_id
        JOIN teams at ON m.away_squad_id = at.squad_id
    """, conn)
    bet_df = bet_df.merge(matches, on="match_id", how="left")

    # Player's team
    player_squads = pd.read_sql_query("""
        SELECT DISTINCT ps.match_id, ps.player_id, t.squad_name AS player_team
        FROM player_stats_2025 ps
        JOIN teams t ON ps.squad_id = t.squad_id
    """, conn)
    bet_df = bet_df.merge(player_squads, on=["match_id", "player_id"], how="left")

    conn.close()

    # 6. Build the match label
    bet_df["match"] = bet_df["home_team"] + " v " + bet_df["away_team"]

    # 7. Format columns for the report
    bet_df["result"] = bet_df["won"].map({1: "WON", 0: "LOST"})
    bet_df["profit"] = bet_df["payout"] - bet_df["stake"]
    bet_df["model_prob_pct"] = (bet_df["model_prob"] * 100).round(1)
    bet_df["implied_prob_pct"] = (bet_df["implied_prob"] * 100).round(1)
    bet_df["edge_pct"] = (bet_df["edge"] * 100).round(1)

    # 8. Sort, compute cumulative P&L, then select columns
    bet_df = bet_df.sort_values(["round_number", "match", "edge_pct"], ascending=[True, True, False])
    bet_df = bet_df.reset_index(drop=True)
    bet_df["cumulative_pnl"] = bet_df["profit"].cumsum().round(2)

    report_cols = [
        "round_number", "match", "display_name", "player_team",
        "position_code", "odds", "model_prob_pct", "implied_prob_pct",
        "edge_pct", "stake", "payout", "profit", "result", "cumulative_pnl",
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
    report_df["Profit"] = report_df["Profit"].round(2)

    # 9. Build round-level summary
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

    # 10. Overall summary row
    total_bets = len(report_df)
    total_wins = (report_df["Result"] == "WON").sum()
    total_staked = report_df["Stake"].sum()
    total_payout = report_df["Payout"].sum()
    total_profit = report_df["Profit"].sum()

    overall = pd.DataFrame([{
        "Metric": "Total Bets",
        "Value": total_bets,
    }, {
        "Metric": "Total Wins",
        "Value": total_wins,
    }, {
        "Metric": "Hit Rate",
        "Value": f"{total_wins / total_bets * 100:.1f}%",
    }, {
        "Metric": "Total Staked",
        "Value": f"${total_staked:,.2f}",
    }, {
        "Metric": "Total Payout",
        "Value": f"${total_payout:,.2f}",
    }, {
        "Metric": "Total Profit",
        "Value": f"${total_profit:,.2f}",
    }, {
        "Metric": "ROI",
        "Value": f"{total_profit / total_staked * 100:.1f}%",
    }, {
        "Metric": "Avg Odds",
        "Value": f"{report_df['Odds'].mean():.2f}",
    }, {
        "Metric": "Avg Edge",
        "Value": f"{report_df['Edge %'].mean():.1f}%",
    }])

    # 11. Write to Excel with multiple sheets
    output_path = "data/backtest_results/2025_bets_report.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="Summary", index=False)
        round_summary.to_excel(writer, sheet_name="By Round", index=False)
        report_df.to_excel(writer, sheet_name="All Bets", index=False)

        # Per-round sheets with match grouping
        for rnd in sorted(report_df["Round"].unique()):
            rnd_df = report_df[report_df["Round"] == rnd].copy()
            sheet_name = f"Round {int(rnd)}"
            if len(sheet_name) > 31:  # Excel sheet name limit
                sheet_name = sheet_name[:31]
            rnd_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Report saved to %s", output_path)
    logger.info(
        "Summary: %d bets, %d wins (%.1f%%), ROI=%.1f%%, Profit=$%.2f",
        total_bets, total_wins, total_wins / total_bets * 100,
        total_profit / total_staked * 100, total_profit,
    )


if __name__ == "__main__":
    main()
