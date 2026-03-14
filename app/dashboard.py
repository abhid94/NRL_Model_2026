"""NRL ATS Model Dashboard.

Interactive Streamlit dashboard for browsing predictions, comparing
bookmaker odds, and viewing bet recommendations.

Usage:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import sqlite3

import pandas as pd
import streamlit as st

from src.config import (
    BOOKMAKER_DISPLAY_NAMES,
    DB_PATH,
    ODDS_API_BOOKMAKERS,
)
from src.db import get_connection, table_exists

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOOKMAKER_COLS = [f"odds_{bk}" for bk in ODDS_API_BOOKMAKERS]
BOOKMAKER_HEADERS = {f"odds_{bk}": BOOKMAKER_DISPLAY_NAMES.get(bk, bk) for bk in ODDS_API_BOOKMAKERS}

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_connection() -> sqlite3.Connection:
    """Return a cached DB connection (safe for Streamlit re-runs)."""
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


@st.cache_data(ttl=300)
def _load_match_schedule(season: int) -> pd.DataFrame:
    """Load match schedule with team names for a season."""
    conn = _get_connection()
    table = f"matches_{season}"
    if not table_exists(conn, table):
        return pd.DataFrame()
    df = pd.read_sql_query(
        f"""
        SELECT m.match_id, m.round_number, m.venue_name,
               ht.squad_name AS home_team, at.squad_name AS away_team
        FROM {table} m
        JOIN teams ht ON m.home_squad_id = ht.squad_id
        JOIN teams at ON m.away_squad_id = at.squad_id
        ORDER BY m.round_number, m.match_id
        """,
        conn,
    )
    return df


@st.cache_data(ttl=300)
def _available_rounds(season: int) -> list[int]:
    """Return sorted list of round numbers for a season."""
    schedule = _load_match_schedule(season)
    if schedule.empty:
        return []
    return sorted(schedule["round_number"].unique().tolist())


def _match_label(row: pd.Series) -> str:
    return f"{row['home_team']} vs {row['away_team']}"


# ---------------------------------------------------------------------------
# Pipeline runner (session-state cached)
# ---------------------------------------------------------------------------

def _run_pipeline(season: int, round_number: int, bankroll: float,
                  flat_stake: float | None, min_edge: float,
                  pull_odds: bool = True) -> None:
    """Execute the weekly pipeline and store results in session state."""
    from src.pipeline.weekly_pipeline import run_weekly_pipeline

    label = "Running pipeline — training model & generating predictions..."
    if not pull_odds:
        label = "Running pipeline (using stored odds)..."

    with st.spinner(label):
        result = run_weekly_pipeline(
            season=season,
            round_number=round_number,
            bankroll=bankroll,
            flat_stake=flat_stake,
            rebuild_features=True,
            pull_odds=pull_odds,
        )

    predictions: pd.DataFrame = result["predictions"]
    bet_card = result["bet_card"]

    # Merge form stats from the feature store built during the pipeline
    predictions = _merge_form_stats(predictions, season, round_number)

    # Merge match info (team names)
    schedule = _load_match_schedule(season)
    if not schedule.empty:
        predictions = predictions.merge(
            schedule[["match_id", "home_team", "away_team", "venue_name"]],
            on="match_id",
            how="left",
        )

    # Build stakes lookup from bet card
    stake_map: dict[int, float] = {}
    bookmaker_map: dict[int, str] = {}
    if bet_card.bets:
        for b in bet_card.bets:
            stake_map[b["player_id"]] = b["stake"]
            if "bookmaker" in b:
                bookmaker_map[b["player_id"]] = b["bookmaker"]
    predictions["stake"] = predictions["player_id"].map(stake_map)

    st.session_state["predictions"] = predictions
    st.session_state["bet_card"] = bet_card
    st.session_state["pipeline_elapsed"] = result["elapsed_seconds"]
    st.session_state["pipeline_key"] = (season, round_number)


def _merge_form_stats(predictions: pd.DataFrame, season: int,
                      round_number: int) -> pd.DataFrame:
    """Merge rolling form stats from the feature store.

    First attempts a (match_id, player_id) merge. If that produces all NaN
    for the form columns (prediction-round match_ids not in the feature store),
    falls back to using each player's most recent historical row.
    """
    if predictions.empty or "match_id" not in predictions.columns:
        return predictions

    from src.config import FEATURE_STORE_DIR

    form_cols = ["rolling_tries_3", "rolling_line_breaks_3", "rolling_attack_tackle_breaks_3",
                  "typical_position_code", "is_positional_change"]
    path = FEATURE_STORE_DIR / f"feature_store_{season}.parquet"
    if not path.exists():
        return predictions

    read_cols = ["match_id", "player_id", "round_number"] + form_cols
    # Only read columns that actually exist in the parquet
    try:
        import pyarrow.parquet as pq
        available = set(pq.read_schema(path).names)
        read_cols = [c for c in read_cols if c in available]
    except Exception:
        pass
    fs = pd.read_parquet(path, columns=read_cols)

    merge_keys = ["match_id", "player_id"]
    new_cols = [c for c in form_cols if c not in predictions.columns and c in fs.columns]
    if not new_cols:
        return predictions

    result = predictions.merge(fs[merge_keys + new_cols], on=merge_keys, how="left")

    # Fallback: if the match_id merge produced all NaN, use latest row per player
    if result[new_cols].isna().all(axis=None) and "round_number" in fs.columns:
        LOGGER.info("Form stats merge by (match_id, player_id) yielded all NaN — "
                     "falling back to latest row per player_id")
        latest = (
            fs.sort_values("round_number")
            .drop_duplicates(subset=["player_id"], keep="last")
        )
        for col in new_cols:
            if col in latest.columns:
                fill_map = latest.set_index("player_id")[col]
                result[col] = result["player_id"].map(fill_map)

    return result


# ---------------------------------------------------------------------------
# Data management helpers
# ---------------------------------------------------------------------------

def _update_team_lists(season: int, round_number: int) -> None:
    """Scrape and ingest team lists for the selected round."""
    from src.ingestion.ingest_team_lists import ingest_round_team_lists

    with st.spinner(f"Updating team lists for {season} Round {round_number}..."):
        try:
            summary = ingest_round_team_lists(round_number=round_number, year=season)
            st.cache_data.clear()
            st.success(
                f"Team lists updated: {summary['n_matched']}/{summary['n_scraped']} "
                f"players matched, {summary['n_inserted']} rows upserted"
            )
            if summary.get("unmatched_players"):
                st.warning(
                    f"Unmatched players: {', '.join(summary['unmatched_players'])}"
                )
        except Exception as exc:
            LOGGER.exception("Team list update failed")
            st.error(f"Team list update failed: {exc}")


def _pull_game_stats(season: int) -> None:
    """Fetch and ingest completed match data from Champion Data."""
    from src.ingestion.ingest_match_data import fetch_and_ingest_completed_matches

    with st.spinner(f"Pulling game stats for {season}..."):
        try:
            summary = fetch_and_ingest_completed_matches(year=season)
            st.cache_data.clear()
            st.success(
                f"Game stats: {summary['n_ingested']} matches ingested, "
                f"{summary['n_failed']} failed, "
                f"{summary['n_pending']} were pending"
            )
            if summary.get("errors"):
                for err in summary["errors"][:5]:
                    st.warning(err)
        except Exception as exc:
            LOGGER.exception("Game stats pull failed")
            st.error(f"Game stats pull failed: {exc}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_pct(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "-"
    return f"{val * 100:.1f}%"


def _format_odds(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "-"
    return f"{val:.2f}"


def _format_stake(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "-"
    return f"${val:.0f}"


def _format_stat(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "-"
    return f"{val:.1f}"


def _style_edge(val: str) -> str:
    """Apply color to edge values."""
    if val == "-":
        return ""
    try:
        num = float(val.replace("%", "").replace("+", ""))
    except (ValueError, AttributeError):
        return ""
    if num >= 5:
        return "background-color: #c6efce; font-weight: bold"
    elif num >= 0:
        return "background-color: #e2efda"
    else:
        return "background-color: #fce4ec"


def _style_model_prob(val: str) -> str:
    """Green gradient for model probability with readable text."""
    if val == "-":
        return ""
    try:
        num = float(val.replace("%", ""))
    except (ValueError, AttributeError):
        return ""
    if num >= 40:
        return "background-color: #2e7d32; color: white; font-weight: bold"
    elif num >= 30:
        return "background-color: #558b2f; color: white; font-weight: bold"
    elif num >= 25:
        return "background-color: #7cb342; color: white"
    elif num >= 15:
        return "background-color: #c5e1a5"
    return ""


def _style_stake(val: str) -> str:
    if val != "-" and val.startswith("$"):
        return "color: #2e7d32; font-weight: bold"
    return ""


def _build_display_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """Build the display-ready DataFrame with formatted columns.

    Returns the display DataFrame and a mapping of row index to the
    bookmaker display-name column that has the best odds for that row.
    """
    rows = []
    best_col_per_row: dict[int, str] = {}

    for idx, (_, r) in enumerate(df.iterrows()):
        # Show "Usual Pos" only when it differs from current position
        pos = r.get("position_code", "")
        typical_pos = r.get("typical_position_code")
        usual_pos = ""
        if pd.notna(typical_pos) and typical_pos and typical_pos != pos:
            usual_pos = str(typical_pos)

        row: dict = {
            "Rank": int(r["match_rank"]) if pd.notna(r.get("match_rank")) else "",
            "Player": r.get("player_name", f"ID:{r['player_id']}"),
            "Pos": pos,
            "Usual Pos": usual_pos,
            "XIII": "Y" if r.get("is_starter") else "",
            "Model %": _format_pct(r.get("model_prob")),
            "Mkt %": _format_pct(r.get("best_implied_prob")),
            "Edge": _format_pct(r.get("edge")),
            "Stake": _format_stake(r.get("stake")),
            "Best Odds": _format_odds(r.get("best_odds")),
            "Best Book": BOOKMAKER_DISPLAY_NAMES.get(
                str(r.get("best_bookmaker", "")), str(r.get("best_bookmaker", "-"))
            ) if pd.notna(r.get("best_bookmaker")) else "-",
            "Tries(3m)": _format_stat(r.get("rolling_tries_3")),
            "LB(3m)": _format_stat(r.get("rolling_line_breaks_3")),
            "TB(3m)": _format_stat(r.get("rolling_attack_tackle_breaks_3")),
        }

        # Per-bookmaker odds columns — track which has the highest value
        best_val = 0.0
        best_header = None
        for col in BOOKMAKER_COLS:
            header = BOOKMAKER_HEADERS[col]
            raw = r.get(col)
            row[header] = _format_odds(raw)
            if raw is not None and not pd.isna(raw) and raw > best_val:
                best_val = raw
                best_header = header

        if best_header is not None and best_val > 0:
            best_col_per_row[idx] = best_header

        rows.append(row)

    return pd.DataFrame(rows), best_col_per_row


def _apply_styles(styler: pd.io.formats.style.Styler,
                   best_col_per_row: dict[int, str] | None = None,
                   ) -> pd.io.formats.style.Styler:
    """Apply conditional formatting to the display table."""
    styler = styler.map(_style_edge, subset=["Edge"])
    styler = styler.map(_style_model_prob, subset=["Model %"])
    styler = styler.map(_style_stake, subset=["Stake"])

    # Highlight the best-odds bookmaker cell per row
    if best_col_per_row:
        bk_display_names = list(BOOKMAKER_HEADERS.values())
        available_bk = [c for c in bk_display_names if c in styler.data.columns]
        if available_bk:
            def _highlight_best(row: pd.Series) -> list[str]:
                best_col = best_col_per_row.get(row.name)
                return [
                    "background-color: #1b5e20; color: white; font-weight: bold"
                    if col == best_col else ""
                    for col in row.index
                ]
            styler = styler.apply(_highlight_best, subset=available_bk, axis=1)

    return styler


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="NRL ATS Dashboard", layout="wide")
    st.title("NRL ATS Betting Dashboard")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")
        season = st.selectbox("Season", [2026, 2025, 2024], index=0)
        rounds = _available_rounds(season)
        if not rounds:
            st.warning(f"No matches found for {season}")
            return
        round_number = st.selectbox("Round", rounds, index=0)

        # Match filter
        schedule = _load_match_schedule(season)
        round_matches = schedule[schedule["round_number"] == round_number]
        match_options = ["All Matches"] + [
            _match_label(row) for _, row in round_matches.iterrows()
        ]
        match_filter = st.selectbox("Match", match_options)

        st.divider()

        bankroll = st.number_input("Bankroll ($)", min_value=1000, max_value=100000,
                                   value=10000, step=1000)
        staking = st.radio("Staking", ["Flat $100", "Quarter Kelly"], index=0)
        flat_stake = 100.0 if staking == "Flat $100" else None
        min_edge = st.slider("Min Edge (pp)", min_value=1, max_value=10, value=5) / 100.0
        pull_odds = st.toggle("Pull latest odds", value=True,
                              help="Yes = fetch fresh odds from APIs. No = use odds already in the database.")

        st.divider()

        run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)

        if run_clicked:
            _run_pipeline(season, round_number, bankroll, flat_stake, min_edge, pull_odds)

        # --- Data Management ---
        st.divider()
        st.subheader("Data Management")

        if st.button("Update Team Lists", use_container_width=True):
            _update_team_lists(season, round_number)

        if st.button("Pull Game Stats", use_container_width=True):
            _pull_game_stats(season)

    # --- Main area ---
    if "predictions" not in st.session_state:
        st.info("Select season & round, then click **Run Pipeline** to generate predictions.")
        return

    predictions: pd.DataFrame = st.session_state["predictions"]
    bet_card = st.session_state["bet_card"]
    elapsed = st.session_state.get("pipeline_elapsed", 0)
    pipeline_key = st.session_state.get("pipeline_key", (None, None))

    st.caption(f"Round {pipeline_key[1]} Predictions — {pipeline_key[0]}  |  "
               f"Pipeline ran in {elapsed:.1f}s")

    # --- Summary metrics ---
    col1, col2, col3, col4 = st.columns(4)
    n_total = len(predictions)
    n_eligible = int(predictions["is_eligible"].sum()) if "is_eligible" in predictions.columns else 0
    n_positive_edge = int((predictions["edge"] > 0).sum()) if "edge" in predictions.columns else 0
    total_staked = bet_card.total_staked if bet_card else 0

    col1.metric("Total Players", n_total)
    col2.metric("Eligible Bets", n_eligible)
    col3.metric("Positive Edge", n_positive_edge)
    col4.metric("Total Stake", f"${total_staked:,.0f}")

    # Projected lineup warning
    if "is_projected_lineup" in predictions.columns and predictions["is_projected_lineup"].any():
        st.warning(
            "Team lists not yet published — showing projected lineup from previous round. "
            "Re-run the pipeline after official team lists are announced for accurate predictions."
        )

    # --- Recommended Bets ---
    if bet_card and bet_card.bets:
        with st.expander(f"Recommended Bets ({len(bet_card.bets)} bets)", expanded=True):
            rec_pids = {b["player_id"] for b in bet_card.bets}
            rec_df = predictions[predictions["player_id"].isin(rec_pids)].copy()
            rec_df = rec_df.sort_values("edge", ascending=False)
            display, best_cols = _build_display_df(rec_df)
            styled = _apply_styles(display.style, best_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.caption(
                f"Exposure: ${bet_card.total_staked:,.0f} "
                f"({bet_card.exposure_pct:.1f}% of ${bankroll:,.0f} bankroll) "
                f"across {bet_card.n_matches_bet} matches"
            )

            # Warn if flat stakes were reduced by exposure cap
            if flat_stake and len(bet_card.bets) > 0:
                actual_avg = bet_card.total_staked / len(bet_card.bets)
                if actual_avg < flat_stake * 0.95:  # >5% reduction
                    st.warning(
                        f"Flat stakes reduced from ${flat_stake:.0f} to "
                        f"~${actual_avg:.0f} per bet due to exposure cap "
                        f"({bet_card.exposure_pct:.1f}% of bankroll)"
                    )
    else:
        st.info("No bets meet the current edge threshold.")

    # Early-season form stats note
    if "season" in predictions.columns:
        pred_season = predictions["season"].iloc[0]
    else:
        pred_season = pipeline_key[0]
    pred_round = pipeline_key[1]
    if pred_round is not None and pred_round <= 4:
        n_prior = max(0, pred_round - 1)
        st.info(
            f"Form stats (Tries/LB/TB 3m) based on {n_prior} match{'es' if n_prior != 1 else ''} "
            f"of {pred_season} data. Model relies on cross-season priors until more rounds are played."
        )

    # --- Per-match tables ---
    st.subheader("Per-Match Predictions")

    # Determine which matches to show
    if match_filter != "All Matches":
        # Parse "Home vs Away" back to match_id
        filtered = round_matches[
            round_matches.apply(lambda r: _match_label(r) == match_filter, axis=1)
        ]
        if not filtered.empty:
            match_ids = filtered["match_id"].tolist()
        else:
            match_ids = predictions["match_id"].unique().tolist()
    else:
        match_ids = sorted(predictions["match_id"].unique().tolist())

    for mid in match_ids:
        match_preds = predictions[predictions["match_id"] == mid].copy()
        if match_preds.empty:
            continue

        # Build match header
        home = match_preds["home_team"].iloc[0] if "home_team" in match_preds.columns else "?"
        away = match_preds["away_team"].iloc[0] if "away_team" in match_preds.columns else "?"
        venue = match_preds["venue_name"].iloc[0] if "venue_name" in match_preds.columns else ""
        n_bets = int(match_preds["stake"].notna().sum())
        bet_tag = f" | {n_bets} bet{'s' if n_bets != 1 else ''}" if n_bets > 0 else ""

        header = f"{home} vs {away}"
        if venue:
            header += f"  ({venue})"
        header += bet_tag

        with st.expander(header, expanded=True):
            # Sort: starters first (is_starter DESC), then by model rank
            sort_cols = []
            sort_asc = []
            if "is_starter" in match_preds.columns:
                sort_cols.append("is_starter")
                sort_asc.append(False)  # starters (1) above reserves (0)
            sort_cols.append("match_rank")
            sort_asc.append(True)
            match_preds = match_preds.sort_values(sort_cols, ascending=sort_asc)
            display, best_cols = _build_display_df(match_preds)
            styled = _apply_styles(display.style, best_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
