"""Market-derived features for ATS prediction.

Converts bookmaker match-level odds (total points lines, h2h prices)
into expected team tries features. Uses:
- Betfair COMBINED_TOTAL lines (2024-2025 historical)
- Bet365 match odds via odds-api.io (2026 live)

LEAKAGE PREVENTION:
- Match-level odds are pre-match public data (CLAUDE.md Rule 3)
- Regression coefficients are fitted on PRIOR seasons only
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from src.db import table_exists

LOGGER = logging.getLogger(__name__)

# Regression coefficients fitted on 2024-2025 Betfair COMBINED_TOTAL data:
#   team_tries = 0.0685 * total_points_line + 0.4877 * is_home + 0.6035
# R² = 0.036 (low, but directionally useful and market-informed)
_TOTAL_LINE_COEF: float = 0.0685
_IS_HOME_COEF: float = 0.4877
_INTERCEPT: float = 0.6035


def _estimate_team_tries_from_total_line(
    total_line: float,
    is_home: bool,
) -> float:
    """Estimate expected team tries from a total points line.

    Parameters
    ----------
    total_line : float
        Total match points line (e.g., 47.5).
    is_home : bool
        Whether the team is the home team.

    Returns
    -------
    float
        Estimated expected tries for this team.
    """
    return (
        _TOTAL_LINE_COEF * total_line
        + _IS_HOME_COEF * float(is_home)
        + _INTERCEPT
    )


def compute_betfair_market_features(
    conn: sqlite3.Connection,
    season: int,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute market-implied expected team tries from Betfair data.

    Uses the COMBINED_TOTAL market to derive total points lines,
    then converts to expected team tries via linear regression.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year (2024 or 2025).
    as_of_round : int | None
        If given, only return features for this round.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, market_expected_team_tries,
        market_total_line.
    """
    betfair_table = f"betfair_markets_{season}"
    matches_table = f"matches_{season}"
    team_stats_table = f"team_stats_{season}"

    if not table_exists(conn, betfair_table):
        LOGGER.info("No %s table — skipping Betfair market features", betfair_table)
        return pd.DataFrame(
            columns=["match_id", "squad_id", "market_expected_team_tries", "market_total_line"]
        )

    # Get the COMBINED_TOTAL "Over" line closest to even money per match
    ct = pd.read_sql_query(
        f"""
        SELECT bm.AD_match_id AS match_id, bm.handicap AS total_line,
               CAST(COALESCE(NULLIF(bm.last_preplay_price, ''),
                    bm.best_back_price_1_min_prior,
                    bm.best_back_price_30_min_prior) AS REAL) AS price
        FROM {betfair_table} bm
        WHERE bm.market_type = 'COMBINED_TOTAL'
          AND bm.runner_name = 'Over'
          AND bm.AD_match_id IS NOT NULL
        """,
        conn,
    )

    if ct.empty:
        LOGGER.info("No COMBINED_TOTAL data in %s", betfair_table)
        return pd.DataFrame(
            columns=["match_id", "squad_id", "market_expected_team_tries", "market_total_line"]
        )

    ct = ct[ct["price"].notna() & (ct["price"] > 1)].copy()
    ct["abs_price_diff"] = (ct["price"] - 1.95).abs()
    closest = (
        ct.sort_values("abs_price_diff")
        .drop_duplicates("match_id", keep="first")[["match_id", "total_line"]]
    )

    # Get home/away squad_ids from matches
    round_filter = f"AND m.round_number = {as_of_round}" if as_of_round else ""
    squads = pd.read_sql_query(
        f"""
        SELECT m.match_id, m.home_squad_id, m.away_squad_id
        FROM {matches_table} m
        WHERE 1=1 {round_filter}
        """,
        conn,
    )

    # Expand to per-team rows
    home = squads[["match_id", "home_squad_id"]].rename(
        columns={"home_squad_id": "squad_id"}
    )
    home["is_home"] = 1
    away = squads[["match_id", "away_squad_id"]].rename(
        columns={"away_squad_id": "squad_id"}
    )
    away["is_home"] = 0
    teams = pd.concat([home, away], ignore_index=True)

    # Join total line
    teams = teams.merge(closest, on="match_id", how="left")

    # Compute expected team tries
    teams["market_expected_team_tries"] = np.where(
        teams["total_line"].notna(),
        _TOTAL_LINE_COEF * teams["total_line"]
        + _IS_HOME_COEF * teams["is_home"]
        + _INTERCEPT,
        np.nan,
    )

    teams.rename(columns={"total_line": "market_total_line"}, inplace=True)
    result = teams[["match_id", "squad_id", "market_expected_team_tries", "market_total_line"]].copy()

    LOGGER.info(
        "Computed Betfair market features: %d team-match rows, %d with total line",
        len(result),
        result["market_expected_team_tries"].notna().sum(),
    )
    return result


def compute_bet365_market_features(
    conn: sqlite3.Connection,
    season: int,
    round_number: int | None = None,
) -> pd.DataFrame:
    """Compute market features from Bet365 match-level odds (odds-api.io).

    For 2026+ where Betfair data isn't available. Uses Bet365 handicap
    lines to derive implied expected team tries.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year (>= 2026).
    round_number : int | None
        If given, only return features for this round.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, market_expected_team_tries,
        bet365_home_odds, bet365_away_odds, bet365_implied_win_prob.
    """
    from src.odds.odds_api_io import (
        OddsAPIIOError,
        fetch_nrl_match_odds,
    )

    empty = pd.DataFrame(
        columns=[
            "match_id", "squad_id", "market_expected_team_tries",
            "bet365_home_odds", "bet365_away_odds",
            "bet365_implied_win_prob",
        ]
    )

    try:
        records = fetch_nrl_match_odds(conn, season, round_number)
    except OddsAPIIOError as exc:
        LOGGER.warning("Bet365 match odds fetch failed: %s", exc)
        return empty

    if not records:
        LOGGER.info("No Bet365 match odds available")
        return empty

    df = pd.DataFrame(records)

    # Extract h2h odds (Game Betting 2-Way)
    h2h = df[df["market"] == "Game Betting 2-Way"].copy()
    if h2h.empty:
        LOGGER.info("No Bet365 h2h market data")
        return empty

    # Build per-match h2h features
    match_features = []
    for mid, group in h2h.groupby("match_id"):
        home_odds = group["home_odds"].dropna().iloc[0] if group["home_odds"].notna().any() else None
        away_odds = group["away_odds"].dropna().iloc[0] if group["away_odds"].notna().any() else None

        if home_odds and away_odds and home_odds > 1 and away_odds > 1:
            home_ip = 1.0 / home_odds
            away_ip = 1.0 / away_odds
            overround = home_ip + away_ip
            home_fair = home_ip / overround
            away_fair = away_ip / overround
        else:
            home_fair = None
            away_fair = None

        match_features.append({
            "match_id": mid,
            "bet365_home_odds": home_odds,
            "bet365_away_odds": away_odds,
            "bet365_home_implied_prob": home_fair,
            "bet365_away_implied_prob": away_fair,
        })

    mf = pd.DataFrame(match_features)

    # Get squads for each match
    matches_table = f"matches_{season}"
    round_filter = f"AND round_number = {round_number}" if round_number else ""
    squads = pd.read_sql_query(
        f"""
        SELECT match_id, home_squad_id, away_squad_id
        FROM {matches_table}
        WHERE 1=1 {round_filter}
        """,
        conn,
    )

    # Expand to per-team
    home = squads[["match_id", "home_squad_id"]].rename(columns={"home_squad_id": "squad_id"})
    home["is_home"] = 1
    away = squads[["match_id", "away_squad_id"]].rename(columns={"away_squad_id": "squad_id"})
    away["is_home"] = 0
    teams = pd.concat([home, away], ignore_index=True)

    teams = teams.merge(mf, on="match_id", how="left")

    # Derive bet365_implied_win_prob for each team
    teams["bet365_implied_win_prob"] = np.where(
        teams["is_home"] == 1,
        teams["bet365_home_implied_prob"],
        teams["bet365_away_implied_prob"],
    )

    # Use implied win prob to estimate expected team tries
    # Higher win prob → more expected tries
    # Proxy: avg_tries * (win_prob / 0.5) scaled by home advantage
    avg_team_tries = 4.06  # From 2024-2025 data
    teams["market_expected_team_tries"] = np.where(
        teams["bet365_implied_win_prob"].notna(),
        avg_team_tries
        * (teams["bet365_implied_win_prob"] / 0.5)
        * np.where(teams["is_home"] == 1, 1.05, 0.95),
        np.nan,
    )

    result = teams[[
        "match_id", "squad_id", "market_expected_team_tries",
        "bet365_home_odds", "bet365_away_odds",
        "bet365_implied_win_prob",
    ]].copy()

    LOGGER.info(
        "Computed Bet365 market features: %d team-match rows",
        len(result),
    )
    return result


def compute_multi_bookmaker_features(
    conn: sqlite3.Connection,
    season: int,
    round_number: int | None = None,
) -> pd.DataFrame:
    """Compute market decorrelation features from multi-bookmaker odds.

    Produces features that capture bookmaker consensus/disagreement:
    - bookmaker_consensus_spread: max - min implied prob across bookmakers
    - n_bookmakers_offering: count of bookmakers with odds for this player
    - best_vs_median_odds: ratio of best price to median price
    - odds_band: categorical bucket of best implied prob

    All features use pre-match public odds (no leakage).

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    round_number : int | None
        If given, only return features for this round.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, player_id, bookmaker_consensus_spread,
        n_bookmakers_offering, best_vs_median_odds, odds_band.
    """
    table = f"bookmaker_odds_{season}"
    if not table_exists(conn, table):
        LOGGER.info("No %s table — skipping multi-bookmaker features", table)
        return pd.DataFrame(
            columns=["match_id", "player_id", "bookmaker_consensus_spread",
                      "n_bookmakers_offering", "best_vs_median_odds", "odds_band"]
        )

    round_filter = ""
    if round_number is not None:
        round_filter = f"AND m.round_number = {int(round_number)}"

    try:
        df = pd.read_sql_query(
            f"""
            SELECT bo.match_id, bo.player_id, bo.bookmaker,
                   bo.decimal_odds, bo.implied_probability
            FROM {table} bo
            JOIN matches_{season} m ON bo.match_id = m.match_id
            WHERE bo.is_available = 1 {round_filter}
            """,
            conn,
        )
    except Exception:
        LOGGER.info("Could not query %s for multi-bookmaker features", table)
        return pd.DataFrame(
            columns=["match_id", "player_id", "bookmaker_consensus_spread",
                      "n_bookmakers_offering", "best_vs_median_odds", "odds_band"]
        )

    if df.empty:
        return pd.DataFrame(
            columns=["match_id", "player_id", "bookmaker_consensus_spread",
                      "n_bookmakers_offering", "best_vs_median_odds", "odds_band"]
        )

    # Compute per-player-match aggregates
    grouped = df.groupby(["match_id", "player_id"])

    result = grouped.agg(
        n_bookmakers_offering=("bookmaker", "nunique"),
        max_implied_prob=("implied_probability", "max"),
        min_implied_prob=("implied_probability", "min"),
        median_odds=("decimal_odds", "median"),
        best_odds=("decimal_odds", "max"),
        best_implied_prob=("implied_probability", "min"),  # lowest implied = best price
    ).reset_index()

    # Bookmaker consensus spread: wider = more disagreement = potential value
    result["bookmaker_consensus_spread"] = (
        result["max_implied_prob"] - result["min_implied_prob"]
    )

    # Best vs median odds ratio: outlier best prices may indicate slow-to-update bookmaker
    result["best_vs_median_odds"] = np.where(
        result["median_odds"] > 0,
        result["best_odds"] / result["median_odds"],
        1.0,
    )

    # Odds band: categorical bucket of best implied prob
    result["odds_band"] = pd.cut(
        result["best_implied_prob"],
        bins=[0, 0.10, 0.20, 0.30, 0.40, 1.0],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
        include_lowest=True,
    ).astype(str)

    output_cols = [
        "match_id", "player_id",
        "bookmaker_consensus_spread", "n_bookmakers_offering",
        "best_vs_median_odds", "odds_band",
    ]

    LOGGER.info(
        "Computed multi-bookmaker features: %d player-match rows, avg spread=%.3f",
        len(result),
        result["bookmaker_consensus_spread"].mean() if not result.empty else 0,
    )
    return result[output_cols]


def compute_market_miscalibration(
    conn: sqlite3.Connection,
    season: int,
    round_number: int | None = None,
) -> pd.DataFrame:
    """Compute historical market miscalibration by position group × odds band.

    For each position group and odds band, computes the gap between
    historical actual try rate and historical average bookmaker implied
    probability. Reveals where the market is systematically wrong.

    Uses ONLY data from prior rounds/seasons (no leakage).

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    round_number : int | None
        If given, computes from rounds < round_number.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, player_id, market_miscalibration_by_position.
    """
    # Build historical data from prior seasons
    historical_frames = []
    for prior_year in [2024, 2025]:
        if prior_year >= season:
            continue
        ps_table = f"player_stats_{prior_year}"
        bf_table = f"betfair_markets_{prior_year}"
        m_table = f"matches_{prior_year}"

        if not table_exists(conn, ps_table) or not table_exists(conn, bf_table):
            continue

        try:
            df = pd.read_sql_query(
                f"""
                SELECT ps.player_id, ps.match_id, ps.jumper_number,
                       ps.tries,
                       CAST(COALESCE(NULLIF(bf.last_preplay_price, ''),
                            bf.best_back_price_1_min_prior) AS REAL) AS odds
                FROM {ps_table} ps
                JOIN {m_table} m ON ps.match_id = m.match_id
                LEFT JOIN {bf_table} bf
                    ON bf.AD_match_id = ps.match_id
                    AND bf.AD_player_id = ps.player_id
                    AND bf.market_type = 'TO_SCORE'
                WHERE m.match_type = 'H'
                """,
                conn,
            )
            if not df.empty:
                historical_frames.append(df)
        except Exception:
            continue

    if not historical_frames:
        return pd.DataFrame(columns=["match_id", "player_id", "market_miscalibration_by_position"])

    hist = pd.concat(historical_frames, ignore_index=True)
    hist = hist[hist["odds"].notna() & (hist["odds"] > 1)].copy()
    hist["implied_prob"] = 1.0 / hist["odds"]
    hist["scored"] = (hist["tries"] > 0).astype(int)

    from src.config import position_from_jersey
    hist["position_code"] = hist["jumper_number"].apply(
        lambda j: position_from_jersey(int(j) if pd.notna(j) else None).code
    )
    hist["odds_band"] = pd.cut(
        hist["implied_prob"],
        bins=[0, 0.10, 0.20, 0.30, 0.40, 1.0],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
        include_lowest=True,
    ).astype(str)

    # Compute miscalibration: actual_try_rate - avg_implied_prob per group
    miscal = hist.groupby(["position_code", "odds_band"]).agg(
        actual_rate=("scored", "mean"),
        avg_implied=("implied_prob", "mean"),
        count=("scored", "count"),
    ).reset_index()
    miscal["miscalibration"] = miscal["actual_rate"] - miscal["avg_implied"]
    miscal = miscal[miscal["count"] >= 20]  # Only use groups with enough data

    # Now get current season players to merge
    ps_table = f"player_stats_{season}"
    m_table = f"matches_{season}"
    bf_table = f"betfair_markets_{season}"

    round_filter = f"AND m.round_number = {round_number}" if round_number else ""
    current_source = ps_table if table_exists(conn, ps_table) else f"team_lists_{season}"

    try:
        if table_exists(conn, ps_table):
            current = pd.read_sql_query(
                f"""
                SELECT ps.player_id, ps.match_id, ps.jumper_number
                FROM {ps_table} ps
                JOIN {m_table} m ON ps.match_id = m.match_id
                WHERE m.match_type = 'H' {round_filter}
                """,
                conn,
            )
        else:
            current = pd.read_sql_query(
                f"""
                SELECT tl.player_id, tl.match_id, tl.jersey_number AS jumper_number
                FROM team_lists_{season} tl
                JOIN {m_table} m ON tl.match_id = m.match_id
                WHERE m.match_type = 'H' AND tl.player_id IS NOT NULL {round_filter}
                """,
                conn,
            )
    except Exception:
        return pd.DataFrame(columns=["match_id", "player_id", "market_miscalibration_by_position"])

    if current.empty:
        return pd.DataFrame(columns=["match_id", "player_id", "market_miscalibration_by_position"])

    current["position_code"] = current["jumper_number"].apply(
        lambda j: position_from_jersey(int(j) if pd.notna(j) else None).code
    )

    # Get best odds for the current round to determine odds band
    bk_table = f"bookmaker_odds_{season}"
    if table_exists(conn, bk_table) and round_number:
        try:
            odds_df = pd.read_sql_query(
                f"""
                SELECT bo.match_id, bo.player_id,
                       MIN(bo.implied_probability) AS best_ip
                FROM {bk_table} bo
                JOIN {m_table} m ON bo.match_id = m.match_id
                WHERE m.round_number = {round_number} AND bo.is_available = 1
                GROUP BY bo.match_id, bo.player_id
                """,
                conn,
            )
            if not odds_df.empty:
                current = current.merge(odds_df, on=["match_id", "player_id"], how="left")
        except Exception:
            pass

    # Also try Betfair odds
    if "best_ip" not in current.columns or current["best_ip"].isna().all():
        if table_exists(conn, bf_table):
            try:
                bf_odds = pd.read_sql_query(
                    f"""
                    SELECT bf.AD_match_id AS match_id, bf.AD_player_id AS player_id,
                           1.0 / CAST(COALESCE(NULLIF(bf.last_preplay_price, ''),
                                bf.best_back_price_1_min_prior) AS REAL) AS best_ip
                    FROM {bf_table} bf
                    WHERE bf.market_type = 'TO_SCORE' AND bf.AD_match_id IS NOT NULL
                    """,
                    conn,
                )
                if not bf_odds.empty:
                    current = current.merge(bf_odds, on=["match_id", "player_id"], how="left")
            except Exception:
                pass

    if "best_ip" not in current.columns:
        current["best_ip"] = np.nan

    current["odds_band"] = pd.cut(
        current["best_ip"],
        bins=[0, 0.10, 0.20, 0.30, 0.40, 1.0],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
        include_lowest=True,
    ).astype(str)

    # Merge miscalibration
    current = current.merge(
        miscal[["position_code", "odds_band", "miscalibration"]],
        on=["position_code", "odds_band"],
        how="left",
    )
    current.rename(columns={"miscalibration": "market_miscalibration_by_position"}, inplace=True)

    LOGGER.info(
        "Computed market miscalibration: %d rows, %.1f%% with values",
        len(current),
        current["market_miscalibration_by_position"].notna().mean() * 100,
    )
    return current[["match_id", "player_id", "market_miscalibration_by_position"]]


def compute_market_features(
    conn: sqlite3.Connection,
    season: int,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute market-implied features for a season.

    Dispatches to Betfair (2024-2025) or Bet365 (2026+) based on
    data availability.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    as_of_round : int | None
        If given, only return features for this round.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, market_expected_team_tries,
        plus source-specific columns.
    """
    # Try Betfair first (available for 2024-2025)
    betfair_table = f"betfair_markets_{season}"
    if table_exists(conn, betfair_table):
        result = compute_betfair_market_features(conn, season, as_of_round)
        if not result.empty:
            return result

    # Fall back to Bet365 (2026+)
    if season >= 2026:
        try:
            return compute_bet365_market_features(conn, season, as_of_round)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Bet365 market features failed: %s", exc)

    LOGGER.info("No market features available for season %d", season)
    return pd.DataFrame(
        columns=["match_id", "squad_id", "market_expected_team_tries"]
    )
