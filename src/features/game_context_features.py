"""Game context feature engineering utilities.

This module computes the highest-priority Tier 1 features for ATS prediction:
- expected_team_tries: team attack × opponent defence × home advantage
- player_try_share: player's rolling share of team tries
- Opponent defensive context features
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Sequence

import numpy as np
import pandas as pd

from .. import db
from ..config import DEFAULT_TEAM_FEATURE_WINDOWS, EWM_SPAN

LOGGER = logging.getLogger(__name__)

# Home advantage multiplier for try expectation (derived from ~1.8pp home advantage)
HOME_ADVANTAGE_MULTIPLIER = 1.095  # ~9.5% boost in expected tries for home teams


@dataclass(frozen=True)
class GameContextConfig:
    """Configuration for game context feature computation."""

    windows: Sequence[int] = DEFAULT_TEAM_FEATURE_WINDOWS
    min_team_tries_for_share: int = 1  # Min team tries to compute valid share
    fillna_value: float | None = 0.0


def fetch_player_team_context(
    connection: Connection, year: int | str
) -> pd.DataFrame:
    """Fetch player stats joined to team and opponent context.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to query.

    Returns
    -------
    pandas.DataFrame
        Player-match rows with team tries, opponent info, and home/away status.
    """
    normalized_year = db.normalize_year(year)
    player_stats = db.get_table("player_stats", normalized_year)
    team_stats = db.get_table("team_stats", normalized_year)
    matches = db.get_table("matches", normalized_year)

    query = f"""
        WITH team_tries AS (
            SELECT match_id, squad_id, tries AS team_tries
            FROM {team_stats}
        )
        SELECT
            ps.match_id,
            ps.player_id,
            ps.squad_id,
            ps.tries AS player_tries,
            ps.jumper_number,
            m.round_number,
            m.home_squad_id,
            m.away_squad_id,
            CASE
                WHEN ps.squad_id = m.home_squad_id THEN m.away_squad_id
                WHEN ps.squad_id = m.away_squad_id THEN m.home_squad_id
                ELSE NULL
            END AS opponent_squad_id,
            CASE
                WHEN ps.squad_id = m.home_squad_id THEN 1
                ELSE 0
            END AS is_home,
            tt.team_tries
        FROM {player_stats} ps
        JOIN {matches} m ON ps.match_id = m.match_id
        LEFT JOIN team_tries tt ON ps.match_id = tt.match_id
            AND ps.squad_id = tt.squad_id
        ORDER BY ps.player_id, m.round_number, ps.match_id
    """
    return db.fetch_df(connection, query)


def _rolling_feature(series: pd.Series, window: int, *, min_periods: int = 1) -> pd.Series:
    """Compute a shifted rolling mean for a series."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def _ewm_feature(series: pd.Series, span: int) -> pd.Series:
    """Compute a shifted exponential weighted moving average."""
    return series.shift(1).ewm(span=span, min_periods=1).mean()


def compute_team_attack_strength(
    connection: Connection,
    year: int | str,
    *,
    windows: Sequence[int] = DEFAULT_TEAM_FEATURE_WINDOWS,
) -> pd.DataFrame:
    """Compute rolling team attack strength (tries scored per match).

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year.
    windows : Sequence[int]
        Rolling window sizes.

    Returns
    -------
    pandas.DataFrame
        Team attack features keyed by (match_id, squad_id).
    """
    normalized_year = db.normalize_year(year)
    team_stats = db.get_table("team_stats", normalized_year)
    matches = db.get_table("matches", normalized_year)

    query = f"""
        SELECT
            ts.match_id,
            ts.squad_id,
            m.round_number,
            ts.tries,
            ts.score
        FROM {team_stats} ts
        JOIN {matches} m ON ts.match_id = m.match_id
        ORDER BY ts.squad_id, m.round_number, ts.match_id
    """
    df = db.fetch_df(connection, query)

    grouped = df.groupby("squad_id", group_keys=False)
    features = df[["match_id", "squad_id", "round_number"]].copy()

    for window in windows:
        features[f"team_attack_tries_{window}"] = grouped["tries"].transform(
            lambda s: _rolling_feature(s, window)
        )
        features[f"team_attack_score_{window}"] = grouped["score"].transform(
            lambda s: _rolling_feature(s, window)
        )

    # EWM team attack features (captures form faster than rolling average)
    features[f"ewm_team_attack_tries_{EWM_SPAN}"] = grouped["tries"].transform(
        lambda s: _ewm_feature(s, EWM_SPAN)
    )

    # League average tries (expanding, for Poisson normalization)
    features["_league_avg_tries"] = (
        df["tries"].shift(1).expanding(min_periods=1).mean()
    )

    return features


def compute_team_defence_weakness(
    connection: Connection,
    year: int | str,
    *,
    windows: Sequence[int] = DEFAULT_TEAM_FEATURE_WINDOWS,
) -> pd.DataFrame:
    """Compute rolling team defence weakness (tries conceded per match).

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year.
    windows : Sequence[int]
        Rolling window sizes.

    Returns
    -------
    pandas.DataFrame
        Team defence features keyed by (match_id, squad_id).
    """
    normalized_year = db.normalize_year(year)
    team_stats = db.get_table("team_stats", normalized_year)
    matches = db.get_table("matches", normalized_year)

    # Get tries conceded by joining opponent stats
    query = f"""
        WITH team_match AS (
            SELECT
                ts.match_id,
                ts.squad_id,
                m.round_number,
                ts.tries,
                ts.score
            FROM {team_stats} ts
            JOIN {matches} m ON ts.match_id = m.match_id
        )
        SELECT
            tm.match_id,
            tm.squad_id,
            tm.round_number,
            opp.tries AS tries_conceded,
            opp.score AS score_conceded
        FROM team_match tm
        LEFT JOIN team_match opp ON tm.match_id = opp.match_id
            AND tm.squad_id != opp.squad_id
        ORDER BY tm.squad_id, tm.round_number, tm.match_id
    """
    df = db.fetch_df(connection, query)

    grouped = df.groupby("squad_id", group_keys=False)
    features = df[["match_id", "squad_id", "round_number"]].copy()

    for window in windows:
        features[f"team_defence_tries_conceded_{window}"] = grouped["tries_conceded"].transform(
            lambda s: _rolling_feature(s, window)
        )
        features[f"team_defence_score_conceded_{window}"] = grouped["score_conceded"].transform(
            lambda s: _rolling_feature(s, window)
        )

    # EWM opponent defence features
    features[f"ewm_opponent_tries_conceded_{EWM_SPAN}"] = grouped["tries_conceded"].transform(
        lambda s: _ewm_feature(s, EWM_SPAN)
    )

    return features


def compute_player_try_share(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_TEAM_FEATURE_WINDOWS,
    min_team_tries: int = 1,
) -> pd.DataFrame:
    """Compute player's rolling share of team tries.

    Parameters
    ----------
    df : pandas.DataFrame
        Player-match data with player_tries and team_tries columns.
    windows : Sequence[int]
        Rolling window sizes.
    min_team_tries : int
        Minimum team tries to compute a valid share (prevents division by zero).

    Returns
    -------
    pandas.DataFrame
        Original df with added try_share features.
    """
    result = df.copy()
    grouped = result.groupby("player_id", group_keys=False)

    for window in windows:
        # Compute rolling player tries and team tries
        rolling_player_tries = grouped["player_tries"].transform(
            lambda s: _rolling_feature(s, window)
        )
        rolling_team_tries = grouped["team_tries"].transform(
            lambda s: _rolling_feature(s, window)
        )

        # Compute share, handling division by zero
        try_share = np.where(
            rolling_team_tries >= min_team_tries,
            rolling_player_tries / rolling_team_tries,
            np.nan,
        )
        result[f"player_try_share_{window}"] = try_share

    return result


def compute_game_context_features(
    connection: Connection,
    year: int | str,
    *,
    config: GameContextConfig | None = None,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute game context features: expected team tries, player try share, opponent context.

    This computes the highest-priority Tier 1 features for ATS prediction:
    1. expected_team_tries = team_attack × opponent_defence_weakness × home_advantage
    2. player_try_share = rolling (player_tries / team_tries)
    3. Opponent defensive context = opponent's rolling tries_conceded

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year to compute features for.
    config : GameContextConfig | None
        Feature configuration settings.
    as_of_round : int | None
        If provided, only return features for this round while using prior rounds
        for history. Useful for weekly incremental updates.

    Returns
    -------
    pandas.DataFrame
        Feature table keyed by match_id and player_id.
    """
    feature_config = config or GameContextConfig()

    # Fetch base player-team context
    LOGGER.info("Fetching player-team context for year=%s", year)
    player_team = fetch_player_team_context(connection, year)

    if player_team.empty:
        raise ValueError("No player-team context found for the requested season.")

    # Note: Do NOT filter player_team here. The shift(1) in rolling features
    # ensures we only use prior data. We filter to as_of_round at the end.

    # Compute team attack strength
    LOGGER.info("Computing team attack strength")
    team_attack = compute_team_attack_strength(
        connection, year, windows=feature_config.windows
    )

    # Compute team defence weakness
    LOGGER.info("Computing team defence weakness")
    team_defence = compute_team_defence_weakness(
        connection, year, windows=feature_config.windows
    )

    # Join team attack to player rows
    features = player_team.merge(
        team_attack,
        on=["match_id", "squad_id"],
        how="left",
        suffixes=("", "_team"),
    )

    # Join opponent defence to player rows (opponent_squad_id → squad_id in team_defence)
    opponent_defence = team_defence.rename(
        columns={
            "squad_id": "opponent_squad_id",
            **{
                f"team_defence_tries_conceded_{w}": f"opponent_tries_conceded_{w}"
                for w in feature_config.windows
            },
            **{
                f"team_defence_score_conceded_{w}": f"opponent_score_conceded_{w}"
                for w in feature_config.windows
            },
        }
    )

    features = features.merge(
        opponent_defence,
        on=["match_id", "opponent_squad_id"],
        how="left",
        suffixes=("", "_opp"),
    )

    # Compute expected_team_tries for each window (geometric mean approach)
    for window in feature_config.windows:
        team_attack_col = f"team_attack_tries_{window}"
        opp_defence_col = f"opponent_tries_conceded_{window}"

        # expected_team_tries = team_attack × opponent_weakness × home_advantage
        # Use geometric mean of attack and opponent weakness, then apply home multiplier
        features[f"expected_team_tries_{window}"] = (
            np.sqrt(
                features[team_attack_col].fillna(0) * features[opp_defence_col].fillna(0)
            )
            * np.where(features["is_home"] == 1, HOME_ADVANTAGE_MULTIPLIER, 1.0)
        )

    # Poisson expected tries (properly normalized by league average)
    # Formula: league_avg * (team_off / league_avg) * (opp_def / league_avg) * home_mult
    # This handles scoring environment changes (e.g., 2026 six-man bench)
    for window in feature_config.windows:
        team_attack_col = f"team_attack_tries_{window}"
        opp_defence_col = f"opponent_tries_conceded_{window}"

        league_avg = features.get("_league_avg_tries", pd.Series(np.nan, index=features.index))
        team_att = features[team_attack_col]
        opp_def = features[opp_defence_col]

        # Avoid division by zero: use NaN where league_avg is 0 or NaN
        safe_league_avg = league_avg.replace(0, np.nan)

        features[f"poisson_expected_team_tries_{window}"] = np.where(
            safe_league_avg.notna() & team_att.notna() & opp_def.notna(),
            safe_league_avg * (team_att / safe_league_avg) * (opp_def / safe_league_avg)
            * np.where(features["is_home"] == 1, HOME_ADVANTAGE_MULTIPLIER, 1.0),
            np.nan,
        )

    # Scoring environment ratio: current season avg tries / historical avg
    # Naturally captures rule changes (e.g., 2026 six-man bench → +7.5% tries)
    if "_league_avg_tries" in features.columns:
        # Historical baseline: ~4.0 tries per team per match (2024-2025 average)
        _HISTORICAL_AVG_TRIES = 4.0
        features["scoring_environment_ratio"] = np.where(
            features["_league_avg_tries"].notna() & (features["_league_avg_tries"] > 0),
            features["_league_avg_tries"] / _HISTORICAL_AVG_TRIES,
            1.0,
        )

    # Compute player try share
    LOGGER.info("Computing player try share")
    features = compute_player_try_share(
        features,
        windows=feature_config.windows,
        min_team_tries=feature_config.min_team_tries_for_share,
    )

    # Select final feature columns
    feature_columns = [
        "match_id",
        "player_id",
        "squad_id",
        "opponent_squad_id",
        "round_number",
        "is_home",
    ]

    # Add all computed features
    for window in feature_config.windows:
        feature_columns.extend([
            f"expected_team_tries_{window}",
            f"poisson_expected_team_tries_{window}",
            f"player_try_share_{window}",
            f"team_attack_tries_{window}",
            f"team_attack_score_{window}",
            f"opponent_tries_conceded_{window}",
            f"opponent_score_conceded_{window}",
        ])

    # Add EWM context features
    feature_columns.extend([
        f"ewm_team_attack_tries_{EWM_SPAN}",
        f"ewm_opponent_tries_conceded_{EWM_SPAN}",
    ])

    # Add scoring environment ratio
    if "scoring_environment_ratio" in features.columns:
        feature_columns.append("scoring_environment_ratio")

    # Only keep columns that actually exist
    feature_columns = [c for c in feature_columns if c in features.columns]

    features = features[feature_columns].copy()

    # Fill NaN values
    if feature_config.fillna_value is not None:
        features = features.fillna(feature_config.fillna_value)

    # Filter to specific round if requested
    if as_of_round is not None:
        features = features[features["round_number"] == as_of_round].copy()

    if features.empty:
        if as_of_round is not None:
            available_rounds = sorted(player_team["round_number"].unique())
            raise ValueError(
                f"No feature rows generated for round {as_of_round}. "
                f"Available rounds: {available_rounds}"
            )
        else:
            raise ValueError("No feature rows generated for the requested season.")

    LOGGER.info(
        "Computed game context features for %s rows (season=%s, round=%s)",
        len(features),
        year,
        as_of_round,
    )
    return features.reset_index(drop=True)


def compute_venue_features(
    connection: Connection,
    year: int | str,
    *,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute venue-level try-scoring features.

    Uses 2024-2025 historical averages. For the current season,
    computes venue stats from prior rounds only (no leakage).

    Features:
    - ``venue_avg_tries_per_match`` — average total tries per match at this venue.

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year.
    as_of_round : int | None
        If given, only return rows for this round.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, venue_name, venue_avg_tries_per_match.
    """
    from src.db import table_exists

    # Gather historical venue tries from prior seasons
    historical_dfs = []
    normalized_year = db.normalize_year(year)
    for prior_year in [2024, 2025]:
        if prior_year >= normalized_year:
            continue
        ts_table = db.get_table("team_stats", prior_year)
        m_table = db.get_table("matches", prior_year)
        if not table_exists(connection, ts_table) or not table_exists(connection, m_table):
            continue
        df = db.fetch_df(connection, f"""
            SELECT m.venue_name, SUM(ts.tries) AS total_tries
            FROM {ts_table} ts
            JOIN {m_table} m ON ts.match_id = m.match_id
            GROUP BY m.match_id, m.venue_name
        """)
        if not df.empty:
            historical_dfs.append(df)

    # Also include current season data from prior rounds
    ts_table = db.get_table("team_stats", normalized_year)
    m_table = db.get_table("matches", normalized_year)
    if table_exists(connection, ts_table) and table_exists(connection, m_table):
        round_filter = f"AND m.round_number < {as_of_round}" if as_of_round else ""
        df = db.fetch_df(connection, f"""
            SELECT m.venue_name, SUM(ts.tries) AS total_tries
            FROM {ts_table} ts
            JOIN {m_table} m ON ts.match_id = m.match_id
            WHERE 1=1 {round_filter}
            GROUP BY m.match_id, m.venue_name
        """)
        if not df.empty:
            historical_dfs.append(df)

    if not historical_dfs:
        LOGGER.info("No historical venue data available for season %s", year)
        return pd.DataFrame(columns=["match_id", "venue_avg_tries_per_match"])

    all_venue = pd.concat(historical_dfs, ignore_index=True)
    venue_avg = (
        all_venue.groupby("venue_name")["total_tries"]
        .mean()
        .reset_index()
        .rename(columns={"total_tries": "venue_avg_tries_per_match"})
    )

    # Get current season matches with venues
    round_filter = f"AND round_number = {as_of_round}" if as_of_round else ""
    matches = db.fetch_df(connection, f"""
        SELECT match_id, venue_name
        FROM {m_table}
        WHERE 1=1 {round_filter}
    """)

    if matches.empty:
        return pd.DataFrame(columns=["match_id", "venue_avg_tries_per_match"])

    result = matches.merge(venue_avg, on="venue_name", how="left")

    # Fill unknown venues with global average
    global_avg = all_venue["total_tries"].mean()
    result["venue_avg_tries_per_match"] = result["venue_avg_tries_per_match"].fillna(global_avg)

    LOGGER.info(
        "Computed venue features for %d matches (season=%s), avg=%.2f tries/match",
        len(result), year, result["venue_avg_tries_per_match"].mean(),
    )
    return result[["match_id", "venue_avg_tries_per_match"]].reset_index(drop=True)


def compute_schedule_features(
    connection: Connection,
    year: int | str,
    *,
    as_of_round: int | None = None,
) -> pd.DataFrame:
    """Compute schedule and fatigue features per team-match.

    Features:
    - ``days_since_last_match`` — short turnaround vs bye-week rest.
    - ``matches_in_last_14_days`` — workload / congestion proxy.

    These use only public scheduling data (no leakage risk).

    Parameters
    ----------
    connection : sqlite3.Connection
        Database connection.
    year : int | str
        Season year.
    as_of_round : int | None
        If given, return only rows for this round.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, squad_id, days_since_last_match,
        matches_in_last_14_days.
    """
    normalized_year = db.normalize_year(year)
    matches = db.get_table("matches", normalized_year)
    team_stats = db.get_table("team_stats", normalized_year)

    query = f"""
        SELECT ts.match_id, ts.squad_id, m.round_number, m.utc_start_time
        FROM {team_stats} ts
        JOIN {matches} m ON ts.match_id = m.match_id
        ORDER BY ts.squad_id, m.utc_start_time
    """
    df = db.fetch_df(connection, query)
    if df.empty:
        return pd.DataFrame(columns=["match_id", "squad_id", "days_since_last_match", "matches_in_last_14_days"])

    df["utc_start_time"] = pd.to_datetime(df["utc_start_time"], utc=True)
    df.sort_values(["squad_id", "utc_start_time"], inplace=True)

    # Days since last match (per team)
    df["prev_match_time"] = df.groupby("squad_id")["utc_start_time"].shift(1)
    df["days_since_last_match"] = (
        (df["utc_start_time"] - df["prev_match_time"]).dt.total_seconds() / 86400
    )

    # Matches in last 14 days (per team, excluding current match)
    def _matches_in_window(group: pd.DataFrame) -> pd.Series:
        counts = []
        times = group["utc_start_time"].values
        for i, t in enumerate(times):
            window_start = t - np.timedelta64(14, "D")
            # Count prior matches within the window (exclude current)
            count = int(((times[:i] >= window_start) & (times[:i] < t)).sum())
            counts.append(count)
        return pd.Series(counts, index=group.index)

    df["matches_in_last_14_days"] = df.groupby("squad_id", group_keys=False).apply(
        _matches_in_window, include_groups=False
    )

    result = df[["match_id", "squad_id", "round_number", "days_since_last_match", "matches_in_last_14_days"]].copy()

    if as_of_round is not None:
        result = result[result["round_number"] == as_of_round]

    result = result.drop(columns=["round_number"])

    LOGGER.info("Computed schedule features for %d team-match rows (season=%s)", len(result), year)
    return result.reset_index(drop=True)
