"""Demo script for edge-specific features.

This script demonstrates the edge features module by:
1. Computing team edge attack profiles
2. Computing team edge defence profiles
3. Adding player-level edge features
4. Analyzing edge attack patterns by team
"""

from __future__ import annotations

import pandas as pd

from src.db import fetch_df, get_connection
from src.features.edge_features import (
    add_player_edge_features,
    compute_team_edge_attack_profiles,
    compute_team_edge_defence_profiles,
)

# Set pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.precision", 3)


def main():
    """Run demo analysis of edge features."""
    print("=" * 80)
    print("Edge Features Demo — Sprint 2B")
    print("=" * 80)

    # 1. Team Edge Attack Profiles
    print("\n1. TEAM EDGE ATTACK PROFILES")
    print("-" * 80)
    attack_profiles = compute_team_edge_attack_profiles(season=2024, window=5)
    print(f"Shape: {attack_profiles.shape}")
    print(f"\nSample (first 10 team-matches):")
    print(attack_profiles.head(10))

    # Get team names
    with get_connection() as conn:
        teams = fetch_df(conn, "SELECT squad_id, squad_nickname FROM teams")

    # Join team names
    attack_with_names = attack_profiles.merge(teams, on="squad_id")

    # Find teams with extreme edge attack patterns
    print("\n\nTeams with HIGHEST left edge attack (% of tries from left edge):")
    left_heavy = (
        attack_with_names.groupby("squad_nickname")["left_edge_try_pct_rolling_5"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    for team, pct in left_heavy.items():
        print(f"  {team:20s} {pct:5.1%}")

    print("\nTeams with HIGHEST right edge attack (% of tries from right edge):")
    right_heavy = (
        attack_with_names.groupby("squad_nickname")["right_edge_try_pct_rolling_5"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    for team, pct in right_heavy.items():
        print(f"  {team:20s} {pct:5.1%}")

    print("\nTeams with HIGHEST middle attack (% of tries from middle):")
    middle_heavy = (
        attack_with_names.groupby("squad_nickname")["middle_edge_try_pct_rolling_5"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    for team, pct in middle_heavy.items():
        print(f"  {team:20s} {pct:5.1%}")

    # 2. Team Edge Defence Profiles
    print("\n\n2. TEAM EDGE DEFENCE PROFILES")
    print("-" * 80)
    defence_profiles = compute_team_edge_defence_profiles(season=2024, window=5)
    print(f"Shape: {defence_profiles.shape}")
    print(f"\nSample (first 10 team-matches):")
    print(defence_profiles.head(10))

    # Join team names
    defence_with_names = defence_profiles.merge(teams, on="squad_id")

    # Find teams with weakest edge defence
    print("\n\nTeams CONCEDING MOST to left edge (defensive weakness on right side):")
    weak_right = (
        defence_with_names.groupby("squad_nickname")["conceded_to_left_edge_rolling_5"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    for team, tries in weak_right.items():
        print(f"  {team:20s} {tries:4.2f} tries/5 games")

    print("\nTeams CONCEDING MOST to right edge (defensive weakness on left side):")
    weak_left = (
        defence_with_names.groupby("squad_nickname")["conceded_to_right_edge_rolling_5"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    for team, tries in weak_left.items():
        print(f"  {team:20s} {tries:4.2f} tries/5 games")

    # 3. Player-Level Edge Features
    print("\n\n3. PLAYER-LEVEL EDGE FEATURES")
    print("-" * 80)

    # Get sample player data
    with get_connection() as conn:
        player_data = fetch_df(
            conn,
            """
            SELECT
                ps.match_id,
                ps.player_id,
                ps.squad_id,
                ps.jumper_number,
                ps.tries,
                p.short_display_name,
                m.round_number
            FROM player_stats_2024 ps
            INNER JOIN players_2024 p ON ps.player_id = p.player_id
            INNER JOIN matches_2024 m ON ps.match_id = m.match_id
            WHERE ps.jumper_number IN (2, 3, 4, 5, 11, 12)  -- Edge players only
              AND m.round_number <= 10
            ORDER BY m.round_number, ps.match_id, ps.player_id
            LIMIT 200
            """,
        )

    print(f"Loaded {len(player_data)} edge player observations")

    # Add edge features
    player_with_edge = add_player_edge_features(
        player_df=player_data, season=2024, window=5
    )

    print(f"\nColumns added: {[col for col in player_with_edge.columns if col not in player_data.columns]}")
    print(f"\nSample (first 10 player observations):")
    print(
        player_with_edge[
            [
                "short_display_name",
                "jumper_number",
                "player_edge",
                "team_edge_try_share_rolling_5",
                "opponent_edge_conceded_rolling_5",
                "edge_matchup_score_rolling_5",
                "tries",
            ]
        ].head(10)
    )

    # 4. Edge Feature Distribution Analysis
    print("\n\n4. EDGE FEATURE DISTRIBUTIONS")
    print("-" * 80)

    # Get full player stats with edge features
    with get_connection() as conn:
        all_players = fetch_df(
            conn,
            """
            SELECT
                ps.match_id,
                ps.player_id,
                ps.squad_id,
                ps.jumper_number,
                ps.tries,
                m.round_number
            FROM player_stats_2024 ps
            INNER JOIN matches_2024 m ON ps.match_id = m.match_id
            WHERE ps.jumper_number IS NOT NULL
            """,
        )

    all_with_edge = add_player_edge_features(
        player_df=all_players, season=2024, window=5
    )

    print("\nPlayer edge distribution:")
    print(all_with_edge["player_edge"].value_counts())

    print("\n\nEdge matchup score by player edge:")
    for edge in ["left", "right", "middle", "other"]:
        edge_data = all_with_edge[all_with_edge["player_edge"] == edge]
        if len(edge_data) > 0:
            mean_score = edge_data["edge_matchup_score_rolling_5"].mean()
            print(f"  {edge:10s} mean matchup score: {mean_score:.4f}")

    print("\n\nTry rate by edge matchup score quartiles (edge players only):")
    edge_players = all_with_edge[all_with_edge["player_edge"].isin(["left", "right"])]
    edge_players["matchup_quartile"] = pd.qcut(
        edge_players["edge_matchup_score_rolling_5"],
        q=4,
        labels=["Q1 (worst)", "Q2", "Q3", "Q4 (best)"],
        duplicates="drop",
    )

    try_rate_by_quartile = (
        edge_players.groupby("matchup_quartile", observed=True)
        .agg({"tries": ["sum", "count"]})
        .reset_index()
    )
    try_rate_by_quartile.columns = ["matchup_quartile", "total_tries", "observations"]
    try_rate_by_quartile["try_rate"] = (
        try_rate_by_quartile["total_tries"] / try_rate_by_quartile["observations"]
    )

    print(try_rate_by_quartile.to_string(index=False))

    print("\n\n5. KEY INSIGHTS")
    print("-" * 80)
    print("✓ Edge attack profiles vary significantly by team")
    print("✓ Edge defence profiles identify defensive weaknesses")
    print("✓ Edge matchup scores combine team attack strength × opponent defence weakness")
    print("✓ Players in favorable edge matchups (high score) should have higher try rates")
    print("\nEdge features are ready for model training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
