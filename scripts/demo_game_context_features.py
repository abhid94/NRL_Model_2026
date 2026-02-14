"""Demo script for game context features.

This demonstrates how to use the new Tier 1 game context features:
- expected_team_tries
- player_try_share
- opponent defensive context
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src import db
from src.config import DB_PATH
from src.features.game_context_features import compute_game_context_features
from src.features.player_features import compute_player_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def main():
    """Demonstrate game context features."""
    LOGGER.info("Connecting to database: %s", DB_PATH)
    conn = db.get_connection(DB_PATH)

    try:
        # Compute game context features for 2024 season
        LOGGER.info("Computing game context features for 2024...")
        game_context = compute_game_context_features(conn, 2024)

        LOGGER.info("Game context features shape: %s", game_context.shape)
        LOGGER.info("Columns: %s", list(game_context.columns))

        # Show summary statistics
        LOGGER.info("\n=== Expected Team Tries (5-match window) ===")
        LOGGER.info(game_context["expected_team_tries_5"].describe())

        LOGGER.info("\n=== Player Try Share (5-match window) ===")
        LOGGER.info(game_context["player_try_share_5"].describe())

        LOGGER.info("\n=== Opponent Tries Conceded (5-match window) ===")
        LOGGER.info(game_context["opponent_tries_conceded_5"].describe())

        # Show top players by expected team tries (indicator of strong attacking context)
        LOGGER.info("\n=== Top 10 Player-Match Observations by Expected Team Tries ===")
        top_expected = game_context.nlargest(10, "expected_team_tries_5")[
            [
                "match_id",
                "player_id",
                "expected_team_tries_5",
                "team_attack_tries_5",
                "opponent_tries_conceded_5",
                "is_home",
            ]
        ]
        LOGGER.info("\n%s", top_expected)

        # Compute player features (now with position fields)
        LOGGER.info("\nComputing player features with position fields...")
        player_features = compute_player_features(conn, 2024)

        LOGGER.info("Player features shape: %s", player_features.shape)

        # Show position distribution
        LOGGER.info("\n=== Position Group Distribution ===")
        LOGGER.info(player_features["position_group"].value_counts())

        LOGGER.info("\n=== Starter vs Bench Distribution ===")
        LOGGER.info(player_features["is_starter"].value_counts())

        # Join game context and player features
        LOGGER.info("\nMerging game context and player features...")
        merged = game_context.merge(
            player_features[
                [
                    "match_id",
                    "player_id",
                    "position_group",
                    "position_code",
                    "is_starter",
                    "rolling_tries_5",
                    "rolling_try_rate_5",
                ]
            ],
            on=["match_id", "player_id"],
            how="left",
        )

        LOGGER.info("Merged features shape: %s", merged.shape)

        # Analyze expected tries by position group
        LOGGER.info("\n=== Expected Team Tries by Position Group ===")
        position_analysis = (
            merged.groupby("position_group")["expected_team_tries_5"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )
        LOGGER.info("\n%s", position_analysis)

        # Analyze player try share by position group
        LOGGER.info("\n=== Player Try Share by Position Group ===")
        share_analysis = (
            merged.groupby("position_group")["player_try_share_5"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )
        LOGGER.info("\n%s", share_analysis)

        # Show correlation between key features
        LOGGER.info("\n=== Feature Correlations (5-match window) ===")
        corr_features = [
            "expected_team_tries_5",
            "player_try_share_5",
            "team_attack_tries_5",
            "opponent_tries_conceded_5",
            "rolling_tries_5",
            "rolling_try_rate_5",
        ]
        correlations = merged[corr_features].corr()["rolling_try_rate_5"].sort_values(
            ascending=False
        )
        LOGGER.info("\n%s", correlations)

        LOGGER.info("\nDemo complete!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
