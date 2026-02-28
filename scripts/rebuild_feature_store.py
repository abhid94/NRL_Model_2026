"""Rebuild feature stores for 2024 and 2025 with new features.

Adds: odds momentum features, discipline features.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import get_connection
from src.features.feature_store import build_multi_season_feature_store
from src.config import FEATURE_STORE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    conn = get_connection()
    logger.info("Rebuilding feature stores with expanded features...")

    result = build_multi_season_feature_store(
        conn=conn,
        seasons=[2024, 2025],
        output_dir=str(FEATURE_STORE_DIR),
        save_combined=True,
    )

    for season, df in result.items():
        logger.info(
            "Season %d: %d rows x %d cols",
            season, len(df), len(df.columns),
        )

    conn.close()
    logger.info("Feature store rebuild complete!")


if __name__ == "__main__":
    main()
