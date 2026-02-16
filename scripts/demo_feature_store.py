"""Demo script for feature store consolidation.

This demonstrates:
- Building a consolidated feature store for 2024 season
- Saving/loading to Parquet format
- Multi-season feature stores
- Train/val splitting
- Feature metadata
"""

import logging
import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.db import get_connection
from src.features.feature_store import (
    build_feature_store,
    save_feature_store,
    load_feature_store,
    build_multi_season_feature_store,
    get_feature_metadata,
    get_train_val_split
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate feature store consolidation."""
    logger.info("=" * 80)
    logger.info("Feature Store Demo — Sprint 2D")
    logger.info("=" * 80)

    conn = get_connection()

    # 1. Build feature store for 2024
    logger.info("\n1. BUILDING FEATURE STORE FOR 2024")
    logger.info("-" * 80)
    df_2024 = build_feature_store(conn, 2024)

    logger.info(f"\nFeature store shape: {df_2024.shape}")
    logger.info(f"Columns: {len(df_2024.columns)}")
    logger.info(f"\nFirst 5 column names: {list(df_2024.columns[:5])}")
    logger.info(f"Last 5 column names: {list(df_2024.columns[-5:])}")

    # 2. Check data quality
    logger.info("\n\n2. DATA QUALITY CHECKS")
    logger.info("-" * 80)

    # No duplicates
    duplicates = df_2024.duplicated(subset=['match_id', 'player_id']).sum()
    logger.info(f"Duplicate (match_id, player_id) pairs: {duplicates}")

    # Target variable distribution
    try_rate = df_2024['scored_try'].mean()
    logger.info(f"Try rate (positive class %): {try_rate:.1%}")

    # Missing values by feature group
    missing_pct = (df_2024.isna().sum() / len(df_2024) * 100).sort_values(ascending=False)
    logger.info(f"\nTop 10 features with missing values:")
    for feat, pct in missing_pct.head(10).items():
        logger.info(f"  {feat:40s} {pct:5.1f}%")

    # 3. Feature metadata
    logger.info("\n\n3. FEATURE METADATA")
    logger.info("-" * 80)
    metadata = get_feature_metadata()

    logger.info(f"Total documented features: {len(metadata)}")
    logger.info(f"\nFeatures by module:")
    for module, count in metadata['module'].value_counts().items():
        logger.info(f"  {module:20s} {count:3d} features")

    logger.info(f"\nFeatures by type:")
    for ftype, count in metadata['feature_type'].value_counts().items():
        logger.info(f"  {ftype:20s} {count:3d}")

    # 4. Save/load test
    logger.info("\n\n4. SAVE/LOAD TEST")
    logger.info("-" * 80)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "feature_store_2024.parquet"
        save_feature_store(df_2024, str(output_path))

        df_loaded = load_feature_store(str(output_path))

        logger.info(f"Original shape: {df_2024.shape}")
        logger.info(f"Loaded shape: {df_loaded.shape}")
        logger.info(f"Shapes match: {df_2024.shape == df_loaded.shape}")

    # 5. Sample observations
    logger.info("\n\n5. SAMPLE OBSERVATIONS")
    logger.info("-" * 80)

    # Show a player who scored a try
    try_scorers = df_2024[df_2024['scored_try'] == 1].head(3)
    logger.info(f"\nSample players who scored tries:")
    display_cols = [
        'match_id', 'player_id', 'position_group', 'rolling_tries_3',
        'expected_team_tries_5', 'player_try_share_5', 'betfair_implied_prob',
        'scored_try'
    ]
    logger.info(f"\n{try_scorers[display_cols].to_string()}")

    # 6. Feature correlations with target
    logger.info("\n\n6. FEATURE CORRELATIONS WITH TARGET")
    logger.info("-" * 80)

    # Select numeric features
    numeric_cols = df_2024.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['match_id', 'player_id', 'scored_try', 'tries']]

    correlations = df_2024[numeric_cols + ['scored_try']].corr()['scored_try'].drop('scored_try')
    top_positive = correlations.nlargest(10)
    top_negative = correlations.nsmallest(10)

    logger.info("\nTop 10 positive correlations with scoring a try:")
    for feat, corr in top_positive.items():
        logger.info(f"  {feat:50s} {corr:+.3f}")

    logger.info("\nTop 10 negative correlations:")
    for feat, corr in top_negative.items():
        logger.info(f"  {feat:50s} {corr:+.3f}")

    # 7. Multi-season feature store
    logger.info("\n\n7. MULTI-SEASON FEATURE STORE")
    logger.info("-" * 80)
    with tempfile.TemporaryDirectory() as tmpdir:
        seasons = [2024, 2025]
        season_dfs = build_multi_season_feature_store(
            conn, seasons, tmpdir, save_combined=True
        )

        logger.info(f"Built feature stores for: {list(season_dfs.keys())}")
        for season, df in season_dfs.items():
            logger.info(f"  {season}: {len(df)} rows")

        # Load combined file
        combined_path = Path(tmpdir) / "feature_store_combined.parquet"
        df_combined = load_feature_store(str(combined_path))
        logger.info(f"\nCombined feature store: {len(df_combined)} rows")

        # 8. Train/val split
        logger.info("\n\n8. TRAIN/VAL SPLIT (Temporal)")
        logger.info("-" * 80)
        train_df, val_df = get_train_val_split(df_combined, [2024], [2025])

        logger.info(f"Train set: {len(train_df)} rows (2024 season)")
        logger.info(f"Val set: {len(val_df)} rows (2025 season)")
        logger.info(f"Train try rate: {train_df['scored_try'].mean():.1%}")
        logger.info(f"Val try rate: {val_df['scored_try'].mean():.1%}")

        # Check no overlap
        train_keys = set(zip(train_df['match_id'], train_df['player_id']))
        val_keys = set(zip(val_df['match_id'], val_df['player_id']))
        overlap = train_keys & val_keys
        logger.info(f"Overlapping observations: {len(overlap)}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Feature store demo complete!")
    logger.info("=" * 80)

    conn.close()


if __name__ == "__main__":
    main()
