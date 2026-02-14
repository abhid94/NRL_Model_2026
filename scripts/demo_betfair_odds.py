"""
Demo Script: Betfair Odds Features

Showcases:
1. Price fallback chain effectiveness
2. Coverage statistics by round
3. Odds vs actual try rate calibration
4. Market efficiency analysis (spread, volume)
5. Comparison of odds sources (last_preplay vs 1min vs 30min)

Run: python3 scripts/demo_betfair_odds.py
"""

import sys
sys.path.append('/Users/abhidutta/Documents/repos/NRL_2026_Model')

import pandas as pd
import numpy as np
from src.db import get_connection
from src.odds.betfair import add_betfair_odds_features, validate_betfair_odds_features

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)


def main():
    print("=" * 80)
    print("BETFAIR ODDS FEATURES DEMO")
    print("=" * 80)

    conn = get_connection()

    # ===== Section 1: Load player observations =====
    print("\n" + "=" * 80)
    print("1. LOADING PLAYER OBSERVATIONS (2024 Season)")
    print("=" * 80)

    query = """
    SELECT
        ps.match_id,
        ps.player_id,
        m.round_number,
        ps.jumper_number,
        ps.position,
        ps.tries,
        ps.side,
        2024 as season
    FROM player_stats_2024 ps
    JOIN matches_2024 m ON ps.match_id = m.match_id
    ORDER BY m.round_number, ps.match_id, ps.jumper_number
    """

    obs = pd.read_sql_query(query, conn)
    print(f"\nTotal player-match observations: {len(obs):,}")
    print(f"Try rate: {(obs['tries'] > 0).mean():.1%}")

    # ===== Section 2: Extract odds features =====
    print("\n" + "=" * 80)
    print("2. EXTRACTING BETFAIR TO_SCORE ODDS")
    print("=" * 80)

    obs_with_odds = add_betfair_odds_features(obs, conn, year=2024)

    # Validate
    validate_betfair_odds_features(obs_with_odds)
    print("✅ Validation passed")

    # ===== Section 3: Coverage analysis =====
    print("\n" + "=" * 80)
    print("3. COVERAGE ANALYSIS")
    print("=" * 80)

    non_null_odds = obs_with_odds['betfair_closing_odds'].notna().sum()
    coverage_pct = (non_null_odds / len(obs_with_odds)) * 100
    print(f"\nOverall coverage: {non_null_odds:,}/{len(obs_with_odds):,} ({coverage_pct:.1f}%)")

    # Coverage by source
    print("\nOdds source distribution:")
    source_dist = obs_with_odds['betfair_odds_source'].value_counts(dropna=False)
    for source, count in source_dist.items():
        pct = (count / len(obs_with_odds)) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    # Fallback chain effectiveness
    fallback_count = source_dist.get('1min', 0) + source_dist.get('30min', 0) + source_dist.get('60min', 0)
    if fallback_count > 0:
        fallback_pct = (fallback_count / non_null_odds) * 100
        print(f"\nFallback chain recovered: {fallback_count:,} records ({fallback_pct:.1f}% of extracted odds)")

    # ===== Section 4: Odds vs actual try rate (calibration) =====
    print("\n" + "=" * 80)
    print("4. ODDS CALIBRATION (Market Accuracy)")
    print("=" * 80)

    # Filter to observations with odds
    obs_w_odds = obs_with_odds[obs_with_odds['betfair_closing_odds'].notna()].copy()

    # Create probability bins
    obs_w_odds['prob_bin'] = pd.cut(
        obs_w_odds['betfair_implied_prob'],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
    )

    calibration = obs_w_odds.groupby('prob_bin', observed=True).agg({
        'betfair_implied_prob': 'mean',
        'tries': lambda x: (x > 0).mean(),
        'match_id': 'count'
    }).rename(columns={'match_id': 'count'})

    print("\nImplied Probability vs Actual Try Rate:")
    print(f"{'Probability Bin':<20} {'Avg Implied Prob':<20} {'Actual Try Rate':<20} {'Count':<10} {'Diff'}")
    print("-" * 90)

    for idx, row in calibration.iterrows():
        diff = row['tries'] - row['betfair_implied_prob']
        diff_str = f"{diff:+.1%}"
        print(f"{idx:<20} {row['betfair_implied_prob']:<20.1%} {row['tries']:<20.1%} {int(row['count']):<10} {diff_str}")

    # Overall calibration error
    calibration_error = abs(obs_w_odds['betfair_implied_prob'] - (obs_w_odds['tries'] > 0).astype(int)).mean()
    print(f"\nMean absolute calibration error: {calibration_error:.3f}")
    print("  (< 0.03 is good, < 0.05 is acceptable)")

    # ===== Section 5: Market efficiency analysis =====
    print("\n" + "=" * 80)
    print("5. MARKET EFFICIENCY ANALYSIS")
    print("=" * 80)

    # Spread analysis
    spread_data = obs_with_odds[obs_with_odds['betfair_spread'].notna()].copy()
    print(f"\nSpread (Back - Lay) statistics (n={len(spread_data):,}):")
    print(f"  Mean spread: {spread_data['betfair_spread'].mean():.3f}")
    print(f"  Median spread: {spread_data['betfair_spread'].median():.3f}")
    print(f"  Min spread: {spread_data['betfair_spread'].min():.3f}")
    print(f"  Max spread: {spread_data['betfair_spread'].max():.3f}")

    # Negative spreads (back < lay) - unusual but can happen
    negative_spreads = spread_data[spread_data['betfair_spread'] < 0]
    if len(negative_spreads) > 0:
        neg_pct = (len(negative_spreads) / len(spread_data)) * 100
        print(f"  ⚠️ Negative spreads: {len(negative_spreads):,} ({neg_pct:.1f}%)")

    # Matched volume analysis
    volume_data = obs_with_odds[obs_with_odds['betfair_total_matched_volume'].notna()].copy()
    print(f"\nMatched volume statistics (n={len(volume_data):,}):")
    print(f"  Mean: ${volume_data['betfair_total_matched_volume'].mean():.2f}")
    print(f"  Median: ${volume_data['betfair_total_matched_volume'].median():.2f}")
    print(f"  Min: ${volume_data['betfair_total_matched_volume'].min():.2f}")
    print(f"  Max: ${volume_data['betfair_total_matched_volume'].max():.2f}")

    # ===== Section 6: Odds by position =====
    print("\n" + "=" * 80)
    print("6. ODDS BY POSITION")
    print("=" * 80)

    # Group by position
    position_analysis = obs_w_odds.groupby('position').agg({
        'betfair_implied_prob': 'mean',
        'betfair_closing_odds': 'mean',
        'tries': lambda x: (x > 0).mean(),
        'match_id': 'count'
    }).rename(columns={'match_id': 'count'})

    # Sort by actual try rate
    position_analysis = position_analysis.sort_values('tries', ascending=False)

    print("\nPosition analysis (sorted by actual try rate):")
    print(f"{'Position':<20} {'Avg Odds':<12} {'Implied Prob':<15} {'Actual Rate':<15} {'Count':<10}")
    print("-" * 80)

    for pos, row in position_analysis.head(15).iterrows():
        print(f"{pos:<20} {row['betfair_closing_odds']:<12.2f} {row['betfair_implied_prob']:<15.1%} {row['tries']:<15.1%} {int(row['count']):<10}")

    # ===== Section 7: Odds source comparison =====
    print("\n" + "=" * 80)
    print("7. ODDS SOURCE COMPARISON")
    print("=" * 80)

    # Compare try rates by odds source
    source_comparison = obs_with_odds[obs_with_odds['betfair_odds_source'].notna()].groupby('betfair_odds_source').agg({
        'betfair_implied_prob': 'mean',
        'tries': lambda x: (x > 0).mean(),
        'match_id': 'count'
    }).rename(columns={'match_id': 'count'})

    print("\nTry rate by odds source:")
    print(f"{'Source':<20} {'Avg Implied Prob':<20} {'Actual Try Rate':<20} {'Count':<10}")
    print("-" * 80)

    for source, row in source_comparison.iterrows():
        print(f"{source:<20} {row['betfair_implied_prob']:<20.1%} {row['tries']:<20.1%} {int(row['count']):<10}")

    # ===== Section 8: Top market movers =====
    print("\n" + "=" * 80)
    print("8. TOP MARKET MOVERS (Highest Matched Volume)")
    print("=" * 80)

    # Get top 10 by matched volume
    top_movers = obs_with_odds.nlargest(10, 'betfair_total_matched_volume')[
        ['player_id', 'match_id', 'round_number', 'position',
         'betfair_closing_odds', 'betfair_implied_prob',
         'betfair_total_matched_volume', 'tries']
    ]

    print("\nTop 10 most traded players (by matched volume):")
    print(f"{'Player ID':<12} {'Round':<8} {'Position':<15} {'Odds':<8} {'Impl Prob':<12} {'Volume':<12} {'Scored?'}")
    print("-" * 90)

    for idx, row in top_movers.iterrows():
        scored = "✅" if row['tries'] > 0 else "❌"
        print(f"{row['player_id']:<12} {row['round_number']:<8} {row['position']:<15} "
              f"{row['betfair_closing_odds']:<8.2f} {row['betfair_implied_prob']:<12.1%} "
              f"${row['betfair_total_matched_volume']:<11.2f} {scored}")

    # ===== Section 9: Edge opportunities =====
    print("\n" + "=" * 80)
    print("9. RETROSPECTIVE EDGE ANALYSIS")
    print("=" * 80)

    # Find biggest overpriced players (market too low, player scored)
    obs_w_odds['scored'] = (obs_w_odds['tries'] > 0).astype(int)
    scorers = obs_w_odds[obs_w_odds['scored'] == 1].copy()
    scorers = scorers[scorers['betfair_implied_prob'] < 0.30]  # Market gave < 30% chance
    top_underpriced = scorers.nlargest(10, 'betfair_closing_odds')[
        ['player_id', 'match_id', 'round_number', 'position',
         'betfair_closing_odds', 'betfair_implied_prob', 'tries']
    ]

    print("\nTop 10 underpriced players (longest odds who scored):")
    print(f"{'Player ID':<12} {'Round':<8} {'Position':<15} {'Odds':<8} {'Implied Prob':<15} {'Tries'}")
    print("-" * 80)

    for idx, row in top_underpriced.iterrows():
        print(f"{row['player_id']:<12} {row['round_number']:<8} {row['position']:<15} "
              f"{row['betfair_closing_odds']:<8.2f} {row['betfair_implied_prob']:<15.1%} {int(row['tries'])}")

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
✅ Successfully extracted Betfair TO_SCORE odds for {coverage_pct:.1f}% of player-match observations

📊 Key Findings:
1. Price fallback chain works: {fallback_pct:.1f}% of odds required fallback (1min/30min/60min)
2. Market calibration: MAE = {calibration_error:.3f} (good if < 0.03)
3. Coverage: {non_null_odds:,}/{len(obs_with_odds):,} observations have odds
4. Average spread: {spread_data['betfair_spread'].mean():.3f} (back - lay)
5. Average matched volume: ${volume_data['betfair_total_matched_volume'].mean():.2f}

🎯 Next Steps:
- Use betfair_implied_prob as a feature (market's assessment)
- Compare model probabilities to betfair_implied_prob to find edges
- Use betfair_spread and betfair_total_matched_volume for bet sizing
- Focus on positions where market calibration is poor (overpriced/underpriced)
    """)

    conn.close()


if __name__ == "__main__":
    main()
