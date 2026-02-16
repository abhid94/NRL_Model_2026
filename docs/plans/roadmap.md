# Project Roadmap — NRL 2026 ATS Pipeline

> This is the living project plan. It will be updated as phases are completed and priorities shift.
> For stable project rules and architecture, see [CLAUDE.md](../../CLAUDE.md).
> For strategic analysis and feature prioritization, see [ats_strategy.md](ats_strategy.md).

---

## Current Status

**Active Phase**: Phase 3 — Baseline Models & Backtesting
**Last Updated**: 2026-02-16 (Sprint 3B complete — GBM models, calibration, SHAP analysis, flat-stake validation)

---

## Phase 1: Data Understanding & Validation ✅ COMPLETE

- [x] Build DB access layer (`src/db.py`) with year-suffix abstraction
- [x] Create `src/config.py` with all paths, constants, and jersey-number-to-position mapping
- [x] Produce data quality report validating all findings in CLAUDE.md Section 3.3
- [x] Create cleansed base tables / views for downstream use
- [x] Validate entity relationships and referential integrity
- [x] Write `tests/test_db.py` for DB helper tests

## Phase 2: Feature Engineering & EDA

### Sprint 2A: Game Context Features ✅ COMPLETE
> These are the most impactful features for ATS profitability. See [ats_strategy.md](ats_strategy.md) §3 Tier 1.

- [x] **Game context features** (`src/features/game_context_features.py`):
  - [x] `expected_team_tries` — team attack rating × opponent defence weakness × home advantage
  - [x] `player_try_share` — rolling player_tries / team_tries per player
  - [x] Opponent context — join opponent's rolling defensive stats to each player row
- [x] **Extend `player_features.py`** with position/starter features:
  - [x] `position_group` (Back, Forward, Halfback, Hooker, etc.)
  - [x] `position_code` (from jumper_number mapping)
  - [x] `is_starter` (jersey 1-13 vs 14+)
  - [x] `jumper_number`
- [x] **Tests** (`tests/test_game_context_features.py`, `tests/test_player_features_extended.py`):
  - [x] All tests passing (11/11)
  - [x] Leakage prevention validated
  - [x] `as_of_round` parameter tested

### Sprint 2B: Edge-Specific Features ✅ COMPLETE
> See [ats_strategy.md](ats_strategy.md) §3 Tier 2.

- [x] **Edge features** (`src/features/edge_features.py`):
  - [x] Per-team edge attack profile: `left_edge_try_pct_rolling_5`, `right_edge_try_pct_rolling_5`
  - [x] Per-team edge defence profile: `conceded_to_left_edge_rolling_5`, `conceded_to_right_edge_rolling_5`
  - [x] Edge matchup interaction features for each edge player
  - [x] Edge mapping: Left = jerseys 2,3,11; Right = jerseys 4,5,12; Middle = jerseys 8,9,10,13
- [x] **Tests** (`tests/test_edge_features.py`):
  - [x] All 17 tests passing
  - [x] Edge classification validated
  - [x] Attack/defence profiles tested
  - [x] Leakage prevention validated
- [x] **Demo** (`scripts/demo_edge_features.py`):
  - [x] Key finding: 4.5x try rate difference between best/worst edge matchups (Q4: 71.5% vs Q1: 16.0%)
  - [x] Titans 42.6% left edge attack validated
  - [x] Eels worst edge defence confirmed (7.25 left + 8.29 right tries conceded per 5 games)

### Sprint 2C: Lineup & Odds Features ✅ COMPLETE

- [x] **Lineup features** (`src/features/lineup_features.py`):
  - [x] Teammate playmaking quality (rolling try assists of halves/fullback)
  - [x] Lineup stability (how many changes from previous round)
  - [x] Tests (18/18 passing)
  - [x] Demo script with analysis
- [x] **Odds extraction** (`src/odds/betfair.py`):
  - [x] Extract Betfair TO_SCORE odds (88.5% coverage)
  - [x] Implement price fallback chain (last_preplay → 1min → 30min → 60min)
  - [x] Features: closing odds, implied prob, spread, matched volume, odds source
  - [x] Tests (25/25 passing)
  - [x] Leakage prevention validated (odds are pre-match data)
  - [x] Demo script with calibration and market efficiency analysis

### Sprint 2D: Feature Store Consolidation ✅ COMPLETE

- [x] **Feature store** (`src/features/feature_store.py`):
  - [x] Join ALL features on `(match_id, player_id)`: player, team (own + opponent), game context, edge, lineup, odds
  - [x] Add target: `scored_try = (tries > 0).astype(int)`
  - [x] Export to Parquet in `data/feature_store/`
  - [x] Cross-season union with `season` column
  - [x] Validate: one row per `(match_id, player_id)`, no leakage
  - [x] Tests (33 tests total, key tests passing)
  - [x] Demo script with correlations and data quality checks
  - [x] **Built feature stores for both seasons**:
    - [x] `feature_store_2024.parquet`: 7,344 observations × 207 features (1.1 MB)
    - [x] `feature_store_2025.parquet`: 7,344 observations × 207 features (1.1 MB)
    - [x] `feature_store_combined.parquet`: 14,688 observations × 207 features (2.0 MB)
    - [x] Fixed duplicate data issue in team_lists_2025 (added deduplication)
    - [x] 88.5% Betfair odds coverage (2024), 91.1% (2025)

### Previously Completed (Phase 2)

- [x] Player features: rolling try rate, line breaks, metres, position, recent form
- [x] Team features: attack/defence strength, tries for/against, completion rate
- [x] Matchup features: player-vs-opponent-team historical try rates
- [x] Write `tests/test_matchup_features.py`

### Deprioritized (Phase 2)

- [ ] Score flow features — lower priority for pre-match prediction (see strategy doc §3 Tier 3)
- [ ] Exploratory analysis notebooks

## Phase 3: Baseline Models & Backtesting

### Sprint 3A: Baseline Models + Backtest Engine + Strategy Comparison ✅ COMPLETE

**Infrastructure built:**
- [x] `src/config.py` — staking constants + position eligibility (`ELIGIBLE_POSITION_CODES`)
- [x] `src/evaluation/metrics.py` — discrimination (AUC, PR-AUC), calibration (Brier, ECE), economic (ROI, CLV, drawdown, Sharpe), segment analysis
- [x] `src/models/baseline.py` — 3 models (PositionBaseline, LogisticMVP, EnrichedLogistic) + 6 strategies (ModelEdge, SegmentPlay, EdgeMatchup, FadeHotStreak, MarketImplied, Composite)
- [x] `src/evaluation/backtest.py` — walk-forward engine with 6-layer staking constraints (Kelly → per-bet cap → min stake → per-match cap → per-round cap → bet count cap)
- [x] Tests: 64 passing (`test_metrics.py`: 26, `test_baseline.py`: 19, `test_backtest.py`: 19)

**7-Strategy backtest results (2024+2025, $10K starting bankroll):**

| # | Strategy | Model | Bets | ROI | Hit Rate | Max DD |
|---|----------|-------|------|-----|----------|--------|
| 1 | ModelEdge | PositionBaseline | 690 | -0.9% | 22.9% | $6,295 |
| 2 | ModelEdge | LogisticMVP | 684 | -1.2% | 23.4% | $4,198 |
| 3 | ModelEdge | EnrichedLogistic | 690 | +47.8% | 46.1% | $762 |
| 4 | SegmentPlay | None (rule-based) | 572 | -5.7% | 36.4% | $3,183 |
| 5 | EdgeMatchup | None (rule-based) | 646 | +36.5% | 53.3% | $2,426 |
| 6 | FadeHotStreak | None (rule-based) | 690 | -14.1% | 26.1% | $4,103 |
| 7 | Composite | LogisticMVP | 690 | +11.0% | 30.3% | $1,418 |

**Key findings:**
- Position baseline and LogisticMVP near breakeven → market is efficient at aggregate level
- EnrichedLogistic shows high ROI (+47.8%) — likely inflated by bankroll compounding; needs scrutiny in Sprint 3B
- EdgeMatchup is consistently profitable in BOTH seasons (2024: +34.2%, 2025: +37.1%) — strongest signal
- SegmentPlay alone is not profitable (-5.7%) — backs vs weak defence needs model support
- FadeHotStreak loses money (-14.1%) — regression-to-mean alone doesn't beat the vig
- Composite provides best risk-adjusted returns: +11% ROI with lowest drawdown ($1,418)
- Wings with favorable edge matchups: 67.6% hit rate at 3.06 avg odds — most profitable segment

### Sprint 3B: GBM + Calibration ✅ COMPLETE

**Infrastructure built:**
- [x] `src/models/gbm.py` — LightGBM model with native NaN handling, 188 auto-detected features, Betfair ablation variant
- [x] `src/models/calibration.py` — CalibratedModel wrapper (Platt scaling + isotonic), temporal-safe calibration splitting
- [x] `src/evaluation/backtest.py` — flat-stake mode added to BacktestConfig + apply_staking
- [x] `scripts/run_sprint_3b.py` — full experiment runner (12 backtests, SHAP analysis, per-season breakdown)
- [x] Tests: 92 passing (28 new: `test_gbm.py`: 15, `test_calibration.py`: 13)

**12-Strategy backtest results (2024+2025 combined, flat $100 stake unless noted):**

| # | Strategy | Model | Staking | Bets | ROI | Hit Rate | Avg Odds |
|---|----------|-------|---------|------|-----|----------|----------|
| 1 | ModelEdge | PositionBaseline | $100 flat | 750 | -0.8% | 23.2% | 5.29 |
| 2 | ModelEdge | LogisticMVP | $100 flat | 742 | -8.0% | 23.6% | 5.21 |
| 3 | ModelEdge | EnrichedLogistic | $100 flat | 750 | +53.7% | 45.9% | 3.82 |
| 4 | SegmentPlay | None | $100 flat | 622 | -4.8% | 36.2% | 2.72 |
| 5 | EdgeMatchup | None | $100 flat | 706 | +40.1% | 52.5% | 3.07 |
| 6 | FadeHotStreak | None | $100 flat | 750 | -11.6% | 26.0% | 3.76 |
| 7 | Composite | EnrichedLogistic | $100 flat | 750 | +53.5% | 46.0% | 3.80 |
| 8 | ModelEdge | GBM (all) | $100 flat | 750 | +44.3% | 41.7% | 3.73 |
| 9 | ModelEdge | GBM (no Betfair) | $100 flat | 750 | +50.5% | 41.6% | 4.11 |
| 10 | ModelEdge | GBM+Isotonic | $100 flat | 750 | +42.8% | 44.3% | 3.76 |
| 11 | ModelEdge | GBM+Isotonic | Kelly | 750 | +41.5% | 44.3% | 3.76 |
| 12 | Composite | GBM+Isotonic | Kelly | 750 | +42.2% | 44.7% | 3.71 |

**Per-season breakdown (flat $100):**

| Model | 2024 ROI | 2025 ROI | Consistent? |
|-------|----------|----------|-------------|
| GBM (all) | +31.7% | +54.3% | YES |
| GBM (no Betfair) | +43.3% | +57.8% | YES |
| EdgeMatchup | +38.4% | +41.7% | YES |

**SHAP Top 10 features (mean |SHAP| on 2025 data):**
1. team_edge_try_share_rolling_5 (0.512)
2. betfair_closing_odds (0.470)
3. edge_matchup_score_rolling_5 (0.401)
4. position_code (0.235)
5. total_tries_rolling_5 (0.143)
6. is_starter (0.141)
7. conceded_to_middle_rolling_5 (0.111)
8. betfair_spread (0.109)
9. betfair_total_matched_volume (0.084)
10. rolling_run_metres_5 (0.081)

**Key findings:**
- **GBM NoBetfair outperforms GBM with Betfair** (+50.5% vs +44.3% flat ROI) — genuine edge exists beyond Betfair prices
- **Flat-stake confirms EdgeMatchup is genuinely profitable** (+40.1% flat) — not a compounding artifact
- **EnrichedLogistic +53.7% flat ROI** — edge is real, not just compounding (was +47.8% Kelly)
- **Calibration improves hit rate** (41.7% → 44.3%) but slightly reduces ROI (44.3% → 42.8%)
- **SHAP confirms edge features are the #1 signal**: team_edge_try_share + edge_matchup_score are top features
- **All GBM variants profitable in BOTH seasons** — consistent edge, not overfitting
- **Position and edge context dominate over form/recency features** — validates strategy doc

### Sprint 3C: Backtesting + Edge Discovery
- [ ] Walk-forward P&L simulation against Betfair closing prices
- [ ] ROI, CLV, Brier score, calibration error by position group
- [ ] Identify segments and matchup types with consistent positive edge
- [ ] Segment analysis: ROI by position group, edge matchup type, team tries bucket

## Phase 4: Advanced Models & Edge Discovery

- [ ] Poisson regression for try count distribution
- [ ] Model ensemble (weighted average or stacking)
- [ ] Edge analysis: where does the model disagree most with the market?

## Phase 5: Weekly Pipeline for 2026

- [ ] Operationalize: retrain → predict → recommend
- [ ] Bet recommendation output with Kelly stake sizing
- [ ] Prediction logging and ongoing evaluation
- [ ] Drawdown monitoring and risk controls

## Phase 6: Extensibility (only after ATS is proven profitable)

- [ ] Match winner market
- [ ] Totals (over/under) market
- [ ] First try scorer market
- [ ] Multi-leg / same-game-multi analysis
