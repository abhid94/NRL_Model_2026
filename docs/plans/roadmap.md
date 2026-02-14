# Project Roadmap — NRL 2026 ATS Pipeline

> This is the living project plan. It will be updated as phases are completed and priorities shift.
> For stable project rules and architecture, see [CLAUDE.md](../../CLAUDE.md).
> For strategic analysis and feature prioritization, see [ats_strategy.md](ats_strategy.md).

---

## Current Status

**Active Phase**: Phase 2 — Feature Engineering & EDA
**Last Updated**: 2026-02-15 (Sprint 2C complete - odds features extracted)

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

### Sprint 2D: Feature Store Consolidation ⬅️ NEXT PRIORITY

- [ ] **Feature store** (`src/features/feature_store.py`):
  - [ ] Join ALL features on `(match_id, player_id)`: player, team (own + opponent), game context, edge, lineup, odds
  - [ ] Add target: `scored_try = (tries > 0).astype(int)`
  - [ ] Export to Parquet in `data/feature_store/`
  - [ ] Cross-season union with `season` column
  - [ ] Validate: one row per `(match_id, player_id)`, no leakage

### Previously Completed (Phase 2)

- [x] Player features: rolling try rate, line breaks, metres, position, recent form
- [x] Team features: attack/defence strength, tries for/against, completion rate
- [x] Matchup features: player-vs-opponent-team historical try rates
- [x] Write `tests/test_matchup_features.py`

### Deprioritized (Phase 2)

- [ ] Score flow features — lower priority for pre-match prediction (see strategy doc §3 Tier 3)
- [ ] Exploratory analysis notebooks

## Phase 3: Baseline Models & Backtesting

### Sprint 3A: Baseline + Logistic Model
- [ ] Position-only baseline model (floor)
- [ ] Logistic regression with 6 MVP features: `position_group`, `expected_team_tries`, `player_try_share`, `is_home`, `is_starter`, `opponent_tries_conceded_5`
- [ ] Walk-forward backtest on 2024 and 2025

### Sprint 3B: GBM + Calibration
- [ ] XGBoost/LightGBM with all features, temporal CV
- [ ] SHAP feature importance analysis
- [ ] Probability calibration (Platt scaling / isotonic regression)

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
