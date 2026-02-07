# Project Roadmap — NRL 2026 ATS Pipeline

> This is the living project plan. It will be updated as phases are completed and priorities shift.
> For stable project rules and architecture, see [CLAUDE.md](../../CLAUDE.md).

---

## Current Status

**Active Phase**: Phase 1 — Data Understanding & Validation
**Last Updated**: 2026-02-07

---

## Phase 1: Data Understanding & Validation

- [ ] Build DB access layer (`src/db.py`) with year-suffix abstraction
- [ ] Create `src/config.py` with all paths, constants, and jersey-number-to-position mapping
- [ ] Produce data quality report validating all findings in CLAUDE.md Section 3.3
- [ ] Create cleansed base tables / views for downstream use
- [ ] Validate entity relationships and referential integrity
- [ ] Write `tests/test_db.py` for DB helper tests

## Phase 2: Feature Engineering & EDA

- [ ] Player features: rolling try rate, line breaks, metres, position, recent form
- [ ] Team features: attack/defence strength, tries for/against, completion rate
- [ ] Matchup features: player-vs-opponent-team historical try rates
- [ ] Lineup features: position in team list, jersey number, named position
- [ ] Score flow features: try-scoring patterns, momentum indicators
- [ ] Build feature store (`data/feature_store/`) with one row per `(match_id, player_id)`
- [ ] Exploratory analysis notebooks
- [ ] Write `tests/test_features.py`

## Phase 3: Baseline Models & Backtesting

- [ ] Walk-forward backtesting engine (round-by-round, strict temporal separation)
- [ ] Position-based baseline model (prior try rates by position group)
- [ ] Logistic regression model
- [ ] Simulated P&L against Betfair closing prices
- [ ] ROI, Brier score, log-loss, calibration plots

## Phase 4: Advanced Models & Edge Discovery

- [ ] XGBoost / LightGBM with hyperparameter tuning (temporal CV)
- [ ] Poisson regression for try count distribution
- [ ] Model ensemble (weighted average or stacking)
- [ ] Probability calibration (Platt scaling / isotonic regression)
- [ ] SHAP analysis for feature importance and edge explanation
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
