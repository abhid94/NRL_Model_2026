# CLAUDE.md — NRL 2026 ATS Betting Edge Discovery Pipeline

> This file is the single source of truth for project principles, scope, quality gates, and guardrails.
> It is loaded into every Claude Code session. Keep it accurate and up-to-date.
> Update this document as the project learns new lessons, uncovers data issues, or refines modeling decisions.

---

## 1. Project Overview

**Project**: NRL 2026 Anytime Try Scorer (ATS) Betting Edge Discovery Pipeline

**Primary objective**: Discover profitable betting edges in the ATS market for the 2026 NRL season by building probability models that outperform bookmaker-implied probabilities by more than the bookmaker margin.

**Secondary objectives**:
- Serve as a hypothesis sandbox for testing and backtesting betting strategies
- Be extensible across seasons (2024 and 2025 data for training/validation, 2026 for live betting)
- Support both player-level and team-level analysis as first-class inputs

**Highest leverage focus areas**:
1. **High-quality data alignment and leakage prevention**
2. **Robust backtesting with realistic execution assumptions** (price slippage, liquidity, timing)

**Betting context**:
- User bets with traditional Australian bookmakers (Sportsbet, Bet365, TAB, etc.)
- Betfair closing prices serve as a **free benchmark only** — not the actual betting venue
- Bankroll range: **$5K–$20K**
- Target market: Anytime Try Scorer (ATS)

---

## 2. Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Database | SQLite (`data/nrl_data.db`) |
| Data manipulation | pandas, numpy |
| ML models | scikit-learn, xgboost, lightgbm |
| Statistical models | statsmodels |
| Explainability | shap |
| Visualization | matplotlib, seaborn |
| Testing | pytest |

**Not in scope**: No UI, no live odds integration, no web scraping (data already ingested).

---

## 3. Database Reference

### 3.1 Location & Pattern

- **Path**: `data/nrl_data.db`
- **Convention**: Year-suffixed tables (e.g., `player_stats_2024`, `player_stats_2025`)
- **Non-suffixed**: `teams` (master reference), `betfair_player_mapping`, `betfair_team_mapping`

### 3.2 Table Schemas

#### `teams` (master reference)
| Column | Type | Notes |
|--------|------|-------|
| squad_id | INTEGER PK | Join key for all team-level data |
| squad_name | TEXT | Full team name |
| squad_nickname | TEXT | e.g., "Roosters" |
| squad_code | TEXT | 3-letter code |

#### `players_{year}`
| Column | Type | Notes |
|--------|------|-------|
| player_id | INTEGER PK | Join key for all player-level data |
| firstname | TEXT | |
| surname | TEXT | |
| display_name | TEXT | |
| short_display_name | TEXT | |

#### `matches_{year}`
| Column | Type | Notes |
|--------|------|-------|
| match_id | INTEGER PK | Universal match identifier |
| match_number | INTEGER | |
| round_number | INTEGER | Season round (1–27 + finals) |
| match_type | TEXT | "Regular Season", "Finals", etc. |
| match_status | TEXT | |
| utc_start_time | TEXT | |
| local_start_time | TEXT | |
| home_squad_id | INTEGER | FK → teams.squad_id |
| away_squad_id | INTEGER | FK → teams.squad_id |
| venue_id | INTEGER | |
| venue_name | TEXT | |
| venue_code | TEXT | |
| period_completed | INTEGER | |
| period_seconds | INTEGER | |
| final_code | TEXT | |
| final_short_code | TEXT | |

#### `player_stats_{year}`
| Column | Type | Notes |
|--------|------|-------|
| match_id | INTEGER | PK (composite with player_id) |
| player_id | INTEGER | PK (composite with match_id) |
| squad_id | INTEGER | Player's team for this match |
| position | TEXT | Named position (e.g., "Fullback") |
| jumper_number | INTEGER | Jersey number |
| tries | INTEGER | **TARGET VARIABLE** |
| try_assists | INTEGER | |
| line_breaks | INTEGER | |
| line_break_assists | INTEGER | |
| tackle_breaks | INTEGER | |
| run_metres | INTEGER | |
| post_contact_metres | INTEGER | |
| metres_gained | INTEGER | |
| kicks_general_play | INTEGER | |
| kick_metres | INTEGER | |
| tackles | INTEGER | |
| missed_tackles | INTEGER | |
| errors | INTEGER | |
| passes | INTEGER | |
| possessions | INTEGER | |
| penalties_conceded | INTEGER | |
| conversions | INTEGER | |
| conversion_attempts | INTEGER | |
| sin_bins | INTEGER | |
| on_reports | INTEGER | |
| sent_offs | INTEGER | |
| bomb_kicks_caught | INTEGER | |
| runs_kick_return | INTEGER | |
| runs_hitup | INTEGER | |
| penalty_goal_attempts | INTEGER | |
| penalty_goals_unsuccessful | INTEGER | |
| field_goals_unsuccessful | INTEGER | |
| kicks_caught | INTEGER | |
| try_debits | INTEGER | |
| try_saves | INTEGER | |
| runs_dummy_half | INTEGER | |
| runs_dummy_half_metres | INTEGER | |
| offloads | INTEGER | |
| goal_line_dropouts | INTEGER | |
| forty_twenty | INTEGER | |
| field_goals | INTEGER | |
| penalty_goals | INTEGER | |
| field_goal_attempts | INTEGER | |
| side | TEXT | "home" or "away" |
| opponent_squad_id | INTEGER | |

#### `team_stats_{year}`
| Column | Type | Notes |
|--------|------|-------|
| match_id | INTEGER | PK (composite with squad_id) |
| squad_id | INTEGER | PK (composite with match_id) |
| score | INTEGER | Total match score |
| completion_rate_percentage | REAL | |
| line_breaks | INTEGER | |
| possession_percentage | REAL | |
| run_metres | INTEGER | |
| tackles | INTEGER | |
| errors | INTEGER | |
| missed_tackles | INTEGER | |
| post_contact_metres | INTEGER | |
| metres_gained | INTEGER | |
| try_assists | INTEGER | |
| line_break_assists | INTEGER | |
| try_saves | INTEGER | |
| tries | INTEGER | Team total tries |
| tackle_breaks | INTEGER | |
| passes | INTEGER | |
| bomb_kicks_caught | INTEGER | |
| kicks_caught | INTEGER | |
| kick_metres | INTEGER | |
| kicks_general_play | INTEGER | |
| field_goal_attempts | INTEGER | |
| field_goals | INTEGER | |
| conversion_attempts | INTEGER | |
| conversions | INTEGER | |
| conversions_unsuccessful | INTEGER | |
| penalty_goal_attempts | INTEGER | |
| penalty_goals | INTEGER | |
| penalty_goals_unsuccessful | INTEGER | |
| penalties_conceded | INTEGER | |
| goal_line_dropouts | INTEGER | |
| forty_twenty | INTEGER | |
| scrum_wins | INTEGER | |
| offloads | INTEGER | |
| runs | INTEGER | |
| runs_normal / runs_normal_metres | INTEGER | |
| runs_hitup / runs_hitup_metres | INTEGER | |
| runs_dummy_half / runs_dummy_half_metres | INTEGER | |
| runs_kick_return / runs_kick_return_metres | INTEGER | |
| time_in_own_half / time_in_opp_half | INTEGER | |
| time_in_own20 / time_in_opp20 | INTEGER | |
| complete_sets / incomplete_sets | INTEGER | |
| handling_errors | INTEGER | |
| set_restarts / set_restarts_ruck / set_restarts_10m | INTEGER | |
| sin_bins / on_reports / sent_offs | INTEGER | |
| possessions | INTEGER | |
| tackleds | INTEGER | |
| ineffective_tackles / tackles_ineffective | INTEGER | |

#### `score_flow_{year}`
| Column | Type | Notes |
|--------|------|-------|
| match_id | INTEGER | |
| period | INTEGER | 1 or 2 |
| squad_id | INTEGER | Scoring team |
| player_id | INTEGER | Scorer |
| score_name | TEXT | "Try", "Conversion", "Penalty Goal", etc. |
| score_points | INTEGER | Points for this score event |
| period_seconds | INTEGER | Time in half |

#### `team_lists_2025`
| Column | Type | Notes |
|--------|------|-------|
| match_id | INTEGER | |
| round_number | INTEGER | |
| squad_id | INTEGER | |
| squad_name | TEXT | |
| player_name | TEXT | |
| player_id | INTEGER | |
| jersey_number | INTEGER | Populated (1–21+) |
| position | TEXT | **ALWAYS EMPTY** — see data quality notes |

#### `betfair_markets_{year}`
| Column | Type | Notes |
|--------|------|-------|
| event_date | TEXT | |
| path | TEXT | |
| event_id | INTEGER | |
| market_type | TEXT | "TO_SCORE", "FIRST_TRY_SCORER", "MATCH_WINNER", etc. |
| market_id | INTEGER | |
| market_name | TEXT | |
| selection_id | INTEGER | |
| runner_name | TEXT | Player/team name on Betfair |
| handicap | REAL | |
| runner_status | TEXT | |
| is_winner | INTEGER | 1 if scored a try |
| total_points | REAL | |
| home_team / away_team | TEXT | |
| home_margin / home_score / away_score | REAL | |
| best_back_price_{60,30,1}_min_prior | REAL | |
| best_lay_price_{60,30,1}_min_prior | REAL | |
| matched_volume_{60,30,1}_min_prior | REAL | |
| total_matched_volume | REAL | |
| last_preplay_price | REAL | **33% empty strings in 2024 TO_SCORE** |
| AD_match_id | INTEGER | Mapped match_id (nullable) |
| AD_player_id | INTEGER | Mapped player_id (nullable) |

#### `bookmaker_odds_2025`
| Column | Type | Notes |
|--------|------|-------|
| odds_id | INTEGER PK | Auto-increment |
| match_id | INTEGER | |
| player_id | INTEGER | |
| bookmaker | TEXT | Currently **Betfair only** |
| market_type | TEXT | Default "anytime_tryscorer" |
| decimal_odds | REAL | CHECK(≥ 1.01 AND ≤ 1000.0) |
| implied_probability | REAL | 1 / decimal_odds |
| odds_timestamp | TEXT | ISO format |
| snapshot_type | TEXT | "opening", "closing", etc. |
| is_available | BOOLEAN | |
| is_verified | BOOLEAN | |

#### Operational tables (reference only)
- `betfair_player_mapping` — maps Betfair runner names → official player_id
- `betfair_team_mapping` — maps Betfair team names → official squad_id
- `match_reports` / `match_reports_2024` — sin bin / on report events
- `ingested_matches_2024` / `ingested_matches_2025` — tracking which matches are ingested

### 3.3 Data Quality Notes

| # | Finding | Detail | Action |
|---|---------|--------|--------|
| 1 | `players_2024` is denormalized | 7,344 rows for 523 unique players (one row per match appearance). `players_2025` is correctly deduplicated (526 rows). | Always `SELECT DISTINCT player_id` or join to `player_stats` directly. |
| 2 | `team_lists_2025.position` is empty | All 5,577 rows have blank position string. Jersey number IS populated. | Infer position from jersey number: 1=FB, 2=Wing, 3=Centre, 4=Centre, 5=Wing, 6=Five-eighth, 7=Halfback, 8=Prop, 9=Hooker, 10=Prop, 11=2nd Row, 12=2nd Row, 13=Lock, 14–17=Interchange, 18+=Reserves. |
| 3 | `bookmaker_odds_2025` is Betfair-only | Despite schema supporting multiple bookmakers, only "Betfair" closing prices exist. | Actual bookmaker odds needed for 2026. Current data = Betfair benchmark only. |
| 4 | Betfair empty `last_preplay_price` | 32.8% (2,337/7,127) of TO_SCORE records in 2024 have empty string (not NULL). | Use fallback chain: `last_preplay_price` → `best_back_price_1_min_prior` → `best_back_price_30_min_prior`. The 1-min-prior field recovers 100% of the gaps. |
| 5 | 8.1% unmapped Betfair runners | 579/7,127 TO_SCORE records in 2024 lack `AD_player_id`. | Mostly fringe/bench players. Low impact on top selections but must be accounted for in coverage metrics. |
| 6 | Try distribution is zero-inflated | 80.9% zeros, 15.8% ones, 2.7% twos, 0.45% threes. Overall try rate: ~19.0%. | Binary classification (`tries > 0`) is appropriate. No resampling needed at 19% positive rate. |
| 7 | Position try rates vary massively | Wing: 47.8%, Fullback: 37.5%, Centre: 27.8%, Prop: 8.6%, Interchange: 5.6%. | Position is a top-tier feature. Must include in all models. |
| 8 | Rounds with fewer matches | Rounds 13, 14, 16, 19, 20 (2024) have 5–7 matches (State of Origin / byes). | Don't treat missing matches as zero performance in rolling windows. Handle bye rounds explicitly. |
| 9 | Home try advantage | Home: 19.9%, Away: 18.1% (~1.8pp difference). | Include `is_home` as a feature. |

### 3.4 Entity Relationships

```
teams (squad_id)
  ├── matches_{year}.home_squad_id
  ├── matches_{year}.away_squad_id
  ├── player_stats_{year}.squad_id
  ├── team_stats_{year}.squad_id
  ├── score_flow_{year}.squad_id
  └── team_lists_2025.squad_id

players_{year} (player_id)
  ├── player_stats_{year}.player_id
  ├── score_flow_{year}.player_id
  ├── team_lists_2025.player_id
  └── betfair_markets_{year}.AD_player_id

matches_{year} (match_id)
  ├── player_stats_{year}.match_id
  ├── team_stats_{year}.match_id
  ├── score_flow_{year}.match_id
  ├── fixtures_2025.match_id
  ├── team_lists_2025.match_id
  └── betfair_markets_{year}.AD_match_id
```

### 3.5 Year-Suffix Abstraction Strategy

To avoid hardcoding year suffixes throughout the codebase:

1. **Python helper** in `src/db.py`: `get_table(base_name, year)` returns the suffixed table name
2. **TEMP views** at session start: create views like `player_stats` → `UNION ALL` of all year tables with a `season` column
3. **Parameterized queries**: always accept `year` or `season` as a parameter

---

## 4. Directory Structure

```
NRL_2026_Model/
├── CLAUDE.md                 # This file — project guide
├── PLAN_PROMPT.md            # Original planning prompt
├── requirements.txt          # Python dependencies
├── .gitignore
├── docs/
│   └── plans/                # Project plans (volatile, updated frequently)
│       └── roadmap.md        # 6-phase project roadmap
├── data/
│   ├── nrl_data.db           # SQLite database (DO NOT commit)
│   ├── feature_store/        # Parquet feature files
│   ├── model_artifacts/      # Trained models, calibrators
│   └── backtest_results/     # Backtest output CSVs
├── src/
│   ├── config.py             # Constants, thresholds, path mappings
│   ├── db.py                 # DB connection, year-suffix helpers
│   ├── features/
│   │   ├── player_features.py
│   │   ├── team_features.py
│   │   ├── matchup_features.py
│   │   ├── lineup_features.py
│   │   ├── score_flow_features.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── baseline.py       # Position-based baseline
│   │   ├── logistic.py       # Logistic regression
│   │   ├── gbm.py            # XGBoost / LightGBM
│   │   ├── poisson.py        # Poisson regression
│   │   ├── ensemble.py       # Model ensemble
│   │   └── calibration.py    # Platt scaling / isotonic
│   ├── evaluation/
│   │   ├── backtest.py       # Walk-forward backtesting engine
│   │   ├── metrics.py        # ROI, Brier, log-loss, calibration
│   │   ├── simulation.py     # Monte Carlo bankroll simulations
│   │   └── leakage_checks.py # Automated leakage detection
│   ├── pipeline/
│   │   ├── weekly_pipeline.py   # End-to-end weekly workflow
│   │   ├── predict_round.py     # Generate predictions for a round
│   │   └── bet_recommendations.py # Edge calc + stake sizing
│   └── odds/
│       ├── betfair.py        # Betfair price extraction + fallback
│       ├── bookmaker.py      # Bookmaker odds processing
│       └── edge.py           # Edge calculation (model prob vs odds)
├── notebooks/                # Exploratory analysis (NOT production code)
└── tests/
    ├── test_leakage.py       # Leakage prevention tests
    ├── test_features.py      # Feature computation tests
    ├── test_db.py            # DB helper tests
    └── test_pipeline.py      # Pipeline integration tests
```

---

## 5. Coding Principles

1. **Every feature must be computed using ONLY pre-match historical data.** No exceptions.
2. **All rolling windows use strict `round_number < current_round` filtering.** Never `<=`.
3. **No global state.** Pass DB connections and configuration explicitly.
4. **Functions should be pure where possible.** Same inputs produce the same outputs.
5. **SQL queries must be parameterized.** No f-string interpolation of user/variable values into SQL.
6. **Type hints on all public functions.**
7. **Docstrings on all public functions** (numpy style).
8. **One responsibility per file.** Keep modules focused and composable.
9. **Fail loudly.** Raise exceptions on unexpected data conditions rather than silently returning defaults.
10. **Log, don't print.** Use Python's `logging` module for all output.

---

## 6. Workspace Hygiene Rules

1. **Delete temporary/scratch files after use.** Don't let exploratory artifacts accumulate.
2. **Never commit the SQLite database to git.** It's in `.gitignore`.
3. **Feature store files** go in `data/feature_store/`, **model artifacts** in `data/model_artifacts/`.
4. **Notebooks are exploratory only.** All production logic lives in `src/`.
5. **No hardcoded paths.** Use `src/config.py` for all paths and constants.
6. **Clean up intermediate DataFrames** in memory when no longer needed (`del df; gc.collect()` for large frames).
7. **Don't commit generated output files** (backtest results, predictions). Add to `.gitignore` as needed.

---

## 7. Leakage Prevention Rules (CRITICAL)

These rules are the most important guardrails in the entire project. Violating any of them invalidates all model evaluation results.

| Rule | Description |
|------|-------------|
| **Rule 1** | When predicting round R, features use ONLY rounds < R of the current season + all prior seasons. |
| **Rule 2** | Target variable (`tries > 0`) is NEVER available as a feature for the same observation. |
| **Rule 3** | Betfair/bookmaker odds for round R ARE allowed (they're pre-match public data), but match OUTCOMES from round R are forbidden. |
| **Rule 4** | Team lists for round R ARE allowed (announced ~24h pre-match), but are only used for lineup-based features. |
| **Rule 5** | Rolling windows must handle bye rounds — don't treat missing rounds as zero performance. Use match sequence, not round sequence. |
| **Rule 6** | Calibration and model selection CV must be **temporal** (walk-forward). NEVER use random k-fold. |
| **Rule 7** | Every feature computation function must accept a `max_round` parameter that enforces the temporal cutoff. |
| **Rule 8** | Automated leakage checks run as part of the test suite (`tests/test_leakage.py`). Every feature must pass these checks before entering the feature store. |

**Enforcement**: Before any model evaluation result is reported, verify that `pytest tests/test_leakage.py` passes.

---

## 8. Weekly Retraining Protocol (2026 Season)

```
Monday/Tuesday (after previous round completes):
  1. INGEST    — User adds new round data to _2026 tables
  2. VALIDATE  — Run data quality checks on ingested data
  3. REBUILD   — Rebuild feature store including new round
  4. RETRAIN   — Retrain model on all available data (2024 + 2025 + 2026 to date)
  5. CALIBRATE — Re-run calibration on recent validation window

Wednesday/Thursday (before next round):
  6. TEAM LISTS — Confirm official team lists (Tuesday ~4pm AEDT)
  7. PREDICT    — Generate predictions for upcoming round
  8. ODDS       — Compare model probabilities to bookmaker odds
  9. OUTPUT     — Generate bet recommendations with stake sizes
  10. MONITOR   — Log predictions for future evaluation
```

**Timing**: The full pipeline (steps 1–10) should complete in < 5 minutes.

**Logging**: Every prediction, recommendation, and actual outcome is logged for ongoing model evaluation.

---

## 9. Model Evaluation & Reporting Standards

Every model report must include the following metrics and thresholds, unless explicitly justified:

- **Discrimination**: AUC, PR-AUC (report both).
- **Calibration**: Brier score, calibration error (target < 0.03).
- **Ranking quality**: Top-decile lift and hit rate versus baseline.
- **Economic performance**: ROI, CLV, and max drawdown from walk-forward backtests.
- **Stability**: Performance by season (2024 and 2025) and by position group.

**Minimum acceptance criteria** (baseline to proceed):
- Positive ROI on both 2024 and 2025 walk-forward backtests.
- CLV positive in aggregate.
- Calibration error < 0.03 overall and < 0.05 by position group.

---

## 10. Quality Gates Checklist

Run through this checklist before trusting any model output:

- [ ] Every table row count matches expected (~7,344 player-match observations per season)
- [ ] No duplicate `(match_id, player_id)` pairs in `player_stats`
- [ ] Referential integrity: all `squad_id` values exist in `teams` table
- [ ] Cross-validate tries: `SUM(player_stats.tries)` ≈ count of "Try" events in `score_flow`
- [ ] Betfair price fallback chain recovers 99%+ of TO_SCORE records
- [ ] Feature store has one row per `(match_id, player_id)` for all player appearances
- [ ] No NaN in features for players with >= 5 prior matches
- [ ] Leakage test passes: features for round R use only data from rounds < R
- [ ] Model calibration error < 0.03 across all probability bins
- [ ] Best model achieves positive ROI on **both** 2024 AND 2025 walk-forward backtests
- [ ] Positive Closing Line Value (CLV) on aggregate across recommended bets
- [ ] Weekly pipeline runs end-to-end in < 5 minutes

---

## 10. Project Plans

Detailed project plans, sprint tasks, and phase milestones live in **`docs/plans/`** — not in this file. Keep those plan files updated as work completes (mark tasks done, add new tasks, and adjust milestones as scope changes).

| Plan | Path | Description |
|------|------|-------------|
| Roadmap | [`docs/plans/roadmap.md`](docs/plans/roadmap.md) | 6-phase project roadmap with task checklists |

Phases at a glance: (1) Data Understanding & Validation → (2) Feature Engineering & EDA → (3) Baseline Models & Backtesting → (4) Advanced Models & Edge Discovery → (5) Weekly Pipeline for 2026 → (6) Extensibility.

**IMPORTANT: Roadmap Update Protocol**

As you complete tasks, you MUST update `docs/plans/roadmap.md` immediately:

1. **Mark tasks complete**: Change `[ ]` to `[x]` when a task is fully done
2. **Update "Last Updated" date**: Set to current date when making changes
3. **Add discovered tasks**: If implementation reveals new subtasks, add them to the roadmap
4. **Update "Current Phase"**: Change the active phase when transitioning between phases
5. **Document blockers**: If a task is blocked, add a note explaining why

This ensures the roadmap is always an accurate reflection of project state, not a stale document.

---

## 11. Staking & Risk Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kelly fraction | 0.25 | Quarter Kelly for bankroll preservation |
| Max stake per bet | 5% of bankroll | Absolute cap regardless of edge |
| Max exposure per round | 20% of bankroll | Total stakes across all bets in a round |
| Max bets per round | 15 | Practical limit for manual placement |
| Min stake | $5 | Bookmaker minimums |
| Min edge threshold | 5 percentage points | Model probability must exceed implied probability by ≥ 5pp |
| Drawdown warning | 15% | Reduce to 15% Kelly (from 25%) |
| Drawdown halt | 25% | Pause betting for 2 rounds, full audit |
| Drawdown stop | 40% | Stop betting entirely, fundamental review |

**Bankroll tracking**: Maintain a running log of bankroll, stakes, and outcomes to monitor drawdown levels.

---

## 12. Key Risks & Uncertainties

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Market efficiency** | ATS markets may be too efficient for sustainable edges after bookmaker margins (15–20%). | Must outperform by MORE than the margin. Walk-forward backtest on 2024 + 2025 validates edge existence before risking capital. |
| **Sample size** | ~7,344 player-match observations per season; sparse for player-specific and head-to-head features. | Use regularization, hierarchical priors, and position-group shrinkage. |
| **Bookmaker margins** | Model must outperform true probability by more than the margin to profit. | Track CLV (Closing Line Value) as a leading indicator of edge. |
| **Account limitations** | Australian bookmakers aggressively limit winning accounts (3–6 month tolerance). | Diversify across bookmakers, keep bet sizes moderate, consider exchange (Betfair) as backup. |
| **Bet correlation** | ATS bets in the same match are correlated (if team scores more, multiple players benefit). | Cap at 3–4 ATS bets per match. Account for correlation in bankroll simulations. |
| **Overfitting** | With limited data, complex models can memorize noise. | Walk-forward validation + strong regularization + simple baseline comparison. |
| **Data quality** | `players_2024` dedup issue, empty positions in `team_lists`, Betfair-only odds. | Document all issues (Section 3.3), build defensive data pipelines, validate at every stage. |
