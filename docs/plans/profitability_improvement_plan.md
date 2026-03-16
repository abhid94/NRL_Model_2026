# Plan: Agents & Tools to Improve NRL 2026 Betting Model Profitability

> **Created**: 2026-03-16
> **Last Updated**: 2026-03-17
> **Branch**: `feature/profitability-improvements`
> **Status**: In Progress

---

## Context

The NRL 2026 ATS betting model is substantially built (365+ features, +19.7% backtested ROI, 40 profitable configs). The 2026 season has started (Rounds 1-2). The goal is to identify **built-in agents** we can run right now AND **external open-source tools** to integrate for more profit.

---

## Part 1: Built-In Agents That Can Help This Project

### Tier 1: High Impact (Run These First)

| Agent | What It Does For This Project | Priority | Status |
|-------|-------------------------------|----------|--------|
| **AI Engineer** | Implement PyMC Bayesian hierarchical model, custom profit-maximizing loss functions for GBM, conformal prediction intervals (MAPIE), hyperparameter optimization (Optuna) | P0 | [ ] |
| **Data Engineer** | Build outcome ingestion pipeline (score_flow -> CLV tracking), incremental feature updates, data quality monitoring, weather data integration | P0 | [ ] |
| **Backend Architect** | Design multi-bookmaker API integration (The Odds API, TAB Studio), automated retraining architecture, opening/closing odds dual-snapshot system | P0 | [ ] |
| **DevOps Automator** | Set up automated weekly retraining (cron/launchd), pipeline monitoring, model staleness alerts, automated odds snapshots | P1 | [ ] |

### Tier 2: Medium Impact

| Agent | What It Does For This Project | Priority | Status |
|-------|-------------------------------|----------|--------|
| **Performance Benchmarker** | Profile weekly pipeline (currently manual, target <5min), optimize feature store build, identify SQLite query bottlenecks | P1 | [ ] |
| **API Tester** | Validate all 8 bookmaker odds API integrations end-to-end, test fallback chains, verify edge calculations match manual spot-checks | P1 | [ ] |
| **Reality Checker** | Audit the +19.7% ROI claim with independent verification, stress-test bootstrap confidence intervals, check for remaining leakage | P1 | [ ] |
| **Experiment Tracker** | Manage A/B testing of 40 profitable configs against live 2026 data, track which configs perform best in-season | P2 | [ ] |
| **Workflow Optimizer** | Optimize the Monday-Thursday weekly workflow (ingest -> validate -> rebuild -> retrain -> predict -> bet), reduce manual steps | P2 | [ ] |

### Tier 3: Nice-to-Have

| Agent | What It Does For This Project | Priority | Status |
|-------|-------------------------------|----------|--------|
| **Frontend Developer** | Improve Streamlit dashboard: add CLV tracking charts, bankroll graph, per-bookmaker performance, model confidence display | P2 | [ ] |
| **Data Analytics Reporter** | Build automated weekly P&L reports, season-to-date ROI dashboard, per-position/per-bookmaker breakdown | P2 | [ ] |
| **Evidence Collector** | Screenshot-verify backtest results, check prediction logs match actual outputs, audit staking constraints | P3 | [ ] |
| **Test Results Analyzer** | Fix known test failures (test_sorted_by_edge), improve test coverage for pipeline integration | P3 | [ ] |
| **Security Engineer** | Audit for data leakage beyond temporal (e.g., feature correlation leakage), review API key management | P3 | [ ] |
| **Technical Writer** | Document the weekly operational procedure, API setup guides, model interpretation guide | P3 | [ ] |

---

## Part 2: External Open-Source Tools & GitHub Repos to Integrate

### Tier 1: Implement Immediately (This Week)

#### 1. `shin` - True Probability Extraction
- **Repo**: https://github.com/mberk/shin
- **Install**: `pip install shin`
- **What**: Shin's method calculates true implied probabilities correcting for favourite-longshot bias. More accurate than naive `1/odds`.
- **Integration**: Replace `implied_probability = 1/decimal_odds` in `src/odds/bookmaker.py` and `src/odds/edge.py`
- **Impact**: +0.5-1pp ROI from better edge calculation
- **Effort**: 1-2 hours
- **Status**: [x] Binary devigging for ATS (Shin wrong for non-mutual-exclusive)

#### 2. `optuna` - Bayesian Hyperparameter Optimization
- **Repo**: https://github.com/optuna/optuna
- **Install**: `pip install optuna`
- **What**: Bayesian optimization with pruning. Your Phase 5B grid search found 40 profitable configs; Optuna finds better ones faster.
- **Integration**: Wrap GBM training in `src/models/gbm.py` with Optuna study
- **Impact**: +1-2pp ROI from better model params
- **Effort**: Half day
- **Status**: [x] src/models/hyperopt.py created

#### 3. Weather Features via `weather-au`
- **Install**: `pip install weather-au`
- **What**: Pull BOM (Bureau of Meteorology) data. Rain reduces NRL try scoring by 15-20%.
- **Integration**: Add `src/features/weather_features.py`, join to match venue/date
- **Impact**: +0.5-1pp ROI (weather is a strong signal the market may underweight)
- **Effort**: 1 day
- **Status**: [ ]

#### 4. Devigging Scripts (BettingIsCool)
- **Source**: https://bettingiscool.com/2024/03/18/a-python-script-to-remove-the-overround-from-bookmaker-odds/
- **What**: Three devigging methods (multiplicative, additive, logarithmic). Logarithmic is most accurate.
- **Integration**: Add to `src/odds/edge.py` for proper margin removal
- **Impact**: +0.5pp ROI from more accurate true probability estimation
- **Effort**: 2-3 hours
- **Status**: [x] Integrated into devig.py (binary + margin correction approach)

### Tier 2: Implement This Week

#### 5. `MAPIE` - Conformal Prediction Intervals
- **Repo**: https://github.com/scikit-learn-contrib/MAPIE
- **Install**: `pip install mapie`
- **What**: Calibrated prediction intervals with guaranteed coverage. Skip bets where interval is too wide.
- **Integration**: Wrap CalibratedModel output in `src/models/calibration.py`
- **Impact**: +1pp ROI from avoiding low-confidence bets
- **Effort**: Half day
- **Status**: [x] src/models/conformal.py created, wired into predict_round

#### 6. `netcal` - Advanced Calibration
- **Repo**: https://github.com/EFS-OpenSource/calibration-framework
- **Install**: `pip install netcal`
- **What**: BBQ, ENIR, and other calibration methods beyond Platt/isotonic
- **Integration**: Test in `src/models/calibration.py` as alternative calibrators
- **Impact**: +0.5pp ROI from tighter calibration
- **Effort**: Half day
- **Status**: [ ]

#### 7. `keeks` - Advanced Kelly Staking
- **Repo**: https://github.com/wdm0006/keeks
- **Install**: `pip install keeks`
- **What**: Implements Fractional Kelly, DrawdownAdjustedKelly, Optimal-F with simulation. Direct upgrade to manual `compute_adaptive_kelly()`.
- **Integration**: Replace/augment `src/pipeline/weekly_pipeline.py` kelly logic
- **Impact**: +0.5-1pp ROI from better stake sizing
- **Effort**: Half day
- **Status**: [ ]

### Tier 3: Implement This Month

#### 8. PyMC Hierarchical Rugby Model
- **Tutorial**: https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html
- **Install**: `pip install pymc`
- **What**: Bayesian hierarchical model for rugby with team attack/defence parameters. Naturally handles small samples, provides uncertainty.
- **Integration**: New model in `src/models/bayesian.py`
- **Impact**: +1-2pp ROI, better uncertainty quantification
- **Effort**: 1 week
- **Status**: [ ]

#### 9. `flumine` - Betfair Trading Framework
- **Repo**: https://github.com/betcode-org/flumine
- **Install**: `pip install flumine`
- **What**: Full Betfair trading framework with paper trading, simulation, live execution
- **Integration**: Could automate Betfair-side of pipeline for exchange betting
- **Impact**: Opens Betfair as execution venue (bypasses bookmaker account limits)
- **Effort**: 1-2 weeks
- **Status**: [ ]

#### 10. `tsfresh` - Automated Time Series Features
- **Repo**: https://github.com/blue-yonder/tsfresh
- **Install**: `pip install tsfresh`
- **What**: Auto-extracts 1,200+ time series features. May find novel signals not in your 365 hand-crafted features.
- **Integration**: Feed player match history into tsfresh, select top features
- **Impact**: Uncertain (+0-3pp ROI), risk of overfitting
- **Effort**: 2-3 days
- **Status**: [ ]

#### 11. OddsHarvester - Historical Odds Scraping
- **Repo**: https://github.com/jordantete/OddsHarvester
- **What**: Scrape historical NRL odds from OddsPortal for richer multi-bookmaker backtesting
- **Impact**: Better backtest fidelity
- **Effort**: 1-2 days
- **Status**: [ ]

#### 12. Custom Profit-Maximizing Loss Function
- **Repo**: https://github.com/charlesmalafosse/sports-betting-customloss
- **What**: Train GBM to optimize betting profit, not log-loss. XGBoost/LightGBM both support custom objectives.
- **Integration**: Add custom objective in `src/models/gbm.py`
- **Impact**: +1-2pp ROI (trains model for the actual goal)
- **Effort**: 2-3 days
- **Status**: [x] src/models/custom_loss.py created

#### 13. Correlated Kelly Portfolio Optimization
- **Repo**: https://github.com/thk3421-models/KellyPortfolio
- **What**: Simultaneous correlated bet optimization (ATS bets in same match are correlated)
- **Integration**: Replace per-bet Kelly in `src/pipeline/bet_recommendations.py` with portfolio optimization
- **Impact**: +0.5-1pp ROI from proper correlation modeling
- **Effort**: 1 week
- **Status**: [ ]

### NRL-Specific GitHub Repos to Study

| Repo | URL | Relevance | Status |
|------|-----|-----------|--------|
| beauhobba/NRL-Data | https://github.com/beauhobba/NRL-Data | NRL scraper + TF models, **has ATS notebook** | [ ] |
| brandonfalconer/NRLPredictionModel | https://github.com/brandonfalconer/NRLPredictionModel | NRL Elo ratings system | [ ] |
| DanielTomaro13/nrlR | https://github.com/DanielTomaro13/nrlR | NRL data back to 1998 | [ ] |
| greerreNFL/nfelo | https://github.com/greerreNFL/nfelo | Market-informed Elo (transferable pattern) | [ ] |
| martineastwood/penaltyblog | https://github.com/martineastwood/penaltyblog | Dixon-Coles model (adapt goals -> tries) | [ ] |

### Academic Papers Worth Implementing

1. **Scoring Patterns in Rugby League** - Poisson models for NRL try distribution
2. **In-Game Win Probabilities for NRL** - Live probability modeling
3. **Moskowitz: "Asset Pricing and Sports Betting"** - Validates momentum/fade strategies
4. **Levitt (2004)** - Why bookmakers take positions (exploit their biases)

---

## Part 3: Critical Gaps to Fix First (Before Any New Tools)

These are **blocking issues** that limit profitability regardless of new tools:

### Gap 1: Outcome Ingestion Pipeline (P0)
- [x] `ingest_outcomes_and_clv()` added to weekly_pipeline.py
- [x] Fetches completed match data via Champion Data API
- [x] Evaluates predictions vs actuals, computes P&L
- **Status**: DONE

### Gap 2: Opening Odds Snapshot (P1)
- [ ] Infrastructure exists but only closing odds captured
- [ ] Line movement feature (opening - closing) can't be calculated
- **Fix**: Call odds API at market open (~6pm day before) AND at close
- **Agent**: Backend Architect + DevOps Automator (for scheduling)

### Gap 3: Automated Weekly Pipeline (P1)
- [ ] Currently manual `python scripts/run_weekly_pipeline.py`
- [ ] Miss a week = miss profitable bets
- **Fix**: launchd/cron job: Tuesday 10am AEDT (post-outcomes), Wednesday 6pm (odds snapshot)
- **Agent**: DevOps Automator

### Gap 4: CLV Tracking Wiring (P1)
- [x] `record_clv()` now called inside `ingest_outcomes_and_clv()`
- [x] Wired to adaptive Kelly via existing `_get_clv_kelly_multiplier()`
- **Status**: DONE

---

## Part 4: Estimated Cumulative ROI Impact

| Change | Est. ROI Impact | Effort | Cumulative | Status |
|--------|----------------|--------|------------|--------|
| Baseline (current) | +19.7% | - | +19.7% | Done |
| Outcome ingestion + CLV | +2-3pp | 2-3 days | +22% | [ ] |
| Shin devigging | +0.5-1pp | 2 hours | +23% | [ ] |
| Weather features | +0.5-1pp | 1 day | +24% | [ ] |
| Optuna hyperparams | +1-2pp | Half day | +25.5% | [ ] |
| MAPIE confidence filter | +1pp | Half day | +26.5% | [ ] |
| Custom profit loss fn | +1-2pp | 2-3 days | +28% | [ ] |
| PyMC Bayesian model | +1-2pp | 1 week | +29.5% | [ ] |
| Opening odds + line movement | +0.5-1pp | 1 day | +30% | [ ] |
| **Realistic target** | | | **+25-30%** | |

*Note: These are estimates with significant uncertainty. Diminishing returns likely. Some features may not stack additively.*

---

## Verification Plan

- [ ] After each tool integration, re-run walk-forward backtest on 2024+2025
- [ ] Verify ROI improvement is positive on BOTH seasons independently
- [ ] Run `pytest tests/test_leakage.py` after every feature addition
- [ ] Track actual 2026 P&L separately from backtested ROI
- [ ] Compare model predictions to Betfair closing prices (CLV) weekly

---

## Recommended Execution Order

1. [ ] **Now**: Run Reality Checker agent on current ROI claims
2. [ ] **Day 1**: Data Engineer agent - outcome ingestion + CLV wiring
3. [ ] **Day 1**: Install `shin`, `optuna` - quick integrations
4. [ ] **Day 2**: AI Engineer agent - Optuna hyperparameter search, MAPIE confidence intervals
5. [ ] **Day 3**: Backend Architect agent - opening odds dual-snapshot design
6. [ ] **Day 3**: Add weather features (`weather-au`)
7. [ ] **Week 2**: AI Engineer agent - custom profit loss function, PyMC Bayesian model
8. [ ] **Week 2**: DevOps Automator agent - automated scheduling
9. [ ] **Week 3**: Experiment Tracker agent - A/B test configs on live 2026 data
10. [ ] **Ongoing**: Data Analytics Reporter - weekly P&L reports
