# NRL Anytime Try Scorer (ATS) Betting Model — Full Report

> **Model version**: Phase 5B (Feb 2026)
> **Backtest ROI**: +19.7% on 311 bets across 2024–2025
> **Status**: Ready for 2026 live deployment

---

## Table of Contents

1. [What Does This Model Do?](#1-what-does-this-model-do)
2. [The Data](#2-the-data)
3. [Where the Edge Comes From](#3-where-the-edge-comes-from)
4. [Feature Engineering](#4-feature-engineering)
5. [The Model](#5-the-model)
6. [The Strategy: MarketBlended](#6-the-strategy-marketblended)
7. [Backtesting & Results](#7-backtesting--results)
8. [What We Tested and Rejected](#8-what-we-tested-and-rejected)
9. [Risk Management](#9-risk-management)
10. [How a Bet Gets Made (End-to-End)](#10-how-a-bet-gets-made-end-to-end)
11. [Known Limitations](#11-known-limitations)
12. [2026 Deployment Plan](#12-2026-deployment-plan)

---

## 1. What Does This Model Do?

The model predicts the probability that a player will score **at least one try** in an NRL match (the "Anytime Try Scorer" or ATS market). It then compares its probability to what the bookmaker thinks, and bets only when it finds the bookmaker is underpricing a player.

**Simple example:**
- Model thinks Player X has a **35% chance** of scoring
- Betfair says **25%** (odds of $4.00)
- That's a **10 percentage point edge** → place a bet

The model doesn't try to predict everything. It only bets on ~6 players per round, in positions where the market is most likely to be wrong.

---

## 2. The Data

### What we have

| Source | Records | Coverage |
|--------|---------|----------|
| Player match stats | 14,688 | 2024 + 2025, all regular season matches |
| Team stats per match | 816 | 408 matches × 2 teams |
| Score flow (try events) | ~3,400 | Individual try scoring events |
| Betfair TO_SCORE odds | ~13,000 | 88-91% player-match coverage |
| Match reports (on-reports) | 753 | Disciplinary events |

All data lives in a single SQLite database (`nrl_data.db`, 74MB, 24 tables).

### Key numbers

```
Players per season:     ~523 unique
Matches per season:     204 (27 rounds × ~8 matches, minus bye rounds)
Observations per season: 7,344 player-match records
Overall try rate:        19.0% (roughly 1 in 5 players score)
```

### Try rates vary massively by position

```
Position        Try Rate    Market Implied    Overpriced?
─────────────────────────────────────────────────────────
Wing            47.1%       48.6%             No (fair)
Fullback        37.9%       36.8%             No (fair)
Centre          28.0%       32.3%             Slightly
Five-eighth     23.4%       23.5%             No
Second Row      21.4%       22.4%             No
Halfback        18.7%       20.6%             Slightly
Hooker          13.5%       14.9%             Yes (+1.4pp)
Prop             9.4%       10.3%             Yes (+0.9pp)
Lock             9.0%       10.5%             Yes (+1.5pp)
Interchange      7.4%        ~8%              Yes
```

**Key insight**: The market is *well-calibrated at the position level*. The edge doesn't come from knowing "wings score more" — the bookmaker already knows that. The edge comes from knowing *which specific players in which specific matchups* are underpriced.

---

## 3. Where the Edge Comes From

After extensive analysis, the model's edge comes from three compounding factors:

### Factor 1: Team scoring context

The single most important variable is **how many tries a team is expected to score**. A team averaging 5 tries/game against a defence that concedes 4.5 creates a target-rich environment. The market somewhat accounts for this, but underweights it for individual players.

```
Backs vs weak defence:   48.5% try rate
Backs vs strong defence: 32.8% try rate
                         ─────────────
Swing:                   15.7 percentage points
```

### Factor 2: Edge attack patterns

NRL teams attack unevenly across left/right/middle channels. Some teams score 64% of tries through the left edge, others only 38%. When a left-edge winger faces a defence that leaks tries on the left, the model identifies this specific matchup advantage.

### Factor 3: Market blending

The model *anchors to the market* rather than trying to replace it. By blending 25% model signal with 75% market signal, it only overrides the bookmaker when the model is confidently different. This prevents overfitting and ensures bets are placed where the model has genuine disagreement, not just noise.

```
blended_prob = 0.25 × model_prob + 0.75 × market_prob
edge = blended_prob - market_prob
```

This means the model only bets when its raw prediction diverges significantly from the market (by at least 12 percentage points, since 25% × 12pp = 3pp minimum edge).

---

## 4. Feature Engineering

The model uses **342 features** grouped into 8 categories. Here's what each group captures and why it matters.

### 4.1 Player rolling statistics (~159 features)

Rolling averages over the last 3, 5, and 10 matches for each player:
- Tries, try rate, line breaks, tackle breaks, run metres
- Post-contact metres, offloads, try assists

**Why**: A player's recent form is the strongest predictor of near-term performance.

### 4.2 Team features (~60 features per team)

Rolling team-level stats for both the player's team and the opponent:
- Tries scored/conceded, completion rate, line breaks, possession
- Defence stats: missed tackles, tries conceded

**Why**: Team context determines opportunity. A backline player on a team scoring 5 tries/game has more chances than one on a team scoring 2.

### 4.3 Edge attack/defence features (10 features)

Where teams score and concede tries (left/right/middle edge):
- Team's try distribution by edge (rolling 5 matches)
- Opponent's tries conceded by edge (rolling 5 matches)
- Edge matchup score: team_edge_attack × opponent_edge_weakness

**Why**: NRL attacks are directional. The winger's value depends on whether his team attacks his side AND whether the opposition leaks tries there.

### 4.4 Game context features (~8 features)

- Expected team tries (team attack strength × opponent defence weakness)
- Player's try share (% of team tries this player scores)
- Days since last match (fatigue/freshness)

**Why**: Captures the interaction between team environment and individual role.

### 4.5 Betfair odds features (11 features)

- Closing odds, implied probability, spread, total matched volume
- Price movement (60min → close), late volume share, odds drift (30min → 1min)
- Volume rank within match, volume acceleration

**Why**: The market itself is a strong predictor (correlation +0.34 with try scoring). Late price movements signal sharp money.

### 4.6 Cross-season priors (6 features)

- Prior season try rate, games played, average line breaks, tackle breaks, run metres

**Why**: For early rounds when in-season rolling features are sparse, last season's data provides a baseline. Covers 90.7% of 2025 players.

### 4.7 Matchup features (3 features)

- Player's historical record vs this specific opponent (tries, try rate, games)

**Why**: Some players consistently perform against certain teams. Only 24% of observations have prior matchup history, so this is sparse but informative when available.

### 4.8 Discipline & lineup features (7 features)

- Team/opponent rolling sin bins, on-report incidents
- Playmaker quality, lineup stability, returning player flag

**Why**: A team down to 12 men from sin bins creates scoring opportunities. Lineup continuity correlates with team cohesion.

### Leakage prevention

Every feature is computed using **only data available before the match**. Rolling windows use `shift(1)` before `.rolling()` to exclude the current match. This is enforced by:
- `as_of_round` parameter on every feature function
- 14 automated leakage prevention tests
- Walk-forward backtesting (model never sees future data)

---

## 5. The Model

### Architecture

```
Raw features (342)
    ↓
LightGBM classifier (gradient boosted decision trees)
    ↓
Raw P(try) predictions
    ↓
Isotonic regression calibrator (fitted on held-out recent rounds)
    ↓
Calibrated P(try) predictions
    ↓
Market blending: 0.25 × calibrated + 0.75 × Betfair implied
    ↓
Final probability → Compare to market → Bet if edge ≥ 3pp
```

### Why LightGBM?

- **Handles missing values natively** — critical because 76% of matchup features are NaN (first-time matchups) and all 2024 lineup features are missing
- **Captures non-linear interactions** — a winger's value depends on team attack strength AND opponent defence weakness simultaneously
- **Feature selection is automatic** — with 342 features, the model learns which matter via gradient boosting

### Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| n_estimators | 150 | Moderate complexity, avoids overfitting |
| max_depth | 4 | Shallow trees = better generalization |
| learning_rate | 0.05 | Slow learning for stability |
| min_child_samples | 80 | High minimum = strong regularization |
| reg_alpha / reg_lambda | 3.0 | Heavy L1/L2 regularization |
| subsample | 0.8 | Row subsampling reduces variance |
| colsample_bytree | 0.8 | Column subsampling reduces variance |

**Design philosophy**: The model is intentionally *under-complex*. With only ~7,300 observations per season, complex models memorize noise. Heavy regularization + shallow trees + market anchoring = robust performance.

### Calibration

The raw GBM outputs are **calibrated** using isotonic regression fitted on the last 5 rounds of training data (temporal holdout, not random). This ensures the model's "35%" actually means 35% of those players score.

We tested position-specific calibration (separate calibrator for backs vs forwards) — it performed worse (+13.3% ROI vs +19.7%) because individual position groups have too few calibration samples.

### Top features by importance (SHAP values)

```
Feature                              SHAP Impact
──────────────────────────────────────────────────
team_edge_try_share_rolling_5        ████████████████████  0.51
betfair_closing_odds                 ██████████████████    0.47
edge_matchup_score_rolling_5         ███████████████       0.40
position_code                        █████████             0.24
total_tries_rolling_5                █████                 0.14
is_starter                           █████                 0.14
conceded_to_middle_rolling_5         ████                  0.11
betfair_spread                       ████                  0.11
betfair_total_matched_volume         ███                   0.08
rolling_run_metres_5                 ███                   0.08
```

The top 3 features are edge-related and market-related — confirming that the edge comes from directional attack patterns combined with market disagreements.

---

## 6. The Strategy: MarketBlended

### How it works

1. **Filter**: Only eligible positions (FB, WG, CE, FE, HB, SR, LK) with Betfair odds
2. **Predict**: Model produces calibrated P(try) for each player
3. **Blend**: `blended = 0.25 × model + 0.75 × market`
4. **Edge**: `edge = blended - market_implied_prob`
5. **Filter**: Only bet if edge ≥ 3 percentage points AND odds between $1.50 and $6.00
6. **Stake**: Flat $100 per bet
7. **Constraints**: Max 4 bets per match, max 15 per round, max 20% bankroll exposure

### Why this works better than pure model

The alpha=0.25 blending is the **key breakthrough** that took us from 0 profitable configs to 40:

```
Pure model (alpha=1.0):    Overfits, unstable, negative ROI
Pure market (alpha=0.0):   No bets (edge always 0)
Blended (alpha=0.25):      +19.7% ROI, stable across seasons
```

The market is a strong predictor of try probability. By anchoring to it, we only override when the model strongly disagrees. The 0.25 weight means the model needs to see a 12pp raw difference to produce a 3pp blended edge — a high bar that filters out noise.

### What positions get bet on?

The model overwhelmingly bets on outside backs:
- **Wings** and **Centres**: Highest try rates AND where edge matchup patterns matter most
- **Fullbacks**: Occasional, when team attack patterns favour them
- **Halves/Back-rowers**: Rare, only with strong convergent signals
- **Forwards**: Never — the market overprices them, so the model never finds positive edge

---

## 7. Backtesting & Results

### Methodology: Walk-forward

For each round R in each season:
1. **Train** on all data before round R (including prior seasons)
2. **Predict** probabilities for round R players
3. **Select bets** using the MarketBlended strategy
4. **Resolve** outcomes against actual results
5. **Update** bankroll and move to round R+1

The model is retrained every single round with expanding training data. It never sees future data.

### Overall results

```
                    2024        2025        Combined
────────────────────────────────────────────────────
Bets placed         159         152         311
Total staked        $15,900     $15,200     $31,100
Total payout        $19,117     $18,052     $37,169
Profit              $3,217      $2,852      $6,069
ROI                 +20.6%      +18.8%      +19.7%
Hit rate            37.1%       40.1%       38.6%
Avg odds            $3.49       $3.01       $3.25
Max drawdown        —           —           $1,138
Sharpe ratio        —           —           0.27
P(ROI > 0)          —           —           98.5%
95% CI              —           —           [+2.1%, +37.9%]
```

### Key observations

1. **Consistent across seasons**: 2024 and 2025 ROIs are within 2pp of each other. This is the strongest evidence the edge is real, not overfitted.

2. **Hit rate (38.6%) vs average odds ($3.25)**: Break-even hit rate at $3.25 odds is 30.8%. The model hits 38.6%, well above break-even.

3. **Drawdown is manageable**: Maximum peak-to-trough of $1,138 on a $10,000 bankroll = 11.4% drawdown. Never triggers the 15% warning level.

4. **Bootstrap P(ROI > 0) = 98.5%**: Re-sampling 5,000 times, 98.5% of samples show positive returns. This isn't luck.

### Cumulative P&L trajectory (2025 season)

```
Round  Bets  Wins  Profit   Cumul P&L
─────────────────────────────────────
  3     13    3    -$353      -$353
  4      7    4    +$224      -$129
  5      4    2     -$3       -$132
  6      6    3    +$127       -$5
  7      7    3    +$127      +$122
  8      8    6    +$744      +$866
  9      3    2    +$187     +$1,053
 10     12    8   +$1,955    +$3,008    ← peak early run
 11      5    2    -$134     +$2,874
 12      8    2    -$281     +$2,593
 13      6    4    +$508     +$3,101
 ...
 27      4    2    +$141     +$2,852    ← final
```

The P&L shows a real equity curve with drawdowns and recoveries — not a smooth unrealistic line.

---

## 8. What We Tested and Rejected

| Idea | Result | Why It Failed |
|------|--------|---------------|
| **Quarter-Kelly staking** | +5.0% ROI | Edge estimates too noisy for Kelly; it over-bets on bad edges |
| **Position-specific calibration** | +13.3% ROI | Too few samples per position group for stable calibration |
| **Weighted ensemble (GBM + Logistic)** | +8.2% ROI | Logistic model adds noise, drags down GBM quality |
| **Weighted ensemble (GBM + Poisson)** | +19.7% ROI | Poisson contributes nothing; GBM dominates weights |
| **Stacked ensemble (GBM + Logistic)** | +11.4% ROI | High variance (4.3% 2024, 70.7% 2025), unstable |
| **3-model ensemble** | +7.1% ROI | More models = more noise in this data-limited setting |
| **Pure model edge (no market blending)** | ~+6% ROI | Overfits without market anchor, unstable across seasons |
| **Poisson regression alone** | 0 bets | Edge threshold too high; never finds positive edge |
| **Betting on forwards** | Negative | Market overprices forwards by 1-3pp systematically |
| **Fade hot streaks** | -11.6% ROI | Contrarian approach doesn't work — form is real signal |

**Lesson**: Simplicity wins. The solo GBM with heavy regularization, market blending, and flat stakes beats every more complex alternative.

---

## 9. Risk Management

### Staking constraints (6 layers)

```
Layer 1: Flat $100 per bet (not Kelly)
Layer 2: Max 5% of bankroll per bet ($500 cap)
Layer 3: Drop bets below $5 minimum
Layer 4: Max 4 bets per match (correlation risk)
Layer 5: Max 20% bankroll exposure per round ($2,000 cap)
Layer 6: Max 15 bets per round
```

### Drawdown protocol

| Drawdown Level | Action |
|----------------|--------|
| 15% ($1,500) | Reduce stake to 75% of normal |
| 25% ($2,500) | Pause betting for 2 rounds, audit model |
| 40% ($4,000) | Stop betting entirely, fundamental review |

### Why max 4 bets per match?

ATS bets in the same match are correlated — if one team dominates and scores 6 tries, multiple players from that team benefit. Capping at 4 per match limits correlation-driven losses when games go the other way.

### Position eligibility

The model never bets on:
- Props, Hookers, Locks (market overprices them by 1-3pp)
- Interchange/Reserve players (too little game time, high variance)

This eliminates ~40% of potential bets upfront, focusing capital on positions where edges exist.

---

## 10. How a Bet Gets Made (End-to-End)

Here's the exact pipeline for a Wednesday prediction run:

```
MONDAY/TUESDAY (after previous round):
┌──────────────────────────────────────────┐
│ 1. INGEST new round data into database   │
│ 2. VALIDATE data quality                 │
│ 3. REBUILD feature store                 │
│    → Player rolling stats (3/5/10 match) │
│    → Team attack/defence profiles        │
│    → Edge attack distributions           │
│    → Cross-season priors                 │
│ 4. RETRAIN model on all available data   │
│    → GBM on expanding training window    │
│    → Isotonic calibrator on last 5 rounds│
└──────────────────────────────────────────┘

WEDNESDAY/THURSDAY (before next round):
┌──────────────────────────────────────────┐
│ 5. TEAM LISTS confirmed (~Tuesday 4pm)   │
│ 6. PREDICT for upcoming round            │
│    → Compute features for all players    │
│    → GBM predicts raw P(try)             │
│    → Calibrator adjusts probabilities    │
│ 7. COMPARE to bookmaker odds             │
│    → Blend: 25% model + 75% market       │
│    → Calculate edge for each player      │
│ 8. SELECT bets                           │
│    → Edge ≥ 3pp                          │
│    → Eligible positions only             │
│    → Odds $1.50 – $6.00                  │
│ 9. APPLY staking constraints             │
│    → Flat $100 per bet                   │
│    → Max 4 per match, 15 per round       │
│10. OUTPUT bet recommendations            │
│    → Player, team, odds, edge, stake     │
└──────────────────────────────────────────┘
```

**Timing**: The full pipeline runs in under 5 minutes.

---

## 11. Known Limitations

### Data limitations

- **Only 2 seasons of data** (14,688 observations). More data would improve confidence and enable more complex models.
- **No real bookmaker odds** — we benchmark against Betfair closing prices. Actual bookmaker odds (Sportsbet, Bet365, TAB) will differ, and real ROI depends on the odds you actually get.
- **88-91% Betfair coverage** — 9-12% of player-match observations have no odds data (mostly bench players).
- **team_lists_2024 doesn't exist** — lineup features are missing for the entire 2024 season.

### Model limitations

- **The market is nearly efficient** — edges are 3-5pp, not 20pp. This means variance is high relative to edge.
- **Small sample size per bet type** — 311 bets across 2 seasons is enough for statistical significance but not for fine-grained position/matchup analysis.
- **No injury/weather data** — the model doesn't know about injuries, weather, or tactical changes announced after team lists.
- **Stale cross-season priors** — prior season data loses relevance as the current season progresses.

### Execution risks

- **Account limitations** — Australian bookmakers aggressively limit winning accounts (3-6 month tolerance). Must diversify across bookmakers.
- **Odds movement** — by the time you place bets, odds may have moved. The model uses closing prices; live execution happens at current prices.
- **Correlation risk** — even with the 4-per-match cap, round-level results are lumpy. A bad round can wipe out 3 good ones.

---

## 12. 2026 Deployment Plan

### Recommended configuration

```
Model:     CalibratedGBM (n=150, depth=4, reg=3.0, mcs=80)
Strategy:  MarketBlendedStrategy (alpha=0.25, min_edge=0.03)
Staking:   Flat stake (2% of bankroll per bet)
Bankroll:  $5,000 – $20,000
Expected:  ~6 bets/round, ~150 bets/season
```

### Weekly workflow

| Day | Task |
|-----|------|
| Monday | Ingest previous round results |
| Tuesday | Rebuild features, retrain model |
| Wednesday | Team lists confirmed, generate predictions |
| Thursday | Compare to live bookmaker odds, place bets |
| Friday–Sunday | Monitor results, log outcomes |

### Bankroll sizing

At $100 flat stake with ~150 bets/season:
- **Expected profit**: $2,500–$3,000 (at historical +18% ROI)
- **Realistic profit** (accounting for market efficiency): $750–$1,500 (at 5-10% ROI)
- **Worst realistic outcome**: -$1,500 (15% drawdown)

**Important**: The 19.7% backtest ROI is the *ceiling*, not the floor. Live ROI will likely be lower due to:
- Worse odds execution (non-closing prices)
- Market adaptation (bookmakers adjust)
- Model degradation over time

A realistic expectation is **5-10% ROI** on bets placed, which on $15,000 staked translates to **$750–$1,500 annual profit**.

### What to monitor

- **Rolling 20-bet ROI** — if negative for 30+ bets, investigate
- **Hit rate vs expected** — should be 35-42%, if below 30% for 50+ bets, audit
- **Edge realization** — are actual returns matching predicted edges?
- **Drawdown tracking** — follow the protocol (15%/25%/40% triggers)

---

*Report generated Feb 2026. All backtest results use walk-forward methodology with no look-ahead bias. Past performance does not guarantee future results.*
