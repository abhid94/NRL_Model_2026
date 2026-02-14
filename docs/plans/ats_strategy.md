# ATS Betting Edge Discovery — Strategic Analysis

> This document captures the strategic analysis behind feature selection and model design for the NRL ATS betting pipeline.
> It should be read alongside [`roadmap.md`](roadmap.md) for execution details.

**Last Updated**: 2026-02-14

---

## 1. The Fundamental Equation

```
P(player scores ATS) = f(expected_team_tries, player_share_of_tries, edge_matchup_modifier)
```

**Component 1 dominates everything.** When a team scores 5+ tries, wing try rates jump to ~65% vs ~27% when the team scores 0-2 tries. This single variable — how many tries will this team score in this match? — explains most of the variance in individual try probability.

---

## 2. Where Market Inefficiencies Exist (Validated on 2024-2025 Data)

| Finding | Evidence | Exploitability |
|---------|----------|---------------|
| Betfair overprices forwards by 3-5pp | Middle actual 10.0% vs implied 13.3%; Bench actual 8.6% vs implied 13.6% | **Don't bet forwards** — dead money for punters |
| Form regression is strong | "Hot" players (40%+ H1) drop 10.4pp in H2; cross-season regression -12.5pp | **Fade recent hot streaks**, especially on forwards |
| Edge attack patterns are real | Titans 64.2% left-edge tries, Tigers 38.5%; meaningful team-level variation | **Edge matchup analysis** can identify underpriced backs |
| Team tries context is underpriced | Backs vs weak defence: 48.5% try rate; vs strong defence: 32.8% (15.7pp swing) | **Team-level game context** is the biggest signal the market may not fully price |
| Back-row at high implied prob is +EV | +11.2% ROI in 2025 for back-rowers priced >30% implied | Niche but consistent across both seasons |

### How Much Edge Do We Need?

Bookmaker margins are 15-20%. If a player's true probability is 30%, the bookie prices at ~35% implied ($2.85). We need our model probability to exceed the bookie's implied by **>5pp** (our stated threshold) to bet.

**Realistic ROI target: 5-8% on bets placed, 2-4% overall.**

---

## 3. Missing Feature Analysis (Ranked by Profit Impact)

### Tier 1 — Build First (Highest Impact)

**1. Expected Team Tries**
- The single most predictive feature. Must combine Team A's attack strength with Team B's defensive weakness.
- Currently: `team_features.py` computes attack/defence stats independently but never creates the INTERACTION.
- Need: `expected_team_tries = f(team_attack_rolling_5, opponent_defence_rolling_5, is_home)`

**2. Player Try Share**
- What fraction of their team's tries does this player score? Relatively stable, captures individual ability within team context.
- `player_try_share = rolling_player_tries / rolling_team_tries`
- Also: position-level try share per team.

**3. Position Group & Starter Status**
- `player_features.py` does NOT include jumper_number, position_code, or is_starter.
- Position explains ~60% of try rate variance (Wing 47.8% vs Prop 8.6%).
- `is_starter` (jersey 1-13) is a proxy for minutes: starters play 65-80 min, bench 20-40 min.

### Tier 2 — Build Second (High Impact)

**4. Edge-Specific Attack/Defence**
- Teams have measurably different left/right edge attack patterns (validated in data).
- Edge mapping: Left = jerseys 2,3,11; Right = jerseys 4,5,12; Middle = jerseys 8,9,10,13
- Need: `team_left_edge_try_pct_rolling_5`, `conceded_to_left_edge_rolling_5`
- Then: `edge_matchup_score = attacking_edge_strength * defending_edge_weakness`

**5. Opponent Defensive Context for Each Player**
- `team_features.py` computes defence stats per team, but these aren't joined to each player row.
- Need: for each player, join their OPPONENT's rolling defensive stats as features.

**6. Betfair Implied Probability**
- `src/odds/` directory doesn't exist yet. Need Betfair prices as both a feature and benchmark.

### Tier 3 — Build Third (Medium Impact)

**7. Teammate Playmaking Quality** — rolling try assists of halves/fullback in the player's team
**8. Score Flow Features** — less useful for pre-match prediction, lower priority
**9. Cross-Season Priors** — player's prior season try rate as a feature for early-season rounds

---

## 4. Match Evaluation Framework

### Stage 1 — Team-Level: How many tries will each team score?

```
expected_tries_A = f(A_attack_rating, B_defence_rating, is_home_A, venue)
expected_tries_B = f(B_attack_rating, A_defence_rating, is_home_B, venue)
```

Uses: rolling team attack metrics (tries, line breaks, tackle breaks, run metres) + opponent's rolling defensive metrics (tries conceded, missed tackles, completion rate).

### Stage 2 — Edge-Level: Where will the tries come from?

```
left_edge_tries_A = expected_tries_A * A_left_edge_try_share * B_right_edge_weakness_modifier
right_edge_tries_A = expected_tries_A * A_right_edge_try_share * B_left_edge_weakness_modifier
middle_tries_A = expected_tries_A * A_middle_try_share * B_middle_weakness_modifier
```

Uses: team edge attack profiles + opponent edge defensive profiles.

### Stage 3 — Player-Level: Who scores?

```
lambda_player = position_expected_tries * player_skill_modifier * form_modifier * minutes_modifier
P(ATS) = 1 - exp(-lambda_player)   # Poisson-inspired
```

Uses: player try share within team, recent form, matchup history, is_starter.

### Stage 4 — Edge Detection: Is it a bet?

```
edge = model_prob - bookmaker_implied_prob
if edge > 5pp AND position in [backs, halves, back_row]: BET
stake = 0.25 * kelly_fraction * bankroll, capped at 5%
```

---

## 5. Bet Selection Rules

1. **Only bet backs, halves, back-rowers** — market systematically overprices forwards
2. **Focus on lopsided matchups** — when strong team faces weak defence, backs are underpriced
3. **Edge matchup filter** — when team's preferred edge faces weak opponent edge, those players get boosted
4. **Fade hot forwards** — a prop who scored 2 tries in 3 games regresses hard
5. **Max 4 ATS bets per match** (correlation risk), max 15 per round

---

## 6. Round-by-Round Approach

- **Rounds 1-3**: Rely on prior-season data + position priors. Be conservative (smaller stakes).
- **Rounds 4-8**: 3-game windows become meaningful. Start ramping up.
- **Rounds 9-15**: All windows populated. Most reliable period. Be most aggressive.
- **Rounds 13-14, 19-20**: State of Origin disruption. Great opportunity (weakened teams = underpriced backs on full-strength teams).
- **Rounds 21+**: Full windows but late-season tactical changes. Maintain discipline.

---

## 7. What Makes This Profitable vs Just Accurate

The model doesn't need to be perfect — it needs to be **directionally better than bookmakers in specific, identifiable segments**:

1. **Backs on dominant teams in lopsided matchups** (team tries context underpriced)
2. **Edge players in favourable edge matchups** (structural signal bookmakers don't granularly price)
3. **Players whose recent form regresses** (bookmakers overweight recency, model uses base rates)

---

## 8. Verification Plan

1. Run feature store build end-to-end, verify one row per (match_id, player_id)
2. Run `pytest tests/test_leakage.py` — all features use only rounds < R
3. Walk-forward backtest on 2024 (train rounds 1-R, predict R+1)
4. Out-of-sample validation on 2025
5. Measure: AUC, Brier score, calibration error (<0.03), ROI, CLV
6. Segment analysis: ROI by position group, by edge matchup type, by team tries bucket
