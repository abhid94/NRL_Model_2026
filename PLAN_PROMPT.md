You are Claude Code (Opus 4.6) operating in PLANNING MODE.

Your task: design, from scratch, a data-driven NRL analytics and modelling pipeline whose primary objective is discovering **profitable betting edges** for upcoming NRL seasons (starting with 2026). This is a clean-slate planning exercise. Do NOT code. Do NOT assume heuristics. Do NOT optimize prematurely. Build a structured plan, reason incrementally, and validate each layer before moving on.

=====================================================
PROJECT GOAL
=====================================================
Primary objective:
- Build a robust analytics foundation for discovering, testing, and validating betting edges in NRL markets, with the explicit goal of maximizing profit in the 2026 season.

Secondary objectives:
- Serve as a sandbox for hypothesis testing and backtesting
- Be extensible across seasons (2025 base dataset, 2024 for validation)
- Support **both player-level and team-level** analysis as first-class inputs

Constraints:
- No UI required
- No live odds integration for now
- No scraping (data already ingested)
- Python + SQLite
- Visuals optional
- Incremental design with validation checks at each stage
- The database will be updated weekly during the 2026 season as each round completes; the model must be retrained with the most up-to-date data each week
- Emulations on 2024/2025 data must be done round-by-round with strict leakage prevention (no data creep from future rounds)
- The **only odds data available** is Betfair closing markets; the primary market to target is `TO_SCORE` / "To Score A Try" (Anytime Try Scorer). `FIRST_TRY_SCORER` exists but is not the main focus.

You are acting as:
- A data scientist (data modelling, feature engineering, validation)
- A betting analyst (market intuition, edge discovery, robustness)

=====================================================
DATABASE OVERVIEW (SQLite)
=====================================================
The database file will live at `data/nrl_data.db` in the project root.
The database uses **year-suffixed tables** (e.g., `players_2024`, `players_2025`) for most seasonal data.

Core tables (non-suffixed):
- `teams`

Year-suffixed tables:
- `players_2024`, `players_2025`
- `matches_2024`, `matches_2025`
- `player_stats_2024`, `player_stats_2025`
- `team_stats_2024`, `team_stats_2025`
- `score_flow_2024`, `score_flow_2025`
- `fixtures_2025`
- `team_lists_2025`

Odds/market tables exist but are **out of scope for initial modelling design**:
- `betfair_markets_2024`, `betfair_markets_2025`
- `bookmaker_odds_2025`
- Mapping helpers: `betfair_player_mapping`, `betfair_team_mapping`

Operational/ingestion tables to ignore for modelling:
- `match_reports`, `match_reports_2024`, `ingested_matches_2024`, `ingested_matches_2025`

-----------------------------------------------------
TABLE: teams
-----------------------------------------------------
Columns:
- squad_id (PK)
- squad_name
- squad_nickname
- squad_code

Usage:
- Master team reference; join key for team-level data

-----------------------------------------------------
TABLES: players_2024 / players_2025
-----------------------------------------------------
Columns:
- player_id (PK)
- firstname
- surname
- display_name
- short_display_name

Usage:
- Master player reference per season

-----------------------------------------------------
TABLES: matches_2024 / matches_2025
-----------------------------------------------------
Columns:
- match_id (PK)
- match_number
- round_number
- match_type
- match_status
- utc_start_time
- local_start_time
- home_squad_id
- away_squad_id
- venue_id
- venue_name
- venue_code
- period_completed
- period_seconds
- final_code
- final_short_code

Usage:
- Canonical match metadata
- Join point for team/player stats, score flow

-----------------------------------------------------
TABLE: fixtures_2025
-----------------------------------------------------
Columns:
- match_id (PK)
- round
- match_name
- local_start_time
- venue
- home_team_id
- home_team_name
- away_team_id
- away_team_name
- match_status

Usage:
- Fixture-oriented view of matches
- Future match planning; cross-check completeness

-----------------------------------------------------
TABLE: team_lists_2025
-----------------------------------------------------
Columns:
- match_id
- round_number
- squad_id
- squad_name
- player_name
- player_id
- jersey_number
- position

Usage:
- Official team lineups per match
- Position-based modelling; selection effects

-----------------------------------------------------
TABLES: player_stats_2024 / player_stats_2025
-----------------------------------------------------
Columns include (non-exhaustive):
- match_id
- player_id
- squad_id
- position
- jumper_number
- tries, try_assists
- line_breaks, line_break_assists, tackle_breaks
- run_metres, post_contact_metres, metres_gained
- kicks_general_play, kick_metres
- tackles, missed_tackles
- errors, passes, possessions
- penalties_conceded
- conversions, conversion_attempts
- sin_bins, on_reports, sent_offs
- plus additional kicking and set-play stats

Usage:
- Player form modelling
- Try scorer analysis
- Attacking vs defensive contributions
- Positional and role-based matchup analysis

-----------------------------------------------------
TABLES: team_stats_2024 / team_stats_2025
-----------------------------------------------------
Columns include:
- match_id
- squad_id
- score
- possession_percentage
- completion_rate_percentage
- run_metres, post_contact_metres, metres_gained
- tackles, missed_tackles
- errors, penalties_conceded
- tries, line_breaks, tackle_breaks
- kick metrics, set stats

Usage:
- Team form and dominance modelling
- Defensive strength / weakness
- Game control, tempo, territory

-----------------------------------------------------
TABLES: score_flow_2024 / score_flow_2025
-----------------------------------------------------
Columns:
- match_id
- period
- squad_id
- player_id
- score_name
- score_points
- period_seconds

Usage:
- Event-level scoring timeline
- Momentum analysis; lead changes
- Player scoring attribution

=====================================================
WHAT YOU SHOULD PRODUCE (PLANNING OUTPUT)
=====================================================
Produce a structured, reasoned plan that **builds the pipeline in layers**. You are encouraged to explore and reason independently beyond the minimum requirements, but stay within the constraints.

Required outputs:
1. **Conceptual data model**
   - How tables relate
   - Analytical layers to sit on top (derived tables, aggregates)

2. **Proposed analytical dimensions** (expand as needed)
   - Player form
   - Team form
   - Matchups
   - Selection effects
   - Role / position effects
   - Contextual game state

3. **Validation checks**
   - Data completeness
   - Grain integrity
   - Leakage prevention
   - Sanity checks for modelling inputs

4. **Roadmap**
   - Phase 1: data understanding & derived metrics
   - Phase 2: exploratory analysis
   - Phase 3: backtesting frameworks
   - Phase 4: modelling candidates (no implementation yet)

=====================================================
PROFIT MAXIMISATION FOCUS
=====================================================
- The end goal is to maximize **risk-adjusted profit** over the 2026 season and make as much money as possible.
- Any strategy proposal should explicitly define: expected edge, staking/risk control, and how profit is measured.
- Use Betfair closing prices for the `TO_SCORE` / "To Score A Try" market as the baseline for edge evaluation and backtesting.

=====================================================
ODDS INTEGRATION (EARLY)
=====================================================
- Incorporate `TO_SCORE` closing prices early in the planning pipeline to ensure features and models target **edge vs price**, not just predictive accuracy.
- Treat `FIRST_TRY_SCORER` markets as a separate, secondary path unless a clear rationale emerges to prioritize them.

=====================================================
GUIDANCE
=====================================================
- Stay in planning mode. Do not write production code.
- Do not assume betting markets or choose odds sources yet.
- Avoid premature optimization or heuristic shortcuts.
- Explicitly call out uncertainties and data risks.
- Prioritize clarity, traceability, and extensibility.
- If you need schema confirmation, ask for DB inspection steps as part of the plan.

=====================================================
DELIVERABLE
=====================================================
- Save your full plan to a Markdown file (no code), and ensure it can stand alone for future reference.
- Create an extensive `CLAUDE.md` file that sets project principles, scope, quality gates, weekly retraining protocol, leakage-prevention rules, workspace hygiene rules (keep the workspace clean; delete temporary files once they are no longer needed), and a checklist to keep the project on track.
