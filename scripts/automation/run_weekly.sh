#!/bin/bash
#
# NRL ATS Weekly Automation Wrapper
#
# Detects the current round from the database and runs either:
#   - "ingest" mode: ingest outcomes for the LAST completed round
#   - "predict" mode: predict for the NEXT upcoming round
#
# Usage:
#   ./scripts/automation/run_weekly.sh ingest
#   ./scripts/automation/run_weekly.sh predict
#   ./scripts/automation/run_weekly.sh predict --bankroll 12000
#
# Install as crontab:
#   # Tuesday 10am AEDT: ingest outcomes
#   0 10 * * 2 cd /Users/abhidutta/Documents/repos/NRL_2026_Model && ./scripts/automation/run_weekly.sh ingest >> data/logs/cron.log 2>&1
#   # Thursday 6pm AEDT: predict next round
#   0 18 * * 4 cd /Users/abhidutta/Documents/repos/NRL_2026_Model && ./scripts/automation/run_weekly.sh predict >> data/logs/cron.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
SEASON=2026

cd "$PROJECT_DIR"

# Create log directory
mkdir -p data/logs

MODE="${1:-predict}"
shift 2>/dev/null || true

# Detect current round from database
LAST_COMPLETED=$($PYTHON -c "
import sqlite3, sys
sys.path.insert(0, '.')
from src.config import DB_PATH
conn = sqlite3.connect(str(DB_PATH))
try:
    # Find the highest round with ingested match data
    row = conn.execute('''
        SELECT MAX(m.round_number)
        FROM matches_${SEASON} m
        JOIN ingested_matches_${SEASON} i ON m.match_id = i.match_id
    ''').fetchone()
    print(row[0] if row and row[0] else 0)
except:
    print(0)
finally:
    conn.close()
" 2>/dev/null)

NEXT_ROUND=$((LAST_COMPLETED + 1))

echo "$(date '+%Y-%m-%d %H:%M:%S') — Mode: $MODE, Season: $SEASON, Last completed: R$LAST_COMPLETED, Next: R$NEXT_ROUND"

if [ "$MODE" = "ingest" ]; then
    echo "Ingesting outcomes for Round $LAST_COMPLETED..."
    $PYTHON scripts/ingest_outcomes.py --season $SEASON --round $LAST_COMPLETED "$@"
elif [ "$MODE" = "predict" ]; then
    echo "Predicting for Round $NEXT_ROUND..."
    $PYTHON scripts/run_weekly_pipeline.py --season $SEASON --round $NEXT_ROUND "$@"
else
    echo "Unknown mode: $MODE (use 'ingest' or 'predict')"
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') — Done."
