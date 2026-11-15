#!/usr/bin/env bash
# ============================================================
# track-dvc-git.sh — DVC Initialization, Data Tracking & Git Commit
# ============================================================

set -u  # error on unset variables; do NOT use -e (matches .bat's permissive flow)

LOG="dvc-setup.log"
echo "=== DVC & Git process started at $(date '+%Y-%m-%d %H:%M:%S') ===" > "$LOG"

# ---------- Helper subroutines (dual logging: console + log file) ----------
Log() {
    echo "$1"
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG"
}

LogHeader() {
    echo "============================================================"
    echo " $1"
    echo "============================================================"
    echo "============================================================" >> "$LOG"
    echo "[$(date '+%H:%M:%S')]  $1" >> "$LOG"
    echo "============================================================" >> "$LOG"
}

LogHeader "FTL Syria Crop Yield - DVC & Git Tracking"
Log "Working dir: $(pwd)"
Log "Log file:    $LOG"
Log ""

# Check if virtual environment exists to run dvc safely
VENV_DVC=".venv/bin/dvc"
if [ -x "$VENV_DVC" ]; then
    DVC_RUN="$VENV_DVC"
    Log "Using virtual environment DVC executable."
else
    DVC_RUN="dvc"
    Log "Virtual environment DVC not found. Falling back to global command."
fi
Log ""

# 1. DVC Init
Log "[1/4] Initializing DVC (--no-scm) ..."
if [ ! -d ".dvc" ]; then
    "$DVC_RUN" init --no-scm
    Log "   + DVC initialized successfully."
else
    Log "   .dvc folder already exists - skipping init."
fi
Log ""

# 2. DVC Add data/raw
Log "[2/4] Tracking raw data with DVC (dvc add data/raw) ..."
if [ -d "data/raw" ]; then
    "$DVC_RUN" add data/raw
    Log "   + data/raw added to DVC successfully."
else
    Log "   [WARN] data/raw folder not found. Skipping dvc add."
fi
Log ""

# 3. Git Add
Log "[3/4] Staging changes in Git (git add .) ..."
git add .
Log "   + All project changes and DVC pointer files staged."
Log ""

# 4. Git Commit
Log "[4/4] Committing changes to Git ..."
git commit -m "chore: initial project bootstrap structure and dvc tracking"
if [ $? -ne 0 ]; then
    Log "   [INFO] No new changes to commit or commit already made."
else
    Log "   + Initial Git commit created successfully."
fi
Log ""

LogHeader "[DONE] DVC data tracking and Git commit completed."
exit 0