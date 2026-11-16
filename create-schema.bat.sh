#!/usr/bin/env bash
# ============================================================
# create-schema.sh — project bootstrap (Linux / macOS)
# Safe to re-run: skips existing files and folders.
# Equivalent to 00-create-schema.bat
# ============================================================

set -u  # error on unset variables; do NOT use -e (we want to match .bat's permissive flow)

LOG="setup.log"
echo "=== Setup started at $(date '+%Y-%m-%d %H:%M:%S') ===" > "$LOG"

# ---------- Helper subroutines (dual logging: console + setup.log) ----------
Log() {
    echo "$1"
    echo "$1" >> "$LOG"
}

LogHeader() {
    echo "============================================================"
    echo " $1"
    echo "============================================================"
    echo "============================================================" >> "$LOG"
    echo " $1" >> "$LOG"
    echo "============================================================" >> "$LOG"
}

LogError() {
    echo "$1"
    echo "$1" >> "$LOG"
}

LogHeader "FTL Syria Crop Yield - Project Bootstrap"
Log "Working dir: $(pwd)"
Log "Log file:    $LOG"
Log ""

PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# ---------- 0. Preflight ----------
Log "[0/5] Preflight checks ..."

if ! command -v python >/dev/null 2>&1; then
    LogError "[ERROR] 'python' not found on PATH. Install Python 3.10+ and retry."
    exit 1
fi

PYVER=$(python --version 2>&1 | awk '{print $2}')
PYMAJ=$(echo "$PYVER" | cut -d. -f1)
PYMIN=$(echo "$PYVER" | cut -d. -f2)

if [ "$PYMAJ" -lt "$PYTHON_MIN_MAJOR" ]; then
    LogError "[ERROR] Python $PYVER found, but ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ required."
    exit 1
fi
if [ "$PYMAJ" -eq "$PYTHON_MIN_MAJOR" ] && [ "$PYMIN" -lt "$PYTHON_MIN_MINOR" ]; then
    LogError "[ERROR] Python $PYVER found, but ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ required."
    exit 1
fi
Log "   Python $PYVER OK."
Log ""

# ---------- 1. Directories ----------
Log "[1/5] Creating directories ..."
for D in \
    data data/raw data/interim data/processed \
    src src/data src/features src/models src/utils \
    models metrics notebooks tests \
    .github .github/workflows
do
    if [ ! -d "$D" ]; then
        mkdir -p "$D"
        Log "   + $D"
    fi
done
Log ""

# ---------- 2. Placeholder files ----------
Log "[2/5] Creating placeholder files ..."
for F in \
    data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep \
    models/.gitkeep metrics/.gitkeep \
    src/__init__.py src/data/__init__.py src/features/__init__.py \
    src/models/__init__.py src/utils/__init__.py \
    src/data/ingest.py src/data/build_dataset.py src/data/validate.py src/data/anomaly_check.py \
    src/features/build_features.py \
    src/models/train.py src/models/evaluate.py \
    src/utils/config.py \
    tests/__init__.py tests/test_data_validation.py \
    notebooks/01_eda.ipynb \
    .github/workflows/daily_retrain.yml \
    dvc.yaml params.yaml README.md
do
    if [ ! -e "$F" ]; then
        : > "$F"
        Log "   + $F"
    fi
done
Log ""

# ---------- 3. Seed .gitignore ----------
Log "[3/5] Checking .gitignore ..."
if [ ! -e ".gitignore" ]; then
    cat > ".gitignore" <<'EOF'
# files and folders starting with
.*
./*

# log files
*.log

# MLflow
mlflow.db
mlruns/
mlartifacts/

# Data (tracked by DVC, not Git)
data/raw/*
data/interim/*
data/processed/*
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep

# Model artifacts (tracked by DVC)
models/*
!models/.gitkeep

# DVC internal cache
.dvc/cache/
.dvc/tmp/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.wheel/

# MLflow local tracking
mlruns/

# IDE
.vscode/
.idea/

# Logs & Dev-Reqs
setup.log
install-deps.log
dev-requirements.txt
EOF
    Log "   + .gitignore created"
else
    Log "   .gitignore already exists (left untouched)"
fi
Log ""

# ---------- 4. Write requirements.txt ----------
Log "[4/5] Checking requirements.txt ..."
if [ ! -e "requirements.txt" ]; then
    cat > "requirements.txt" <<'EOF'
# ---- Core data & ML ----
pyyaml>=6.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0

# ---- MLOps stack ----
dvc>=3.0
mlflow>=2.10
pandera>=0.19
pyod>=1.1

# ---- Serving (optional) ----
fastapi>=0.110
uvicorn>=0.29

# ---- Testing ----
pytest>=8.0

# ---- Utilities ----
requests>=2.31
python-dotenv>=1.0
EOF
    Log "   + requirements.txt created"
else
    Log "   requirements.txt already exists (left untouched)"
fi
Log ""

# ---------- 5. Git Init ----------
Log "[5/5] Initializing Git repository ..."
if [ ! -d ".git" ]; then
    git init >> "$LOG" 2>&1
    Log "   + Initialized Git repository"
else
    Log "   Git repository already exists"
fi
Log ""

LogHeader "[DONE] Project structure & Git initialized."
Log ""
Log "Next steps:"
Log "  1. Create and activate your virtual environment however you prefer (e.g., python3 -m venv .venv then source .venv/bin/activate)."
Log "  2. Run '01-install-deps.sh' to verify and install packages."
Log "  3. Run '02-track-dvc-git.sh' to initialize DVC and make your initial commit."