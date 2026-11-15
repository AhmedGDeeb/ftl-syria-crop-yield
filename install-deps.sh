#!/usr/bin/env bash
# ============================================================
# install-deps.sh — Smart Offline/Online Dependency Manager
# Assumes your virtual environment is already activated by you.
# ============================================================

set -u  # error on unset variables; do NOT use -e (matches .bat's permissive flow)

LOG="install-deps.log"
echo "=== Installation started at $(date '+%Y-%m-%d %H:%M:%S') ===" > "$LOG"

# ---------- Helper subroutines (dual timed logging: console + log file) ----------
Log() {
    echo "[$(date '+%H:%M:%S')] $1"
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG"
}

LogHeader() {
    echo "============================================================"
    echo " $1"
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] ============================================================" >> "$LOG"
    echo "[$(date '+%H:%M:%S')]  $1" >> "$LOG"
    echo "[$(date '+%H:%M:%S')] ============================================================" >> "$LOG"
}

LogError() {
    echo "[$(date '+%H:%M:%S')] $1"
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG"
}

LogHeader "FTL Syria Crop Yield - Smart Dependency Manager"
Log "Working dir: $(pwd)"
Log "Log file:    $LOG"
Log ""

WHEEL_DIR=".wheel"

# 0. Verify python is accessible from active environment
Log "[0/4] Checking active Python environment ..."
if ! python --version >/dev/null 2>&1; then
    LogError "[ERROR] 'python' command not found. Please activate your virtual environment first!"
    exit 1
fi
PYVER=$(python --version 2>&1 | awk '{print $2}')
Log "   Active Python version: $PYVER"
Log ""

# 1. Check if requirements are already installed
Log "[1/4] Checking if requirements are already installed ..."

# Same python snippet as the .bat (the second, "simplified robust" one —
# the first snippet in the .bat uses invalid Python syntax and always fails
# silently; only the second one is effective. Kept both is impossible in
# bash without divergence, so we run the second only. This matches the
# actual behavior of the .bat: the first snippet never succeeds.)
python -c "import importlib.metadata, re, sys; reqs = [line.strip() for line in open('requirements.txt') if line.strip() and not line.startswith('#')]; missing = 0; [missing := missing + 1 for r in reqs if (lambda n: (try_except := [1 for _ in [0] if __import__('importlib.metadata').metadata.version(n)] or [0])[-1] == 0)(re.split(r'[>=<~!]', r)[0].strip())]; sys.exit(missing)" >/dev/null 2>&1

if [ $? -eq 0 ]; then
    Log "   [SUCCESS] All requirements are already installed in this environment!"
    generate_freeze=1
else
    Log "   Some requirements are missing or need updating. Proceeding with installation checks..."
    generate_freeze=0
fi
Log ""

if [ "$generate_freeze" -eq 1 ]; then
    # Skip install logic, jump to freeze (matches `goto :generate_freeze`)
    :
else

    # 2. Check wheel cache folder
    Log "[2/4] Checking wheel cache folder ..."
    if [ ! -d "$WHEEL_DIR" ]; then
        mkdir -p "$WHEEL_DIR"
        Log "   + Created $WHEEL_DIR directory."
    else
        Log "   $WHEEL_DIR directory already exists."
    fi
    Log ""

    # 3. Check local wheels vs internet
    Log "[3/4] Running installation logic ..."
    HAS_WHEELS=0
    shopt -s nullglob
    for f in "$WHEEL_DIR"/*.whl; do
        HAS_WHEELS=1
    done
    shopt -u nullglob

    if [ "$HAS_WHEELS" -eq 1 ]; then
        Log "   Local wheels found in $WHEEL_DIR. Installing offline ..."
        python -m pip install --no-index --find-links="$WHEEL_DIR" -r requirements.txt

        if [ $? -ne 0 ]; then
            Log "   [WARN] Offline installation failed. Falling back to internet download..."
            download_and_install=1
        else
            Log "   [SUCCESS] Packages installed successfully from local cache!"
            download_and_install=0
        fi
    else
        Log "   No local wheels found in $WHEEL_DIR."
        download_and_install=1
    fi

    if [ "$download_and_install" -eq 1 ]; then
        Log "   Downloading packages from the internet to '$WHEEL_DIR'..."
        python -m pip download -r requirements.txt -d "$WHEEL_DIR"
        if [ $? -ne 0 ]; then
            LogError "[ERROR] Failed to download packages. Check internet connection."
            exit 1
        fi

        Log "   Installing downloaded packages from local cache..."
        python -m pip install --no-index --find-links="$WHEEL_DIR" -r requirements.txt
        if [ $? -ne 0 ]; then
            LogError "[ERROR] Installation from downloaded wheels failed."
            exit 1
        fi
        Log "   [SUCCESS] Packages downloaded and installed successfully!"
    fi

fi

# :generate_freeze
Log ""
Log "[4/4] Generating pinned dependency snapshot (dev-requirements.txt) ..."
python -m pip freeze > dev-requirements.txt
Log "   + dev-requirements.txt created successfully."

LogHeader "[DONE] Environment check, wheels cache, and dev-requirements.txt are ready."
exit 0