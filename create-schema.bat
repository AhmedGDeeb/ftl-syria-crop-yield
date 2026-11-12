@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM create-schema.bat — project bootstrap (Windows CMD/PowerShell)
REM ============================================================

set "LOG=setup.log"
> "%LOG%" echo === Setup started at %DATE% %TIME% ===

call :LogHeader "FTL Syria Crop Yield - Project Bootstrap"
call :Log "Working dir: %CD%"
call :Log "Log file:    %LOG%"
call :Log ""

set "PYTHON_MIN_MAJOR=3"
set "PYTHON_MIN_MINOR=10"

REM ---------- 0. Preflight ----------
call :Log "[0/6] Preflight checks ..."

where python >nul 2>&1
if errorlevel 1 (
    call :LogError "[ERROR] 'python' not found on PATH. Install Python 3.10+ and retry."
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set "PYMAJ=%%a"
    set "PYMIN=%%b"
)
if !PYMAJ! LSS %PYTHON_MIN_MAJOR% (
    call :LogError "[ERROR] Python !PYVER! found, but %PYTHON_MIN_MAJOR%.%PYTHON_MIN_MINOR%+ required."
    exit /b 1
)
if !PYMAJ! EQU %PYTHON_MIN_MAJOR% if !PYMIN! LSS %PYTHON_MIN_MINOR% (
    call :LogError "[ERROR] Python !PYVER! found, but %PYTHON_MIN_MAJOR%.%PYTHON_MIN_MINOR%+ required."
    exit /b 1
)
call :Log "   Python !PYVER! OK."
call :Log ""

REM ---------- 1. Directories ----------
call :Log "[1/6] Creating directories ..."
for %%D in (
    data data\raw data\interim data\processed
    src src\data src\features src\models src\utils
    models metrics notebooks tests
    .github .github\workflows
) do (
    if not exist "%%D" (
        mkdir "%%D"
        call :Log "   + %%D"
    )
)
call :Log ""

REM ---------- 2. Placeholder files ----------
call :Log "[2/6] Creating placeholder files ..."
for %%F in (
    data\raw\.gitkeep data\interim\.gitkeep data\processed\.gitkeep
    models\.gitkeep metrics\.gitkeep
    src\__init__.py src\data\__init__.py src\features\__init__.py
    src\models\__init__.py src\utils\__init__.py
    src\data\ingest.py src\data\validate.py src\data\anomaly_check.py
    src\features\build_features.py
    src\models\train.py src\models\evaluate.py
    src\utils\config.py
    tests\__init__.py tests\test_data_validation.py
    notebooks\01_eda.ipynb
    .github\workflows\daily_retrain.yml
    dvc.yaml params.yaml README.md
) do (
    if not exist "%%F" (
        type nul > "%%F"
        call :Log "   + %%F"
    )
)
call :Log ""

REM ---------- 3. Seed .gitignore ----------
call :Log "[3/6] Checking .gitignore ..."
if not exist ".gitignore" (
    (
        echo # Data ^(tracked by DVC, not Git^)
        echo data/raw/*
        echo data/interim/*
        echo data/processed/*
        echo !data/raw/.gitkeep
        echo !data/interim/.gitkeep
        echo !data/processed/.gitkeep
        echo.
        echo # Model artifacts ^(tracked by DVC^)
        echo models/*
        echo !models/.gitkeep
        echo.
        echo # DVC internal cache
        echo .dvc/cache/
        echo .dvc/tmp/
        echo.
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo *.egg-info/
        echo .venv/
        echo venv/
        echo.
        echo # MLflow local tracking
        echo mlruns/
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo.
        echo # Logs
        echo setup.log
    ) > ".gitignore"
    call :Log "   + .gitignore created"
) else (
    call :Log "   .gitignore already exists ^(left untouched^)"
)
call :Log ""

REM ---------- 4. Write requirements.txt ----------
call :Log "[4/6] Checking requirements.txt ..."
if not exist "requirements.txt" (
    (
        echo # ---- Core data ^& ML ----
        echo pandas^>=2.0
        echo numpy^>=1.24
        echo scikit-learn^>=1.3
        echo xgboost^>=2.0
        echo.
        echo # ---- MLOps stack ----
        echo dvc^>=3.0
        echo dvc-git^>=3.0
        echo mlflow^>=2.10
        echo pandera^>=0.19
        echo pyod^>=1.1
        echo.
        echo # ---- Serving ^(optional^) ----
        echo fastapi^>=0.110
        echo uvicorn^>=0.29
        echo.
        echo # ---- Testing ----
        echo pytest^>=8.0
        echo.
        echo # ---- Utilities ----
        echo requests^>=2.31
        echo python-dotenv^>=1.0
    ) > "requirements.txt"
    call :Log "   + requirements.txt created"
) else (
    call :Log "   requirements.txt already exists ^(left untouched^)"
)
call :Log ""

REM ---------- 5. Git and DVC Setup ----------
call :Log "[5/6] Initializing Git and DVC ..."

if not exist ".git" (
    git init >> "%LOG%" 2>&1
    call :Log "   + Initialized Git repository"
) else (
    call :Log "   Git repository already exists"
)

if not exist ".dvc" (
    dvc init --no-scm >> "%LOG%" 2>&1
    call :Log "   + Initialized DVC"
) else (
    call :Log "   DVC already initialized"
)

if exist "data\raw" (
    dvc add data\raw >> "%LOG%" 2>&1
    call :Log "   + Added data\raw to DVC"
)
call :Log ""

REM ---------- 6. Initial Git Commit ----------
call :Log "[6/6] Making first Git commit ..."
git add . >> "%LOG%" 2>&1
git commit -m "chore: initial project bootstrap structure" >> "%LOG%" 2>&1
call :Log "   + Created initial git commit"
call :Log ""

call :LogHeader "[DONE] Project structure & version control complete."
call :Log ""
call :Log "Next steps (Virtual Environment & Dependencies):"
call :Log "  1. python -m venv .venv"
call :Log "  2. .venv\Scripts\activate"
call :Log "  3. python -m pip install --upgrade pip"
call :Log "  4. pip install -r requirements.txt"

endlocal
exit /b 0

REM ============================================================
REM Helper Subroutines for Dual Logging (Console + setup.log)
REM ============================================================
:Log
echo %~1
echo %~1 >> "%LOG%"
goto :eof

:LogHeader
echo ============================================================
echo  %~1
echo ============================================================
echo ============================================================ >> "%LOG%"
echo  %~1 >> "%LOG%"
echo ============================================================ >> "%LOG%"
goto :eof

:LogError
echo %~1
echo %~1 >> "%LOG%"
goto :eof