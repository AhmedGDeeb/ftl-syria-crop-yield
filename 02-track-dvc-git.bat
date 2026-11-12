@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM track-dvc-git.bat — DVC Initialization, Data Tracking & Git Commit
REM ============================================================

set "LOG=dvc-setup.log"
echo === DVC & Git process started at %DATE% %TIME% === > "%LOG%"

call :LogHeader "FTL Syria Crop Yield - DVC & Git Tracking"
call :Log "Working dir: %CD%"
call :Log "Log file:    %LOG%"
call :Log ""

REM Check if virtual environment exists to run dvc safely
set "VENV_DVC=.venv\Scripts\dvc.exe"
if exist "%VENV_DVC%" (
    set "DVC_RUN=%VENV_DVC%"
    call :Log "Using virtual environment DVC executable."
) else (
    set "DVC_RUN=dvc"
    call :Log "Virtual environment DVC not found. Falling back to global command."
)
call :Log ""

REM 1. DVC Init
call :Log "[1/4] Initializing DVC (--no-scm) ..."
if not exist ".dvc" (
    "%DVC_RUN%" init --no-scm
    call :Log "   + DVC initialized successfully."
) else (
    call :Log "   .dvc folder already exists - skipping init."
)
call :Log ""

REM 2. DVC Add data\raw
call :Log "[2/4] Tracking raw data with DVC (dvc add data\raw) ..."
if exist "data\raw" (
    "%DVC_RUN%" add data\raw
    call :Log "   + data\raw added to DVC successfully."
) else (
    call :Log "   [WARN] data\raw folder not found. Skipping dvc add."
)
call :Log ""

REM 3. Git Add
call :Log "[3/4] Staging changes in Git (git add .) ..."
git add .
call :Log "   + All project changes and DVC pointer files staged."
call :Log ""

REM 4. Git Commit
call :Log "[4/4] Committing changes to Git ..."
git commit -m "chore: initial project bootstrap structure and dvc tracking"
if errorlevel 1 (
    call :Log "   [INFO] No new changes to commit or commit already made."
) else (
    call :Log "   + Initial Git commit created successfully."
)
call :Log ""

call :LogHeader "[DONE] DVC data tracking and Git commit completed."
endlocal
exit /b 0

REM ============================================================
REM Helper Subroutines for Dual Logging with Timestamps
REM ============================================================
:Log
echo %~1
echo [%TIME%] %~1 >> "%LOG%"
goto :eof

:LogHeader
echo ============================================================
echo  %~1
echo ============================================================
echo ============================================================ >> "%LOG%"
echo [%TIME%]  %~1 >> "%LOG%"
echo ============================================================ >> "%LOG%"
goto :eof