@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM install-deps.bat — Smart Offline/Online Dependency Manager
REM Assumes your virtual environment is already activated by you.
REM ============================================================

set "LOG=install-deps.log"
> "%LOG%" echo === Installation started at %DATE% %TIME% ===

call :LogHeader "FTL Syria Crop Yield - Smart Dependency Manager"
call :Log "Working dir: %CD%"
call :Log "Log file:    %LOG%"
call :Log ""

set "WHEEL_DIR=.wheel"

REM 0. Verify python is accessible from active environment
call :Log "[0/4] Checking active Python environment ..."
python --version >nul 2>&1
if errorlevel 1 (
    call :LogError "[ERROR] 'python' command not found. Please activate your virtual environment first!"
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do call :Log "   Active Python version: %%v"
call :Log ""

REM 1. Check if requirements are already installed
call :Log "[1/4] Checking if requirements are already installed ..."
python -c "import importlib.metadata, re, sys; reqs = [line.strip() for line in open('requirements.txt') if line.strip() and not line.startswith('#')]; missing = []; [missing.append(r) for r in reqs if not (lambda name: __import__('pkgutil').find_loader(name) or any(importlib.metadata.distributions()))(re.split(r'[>=<~!]', r)[0].strip()) rescue None]" >nul 2>&1

REM Simplified robust check via python snippet:
python -c "import importlib.metadata, re, sys; reqs = [line.strip() for line in open('requirements.txt') if line.strip() and not line.startswith('#')]; missing = 0; [missing := missing + 1 for r in reqs if (lambda n: (try_except := [1 for _ in [0] if __import__('importlib.metadata').metadata.version(n)] or [0])[-1] == 0)(re.split(r'[>=<~!]', r)[0].strip())]; sys.exit(missing)" >nul 2>&1

if not errorlevel 1 (
    call :Log "   [SUCCESS] All requirements are already installed in this environment!"
    goto :generate_freeze
) else (
    call :Log "   Some requirements are missing or need updating. Proceeding with installation checks..."
)
call :Log ""

REM 2. Check wheel cache folder
call :Log "[2/4] Checking wheel cache folder ..."
if not exist "%WHEEL_DIR%" (
    mkdir "%WHEEL_DIR%"
    call :Log "   + Created %WHEEL_DIR% directory."
) else (
    call :Log "   %WHEEL_DIR% directory already exists."
)
call :Log ""

REM 3. Check local wheels vs internet
call :Log "[3/4] Running installation logic ..."
set "HAS_WHEELS=0"
for %%f in ("%WHEEL_DIR%\*.whl") do (
    set "HAS_WHEELS=1"
)

if "!HAS_WHEELS!"=="1" (
    call :Log "   Local wheels found in %WHEEL_DIR%. Installing offline ..."
    python -m pip install --no-index --find-links="%WHEEL_DIR%" -r requirements.txt
    
    if errorlevel 1 (
        call :Log "   [WARN] Offline installation failed. Falling back to internet download..."
        goto :download_and_install
    ) else (
        call :Log "   [SUCCESS] Packages installed successfully from local cache!"
        goto :generate_freeze
    )
) else (
    call :Log "   No local wheels found in %WHEEL_DIR%."
    goto :download_and_install
)

:download_and_install
call :Log "   Downloading packages from the internet to '%WHEEL_DIR%'..."
python -m pip download -r requirements.txt -d "%WHEEL_DIR%"
if errorlevel 1 (
    call :LogError "[ERROR] Failed to download packages. Check internet connection."
    exit /b 1
)

call :Log "   Installing downloaded packages from local cache..."
python -m pip install --no-index --find-links="%WHEEL_DIR%" -r requirements.txt
if errorlevel 1 (
    call :LogError "[ERROR] Installation from downloaded wheels failed."
    exit /b 1
)
call :Log "   [SUCCESS] Packages downloaded and installed successfully!"

:generate_freeze
call :Log ""
call :Log "[4/4] Generating pinned dependency snapshot (dev-requirements.txt) ..."
python -m pip freeze > dev-requirements.txt
call :Log "   + dev-requirements.txt created successfully."

call :LogHeader "[DONE] Environment check, wheels cache, and dev-requirements.txt are ready."
endlocal

REM ============================================================
REM Helper Subroutines for Dual Timed Logging (Console + Log File)
REM ============================================================
:Log
echo [%TIME%] %~1
echo [%TIME%] %~1 >> "%LOG%"
goto :eof

:LogHeader
echo ============================================================
echo  %~1
echo ============================================================
echo [%TIME%] ============================================================ >> "%LOG%"
echo [%TIME%]  %~1 >> "%LOG%"
echo [%TIME%] ============================================================ >> "%LOG%"
goto :eof

:LogError
echo [%TIME%] %~1
echo [%TIME%] %~1 >> "%LOG%"
goto :eof