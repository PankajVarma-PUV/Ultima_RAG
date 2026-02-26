@echo off
:: UltimaRAG — Multi-Agent RAG System
:: Copyright (C) 2026 Pankaj Varma
::
:: This program is free software: you can redistribute it and/or modify
:: it under the terms of the GNU Affero General Public License as published by
:: the Free Software Foundation, either version 3 of the License, or
:: (at your option) any later version.
::
:: This program is distributed in the hope that it will be useful,
:: but WITHOUT ANY WARRANTY; without even the implied warranty of
:: MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
:: GNU Affero General Public License for more details.
::
:: You should have received a copy of the GNU Affero General Public License
:: along with this program.  If not, see <https://www.gnu.org/licenses/>.

setlocal enabledelayedexpansion

REM ==========================================================
REM UltimaRAG - Startup Script (Windows 11)
REM ==========================================================

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"
cd /d "%BASE_DIR%"

set "PORT=8000"
set "RELOAD_FLAG=--reload"

echo ==========================================================
echo  UltimaRAG: Metacognitive Intelligence System
echo ==========================================================
echo  Project : %BASE_DIR%
echo  Port    : %PORT%
echo ==========================================================
echo.

REM ----------------------------------------------------------
REM STEP 0: Port check
REM ----------------------------------------------------------
echo [0/6] Checking port %PORT%...
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [WARNING] Port %PORT% is in use. Close other instances first.
) else (
    echo [OK] Port %PORT% is free.
)
echo.

REM ----------------------------------------------------------
REM STEP 1: Python check
REM ----------------------------------------------------------
echo [1/6] Checking Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PY_VER=%%V"
echo [OK] Python %PY_VER% found.
echo.

REM ----------------------------------------------------------
REM STEP 2: Virtual environment
REM ----------------------------------------------------------
set "VENV_DIR=%BASE_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

echo [2/6] Checking virtual environment...
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM ----------------------------------------------------------
REM STEP 3: Activate virtual environment
REM ----------------------------------------------------------
echo [3/6] Activating virtual environment...
if not exist "%VENV_ACTIVATE%" (
    echo [WARNING] Activate script missing - recreating venv...
    rmdir /s /q "%VENV_DIR%" >nul 2>&1
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to recreate virtual environment.
        pause
        exit /b 1
    )
)
call "%VENV_ACTIVATE%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment active.
echo.

REM ----------------------------------------------------------
REM STEP 4: Upgrade pip
REM ----------------------------------------------------------
echo [4/6] Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] pip upgrade failed - continuing anyway.
) else (
    echo [OK] pip is up to date.
)
echo.

REM ----------------------------------------------------------
REM STEP 5: Install all dependencies
REM ----------------------------------------------------------
echo [5/6] Installing dependencies...
echo.

REM ==========================================================
REM 5.1: FIND nvidia-smi.exe ON DISK
REM
REM  WHY THIS APPROACH:
REM  On Windows 11 with modern NVIDIA drivers, nvidia-smi.exe
REM  is NOT in PATH. It sits in one of these locations:
REM    - C:\Windows\System32\nvidia-smi.exe        (most common)
REM    - C:\Program Files\NVIDIA Corporation\NVSMI\ (CUDA toolkit)
REM    - C:\Windows\System32\DriverStore\FileRepository\nv*\
REM      (hashed folder, changes every driver update)
REM
REM  We use batch-native IF EXIST and DIR /S /B to find it
REM  without relying on Python temp files (which were the
REM  root cause of all previous detection failures).
REM ==========================================================
echo [INFO] Searching for nvidia-smi.exe...
set "SMI_EXE="

REM --- Check 1: Already on PATH ---
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "delims=" %%P in ('where nvidia-smi 2^>nul') do (
        if "!SMI_EXE!"=="" set "SMI_EXE=%%P"
    )
    if not "!SMI_EXE!"=="" echo [INFO] Found on PATH: !SMI_EXE!
)

REM --- Check 2: C:\Windows\System32 (most common modern location) ---
if "!SMI_EXE!"=="" (
    if exist "C:\Windows\System32\nvidia-smi.exe" (
        set "SMI_EXE=C:\Windows\System32\nvidia-smi.exe"
        echo [INFO] Found in System32: !SMI_EXE!
    )
)

REM --- Check 3: NVSMI folder (CUDA toolkit / older drivers) ---
if "!SMI_EXE!"=="" (
    if exist "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
        set "SMI_EXE=C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        echo [INFO] Found in NVSMI: !SMI_EXE!
    )
)

REM --- Check 4: DriverStore FileRepository (Windows 11 modern drivers) ---
REM  Uses DIR /S /B to recursively search all nv* subfolders.
REM  This handles the hashed folder name that changes per driver version.
if "!SMI_EXE!"=="" (
    echo [INFO] Searching DriverStore (this may take a moment^)...
    for /f "delims=" %%F in ('dir /s /b "C:\Windows\System32\DriverStore\FileRepository\nvidia-smi.exe" 2^>nul') do (
        if "!SMI_EXE!"=="" set "SMI_EXE=%%F"
    )
    if not "!SMI_EXE!"=="" echo [INFO] Found in DriverStore: !SMI_EXE!
)

REM ==========================================================
REM 5.2: GET CUDA VERSION FROM nvidia-smi -q
REM
REM  nvidia-smi -q gives structured key:value output.
REM  No pipe characters - safe to parse with findstr + for /f.
REM  The CUDA Version line looks like:
REM    "CUDA Version                              : 12.4"
REM  With delims=: and space, tokens are:
REM    token1=CUDA  token2=Version  token3=12.4
REM ==========================================================
set "TORCH_URL=CPU"
set "CUDA_VER="
set "CUDA_MAJOR=0"
set "CUDA_MINOR=0"

if "!SMI_EXE!"=="" goto :no_gpu

echo [INFO] Running: !SMI_EXE! -q
set "SMI_OUT=%TEMP%\ultima_smi_out.txt"
"!SMI_EXE!" -q > "%SMI_OUT%" 2>nul

if not exist "%SMI_OUT%" goto :no_gpu

REM Extract the CUDA version number
for /f "tokens=3 delims=: " %%V in ('findstr /i "CUDA Version" "%SMI_OUT%"') do (
    if "!CUDA_VER!"=="" set "CUDA_VER=%%V"
)
del "%SMI_OUT%" >nul 2>&1

if "!CUDA_VER!"=="" goto :no_gpu
if "!CUDA_VER!"=="N/A" goto :no_gpu

echo [OK] CUDA Version detected: !CUDA_VER!

REM Split version into major and minor
for /f "tokens=1 delims=." %%M in ("!CUDA_VER!") do set "CUDA_MAJOR=%%M"
set "CUDA_MINOR=0"
for /f "tokens=2 delims=." %%N in ("!CUDA_VER!") do set "CUDA_MINOR=%%N"

REM ==========================================================
REM 5.3: MAP CUDA VERSION TO PYTORCH WHEEL INDEX URL
REM  CUDA 11.x         -> cu118
REM  CUDA 12.0 - 12.3  -> cu121
REM  CUDA 12.4+        -> cu124
REM ==========================================================
if "!CUDA_MAJOR!"=="11" (
    set "TORCH_URL=https://download.pytorch.org/whl/cu118"
    set "EXPECTED_CUDA_TAG=cu118"
    set "EXPECTED_TORCH_CUDA_MAJOR=11"
    goto :gpu_found
)
if "!CUDA_MAJOR!"=="12" (
    if !CUDA_MINOR! GEQ 4 (
        set "TORCH_URL=https://download.pytorch.org/whl/cu124"
        set "EXPECTED_CUDA_TAG=cu124"
        set "EXPECTED_TORCH_CUDA_MAJOR=12"
    ) else (
        set "TORCH_URL=https://download.pytorch.org/whl/cu121"
        set "EXPECTED_CUDA_TAG=cu121"
        set "EXPECTED_TORCH_CUDA_MAJOR=12"
    )
    goto :gpu_found
)
REM Unknown CUDA major - use cu121 as safe default for RTX cards
set "TORCH_URL=https://download.pytorch.org/whl/cu121"
set "EXPECTED_CUDA_TAG=cu121"
set "EXPECTED_TORCH_CUDA_MAJOR=12"
goto :gpu_found

:no_gpu
echo [INFO] No NVIDIA GPU detected. CPU-only PyTorch will be installed.
set "EXPECTED_CUDA_TAG=CPU"
set "EXPECTED_TORCH_CUDA_MAJOR=0"
goto :run_preflight

:gpu_found
echo [OK] GPU detected - PyTorch wheel target: !TORCH_URL!

REM ==========================================================
REM 5.4: PRE-FLIGHT TORCH VERIFICATION GATE
REM
REM  PURPOSE:
REM  On a clean re-run where torch is already correctly installed
REM  (GPU build, CUDA available, correct CUDA major version),
REM  there is zero reason to uninstall and re-download ~2.5 GB.
REM
REM  WHY importlib.util.find_spec AND NOT try/except:
REM  Batch line-continuation (^) collapses multi-line strings
REM  into a single space-separated line. Python's try/except
REM  requires the except clause on a NEW indented line - it
REM  cannot appear after a semicolon. Any attempt to write
REM  try/except across ^ continuation lines produces a Python
REM  SyntaxError, causing the check to always exit with code 1
REM  and therefore always trigger reinstall. This bug silently
REM  defeats the entire optimisation.
REM
REM  importlib.util.find_spec('torch') returns None if torch
REM  is not installed - no exception, no indentation needed.
REM  Combined with conditional expressions, this makes the
REM  entire three-step check a valid Python one-liner.
REM
REM  EXPANSION NOTE (%% vs !!):
REM  %EXPECTED_TORCH_CUDA_MAJOR% uses normal %% expansion, not
REM  delayed !! expansion. This is correct because:
REM    - This is a single non-parenthesised command line.
REM    - %% expansion occurs when the line is reached.
REM    - The variable is fully set before this line runs.
REM  Using !! inside a quoted -c string passed to Python would
REM  not expand correctly in all batch contexts.
REM
REM  Exit codes:
REM    0  = All checks passed -> SKIP reinstall
REM    1  = torch not found (find_spec returned None)
REM    2  = torch found but cuda.is_available() is False
REM    3  = CUDA available but major version mismatches driver
REM ==========================================================
:run_preflight
echo.
echo [INFO] Running PyTorch pre-flight verification...
set "SKIP_TORCH_INSTALL=0"

if "!EXPECTED_CUDA_TAG!"=="CPU" (
    REM ---------------------------------------------------
    REM CPU path: probe torch existence via find_spec.
    REM Exits 0 if torch is present, 1 if not.
    REM No try/except needed - no SyntaxError risk.
    REM ---------------------------------------------------
    "%VENV_PYTHON%" -c "import sys,importlib.util;sys.exit(0 if importlib.util.find_spec('torch') is not None else 1)" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [OK] CPU torch already installed - skipping reinstall.
        set "SKIP_TORCH_INSTALL=1"
    ) else (
        echo [INFO] torch not found - will install.
    )
) else (
    REM ---------------------------------------------------
    REM GPU path: three checks in one valid Python one-liner.
    REM
    REM  s = find_spec('torch')
    REM  -> if None: exit 1  (not installed)
    REM  -> else: import torch
    REM     -> if not cuda available: exit 2  (CPU build)
    REM     -> get installed CUDA major from torch.version.cuda
    REM        -> if matches expected: exit 0  (all good)
    REM        -> else: exit 3  (wrong wheel tier)
    REM
    REM  sys.exit() raises SystemExit so code after it in the
    REM  same semicolon chain never executes - this is the
    REM  correct and safe way to use conditional exits inline.
    REM ---------------------------------------------------
    "%VENV_PYTHON%" -c "import sys,importlib.util;s=importlib.util.find_spec('torch');sys.exit(1) if s is None else None;import torch;sys.exit(2) if not torch.cuda.is_available() else None;m=(torch.version.cuda or '0').split('.')[0];sys.exit(0 if m=='%EXPECTED_TORCH_CUDA_MAJOR%' else 3)" >nul 2>&1
    set "PREFLIGHT_CODE=!ERRORLEVEL!"

    if "!PREFLIGHT_CODE!"=="0" (
        REM All checks passed - torch is healthy, correct build, CUDA live
        echo [OK] PyTorch is already correctly installed with CUDA support.
        for /f "delims=" %%V in ('"%VENV_PYTHON%" -c "import torch;print(torch.__version__)" 2^>nul') do set "EXISTING_TORCH_VER=%%V"
        for /f "delims=" %%C in ('"%VENV_PYTHON%" -c "import torch;print(torch.version.cuda)" 2^>nul') do set "EXISTING_TORCH_CUDA=%%C"
        echo [OK] Installed : PyTorch !EXISTING_TORCH_VER! ^| Compiled CUDA !EXISTING_TORCH_CUDA!
        echo [OK] Expected  : CUDA major %EXPECTED_TORCH_CUDA_MAJOR% ^(%EXPECTED_CUDA_TAG%^)
        echo [SKIP] No reinstall needed. Jumping to requirements check.
        set "SKIP_TORCH_INSTALL=1"
    ) else if "!PREFLIGHT_CODE!"=="1" (
        echo [INFO] torch is not installed - will perform full install.
    ) else if "!PREFLIGHT_CODE!"=="2" (
        echo [WARNING] torch found but CUDA is NOT available - CPU build detected.
        echo [INFO] Will uninstall and reinstall the correct GPU build.
    ) else if "!PREFLIGHT_CODE!"=="3" (
        echo [WARNING] torch found but CUDA major version does not match your driver.
        echo [INFO] Installed build targets a different CUDA generation.
        echo [INFO] Will uninstall and reinstall the correct GPU build.
    )
)
echo.

REM ==========================================================
REM 5.5: INSTALL requirements.txt
REM
REM  Always runs so any missing non-torch packages are resolved.
REM
REM  If SKIP_TORCH_INSTALL=1 (torch is healthy), pip sees torch
REM  as already satisfied and leaves it untouched. PEP 440 local
REM  versions (the +cuXXX suffix on GPU builds) compare greater
REM  than the plain release so == and >= constraints in
REM  requirements.txt are both satisfied. pip will not re-download.
REM
REM  --quiet is intentionally absent. If this step fails you need
REM  to see the full pip output to understand why. The error
REM  message "review the output above" is only useful when there
REM  IS output above to review.
REM ==========================================================
echo [INFO] Installing project dependencies from requirements.txt...
if "!SKIP_TORCH_INSTALL!"=="1" (
    echo [INFO] ^(torch is healthy - pip will not touch it^)
)
"%VENV_PYTHON%" -m pip install -r "%BASE_DIR%\requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] requirements.txt install failed. Review the pip output above.
    pause
    exit /b 1
)
echo [OK] requirements.txt done.
echo.

REM ==========================================================
REM 5.6: CONDITIONAL TORCH UNINSTALL + REINSTALL
REM
REM  ONLY entered when pre-flight determined torch is absent,
REM  broken, or the wrong CUDA build.
REM
REM  When SKIP_TORCH_INSTALL=1 the entire block is bypassed.
REM  A correctly installed GPU torch is NEVER touched on a
REM  healthy re-run. No more unnecessary 2.5 GB downloads.
REM ==========================================================
if "!SKIP_TORCH_INSTALL!"=="1" (
    echo [SKIP] PyTorch reinstall bypassed - existing installation is valid.
    echo.
    goto :final_verify
)

echo [INFO] Clearing any existing torch packages before clean install...
"%VENV_PYTHON%" -m pip uninstall torch torchvision torchaudio -y >nul 2>&1
echo [OK] Existing torch cleared.
echo.

if "!TORCH_URL!"=="CPU" (
    echo [INFO] Installing CPU-only PyTorch...
    echo.
    "%VENV_PYTHON%" -m pip install torch torchvision torchaudio
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] PyTorch install failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo [OK] CPU PyTorch installed.
) else (
    echo [INFO] Installing GPU PyTorch from: !TORCH_URL!
    echo [INFO] Download is ~2.5 GB - please be patient.
    echo.
    "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url "!TORCH_URL!"
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [WARNING] GPU install failed. Falling back to CPU PyTorch...
        "%VENV_PYTHON%" -m pip install torch torchvision torchaudio
        if !ERRORLEVEL! NEQ 0 (
            echo [ERROR] CPU fallback also failed. Check your internet connection.
            pause
            exit /b 1
        )
        echo [OK] CPU fallback installed.
    ) else (
        echo [OK] GPU PyTorch installed.
    )
)
echo.

REM ==========================================================
REM 5.7: FINAL VERIFICATION
REM
REM  Runs after any new install to confirm the environment is
REM  healthy before launching the server.
REM  Also runs on the SKIP path as a lightweight sanity check
REM  (~200ms import cost) to confirm nothing changed since the
REM  pre-flight ran (e.g. requirements.txt re-installing a
REM  different torch as a transitive dependency).
REM ==========================================================
:final_verify
echo [INFO] Verifying final PyTorch build...
"%VENV_PYTHON%" -c "import torch;v=torch.__version__;c=torch.cuda.is_available();print('[OK] PyTorch ' + v + ' | CUDA available: ' + str(c));exit(0 if c else 2)"
set "VERIFY=!ERRORLEVEL!"

if "!VERIFY!"=="0" (
    echo [OK] SUCCESS - GPU confirmed, models will run on your RTX GPU.
) else if "!VERIFY!"=="2" (
    if "!TORCH_URL!"=="CPU" (
        echo [OK] CPU PyTorch confirmed - no GPU on this machine.
    ) else (
        echo.
        echo [WARNING] GPU torch installed but CUDA still shows False.
        echo [WARNING] This typically means your NVIDIA driver needs updating.
        echo [WARNING] Get the latest driver at: https://www.nvidia.com/drivers
        echo [WARNING] After updating the driver: delete .venv and re-run this script.
        echo.
    )
) else (
    echo [ERROR] PyTorch failed to import. Delete .venv and re-run.
    pause
    exit /b 1
)
echo.

REM ----------------------------------------------------------
REM STEP 6: Create data directories
REM ----------------------------------------------------------
echo [6/6] Verifying data directories...
if not exist "%BASE_DIR%\data"           mkdir "%BASE_DIR%\data"
if not exist "%BASE_DIR%\data\Ultima_db" mkdir "%BASE_DIR%\data\Ultima_db"
if not exist "%BASE_DIR%\Credentials"    mkdir "%BASE_DIR%\Credentials"
echo [OK] Directories ready.
echo.

REM ----------------------------------------------------------
REM Launch
REM ----------------------------------------------------------
echo ==========================================================
echo  Launching UltimaRAG
echo ==========================================================
echo  URL    : http://127.0.0.1:%PORT%
echo  Status : Starting (first load takes 30-60 seconds)
echo ==========================================================
echo.

"%VENV_PYTHON%" -m uvicorn src.api.main:app %RELOAD_FLAG% --host 127.0.0.1 --port %PORT%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FATAL] UltimaRAG crashed (Exit code: %ERRORLEVEL%^)
    echo.
    echo  Common causes:
    echo   1. Ollama not running      ^>  run: ollama serve
    echo   2. Port already in use     ^>  run: netstat -ano ^| findstr :%PORT%
    echo   3. Corrupt environment     ^>  delete .venv and re-run this script
    echo.
    pause
    exit /b %ERRORLEVEL%
)

endlocal
