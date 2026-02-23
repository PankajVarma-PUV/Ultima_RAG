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
echo  Ultima_RAG: Metacognitive Intelligence System
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
    goto :gpu_found
)
if "!CUDA_MAJOR!"=="12" (
    if !CUDA_MINOR! GEQ 4 (
        set "TORCH_URL=https://download.pytorch.org/whl/cu124"
    ) else (
        set "TORCH_URL=https://download.pytorch.org/whl/cu121"
    )
    goto :gpu_found
)
REM Unknown CUDA major - use cu121 as safe default for RTX cards
set "TORCH_URL=https://download.pytorch.org/whl/cu121"
goto :gpu_found

:no_gpu
echo [INFO] No NVIDIA GPU detected. CPU-only PyTorch will be installed.
goto :install_requirements

:gpu_found
echo [OK] GPU detected - PyTorch wheel: !TORCH_URL!

REM ==========================================================
REM 5.4: INSTALL requirements.txt
REM
REM  Runs BEFORE torch install on purpose. Packages like
REM  sentence-transformers and easyocr declare torch as a
REM  dependency, pulling in a CPU build from PyPI. That is
REM  intentional - step 5.5 force-removes it immediately after.
REM ==========================================================
:install_requirements
echo.
echo [INFO] Installing project dependencies from requirements.txt...
echo [INFO] ^(Any torch pulled in here will be replaced in the next step^)
"%VENV_PYTHON%" -m pip install -r "%BASE_DIR%\requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] requirements.txt install failed. Check output above.
    pause
    exit /b 1
)
echo [OK] requirements.txt done.
echo.

REM ==========================================================
REM 5.5: FORCE-UNINSTALL ALL EXISTING TORCH PACKAGES
REM
REM  Removes any CPU torch that was pulled in as a transitive
REM  dependency. Forces pip to do a clean install next step
REM  rather than skipping with "already satisfied".
REM ==========================================================
echo [INFO] Removing any existing torch packages...
"%VENV_PYTHON%" -m pip uninstall torch torchvision torchaudio -y >nul 2>&1
echo [OK] Existing torch cleared.
echo.

REM ==========================================================
REM 5.6: INSTALL CORRECT TORCH BUILD - ALWAYS THE FINAL STEP
REM
REM  Nothing runs after this so nothing can overwrite it.
REM ==========================================================
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
    echo [INFO] Installing GPU PyTorch: !TORCH_URL!
    echo [INFO] Download is ~2.5 GB - please be patient.
    echo.
    "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url "!TORCH_URL!"
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [WARNING] GPU install failed. Falling back to CPU...
        "%VENV_PYTHON%" -m pip install torch torchvision torchaudio
        if !ERRORLEVEL! NEQ 0 (
            echo [ERROR] CPU fallback also failed.
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
REM ==========================================================
echo [INFO] Verifying final PyTorch build...
"%VENV_PYTHON%" -c "import torch; v=torch.__version__; c=torch.cuda.is_available(); print('[OK] PyTorch ' + v + ' | CUDA available: ' + str(c)); exit(0 if c else 2)"
set "VERIFY=%ERRORLEVEL%"

if "!VERIFY!"=="0" (
    echo [OK] SUCCESS - GPU confirmed, models will run on your RTX 3060.
) else if "!VERIFY!"=="2" (
    if "!TORCH_URL!"=="CPU" (
        echo [OK] CPU PyTorch confirmed - no GPU on this machine.
    ) else (
        echo.
        echo [WARNING] GPU torch installed but CUDA still shows False.
        echo [WARNING] This means your NVIDIA driver needs updating.
        echo [WARNING] Get the latest driver at: https://www.nvidia.com/drivers
        echo [WARNING] After updating driver: delete .venv and re-run.
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
echo  Launching Ultima_RAG
echo ==========================================================
echo  URL    : http://127.0.0.1:%PORT%
echo  Status : Starting (first load takes 30-60 seconds)
echo ==========================================================
echo.

"%VENV_PYTHON%" -m uvicorn src.api.main:app %RELOAD_FLAG% --host 127.0.0.1 --port %PORT%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FATAL] Ultima_RAG crashed (Exit code: %ERRORLEVEL%^)
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
