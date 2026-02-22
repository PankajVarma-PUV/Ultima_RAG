@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM SentinelRAG - Automated SOTA Startup Script (Windows)
REM ==========================================================

:: Get the absolute path of the directory containing this batch file
set "BASE_DIR=%~dp0"
:: Remove trailing backslash if present
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"

:: Change to project directory
cd /d "%BASE_DIR%"

set "PORT=8000"
set "RELOAD_FLAG=--reload"
echo [INFO] Hot-Reload enabled by default.

echo ==========================================================
echo 🧠 SentinelRAG: Metacognitive Intelligence System
echo ==========================================================
echo Project Location: %BASE_DIR%
echo Target Port:     %PORT%
echo ==========================================================
echo.

:: ================== STEP 0: Port Check ====================
echo [0/6] Checking if port %PORT% is already in use...
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [WARNING] Port %PORT% is ALREADY IN USE.
    echo [INFO] This usually causes uvicorn to crash. 
    echo [INFO] Please close any other running instances of SentinelRAG.
    echo.
) else (
    echo [OK] Port %PORT% is available.
)
echo.

:: ================== STEP 1: Python Check ==================
echo [1/6] Verifying Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%
echo.

:: =============== STEP 2: Virtual Environment ==============
set "VENV_DIR=%BASE_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

echo [2/6] Checking virtual environment...
if not exist "%VENV_DIR%" (
    echo [INFO] Virtual environment not found. Creating new one...
    python -m venv "%VENV_DIR%"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully
) else (
    echo [OK] Virtual environment found
)
echo.

:: ============== STEP 3: Activate Environment ==============
echo [3/6] Activating virtual environment...
if not exist "%VENV_ACTIVATE%" (
    echo [ERROR] Activation script not found. Recreating environment...
    rmdir /s /q "%VENV_DIR%"
    python -m venv "%VENV_DIR%"
)

call "%VENV_ACTIVATE%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

:: ============== STEP 4: Upgrade pip ======================
echo [4/6] Ensuring pip is up to date...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
) else (
    echo [OK] pip is up to date
)
echo.

:: ============ STEP 5: Dependency Installation =============
echo [5/6] Checking dependencies (Hardware-Aware)...

REM Check if torch and other core modules are functional
"%VENV_PYTHON%" -c "import torch; import lancedb; import langgraph; from fpdf import FPDF" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :dependencies_ok

echo [INFO] Dependencies missing or misconfigured. Initializing setup...

REM --- SMART HARDWARE DETECTION ---
set "USE_GPU=0"
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] NVIDIA GPU Detected.
    set "USE_GPU=1"
) else (
    echo [INFO] No NVIDIA GPU detected. Using CPU mode.
)

REM --- DYNAMIC TORCH INSTALLATION ---
if "%USE_GPU%"=="1" (
    echo [INFO] Installing GPU-Optimized Torch...
    "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
) else (
    echo [INFO] Installing standard Torch (CPU)...
    "%VENV_PIP%" install torch torchvision torchaudio --quiet
)

REM Install remaining requirements
echo [INFO] Installing remaining dependencies...
"%VENV_PIP%" install -r "%BASE_DIR%\requirements.txt"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo [OK] All dependencies installed successfully.
goto :dependencies_done

:dependencies_ok
echo [OK] All core dependencies verified.

:dependencies_done
echo.
echo.

:: ============== STEP 6: Data Directory Setup ==============
echo [6/6] Verifying data directories...
if not exist "%BASE_DIR%\data" mkdir "%BASE_DIR%\data"
if not exist "%BASE_DIR%\data\sentinel_db" mkdir "%BASE_DIR%\data\sentinel_db"
if not exist "%BASE_DIR%\Credentials" mkdir "%BASE_DIR%\Credentials"
echo [OK] Data directories ready
echo.

:: =================== Launch Application ===================
echo ==========================================================
echo 🚀 Starting SentinelRAG Stack...
echo ==========================================================
echo 📍 Dashboard: http://127.0.0.1:%PORT%
echo 🧠 Brain:     INITIALIZING (Please Wait)
echo ==========================================================
echo.
echo NOTE: Initial model loading may take 30-60 seconds...
echo.

:: Launch using the virtual environment's Python
:: We use explicit host 127.0.0.1 to avoid binding issues on some Windows setups
:: We use --no-reload by default to ensure maximum stability on Windows
"%VENV_PYTHON%" -m uvicorn src.api.main:app %RELOAD_FLAG% --host 127.0.0.1 --port %PORT%

:: Handle exit
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FATAL ERROR] SentinelRAG stopped unexpectedly (Exit Code: %ERRORLEVEL%)
    echo.
    echo Troubleshooting Steps:
    echo 1. Is Ollama running? Run: ollama serve
    echo 2. Is port %PORT% occupied? Check with: netstat -ano ^| findstr :%PORT%
    echo 3. Missing Models? Models are downloaded on first run.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

endlocal