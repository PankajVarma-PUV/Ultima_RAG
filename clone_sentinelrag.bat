@echo off
SET REPO_URL=https://github.com/PankajVarma-PUV/SentinelRAG.git

echo 🛡️ SentinelRAG Repository Cloner
echo ---------------------------------
echo 📂 Target: %REPO_URL%

:: Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git is not installed. Please install it from https://git-scm.com/
    pause
    exit /b
)

echo 🚀 Cloning repository...
:: Use quotes around the path to handle spaces in folder names correctly
git clone "%REPO_URL%" "%~dp0SentinelRAG_Clone"

if %errorlevel% equ 0 (
    echo ---------------------------------
    echo ✅ SUCCESS! Repository cloned into 'SentinelRAG_Clone' folder.
    echo 📂 Path: "%~dp0SentinelRAG_Clone"
    echo.
    echo 💡 NEXT STEP:
    echo 1. Go into the 'SentinelRAG_Clone' folder.
    echo 2. Run 'run_sentinelrag.bat' to install dependencies and start the app.
) else (
    echo ---------------------------------
    echo ❌ FAILED to clone. Check your internet or repository visibility.
)

pause
