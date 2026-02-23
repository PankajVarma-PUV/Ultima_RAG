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

SET REPO_URL=https://github.com/PankajVarma-PUV/Ultima_RAG.git

echo 🛡️ UltimaRAG Repository Cloner
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
:: Overwrite logic: Remove existing folder if it exists
if exist "%~dp0Ultima_RAG" (
    echo 🗑️ Existing Ultima_RAG folder detected. Overwriting...
    rmdir /s /q "%~dp0Ultima_RAG"
)

:: Use quotes around the path to handle spaces in folder names correctly
git clone "%REPO_URL%" "%~dp0Ultima_RAG"

if %errorlevel% equ 0 (
    echo ---------------------------------
    echo ✅ SUCCESS! Repository cloned into 'Ultima_RAG' folder.
    echo 📂 Path: "%~dp0Ultima_RAG"
    echo.
    echo 💡 NEXT STEP:
    echo 1. Go into the 'Ultima_RAG' folder.
    echo 2. Run 'run_Ultima_RAG.bat' to install dependencies and start the app.
) else (
    echo ---------------------------------
    echo ❌ FAILED to clone. Check your internet or repository visibility.
)

pause

