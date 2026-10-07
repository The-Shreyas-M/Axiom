@echo off
setlocal
cd /d "%~dp0"
title Axiom
echo ==================================================
echo   Axiom - offline research paper assistant
echo ==================================================

where python >nul 2>nul
if errorlevel 1 (
  echo [!] Python 3.10+ not found. Install it from https://www.python.org/downloads/ ^(tick "Add to PATH"^) and re-run.
  pause & exit /b 1
)

where ollama >nul 2>nul
if errorlevel 1 (
  echo [..] Ollama not found - trying to install it with winget...
  winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
  where ollama >nul 2>nul
  if errorlevel 1 (
    echo [!] Please install Ollama from https://ollama.com/download then re-run run.bat
    pause & exit /b 1
  )
)

if not exist .venv\Scripts\python.exe (
  echo [..] First run: creating environment and installing packages ^(a few minutes^)...
  python -m venv .venv || ( echo [!] venv failed & pause & exit /b 1 )
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\python -m pip install -q -r requirements.txt || ( echo [!] pip install failed & pause & exit /b 1 )
)

rem Make sure the Ollama server is up (it normally auto-starts on Windows)
curl -s http://localhost:11434/api/tags >nul 2>nul || ( start "" /min ollama serve & timeout /t 4 >nul )

if not exist data\.setup_done (
  echo [..] First run: downloading models ^(one time, needs internet^)...
  .venv\Scripts\python -W ignore -m axiom.setup || ( echo [!] Setup incomplete - fix the message above and re-run. & pause & exit /b 1 )
  if not exist data mkdir data
  echo ok> data\.setup_done
)

echo [OK] Starting Axiom at http://127.0.0.1:8000  ^(use the Quit button in the app to stop^)
.venv\Scripts\python -W ignore -m axiom.server
