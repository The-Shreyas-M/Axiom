#!/usr/bin/env bash
# Axiom launcher for macOS/Linux.  Usage: ./run.sh
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3.10+ first"; exit 1; }
command -v ollama  >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
[ -d .venv ] || { python3 -m venv .venv; .venv/bin/pip install -q --upgrade pip; .venv/bin/pip install -q -r requirements.txt; }
curl -s localhost:11434/api/tags >/dev/null || { (ollama serve >/dev/null 2>&1 &); sleep 4; }
[ -f data/.setup_done ] || { .venv/bin/python -W ignore -m axiom.setup && mkdir -p data && touch data/.setup_done; }
echo "Axiom running at http://127.0.0.1:8000 (Quit button in the app stops it)"
.venv/bin/python -W ignore -m axiom.server
