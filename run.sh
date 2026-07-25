#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# MammoDoctor — one-command launcher
# Usage:  ./run.sh            (launches the doctor-facing app)
#         ./run.sh app        (launches the lightweight app.py)
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

APP="mammo_doctor.py"
[ "${1:-}" = "app" ] && APP="app.py"

# Pick a Python: prefer the project venv, then the 'aicd' conda env, then PATH.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "/opt/anaconda3/envs/aicd/bin/python" ]; then
    PY="/opt/anaconda3/envs/aicd/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

echo "▶  Python : $PY"
echo "▶  App    : $APP"

# Ensure Streamlit is importable; if not, tell the user how to install.
if ! "$PY" -c "import streamlit" 2>/dev/null; then
    echo "✗  Streamlit not found in this environment."
    echo "   Install dependencies first:"
    echo "     $PY -m pip install -r requirements.txt"
    exit 1
fi

exec "$PY" -m streamlit run "$APP" \
    --server.port "${PORT:-8501}" \
    --server.address "127.0.0.1" \
    --browser.gatherUsageStats false
