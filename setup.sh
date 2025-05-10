#!/usr/bin/env bash
set -euo pipefail
VENV=".venv"
REQ_ROOT="requirements.txt"
OS_REQ="requirements/linux.txt"
LOCK_FILE="requirements/lock.txt"

if [[ "$(uname -s)" == "Darwin" ]]; then
  OS_REQ="requirements/linux.txt"     # macOS uses linux.txt for now
elif [[ -f /etc/runpod.info ]]; then
  OS_REQ="requirements/runpod.txt"
fi

PY_CMD="python3"
echo "Using $(${PY_CMD} --version)"

# ── create venv if missing ────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
  "${PY_CMD}" -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# ── upgrade pip & pip-tools ───────────────────────────────────────────
pip install --quiet --upgrade pip pip-tools

# ── install base + OS reqs ────────────────────────────────────────────
pip install --quiet -r "$REQ_ROOT" -r "$OS_REQ"

# ── plugin / user extras ──────────────────────────────────────────────
python scripts/ensure_deps.py

# ── generate / refresh lock ───────────────────────────────────────────
pip-compile --quiet -o "$LOCK_FILE" "$REQ_ROOT" "$OS_REQ"
echo "Lock written to $LOCK_FILE"

echo
echo "✓ Environment ready and activated."
echo "  Run:  python main.py --config configs/quickstart.json --dry-run"
echo "  Deactivate with:  deactivate"
