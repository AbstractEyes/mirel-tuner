"""
ensure_deps.py  ·  v1.1  (single requirements file + cached lock)

CURRENT SCOPE
──────────────────────────────────────────────
• Installs dependencies from repo-root `requirements.txt`.
• Platform markers (PEP 508) auto-select OS-specific wheels.
• Plugin + user requirements are stubbed; touching them prints a
  warning for a future plugin loader.
• Generates a per-OS lock under   .cache/locks/lock-<os>.txt
  (folder auto-created).  Uses --strip-extras to silence pip-tools 8.0
  deprecation warning.
• Idempotent: respects $MIREL_DEPS_OK inside a single process.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT        = Path(__file__).resolve().parents[1]
REQ_FILE    = ROOT / "requirements.txt"

CACHE_DIR   = ROOT / ".cache" / "locks"
LOCK_FILE   = CACHE_DIR / f"lock-{platform.system().lower()}.txt"

SENTINEL    = "MIREL_DEPS_OK"

# ── plugin & user stubs ─────────────────────────────────────────────
PLUG_CFG    = ROOT / "user" / "plugins_enabled.json"
USER_REQ    = ROOT / "user" / "requirements.txt"

def _warn_stub(path: Path, label: str) -> None:
    print(f"[warn] {label} support coming in a future update — '{path}' ignored.", file=sys.stderr)

# ────────────────────────────────────────────────────────────────────

def _pip_install() -> None:
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check", "--exists-action=i",
            "-r", str(REQ_FILE)
        ]
    )

def _refresh_lock() -> None:
    try:
        import piptools  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pip-tools"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(
        [
            "pip-compile",
            "--quiet",
            "--strip-extras",          # future default; silences warning
            "-o", str(LOCK_FILE),
            str(REQ_FILE),
        ]
    )

def satisfy_once() -> None:
    if os.environ.get(SENTINEL):
        return

    if PLUG_CFG.exists():
        _warn_stub(PLUG_CFG, "Plugin")
    if USER_REQ.exists():
        _warn_stub(USER_REQ, "User requirements")

    _pip_install()
    _refresh_lock()
    os.environ[SENTINEL] = "1"

# ── CLI entry ───────────────────────────────────────────────────────
if __name__ == "__main__":
    satisfy_once()
    print(f"✓ Dependencies satisfied. Lock written to {LOCK_FILE}")
