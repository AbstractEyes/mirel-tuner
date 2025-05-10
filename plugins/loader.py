"""
plugins.loader  –  central registry + dynamic import
• Ensures all plugin deps are installed (via ensure_deps.satisfy_once())
• Imports each plugin’s entry-point
"""
from __future__ import annotations
import importlib, json, sys
from pathlib import Path

from scripts import ensure_deps  # relative import works because scripts is a package

ROOT = Path(__file__).resolve().parents[1]
PLUG_CFG = ROOT / "user" / "plugins_enabled.json"

def _iter_enabled():
    if not PLUG_CFG.exists():
        return []
    return json.loads(PLUG_CFG.read_text())

def load_all():
    # ── 1. Make sure environment is satisfied
    ensure_deps.satisfy_once()

    # ── 2. Import enabled plugins
    for plug_name in _iter_enabled():
        meta_path = ROOT / "plugins" / plug_name / "plugin.toml"
        if not meta_path.exists():
            print(f"[warn] plugin '{plug_name}' missing plugin.toml – skipped")
            continue

        # read entry = "pkg.module:func"
        try:
            import tomllib  # py 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # back-compat

        entry = tomllib.loads(meta_path.read_text())["plugin"]["entry"]
        mod_name, func_name = entry.split(":", 1)
        mod = importlib.import_module(mod_name)
        register_fn = getattr(mod, func_name)
        register_fn()
        print(f"✓ plugin loaded: {plug_name}")

# expose convenience on import
load_all()
