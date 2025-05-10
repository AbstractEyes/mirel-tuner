"""
ModelCache  –  HF-Hub LRU model snapshot manager
------------------------------------------------
• snapshot_download() → local_dir
• get()             → path (raises if absent and download=False)
• evict()           → free N oldest entries or by bytes
• offload_to_cpu()  → move tensors out of GPU for idle models
"""
from __future__ import annotations
import shutil, time, json
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import snapshot_download, hf_hub_download

CACHE_ROOT = Path.home() / ".cache" / "mirel_tuner"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

_METADATA = CACHE_ROOT / "meta.json"
_DEFAULT_LIMIT_GB = 40  # hard cap (can tune via config)

def _load_meta() -> Dict[str, float]:
    if _METADATA.exists():
        return json.loads(_METADATA.read_text())
    return {}

def _save_meta(meta: Dict[str, float]) -> None:
    _METADATA.write_text(json.dumps(meta))

class ModelCache:
    """Simple path-only LRU cache (expandable to RAM/VRAM semantics)."""

    def __init__(self, size_limit_gb: int = _DEFAULT_LIMIT_GB):
        self.size_limit = size_limit_gb * (1024**3)
        self.meta = _load_meta()

    # ────────────────────────────────────── public API

    def get(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        allow_download: bool = True,
        **snapshot_kwargs,
    ) -> Path:
        """Return local snapshot path, downloading if absent."""
        key = f"{repo_id}@{revision or 'main'}"
        if key not in self.meta:
            if not allow_download:
                raise FileNotFoundError(key)
            path = snapshot_download(
                repo_id, revision=revision, local_dir=CACHE_ROOT / key, **snapshot_kwargs
            )
            self.meta[key] = time.time()
            _save_meta(self.meta)
            self._evict_if_full()
        else:
            path = CACHE_ROOT / key
            self.meta[key] = time.time()  # touch
            _save_meta(self.meta)
        return Path(path)

    def evict(self, n: int = 1) -> None:
        """Evict *n* oldest snapshots."""
        for key in sorted(self.meta, key=self.meta.get)[:n]:
            shutil.rmtree(CACHE_ROOT / key, ignore_errors=True)
            del self.meta[key]
        _save_meta(self.meta)

    def offload_to_cpu(self, pipeline) -> None:
        """Move modules to CPU (placeholder — extend per-module)."""
        import torch
        for _, module in pipeline.components.items():
            if hasattr(module, "to"):
                module.to(torch.device("cpu"))

    # ────────────────────────────────────── helpers

    def _evict_if_full(self) -> None:
        total = sum(p.stat().st_size for p in CACHE_ROOT.rglob("*") if p.is_file())
        while total > self.size_limit and self.meta:
            self.evict(1)
            total = sum(p.stat().st_size for p in CACHE_ROOT.rglob("*") if p.is_file())
