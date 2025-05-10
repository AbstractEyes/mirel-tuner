"""
ModelCache v2 – project-scoped cache that re-uses global HF blob store.
"""
from __future__ import annotations
import json, shutil, time, os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download, hf_hub_download

# ── paths --------------------------------------------------------------------
PROJECT_CACHE = Path.home() / ".cache" / "mirel_tuner"   # symlinks / filtered tree
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)

HF_CACHE_BASE = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"  # object store

_META = PROJECT_CACHE / "meta.json"

def _load_meta() -> Dict[str, float]:
    return json.loads(_META.read_text()) if _META.exists() else {}

def _save_meta(meta: Dict[str, float]) -> None:
    _META.write_text(json.dumps(meta))

_DEFAULT_LIMIT_GB = 40

class ModelCache:
    def __init__(self, size_limit_gb=_DEFAULT_LIMIT_GB, cache_root: Path | None = None):
        self.cache_root = cache_root or Path.home() / ".cache" / "mirel_tuner"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit_gb * (1024**3)
        self.meta_file = self.cache_root / "meta.json"
        self.meta = json.loads(self.meta_file.read_text()) if self.meta_file.exists() else {}

    # ------------------------------------------------------------------
    def get(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        allow_download: bool = True,
    ) -> Path:

        key = f"{repo_id}@{revision or 'main'}"
        local_dir = PROJECT_CACHE / key

        if key not in self.meta:
            if not allow_download:
                raise FileNotFoundError(key)

            if allow_patterns is None:   # default diffusers filter
                allow_patterns = [
                    "model_index.json",
                    "diffusion_*_model.*",
                    "text_encoder/*",
                    "vae/*",
                    "tokenizer/*",
                ]
                ignore_patterns = ["*.msgpack", "*.onnx", "*.bin.index.*"]

            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=HF_CACHE_BASE,        # global blob store
                local_dir=local_dir,            # project view
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                resume_download=True,
            )
            self.meta[key] = time.time()
            _save_meta(self.meta)
            self._evict_if_full()
        else:
            self.meta[key] = time.time()
            _save_meta(self.meta)

        return local_dir

    # ------------------------------------------------------------------
    def evict(self, n: int = 1) -> None:
        for key in sorted(self.meta, key=self.meta.get)[:n]:
            shutil.rmtree(PROJECT_CACHE / key, ignore_errors=True)
            del self.meta[key]
        _save_meta(self.meta)

    def _evict_if_full(self) -> None:
        total = sum(p.stat().st_size for p in PROJECT_CACHE.rglob("*") if p.is_file())
        while total > self.size_limit and self.meta:
            self.evict(1)
            total = sum(p.stat().st_size for p in PROJECT_CACHE.rglob("*") if p.is_file())
