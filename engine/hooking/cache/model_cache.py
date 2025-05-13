##############################################
from __future__ import annotations
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

from huggingface_hub import hf_hub_download, list_repo_files

##############################################
HF_CACHE_BASE = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
_DEFAULT_LIMIT_GB = 40

##############################################
class ModelCache:

    """
        Resolve and stage exactly the weight blobs a Diffusers pipeline needs.
        • Reads model_index.json first (cheap).
        • Determines weight filenames via tensor_files / weight_files.
        • If none listed (e.g. SD-XL) scans repo once for *.safetensors or *.bin
          inside folders referenced by model_index.json.
        • Downloads each blob once into HF cache, then hard-links into
          project cache (root).
        • LRU eviction by total bytes.
    """

    def __init__(self,
                 root: Path | str | None = None,
                 size_limit_gb: int = _DEFAULT_LIMIT_GB):
        self.root = Path(root).expanduser() if root else Path.home() / ".cache" / "mirel_tuner"
        self.root.mkdir(parents=True, exist_ok=True)

        self.size_limit = size_limit_gb * 1024 ** 3
        self.meta_file  = self.root / "meta.json"
        self.meta: Dict[str, float] = json.loads(self.meta_file.read_text()) if self.meta_file.exists() else {}

    def stage(
            self,
            repo_id: str,
            revision: str | None = None,
            patch_fn: Optional[Callable[[Path], None]] = None,
    ) -> Path:
        key = f"{repo_id}@{revision or 'main'}"
        loc = self.root / key

        if key not in self.meta:
            loc.mkdir(parents=True, exist_ok=True)

            # 1) download model_index.json only
            idx = Path(
                hf_hub_download(
                    repo_id,
                    filename="model_index.json",
                    revision=revision,
                    cache_dir=HF_CACHE_BASE,
                )
            )
            idx_dst = loc / "model_index.json"
            if not idx_dst.exists():
                os.link(idx, idx_dst)

            # 2) resolve every weight + config file we need
            files = self._resolve_weights(repo_id, revision, idx)

            # 3) download or link each file, preserving sub-folders
            for fname in files:
                blob = hf_hub_download(
                    repo_id,
                    filename=fname,
                    revision=revision,
                    cache_dir=HF_CACHE_BASE,
                )
                dst = loc / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.link(blob, dst)

            if patch_fn:
                patch_fn(loc)

            self.meta[key] = time.time()
            self._save_meta()
            self._evict_if_full()
        else:
            self.meta[key] = time.time()
            self._save_meta()

        return loc

    def _resolve_weights(
            self, repo_id: str, revision: str | None, idx_path: Path
    ) -> List[str]:
        data = json.loads(idx_path.read_text())

        needed: set[str] = set()
        folders: set[str] = set()

        for comp in data.values():
            if isinstance(comp, dict):
                needed.update(comp.get("weights", []))
                needed.update(comp.get("weight_files", []))
                needed.update(comp.get("tensor_files", []))
            elif isinstance(comp, list):
                folders.update({p.strip("/") for p in comp})

        # include root-level config.json
        needed.add("config.json")

        # pull weights + config.json from referenced folders
        if folders:
            repo_files = list_repo_files(repo_id, revision=revision)
            for f in repo_files:
                if not any(f.startswith(f"{d}/") for d in folders):
                    continue
                if f.endswith((".safetensors", ".bin", ".json")):
                    needed.add(f)

        if not needed:
            raise RuntimeError(
                f"No weight or config files resolved for {repo_id}@{revision or 'main'}"
            )

        return sorted(needed)

    ##########################################
    def _save_meta(self) -> None:
        self.meta_file.write_text(json.dumps(self.meta))

    ##########################################
    def evict(self, n: int = 1) -> None:
        for key in sorted(self.meta, key=self.meta.get)[:n]:
            shutil.rmtree(self.root / key, ignore_errors=True)
            del self.meta[key]
        self._save_meta()

    ##########################################
    def _evict_if_full(self) -> None:
        total = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        while total > self.size_limit and self.meta:
            self.evict(1)
            total = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
