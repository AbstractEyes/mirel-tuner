"""
ModelCache v3.1  –  lazy, metadata-driven downloads
────────────────────────────────────────────────────
• Reads only model_index.json first.
• Determines required weight filenames:
      – if weights/tensor_files listed → use them directly
      – else (e.g. SD-XL) scan repo once and pick *.safetensors / *.bin
        inside the component folders named in model_index.json.
• Downloads each blob exactly once into the global HF cache,
  then links it into the project cache (cfg["cache_dir"] or default).
• Per-OS LRU eviction by total bytes.
"""

from __future__ import annotations
import json, os, shutil, time
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download, list_repo_files

HF_CACHE_BASE = (
    Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
)
_DEFAULT_LIMIT_GB = 40


class ModelCache:
    def __init__(
        self,
        size_limit_gb: int = _DEFAULT_LIMIT_GB,
        cache_root: Path | None = None,
    ):
        self.cache_root = cache_root or Path.home() / ".cache" / "mirel_tuner"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.size_limit = size_limit_gb * (1024**3)
        self.meta_file = self.cache_root / "meta.json"
        self.meta: Dict[str, float] = (
            json.loads(self.meta_file.read_text()) if self.meta_file.exists() else {}
        )

    # ──────────────────────────────────────────────────────────────
    def get(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        allow_download: bool = True,
    ) -> Path:
        key = f"{repo_id}@{revision or 'main'}"
        local_dir = self.cache_root / key

        if key not in self.meta:
            if not allow_download:
                raise FileNotFoundError(key)

            local_dir.mkdir(parents=True, exist_ok=True)

            # 1) Fetch model_index.json only
            idx_path = Path(
                hf_hub_download(
                    repo_id,
                    filename="model_index.json",
                    revision=revision,
                    cache_dir=HF_CACHE_BASE,
                    local_dir=local_dir,
                )
            )

            # 2) Resolve weight filenames
            weights = self._resolve_weights(repo_id, revision, idx_path)

            # 3) Download/link each weight file
            for fname in weights:
                hf_hub_download(
                    repo_id,
                    filename=fname,
                    revision=revision,
                    cache_dir=HF_CACHE_BASE,
                    local_dir=local_dir,
                )

            self.meta[key] = time.time()
            self._save_meta()
            self._evict_if_full()
        else:
            self.meta[key] = time.time()
            self._save_meta()

        return local_dir

    # ───────────────────────── helpers ───────────────────────────
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

        # If only folders listed, enumerate repo once
        if folders and not needed:
            repo_files = list_repo_files(repo_id, revision=revision)
            for f in repo_files:
                if any(f.startswith(folder + "/") for folder in folders) and f.endswith(
                    (".safetensors", ".bin")
                ):
                    needed.add(f)

        if not needed:
            raise RuntimeError(
                f"No weight files resolved for {repo_id}@{revision or 'main'} "
                f"via {idx_path}. Cannot proceed."
            )

        return sorted(needed)

    def _save_meta(self) -> None:
        self.meta_file.write_text(json.dumps(self.meta))

    def evict(self, n: int = 1) -> None:
        for key in sorted(self.meta, key=self.meta.get)[:n]:
            shutil.rmtree(self.cache_root / key, ignore_errors=True)
            del self.meta[key]
        self._save_meta()

    def _evict_if_full(self) -> None:
        total = sum(
            p.stat().st_size for p in self.cache_root.rglob("*") if p.is_file()
        )
        while total > self.size_limit and self.meta:
            self.evict(1)
            total = sum(
                p.stat().st_size for p in self.cache_root.rglob("*") if p.is_file()
            )
