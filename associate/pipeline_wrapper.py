"""
pipeline_wrapper.py
────────────────────
Builds the Diffusers pipeline, handles device selection,
and forwards cache_dir / revision directly to Hugging Face Hub.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import torch

from associate.containers.diffusion import DiffusionPipelineContainer, DeviceSpec


class PipelineWrapper:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # ── device negotiation ─────────────────────────────────────────
        if torch.cuda.is_available():
            main_dev = cfg.get("devices", {}).get("primary", "cuda:0")
            off_dev = cfg.get("devices", {}).get("offload", "cpu")
        else:
            main_dev, off_dev = "cpu", None
        self.device_spec = DeviceSpec(main_dev, off_dev)

        # ── model location & HF cache path ─────────────────────────────
        model_repo = cfg["model"]["repo_id"]
        model_rev = cfg["model"].get("revision")
        cache_dir = (
            Path(cfg["cache_dir"]).expanduser() if cfg.get("cache_dir") else None
        )

        # ── build container ────────────────────────────────────────────
        self.container = DiffusionPipelineContainer(
            model_path=model_repo,
            revision=model_rev,
            cache_dir=str(cache_dir) if cache_dir else None,
            device_spec=self.device_spec,
            dtype=getattr(torch, cfg["precision"]),
        )

        # ── optional LoRAs ─────────────────────────────────────────────
        if cfg.get("lora"):
            self.container.apply_lora([Path(p) for p in cfg["lora"]])

        # ── optional scheduler swap ────────────────────────────────────
        sched_name = cfg.get("scheduler", "DDPMScheduler")
        if sched_name != "DDPMScheduler":
            self._swap_scheduler(sched_name)

    # ------------------------------------------------------------------
    def _swap_scheduler(self, name: str) -> None:
        from diffusers import DDPMScheduler  # fallback

        sched_cls = getattr(__import__("diffusers", fromlist=[name]), name, DDPMScheduler)
        self.container.pipeline.scheduler = sched_cls.from_config(
            self.container.pipeline.scheduler.config
        )
