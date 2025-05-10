"""
PipelineWrapper orchestrates:
  1. Model caching
  2. Container instantiation
  3. Adapter / scheduler injection
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import torch

from associate.model_cache import ModelCache
from associate.containers.diffusion import DiffusionPipelineContainer, DeviceSpec

class PipelineWrapper:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.cache = ModelCache(size_limit_gb=cfg.get("cache_limit_gb", 40))

        # ── device negotiation
        if torch.cuda.is_available():
            main_dev = cfg.get("devices", {}).get("primary", "cuda:0")
            off_dev  = cfg.get("devices", {}).get("offload", "cpu")
        else:
            main_dev, off_dev = "cpu", None
        self.device_spec = DeviceSpec(main_dev, off_dev)

        # ── fetch model snapshot
        model_path = self.cache.get(
            cfg["model"]["repo_id"],
            revision=cfg["model"].get("revision"),
        )

        # ── build container
        self.container = DiffusionPipelineContainer(
            model_path=model_path,
            device_spec=self.device_spec,
            dtype=getattr(torch, cfg["precision"]),
        )

        # ── apply LoRAs if any
        if cfg.get("lora"):
            self.container.apply_lora([Path(p) for p in cfg["lora"]])

        # ── optional scheduler swap (vanilla diffusers first)
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
