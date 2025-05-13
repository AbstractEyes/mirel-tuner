# associate/pipeline_wrapper.py
##############################################
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from associate.model_cache import ModelCache
from associate.layer_hub import LayerHub
from engine.bus import get_bus
from associate.pipeline.diffusion import (
    DiffusionPipelineContainer,
    DeviceSpec,
)

bus = get_bus()
##############################################
class PipelineWrapper:
    """Build a Diffusers pipeline, expose LayerHub, and react to bus events."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._build_pipeline()
        self._register_bus_handlers()

    ##########################################
    def _build_pipeline(self) -> None:
        # device selection
        if torch.cuda.is_available():
            main_dev = self.cfg.get("devices", {}).get("primary", "cuda:0")
            off_dev = self.cfg.get("devices", {}).get("offload", "cpu")
        else:
            main_dev, off_dev = "cpu", None
        self.device_spec = DeviceSpec(main_dev, off_dev)

        # stage weights locally
        cache_root = self.cfg.get("cache_dir", "~/.cache/mirel_tuner")
        staged_dir = ModelCache(cache_root).stage(
            self.cfg["model"]["repo_id"],
            revision=self.cfg["model"].get("revision"),
        )

        # build Diffusers container
        self.container = DiffusionPipelineContainer(
            model_path=str(staged_dir),
            revision=None,
            cache_dir=None,
            device_spec=self.device_spec,
            dtype=getattr(torch, self.cfg["precision"]),
        )

        # apply startup LoRAs
        if self.cfg.get("lora"):
            self.container.apply_lora([Path(p) for p in self.cfg["lora"]])

        # optional scheduler swap
        sched_name = self.cfg.get("scheduler", "DDPMScheduler")
        if sched_name != "DDPMScheduler":
            self._swap_scheduler(sched_name)

        # attach LayerHub and announce readiness
        self.hub: Optional[LayerHub] = None
        if hasattr(self.container.pipeline, "unet"):
            self.hub = LayerHub(self.container.pipeline.unet)
        bus.emit("model_ready", hub=self.hub, pipeline=self.container.pipeline)

    ##########################################
    def _register_bus_handlers(self) -> None:
        @bus.on("apply_lora")
        async def _apply_lora(event, payload):
            paths: List[str] = payload["paths"]
            self.container.apply_lora([Path(p) for p in paths])

        @bus.on("swap_scheduler")
        async def _swap_sched(event, payload):
            self._swap_scheduler(payload["name"])

        @bus.on("set_precision")
        async def _set_precision(event, payload):
            dtype = getattr(torch, payload["dtype"])
            self.container.pipeline.to(dtype)

        @bus.on("pipeline_offload")
        async def _pipeline_offload(event, payload):
            dev = payload.get("device", "cpu")
            self.container.pipeline.to(dev)

        @bus.on("swap_layer")
        async def _swap_layer(event, payload):
            if self.hub is None:
                return
            self.hub.swap(payload["path"], payload["new_module"])

    ##########################################
    def _swap_scheduler(self, name: str) -> None:
        from diffusers import DDPMScheduler
        sched_cls = getattr(__import__("diffusers", fromlist=[name]),
                            name, DDPMScheduler)
        self.container.pipeline.scheduler = sched_cls.from_config(
            self.container.pipeline.scheduler.config
        )
##############################################
