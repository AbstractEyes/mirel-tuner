# associate/containers/diffusion.py
##############################################
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline

from engine.bus import get_bus

bus = get_bus()
##############################################
@dataclass
class DeviceSpec:
    primary: str            # "cuda:0" or "cpu"
    offload: Optional[str] = None            # off-GPU staging device
##############################################
class DiffusionPipelineContainer:
    """
    Thin utility around Diffusers pipelines that

      • accepts either a repo ID / local dir OR an already-built pipeline
      • applies LoRAs
      • fast device swap / off-load helpers
      • broadcasts 'pipeline_moved' when .to() is called
    """

    def __init__(
        self,
        model_path: str | None = None,       # repo ID or folder
        *,
        pipeline: DiffusionPipeline | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        device_spec: DeviceSpec,
        dtype: torch.dtype = torch.float16,
        disable_safety_checker: bool = True,
        enable_xformers: bool = True,
    ):
        if pipeline is None:
            variant = "fp16" if dtype == torch.float16 else None
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                revision=revision,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                variant=variant,
                safety_checker=None if disable_safety_checker else None,
            )

        self.pipeline = pipeline
        self.device_spec = device_spec

        # move to primary device
        self.pipeline.to(self.device_spec.primary)
        bus.emit("pipeline_moved", device=self.device_spec.primary)

        # optional xformers
        if enable_xformers and hasattr(
            self.pipeline, "enable_xformers_memory_efficient_attention"
        ):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as err:                      # noqa: BLE001
                print("[warn] xFormers unavailable:", err)

    ##########################################
    # class-level constructor from existing pipeline
    @classmethod
    def from_existing(
        cls,
        pipe: DiffusionPipeline,
        device_spec: DeviceSpec,
    ) -> "DiffusionPipelineContainer":
        return cls(
            model_path=None,
            pipeline=pipe,
            device_spec=device_spec,
            revision=None,
            cache_dir=None,
        )

    ##########################################
    # LoRA utility
    def apply_lora(self, adapters: List[Path]) -> None:
        for lora in adapters:
            self.pipeline.load_lora_weights(str(lora))

    ##########################################
    # device helpers
    def to(self, device: str | torch.device) -> None:
        self.pipeline.to(device)
        bus.emit("pipeline_moved", device=str(device))

    def offload(self) -> None:
        if self.device_spec.offload:
            self.to(self.device_spec.offload)

    ##########################################
    # scheduler swap helper
    def swap_scheduler(self, name: str) -> None:
        from diffusers import DDPMScheduler
        sched_cls = getattr(
            __import__("diffusers", fromlist=[name]),
            name,
            DDPMScheduler,
        )
        self.pipeline.scheduler = sched_cls.from_config(
            self.pipeline.scheduler.config
        )
##############################################
