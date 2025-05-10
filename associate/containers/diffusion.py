"""
diffusion.py
────────────
Thin wrapper around diffusers pipelines with device control.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline


@dataclass
class DeviceSpec:
    primary: str          # e.g. "cuda:0" or "cpu"
    offload: Optional[str] = None


class DiffusionPipelineContainer:
    def __init__(
        self,
        model_path: str,                 # repo ID or local folder
        revision: str | None,
        cache_dir: str | None,
        device_spec: DeviceSpec,
        dtype: torch.dtype = torch.float16,
        disable_safety_checker: bool = True,
        enable_xformers: bool = True,
    ):
        variant = "fp16" if dtype == torch.float16 else None

        self.device_spec = device_spec
        self.pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            variant=variant,
            safety_checker=None if disable_safety_checker else None,
        )

        self.pipeline.to(device_spec.primary)

        if enable_xformers and hasattr(
            self.pipeline, "enable_xformers_memory_efficient_attention"
        ):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] xFormers unavailable: {e}")

    # ── extension hooks ───────────────────────────────────────────────
    def apply_lora(self, adapters: List[Path]) -> None:
        for lora in adapters:
            self.pipeline.load_lora_weights(str(lora))

    def offload(self) -> None:
        if self.device_spec.offload:
            self.pipeline.to(self.device_spec.offload)
