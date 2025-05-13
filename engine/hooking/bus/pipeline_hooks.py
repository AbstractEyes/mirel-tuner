# associate/pipeline_hooks.py
##############################################
from __future__ import annotations
from typing import Protocol, List, Dict, Any
from diffusers import DiffusionPipeline
from associate.layer_hub import LayerHub
##############################################
class PipelineHook(Protocol):
    """
    Uniform interface for run-time pipeline events.
    Any method may be omitted; ComposePipelineHook will skip it.
    """

    # called once right after the full pipeline + LayerHub exist
    def on_pipeline_ready(self,
                          pipeline: DiffusionPipeline,
                          hub: LayerHub,
                          **ctx) -> None: ...

    # emitted each time LoRA weights are applied
    def on_lora_applied(self,
                        pipeline: DiffusionPipeline,
                        paths: List[str],
                        **ctx) -> None: ...

    # when .swap_scheduler() succeeds
    def on_scheduler_swapped(self,
                             pipeline: DiffusionPipeline,
                             name: str,
                             **ctx) -> None: ...

    # when precision or device moves
    def on_pipeline_moved(self,
                          pipeline: DiffusionPipeline,
                          device: str,
                          **ctx) -> None: ...

##############################################
class BasePipelineHook:
    """No-op base so users can subclass only what they need."""
    def on_pipeline_ready(self, pipeline, hub, **ctx): ...
    def on_lora_applied(self, pipeline, paths, **ctx): ...
    def on_scheduler_swapped(self, pipeline, name, **ctx): ...
    def on_pipeline_moved(self, pipeline, device, **ctx): ...

##############################################
class ComposePipelineHook(BasePipelineHook):
    """Chain multiple PipelineHook objects together."""
    def __init__(self, hooks: List[PipelineHook]):
        self.hooks = hooks

    def on_pipeline_ready(self, pipeline, hub, **ctx):
        for h in self.hooks:
            if hasattr(h, "on_pipeline_ready"):
                h.on_pipeline_ready(pipeline, hub, **ctx)

    def on_lora_applied(self, pipeline, paths, **ctx):
        for h in self.hooks:
            if hasattr(h, "on_lora_applied"):
                h.on_lora_applied(pipeline, paths, **ctx)

    def on_scheduler_swapped(self, pipeline, name, **ctx):
        for h in self.hooks:
            if hasattr(h, "on_scheduler_swapped"):
                h.on_scheduler_swapped(pipeline, name, **ctx)

    def on_pipeline_moved(self, pipeline, device, **ctx):
        for h in self.hooks:
            if hasattr(h, "on_pipeline_moved"):
                h.on_pipeline_moved(pipeline, device, **ctx)
##############################################
