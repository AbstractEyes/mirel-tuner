# engine/grad_pool.py
##############################################
from __future__ import annotations
import torch
from typing import Dict, Any, List, Optional

from engine.bus import emit, on   # global event bus

##############################################
class GradPool:
    """
    Central registry for tensors / modules that need:
      • gradient checkpointing
      • micro-batch gradient accumulation
      • CPU↔GPU off-load to manage VRAM

    Hooks or training code call:
      pool.register_module(path, module)
      pool.offload(path)                 (to CPU)
      pool.load(path, device="cuda:0")   (back to GPU)
      pool.accumulate(path, grad)
    """

    def __init__(self):
        self._mods: Dict[str, torch.nn.Module] = {}
        self._accum: Dict[str, torch.Tensor] = {}

    ##########################################
    def register_module(self, path: str, mod: torch.nn.Module) -> None:
        self._mods[path] = mod

    ##########################################
    # checkpoint: run fn under torch.utils.checkpoint
    def checkpoint(self, path: str, fn, *args, **kwargs):
        mod = self._mods[path]
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)

    ##########################################
    # accumulation helpers
    def accumulate(self, path: str, grad: torch.Tensor) -> None:
        self._accum[path] = self._accum.get(path, 0) + grad.detach()

    def flush(self, scale: float = 1.0) -> None:
        for path, g in self._accum.items():
            mod = self._mods[path]
            for p in mod.parameters():
                if p.grad is None:
                    p.grad = g.clone() * scale
                else:
                    p.grad.add_(g, alpha=scale)
        self._accum.clear()

    ##########################################
    # off-load helpers
    def offload(self, path: str, device: str = "cpu") -> None:
        self._mods[path].to(device)

    def load(self, path: str, device: str) -> None:
        self._mods[path].to(device)

##############################################
# singleton instance
pool = GradPool()

##############################################
# optional automatic accumulation via bus events
@on("after_bwd")
async def _auto_accum(event, payload):
    if not payload.get("accum_path"):   # hook may pass a path list
        return
    for path, grad in payload["accum_path"]:
        pool.accumulate(path, grad)

@on("optim_step")
async def _flush_accum(event, payload):
    pool.flush()
##############################################
