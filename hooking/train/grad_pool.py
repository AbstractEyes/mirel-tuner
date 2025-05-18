from __future__ import annotations
import torch
from typing import Dict, Any, Callable

class GradPool:
    """
    Central place to:
      • register tensors that need gradient checkpointing
      • accumulate grads across micro-batches
      • offload & swap modules to CPU to fit into VRAM
    Hooks call .register(tensor, tag="vae") or .register_module(mod, tag="block.3")
    """
    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._accum: Dict[str, torch.Tensor] = {}

    # ----------------------------------------------------------------
    def register(self, tensor: torch.Tensor, tag: str):
        # store ref; later .checkpoint(tag) can be applied
        self._registry[tag] = tensor

    def register_module(self, mod: torch.nn.Module, tag: str):
        self._registry[tag] = mod

    # ----------------------------------------------------------------
    def checkpoint(self, tag: str):
        t = self._registry[tag]
        return torch.utils.checkpoint.checkpoint(lambda: t, use_reentrant=False)

    def accumulate(self, tag: str, grad: torch.Tensor):
        self._accum[tag] = self._accum.get(tag, 0) + grad

    def step(self, scale: float = 1.0):
        """Flush accumulated gradients back to registered objects."""
        for tag, grad in self._accum.items():
            obj = self._registry.get(tag)
            if isinstance(obj, torch.nn.Module):
                for p in obj.parameters():
                    if p.grad is None:
                        p.grad = grad.clone() * scale
                    else:
                        p.grad.add_(grad, alpha=scale)
            elif torch.is_tensor(obj):
                if getattr(obj, "grad", None) is None:
                    obj.grad = grad.clone() * scale
                else:
                    obj.grad.add_(grad, alpha=scale)
        self._accum.clear()

    # ----------------------------------------------------------------
    def offload(self, tag: str, device="cpu"):
        obj = self._registry[tag]
        if isinstance(obj, torch.nn.Module):
            obj.to(device)
        elif torch.is_tensor(obj):
            self._registry[tag] = obj.to(device)
