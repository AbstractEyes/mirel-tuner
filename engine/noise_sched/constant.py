"""Constant-variance Gaussian noise scheduler.

Adds N(0, σ²) noise to `state["latent"]` every call, where σ is fixed.
Useful for smoke tests and as a deterministic baseline.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..base_sched import BaseScheduler


class ConstantNoiseScheduler(BaseScheduler):
    def __init__(self, sigma: float = 1.0, seed: int | None = None) -> None:
        self.sigma = float(sigma)
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def step(self, t_idx: int, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        latent = state.get("latent")
        if latent is None:
            raise KeyError("State dict must contain a `latent` tensor.")

        noise = torch.randn_like(latent, generator=self._generator) * self.sigma
        state["latent"] = latent + noise
        return state
