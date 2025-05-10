"""Abstract scheduler interface for LOBA.

All concrete schedulers—noise, timestep, layer, LR—must inherit
from `BaseScheduler` and implement the `step` method.

Signature
---------
step(t_idx: int, state: dict[str, Any], **kwargs) -> dict[str, Any]

* `t_idx`   : current integer timestep (0-based, monotonically increasing or scheduled)
* `state`   : mutable dict carrying in-progress tensors or metadata
* `kwargs`  : scheduler-specific extras (e.g., noise_scale)

The method **must** return the (potentially mutated) `state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseScheduler(ABC):
    """Minimal contract for every scheduler variant."""

    @abstractmethod
    def step(self, t_idx: int, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute one scheduler tick and return updated state."""
        raise NotImplementedError

    def reset(self) -> None:  # optional override
        """Reset internal buffers—called at epoch boundaries."""
        pass
