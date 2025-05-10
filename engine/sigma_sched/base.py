from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseSigmaSchedule(ABC):
    """Return sigma (std-dev) for a given timestep index."""

    @abstractmethod
    def sigma(self, t_idx: int, total_steps: int) -> float:
        ...

    # optional: mutate internal params on-the-fly
    def update(self, **kwargs): ...
