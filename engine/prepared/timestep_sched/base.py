from __future__ import annotations
from abc import ABC, abstractmethod

class BaseTimestepSchedule(ABC):
    def __init__(self, total_steps: int):
        self.total = total_steps
        self.cursor = 0

    @abstractmethod
    def step(self) -> int:  # returns next t_idx
        ...

    def reset(self):
        self.cursor = 0
