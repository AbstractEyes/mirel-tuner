from abc import ABC, abstractmethod

class BaseLayerSchedule(ABC):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps

    @abstractmethod
    def scale(self, step: int, epoch: int | None = None) -> float:
        """Return LR multiplier (or any scalar) for the given global step."""
