from .base import BaseLayerSchedule

class LinearLRLayerSchedule(BaseLayerSchedule):
    def __init__(self, start: float, end: float, total_steps: int):
        super().__init__(total_steps)
        self.start, self.end = start, end

    def scale(self, step: int, epoch=None):
        frac = min(step / max(self.total_steps - 1, 1), 1.0)
        return self.start + (self.end - self.start) * frac
