from .base import BaseTimestepSchedule

class LinearTimestepSchedule(BaseTimestepSchedule):
    def step(self) -> int:
        t = self.cursor % self.total
        self.cursor += 1
        return t
