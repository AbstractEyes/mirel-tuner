from .base import BaseSigmaSchedule


class LinearSigmaSchedule(BaseSigmaSchedule):
    """σ(t) = start + (end-start) * (t / (T-1))."""

    def __init__(self, start: float = 1.0, end: float = 0.01):
        self.start = float(start)
        self.end = float(end)

    def sigma(self, t_idx: int, total_steps: int) -> float:
        frac = t_idx / max(total_steps - 1, 1)
        return self.start + (self.end - self.start) * frac

    def update(self, *, start: float | None = None, end: float | None = None, **_):
        if start is not None:
            self.start = float(start)
        if end is not None:
            self.end = float(end)
