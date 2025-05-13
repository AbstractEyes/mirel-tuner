from .constant import ConstantNoiseScheduler
from diffusers import DDPMScheduler

_SCHED_REGISTRY = {
    "constant": ConstantNoiseScheduler,
    "ddpmscheduler": DDPMScheduler,
}


def get_scheduler(name: str, **kwargs):
    try:
        cls = _SCHED_REGISTRY[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown noise scheduler '{name}'.") from e
    return cls(**kwargs)
