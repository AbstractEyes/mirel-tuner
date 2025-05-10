from .linear import LinearTimestepSchedule

_TS_REG = {"linear": LinearTimestepSchedule}

def get_timestep_schedule(name: str, **cfg):
    try:
        return _TS_REG[name.lower()](**cfg)
    except KeyError as e:
        raise ValueError(f"Unknown timestep schedule '{name}'.") from e
