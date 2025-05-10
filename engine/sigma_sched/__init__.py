from .linear import LinearSigmaSchedule

_SIGMA_REGISTRY = {
    "linear": LinearSigmaSchedule,
}

def get_sigma_schedule(name: str, **kwargs):
    try:
        return _SIGMA_REGISTRY[name.lower()](**kwargs)
    except KeyError as e:
        raise ValueError(f"Unknown sigma schedule '{name}'.") from e
