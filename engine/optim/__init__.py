import torch
from .surge_adafactor import SurgeAdafactor

_OPTIM_REG = {
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "surge_adafactor": SurgeAdafactor,  # placeholder
}

def get_optimizer(name: str, params, **cfg):
    try:
        return _OPTIM_REG[name.lower()](params, **cfg)
    except KeyError as e:
        raise ValueError(f"Unknown optimiser '{name}'.") from e
