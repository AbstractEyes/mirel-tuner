import torch

_LR_REG = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "constant": torch.optim.lr_scheduler.LambdaLR,
}

def get_lr_scheduler(name: str, optimizer, **cfg):
    try:
        return _LR_REG[name.lower()](optimizer, **cfg)
    except KeyError as e:
        raise ValueError(f"Unknown LR scheduler '{name}'.") from e
