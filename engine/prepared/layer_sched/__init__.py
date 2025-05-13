from .linear_lr import LinearLRLayerSchedule
_LAYER_SCHED = {"linear_lr": LinearLRLayerSchedule}

def get_layer_schedule(name: str, **cfg):
    try:
        return _LAYER_SCHED[name](**cfg)
    except KeyError as e:
        raise ValueError(f"Unknown layer schedule '{name}'.")
