import re, torch
from typing import List, Dict, Any
from hooking.bus.train_hooks import BaseHook
from engine.layer_sched import get_layer_schedule

class BaseLayerHook(BaseHook):
    """Controls a subset of model parameters via regex patterns."""

    def __init__(self, model: torch.nn.Module, patterns: List[str], schedule_cfg: Dict[str, Any]):
        self.patterns = [re.compile(p) for p in patterns]
        self.schedule = get_layer_schedule(**schedule_cfg)
        self._collect_params(model)

    # ---------------------------------------------------------------- #
    def _collect_params(self, model: torch.nn.Module):
        self.param_refs = []  # (param, base_lr, optim_group)
        for name, param in model.named_parameters():
            if any(pat.search(name) for pat in self.patterns):
                self.param_refs.append(param)

    # ---------------------------------------------------------------- #
    async def on_optim_step_start(self, trainer, **ctx):
        scale = self.schedule.scale(trainer.state["step"], trainer.state["epoch"])
        for pg in trainer.optimiser.param_groups:
            for p in pg["params"]:
                if p in self.param_refs:
                    pg["lr"] = pg.get("base_lr", pg["lr"]) * scale

class LayerHookManager:
    def __init__(self, hooks: List[BaseLayerHook]):
        self.hooks = hooks

    async def apply(self, trainer):
        for h in self.hooks:
            await h.on_optim_step_start(trainer=trainer)
