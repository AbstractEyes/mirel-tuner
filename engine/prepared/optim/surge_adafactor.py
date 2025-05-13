import math
from torch.optim.optimizer import Optimizer

class SurgeAdafactor(Optimizer):
    def __init__(self, params, lr: float = 1e-3, beta2: float = 0.999):
        defaults = dict(lr=lr, beta2=beta2)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        # TODO: real update logic
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad, alpha=-group["lr"])
        return loss
