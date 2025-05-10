import torch
from ..base_sched import BaseScheduler
from ..sigma_sched import get_sigma_schedule


class DDPMVPredScheduler(BaseScheduler):
    """Classic DDPM σ schedule with v-prediction target."""

    def __init__(self, total_steps: int, sigma_sched_cfg: Dict[str, Any] | None = None):
        self.total_steps = total_steps
        self.sigma_sched = get_sigma_schedule(**(sigma_sched_cfg or dict(name="linear")))
        self.generator = torch.Generator()

    def step(self, t_idx: int, state, **kw):
        latent = state["latent"]
        sigma = self.sigma_sched.sigma(t_idx, self.total_steps)
        noise = torch.randn_like(latent, generator=self.generator) * sigma
        state["latent"] = latent + noise
        state["sigma"] = sigma  # expose to loss/metrics
        return state
