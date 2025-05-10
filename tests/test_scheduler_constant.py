import torch

from engine.noise_sched import get_scheduler

def test_constant_noise_scheduler_determinism():
    latent = torch.zeros(1, 4, 8, 8)
    sched = get_scheduler("constant", sigma=0.5, seed=42)
    state = {"latent": latent}

    out1 = sched.step(0, state.copy())["latent"]
    out2 = sched.step(0, {"latent": torch.zeros_like(latent)})["latent"]

    # Same seed → identical noise
    assert torch.allclose(out1, out2)
