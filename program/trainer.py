# program/trainer.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
from torch.utils.data import DataLoader

from engine.noise_sched import get_scheduler
from hooking.bus.train_hooks import Hook
from engine.async_utils import run_async


@dataclass
class Trainer:
    # ───────────────────────────────────────── public args ─────────────
    model: torch.nn.Module
    optimiser: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
    noise_sched_cfg: Dict[str, Any]

    device: str | torch.device = "cuda"

    # concurrency
    concurrency: Literal["sync", "thread", "process"] = "sync"
    workers: int = 4

    # hooks
    hooks: List[Hook] = field(default_factory=list)

    # per-bucket min-SNR gamma offsets  e.g. { "(512,512)": -1.0, "(768,768)": +0.5 }
    gamma_map: Dict[str, float] = field(default_factory=dict)
    base_gamma: float = 7.0

    # ───────────────────────────────────────── internal fields ─────────
    _executor_pool: Optional[ThreadPoolExecutor | ProcessPoolExecutor] = field(init=False, default=None)
    _loop: Optional[asyncio.AbstractEventLoop] = field(init=False, default=None)
    state: Dict[str, Any] = field(init=False, default_factory=dict)

    # ───────────────────────────────────────── lifecycle ───────────────
    def __post_init__(self):
        self.device = torch.device(self.device)
        self.model.to(self.device)

        self.noise_sched = get_scheduler(**self.noise_sched_cfg)

        if self.concurrency == "thread":
            self._executor_pool = ThreadPoolExecutor(max_workers=self.workers)
        elif self.concurrency == "process":
            self._executor_pool = ProcessPoolExecutor(max_workers=self.workers)

        self.state.update(step=0, epoch=0, snr_scale=1.0)

    # ───────────────────────────────────────── hook dispatcher ─────────
    async def _dispatch(self, event: str, *args, **kwargs):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        tasks = []
        for hk in self.hooks:
            cb = getattr(hk, event, None)
            if cb is None:
                continue
            tasks.append(
                run_async(
                    cb,
                    *args,
                    executor=self._executor_pool,
                    loop=self._loop,
                    trainer=self,
                    **kwargs,
                )
            )
        if tasks:
            await asyncio.gather(*tasks)

    # ───────────────────────────────────────── public fit ──────────────
    async def _fit_async(self, train_loader: DataLoader, epochs: int):
        await self._dispatch("on_fit_start")
        for epoch in range(epochs):
            self.state["epoch"] = epoch
            await self._train_epoch_async(train_loader)
        await self._dispatch("on_fit_end")

    def fit(self, train_loader: DataLoader, epochs: int = 1):
        asyncio.run(self._fit_async(train_loader, epochs))

    # ───────────────────────────────────────── epoch loops ─────────────
    async def _train_epoch_async(self, loader: DataLoader):
        await self._dispatch("on_epoch_start", self.state["epoch"])
        self.model.train()

        for batch_idx, batch in enumerate(loader):
            self.state["step"] += 1

            await self._dispatch(
                "on_batch_start",
                batch_idx=batch_idx,
                batch_meta=batch.get("bucket_key"),
            )

            # move tensors to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device, non_blocking=True)

            # choose latent key
            latent = batch.get("latent", batch.get("image"))
            if latent is None:
                raise KeyError("Batch must contain 'latent' or 'image' tensor.")

            # resolve per-bucket gamma
            bucket_key = batch.get("bucket_key")
            gamma = self.base_gamma + self.gamma_map.get(str(bucket_key), 0.0)

            # noise schedule step
            sched_state = self.noise_sched.step(
                idx=batch_idx,
                state={"latent": latent, "gamma": gamma},
            )
            noisy = sched_state["latent"]

            # forward
            outputs = self.model(noisy)
            await self._dispatch("on_forward_end", outputs=outputs)

            loss = self.loss_fn(outputs, batch)
            await self._dispatch("on_loss_computed", loss=float(loss))

            # backward
            loss.backward()
            await self._dispatch("on_backward_end")

            await self._dispatch("on_optim_step_start")
            self.optimiser.step()
            self.optimiser.zero_grad()
            await self._dispatch("on_optim_step_end")

            await self._dispatch("on_step_end")

        await self._dispatch("on_epoch_end", self.state["epoch"])

    # ───────────────────────────────────────── utils ───────────────────
    def add_hook(self, hook: Hook):
        self.hooks.append(hook)

    def close(self):
        if self._executor_pool:
            self._executor_pool.shutdown()
