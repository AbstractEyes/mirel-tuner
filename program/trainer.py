# program/trainer.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from engine.noise_sched import get_scheduler
from .hooks import Hook
from .async_utils import run_async


@dataclass
class Trainer:
    # -------------------------------------------------- public ctor args
    model: torch.nn.Module
    optimiser: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    noise_sched_cfg: Dict[str, Any]
    device: str | torch.device = "cuda"

    # concurrency
    concurrency: Literal["sync", "thread", "process"] = "sync"
    workers: int = 4

    # hooks
    hooks: List[Hook] = field(default_factory=list)

    # -------------------------------------------------- internal fields
    _executor_pool: Optional[ThreadPoolExecutor | ProcessPoolExecutor] = field(
        init=False, default=None
    )
    state: Dict[str, Any] = field(init=False, default_factory=dict)

    # -------------------------------------------------- lifecycle
    def __post_init__(self):
        self.device = torch.device(self.device)
        self.model.to(self.device)
        self.noise_sched = get_scheduler(**self.noise_sched_cfg)

        if self.concurrency == "thread":
            self._executor_pool = ThreadPoolExecutor(max_workers=self.workers)
        elif self.concurrency == "process":
            self._executor_pool = ProcessPoolExecutor(max_workers=self.workers)

        # global state visible to hooks / schedulers
        self.state.update(
            step=0,
            epoch=0,
            device=self.device,
            snr_scale=1.0,
        )

    # -------------------------------------------------- hook dispatcher
    async def _dispatch(self, event: str, *args, **kwargs):
        tasks = []
        for hk in self.hooks:
            cb = getattr(hk, event, None)
            if cb is None:
                continue
            tasks.append(
                run_async(
                    cb,
                    *args,
                    trainer=self,
                    executor=self._executor_pool,
                    loop=asyncio.get_running_loop(),
                    **kwargs,
                )
            )
        if tasks:
            await asyncio.gather(*tasks)

    # -------------------------------------------------- fit / validate
    async def _fit_async(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
    ):
        await self._dispatch("on_fit_start")
        for epoch in range(epochs):
            self.state["epoch"] = epoch
            await self._train_epoch_async(train_loader)
            if val_loader is not None:
                await self._validate_epoch_async(val_loader)
        await self._dispatch("on_fit_end")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
    ):
        """Synchronous wrapper."""
        asyncio.run(self._fit_async(train_loader, val_loader, epochs))

    # -------------------------------------------------- train / val impl
    async def _train_epoch_async(self, loader: DataLoader):
        await self._dispatch("on_epoch_start")
        self.model.train()

        for batch_idx, batch in enumerate(loader):
            self.state["step"] += 1
            await self._dispatch("on_batch_start", batch_idx=batch_idx, batch_meta=batch.get("bucket_meta"))

            # --- CPU → GPU transfer
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            # --- optional snr_scale from DataHook meta
            if "bucket_meta" in batch and "snr" in batch["bucket_meta"]:
                self.state["snr_scale"] = batch["bucket_meta"]["snr"]

            # --- noise schedule
            latent = batch["latent"]
            batch_state = {"latent": latent, "snr_scale": self.state["snr_scale"]}
            batch_state = self.noise_sched.step(batch_idx, batch_state)
            noisy_latent = batch_state["latent"]

            # --- forward / loss / backward
            outputs = self.model(noisy_latent)
            await self._dispatch("on_forward_end", outputs=outputs)

            loss = self.loss_fn(outputs, batch["target"])
            await self._dispatch("on_loss_computed", loss=loss.item())

            loss.backward()
            await self._dispatch("on_backward_end")

            self.optimiser.step()
            self.optimiser.zero_grad()
            await self._dispatch("on_step_end")

        await self._dispatch("on_epoch_end")

    async def _validate_epoch_async(self, loader: DataLoader):
        await self._dispatch("on_validation_start")
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                # validation logic omitted for brevity
                pass
        await self._dispatch("on_validation_end")

    # -------------------------------------------------- utils
    def add_hook(self, hook: Hook):
        self.hooks.append(hook)

    def close(self):
        if self._executor_pool:
            self._executor_pool.shutdown()
