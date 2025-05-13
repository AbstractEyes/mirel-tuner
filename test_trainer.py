#!/usr/bin/env python
##############################################
# test_trainer.py – end-to-end sanity run
# Uses:
#   • ModelCache  ( weight staging )
#   • PipelineWrapper ( bus-aware, LayerHub inside )
#   • BucketDataset  ( multi-resolution, DataHook chain )
#   • Trainer  ( emits events via BusRelayHook )
#   • Engine Bus  ( single comms spine )
##############################################

import json, random
from pathlib import Path
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import DataLoader

from engine.bus import emit, on
from associate.pipeline.pipeline_wrapper import PipelineWrapper
from engine.hooking.dataset import (
    BucketDataset, BucketConfig, ComposeCallback,
    ImageLoadCB, ToDeviceCB
)
from program.trainer import Trainer
from engine.hooking.bus import BaseHook

##############################################
# 1) load config
cfg = json.loads(Path("configs/default_config.json").read_text())

##############################################
# 2) build pipeline (stages weights, emits 'model_ready')
pipe_wrap = PipelineWrapper(cfg)
unet      = pipe_wrap.container.pipeline.unet

##############################################
# 3) build bucketed dataset
def build_manifest(root: str):
    lst=[]
    for p in Path(root).rglob("*.png"):
        with Image.open(p) as im: w,h=im.size
        lst.append({"path":str(p),"width":w,"height":h})
    if not lst: raise RuntimeError("no PNG files in data_dir")
    return lst

manifest = build_manifest(cfg["data_dir"])
bucket_cfg = BucketConfig(
    resolutions   = cfg["bucket"]["resolutions"],
    batch_size    = cfg["batch_size"],
    target_ratios = cfg["bucket"].get("target_ratios", {})
)

class BucketEventCB(ImageLoadCB):
    def on_batch_collate(self, batch, *, bucket_key, **ctx):
        emit("bucket_enter", key=bucket_key)
        out = super().on_batch_collate(batch, bucket_key=bucket_key)
        emit("bucket_exit", key=bucket_key)
        return out

callbacks = ComposeCallback([
    BucketEventCB(),
    ToDeviceCB(device="cpu")
])

dataset = BucketDataset(manifest, bucket_cfg, callbacks)
loader  = DataLoader(dataset,
                     batch_size=None,
                     num_workers=0,
                     pin_memory=True)

##############################################
# 4) trainer -> bus relay hook
class BusRelayHook(BaseHook):
    async def on_epoch_start(self, epoch_idx, trainer):
        emit("epoch_start", idx=epoch_idx, broadcast=True)
    async def on_epoch_end(self, epoch_idx, trainer):
        emit("epoch_end", idx=epoch_idx, broadcast=True)
    async def on_step_end(self, batch_idx, trainer):
        emit("step_end", idx=batch_idx)

##############################################
# 5) simple logger listening on bus
@on("step_end")
async def log_step(event, payload):
    if payload["idx"] % 5 == 0:
        print("step", payload["idx"])

##############################################
# 6) optimiser, loss, trainer
optim   = torch.optim.AdamW(unet.parameters(), lr=cfg["learning_rate"])
loss_fn = lambda out, batch: nn.functional.mse_loss(out, batch["image"])

trainer = Trainer(model           = unet,
                  optimiser       = optim,
                  loss_fn         = loss_fn,
                  noise_sched_cfg = {"name":"constant"},
                  device          = "cpu",
                  hooks           = [BusRelayHook()],
                  concurrency     = "sync",
                  workers         = 0)

##############################################
# 7) run
random.seed(cfg.get("seed", 42))
torch.manual_seed(cfg.get("seed", 42))
trainer.fit(loader, epochs=cfg["epochs"])
print("✓ bus-centric smoke test finished")
##############################################
