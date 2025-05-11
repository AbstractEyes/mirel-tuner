#!/usr/bin/env python
"""
test_trainer.py – bucket-based smoke test that drives

  • dataset.buckets.BucketDataset  (multi-resolution, per-bucket hooks)
  • program.trainer.Trainer        (async fit loop + Hook dispatch)
  • associate.pipeline_wrapper.PipelineWrapper (SDXL container)

Run:  python test_trainer.py
"""

# ── stdlib / third-party ───────────────────────────────────────────────
import json
from pathlib import Path
import itertools
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# ── project imports ────────────────────────────────────────────────────
from associate.pipeline_wrapper import PipelineWrapper
from program.trainer import Trainer
from program.hooks import BaseHook, Hook
from dataset.buckets import (
    BucketDataset, BucketConfig, ComposeCallback,
    ImageLoadCB, ToDeviceCB,
)

# ===============================================================
# 1) Load config  ------------------------------------------------
# ===============================================================
CFG_PATH = Path("configs/default_config.json")
cfg = json.loads(CFG_PATH.read_text())

required = ["batch_size", "epochs", "learning_rate",
            "data_dir"]                     # data_dir now required
for k in required:
    if k not in cfg:
        raise KeyError(f"Config missing key: {k}")

def main():
    # set seed for reproducibility
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ===============================================================
    # 2) Build SDXL UNet via PipelineWrapper  ------------------------
    # ===============================================================
    pipe_wrapper = PipelineWrapper(cfg)
    unet = pipe_wrapper.container.pipeline.unet        # torch.nn.Module

    # ===============================================================
    # 3) BucketDataset + DataLoader  --------------------------------
    # ===============================================================
    def build_manifest(root: str):
        recs = []
        for p in Path(root).rglob("*.png"):
            with Image.open(p) as im:
                w, h = im.size
            recs.append({"path": str(p), "width": w, "height": h})
        if not recs:
            raise RuntimeError(f"No .png files found in {root}")
        return recs

    manifest = build_manifest(cfg["data_dir"])

    bucket_cfg = BucketConfig(
        resolutions=[(512, 512), (640, 640), (768, 768)],
        batch_size=cfg["batch_size"],
        target_ratios={(512, 512): 0.3, (640, 640): 0.5, (768, 768): 0.2},
    )

    callbacks = ComposeCallback([
        ImageLoadCB(),                         # resize → tensor per bucket
        ToDeviceCB(device="cpu"),              # push tensors to CPU for test
    ])

    dataset = BucketDataset(manifest, bucket_cfg, callbacks)
    loader  = DataLoader(dataset,
                         batch_size=None,      # each yield is a ready batch
                         num_workers=0,
                         pin_memory=True)

    # ===============================================================
    # 4) Optimiser, loss, noise scheduler config  -------------------
    # ===============================================================
    optim = torch.optim.AdamW(unet.parameters(), lr=cfg["learning_rate"])
    loss_fn = lambda out, batch: nn.functional.mse_loss(out, batch["image"])
    noise_sched_cfg = {"name": "constant"}     # ConstantNoiseScheduler

    # ===============================================================
    # 5) Optional Trainer Hooks  ------------------------------------
    # ===============================================================
    class LoggingHook(BaseHook):
        async def on_forward_end(self, outputs, trainer):
            print(f"[LOG] step {trainer.state['step']:>4} "
                  f"shape {outputs.shape} bucket={trainer.last_bucket}")

    hooks: list[Hook] = [LoggingHook()]

    # ===============================================================
    # 6) Instantiate & run Trainer  ---------------------------------
    # ===============================================================
    trainer = Trainer(
        model           = unet,
        optimiser       = optim,
        loss_fn         = loss_fn,
        noise_sched_cfg = noise_sched_cfg,
        device          = "cpu",           # CPU smoke test
        concurrency     = "sync",
        workers         = 0,
        hooks           = hooks,
    )

    trainer.fit(loader, epochs=cfg["epochs"])
    print("\n✓ Bucket-based trainer completed without error")

if __name__ == "__main__":
    main()
    # ─────────────────────────────────────────────────────────────────
    # self-test: generating 15 dummy images …
    # batches/epoch: 8
    # batch 0: res=(512, 512) tensor=(2, 3, 512, 512)
    # batch 1: res=(640, 640) tensor=(2, 3, 640, 640)
    # batch 2: res=(768, 768) tensor=(2, 3, 768, 768)
    # batch 3: res=(512, 512) tensor=(2, 3, 512, 512)
    # batch 4: res=(640, 640) tensor=(2, 3, 640, 640)
    # batch 5: res=(768, 768) tensor=(2, 3, 768, 768)
    # batch 6: res=(512, 512) tensor=(2, 3, 512, 512)
    # batch 7: res=(640, 640) tensor=(2, 3, 640, 640)
