"""Dataset-side hook / injection system for LOBA.

A DataHook can:
  • mutate an individual sample dict after disk read
  • augment / generate additional modalities (e.g. masks, text tokens)
  • modify the collated batch just before it’s returned by DataLoader
"""

from __future__ import annotations
from typing import Protocol, Dict, Any, List


class DataHook(Protocol):
    # -------- sample level --------
    def on_sample_load(self, sample: Dict[str, Any], **ctx) -> Dict[str, Any]: ...
    def on_sample_ready(self, sample: Dict[str, Any], **ctx) -> Dict[str, Any]: ...

    # -------- batch level ---------
    def on_batch_collate(self, batch: List[Dict[str, Any]], **ctx) -> Dict[str, Any]: ...


class BaseDataHook:
    """No-op base so users can subclass and implement only what they need."""

    def on_sample_load(self, sample, **ctx):
        return sample

    def on_sample_ready(self, sample, **ctx):
        return sample

    def on_batch_collate(self, batch, **ctx):
        # default collate: stack common tensor keys; leave objects untouched
        import torch

        collated: Dict[str, Any] = {}
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            if torch.is_tensor(vals[0]):
                collated[k] = torch.stack(vals)
            else:
                collated[k] = vals
        return collated


class ComposeHook(BaseDataHook):
    """Chain multiple hooks together."""

    def __init__(self, hooks: List[DataHook]):
        self.hooks = hooks

    # cascade through each hook in order
    def on_sample_load(self, sample, **ctx):
        for h in self.hooks:
            sample = h.on_sample_load(sample, **ctx)
        return sample

    def on_sample_ready(self, sample, **ctx):
        for h in self.hooks:
            sample = h.on_sample_ready(sample, **ctx)
        return sample

    def on_batch_collate(self, batch, **ctx):
        for h in self.hooks:
            batch = h.on_batch_collate(batch, **ctx)
        return batch
