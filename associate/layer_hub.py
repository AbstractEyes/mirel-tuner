# associate/layer_hub.py
##############################################
from __future__ import annotations
import torch
from typing import Dict, Callable, Any, List

from engine.bus import emit   # global event bus

##############################################
class LayerHub:
    """
    Runtime registry of every nn.Module inside a model/pipeline.

    • Call reindex() once after construction and after every swap.
    • get(<path>) returns the live nn.Module (by ref).
    • swap(<path>, new_module) hot-replaces a block and re-indexes.
    • tag(<path>, **kv) stores arbitrary metadata on that layer.

    Paths use the dotted names from .named_modules():
        "unet.down_blocks.0.attentions.1"
    """

    def __init__(self, root: torch.nn.Module):
        self.root = root
        self._index: Dict[str, torch.nn.Module] = {}
        self.reindex()

    ##########################################
    def reindex(self) -> None:
        self._index.clear()
        for name, mod in self.root.named_modules():
            self._index[name] = mod

    ##########################################
    def get(self, path: str) -> torch.nn.Module:
        return self._index[path]

    ##########################################
    def swap(self, path: str, new_mod: torch.nn.Module) -> None:
        """
        Replace the module at <path> with new_mod.
        Emits 'layer_swapped' with old and new modules.
        """
        parent_path, attr = path.rsplit(".", 1)
        parent = self.root if parent_path == "" else self._index[parent_path]

        old_mod = getattr(parent, attr)
        setattr(parent, attr, new_mod)
        self.reindex()

        emit("layer_swapped",
             path=path,
             old_module=old_mod,
             new_module=new_mod)

    ##########################################
    def tag(self, path: str, **meta) -> None:
        """
        Attach arbitrary metadata to a layer (e.g. device plan).
        """
        mod = self._index[path]
        for k, v in meta.items():
            setattr(mod, f"_hub_{k}", v)

    ##########################################
    # helper: iterate over layers that match a predicate
    def iter(self, filter_fn: Callable[[str, torch.nn.Module], bool]):
        for n, m in self._index.items():
            if filter_fn(n, m):
                yield n, m
##############################################
