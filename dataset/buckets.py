"""
dataset/buckets.py
──────────────────
Callback-first bucketing system, multi-GPU & multi-resolution ready.
No torchvision required.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Protocol, runtime_checkable

import random, math, shutil, tempfile
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

# ───────────────────────────── helpers ────────────────────────────────
def _chunk(seq: Sequence[int], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def _shard_indices(ids: torch.Tensor, dist, batch: int):
    trim = (ids.numel() // (dist.world * batch)) * (dist.world * batch)
    ids  = ids[:trim]
    return ids[dist.rank::dist.world].clone()

# ─────────────────────── callback protocol & base ─────────────────────
@runtime_checkable
class Callback(Protocol):
    def on_manifest(self, meta: Dict[str, Any]) -> Dict[str, Any]: ...
    def on_bucket_enter(self, bucket_key, epoch: int, **ctx): ...
    def on_bucket_exit(self, bucket_key, **ctx): ...
    def on_sample_load(self, sample: Dict[str, Any], **ctx): ...
    def on_sample_ready(self, sample: Dict[str, Any], **ctx): ...
    def on_batch_collate(self, batch: List[Dict[str, Any]], **ctx): ...

class BaseCallback:
    def on_manifest(self, meta): return meta
    def on_bucket_enter(self, *a, **k): ...
    def on_bucket_exit(self, *a, **k): ...
    def on_sample_load(self, s, **k):  return s
    def on_sample_ready(self, s, **k): return s
    def on_batch_collate(self, batch, **ctx):
        keys = batch[0].keys()
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [b[k] for b in batch]
            out[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
        out["bucket_key"] = ctx.get("bucket_key")
        return out

class ComposeCallback(BaseCallback):
    def __init__(self, cbs: List[Callback]): self.cbs = cbs
    def on_manifest(self, meta):
        for cb in self.cbs: meta = cb.on_manifest(meta); return meta
    def on_bucket_enter(self, key, epoch, **ctx):
        for cb in self.cbs: cb.on_bucket_enter(key, epoch, **ctx)
    def on_bucket_exit(self, key, **ctx):
        for cb in self.cbs: cb.on_bucket_exit(key, **ctx)
    def on_sample_load(self, s, **ctx):
        for cb in self.cbs: s = cb.on_sample_load(s, **ctx); return s
    def on_sample_ready(self, s, **ctx):
        for cb in self.cbs: s = cb.on_sample_ready(s, **ctx); return s
    def on_batch_collate(self, batch, **ctx):
        for cb in self.cbs:
            if not isinstance(batch, list): break
            batch = cb.on_batch_collate(batch, **ctx)
        if isinstance(batch, dict) and "bucket_key" not in batch:
            batch["bucket_key"] = ctx.get("bucket_key")
        return batch

# ───────────────────────── bucket structs ─────────────────────────────
@dataclass
class BucketConfig:
    resolutions: List[tuple[int,int]]
    batch_size: int = 4
    shuffle: bool = True
    target_ratios: Optional[Dict[tuple[int,int], float]] = None
    mix_after_epochs: int = 0
@dataclass
class _Bucket: key: tuple[int,int]; idxs: torch.Tensor; ratio: float; ptr: int = 0
@dataclass
class DistInfo: rank:int=0; world:int=1; seed:int=42

# ───────────────────────── BucketDataset ──────────────────────────────
class BucketDataset:
    def __init__(self, manifest: List[Dict[str,Any]], cfg: BucketConfig, cb: ComposeCallback):
        self.mani = [cb.on_manifest(m.copy()) for m in manifest]
        self.cfg, self.cb = cfg, cb
        self.buckets: List[_Bucket] = []
        self._build_buckets()

    def _closest_res(self, w,h):
        return min(self.cfg.resolutions, key=lambda r: abs(r[0]-w)+abs(r[1]-h))
    def _ratio_for(self, res): return (self.cfg.target_ratios or {}).get(res,1.0)

    def _build_buckets(self):
        tmp: Dict[tuple[int,int], List[int]] = {}
        for i,m in enumerate(self.mani):
            res = self._closest_res(m["width"], m["height"])
            tmp.setdefault(res,[]).append(i)
        rng = random.Random(123)
        for res,lst in tmp.items():
            rng.shuffle(lst)
            self.buckets.append(_Bucket(res, torch.tensor(lst), self._ratio_for(res)))

    def _load_sample(self, meta, bucket_key):
        img = Image.open(meta["path"]).convert("RGB")
        s = {"image": img, "meta": meta}
        s = self.cb.on_sample_load(s, meta=meta)
        s = self.cb.on_sample_ready(s, meta=meta, bucket_key=bucket_key)
        return s

    def __len__(self): return sum(len(b.idxs) for b in self.buckets)//self.cfg.batch_size

    def __iter__(self):
        import torch.distributed as dist
        from torch.utils.data import get_worker_info
        worker=get_worker_info(); w_rank=worker.id if worker else 0
        d_rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        d_world=dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        di=DistInfo(d_rank,d_world)
        epoch=0
        while True:
            g=torch.Generator().manual_seed(di.seed+epoch)
            perm=torch.randperm(len(self.mani),generator=g)
            if di.world>1: dist.broadcast(perm,src=0)
            local=_shard_indices(perm,di,self.cfg.batch_size)
            rng=random.Random(di.seed+epoch+19*d_rank)
            buckets=[]
            shard_map={}
            for idx in local:
                res=self._closest_res(self.mani[idx]["width"], self.mani[idx]["height"])
                shard_map.setdefault(res,[]).append(idx.item())
            for res,lst in shard_map.items():
                rng.shuffle(lst)
                buckets.append(_Bucket(res, torch.tensor(lst), self._ratio_for(res)))
            for b in buckets: self.cb.on_bucket_enter(b.key,epoch,rank=d_rank)
            steps=sum(len(b.idxs) for b in buckets)//self.cfg.batch_size
            for _ in range(steps):
                weights=torch.tensor([b.ratio*len(b.idxs) for b in buckets])
                b=buckets[int(torch.multinomial(weights,1))]
                s,e=b.ptr,b.ptr+self.cfg.batch_size
                if e>len(b.idxs): rng.shuffle(b.idxs); s,e=0,self.cfg.batch_size
                ids=b.idxs[s:e]; b.ptr=e
                raw=[self._load_sample(self.mani[i], b.key) for i in ids]
                yield self.cb.on_batch_collate(raw,bucket_key=b.key,rank=d_rank,world=d_world)
            for b in buckets: self.cb.on_bucket_exit(b.key,rank=d_rank)
            epoch+=1

# ───────────────────── resolution-aware Image callback ────────────────
class ImageLoadCB(BaseCallback):
    def _rand_crop(self,img,tgt_w,tgt_h):
        l=random.randint(0,img.width -tgt_w); u=random.randint(0,img.height-tgt_h)
        return img.crop((l,u,l+tgt_w,u+tgt_h))
    def on_sample_load(self,sample,**_):
        if isinstance(sample["image"],(str,Path)):
            sample["image"]=Image.open(sample["image"]).convert("RGB")
        return sample
    def on_sample_ready(self,sample,*,bucket_key,**_):
        tgt_w,tgt_h=bucket_key
        img:Image.Image=sample["image"]; w,h=img.size
        scale=max(tgt_w/w,tgt_h/h)
        img=img.resize((int(w*scale),int(h*scale)),Image.BICUBIC)
        img=self._rand_crop(img,tgt_w,tgt_h)
        arr=np.asarray(img,dtype=np.float32)/255.0
        ten=torch.from_numpy(arr).permute(2,0,1).mul_(2).sub_(1)
        sample["image"]=ten; return sample
    def on_batch_collate(self,batch,*,bucket_key,**_):
        imgs=torch.stack([b["image"] for b in batch])
        return {"image":imgs,"bucket_key":bucket_key}

class ToDeviceCB(BaseCallback):
    def __init__(self,device="cuda:0",non_block=True):
        self.dev,self.nb=device,non_block
    def on_sample_ready(self,s,**_):
        for k,v in s.items():
            if torch.is_tensor(v): s[k]=v.to(self.dev,non_blocking=self.nb)
        return s

# ─────────────────────────── self-test main ───────────────────────────
if __name__=="__main__":  # pragma: no cover
    print("self-test: generating 15 dummy images …")
    tmp=tempfile.mkdtemp()
    try:
        sizes=[(512,512)]*3 + [(640,640)]*7 + [(768,768)]*5
        mani=[]
        for i,(w,h) in enumerate(sizes):
            p=Path(tmp)/f"img_{i}_{w}.png"
            Image.new("RGB",(w,h),(i*17%255,100,150)).save(p)
            mani.append({"path":str(p),"width":w,"height":h})
        cfg=BucketConfig(resolutions=[(512,512),(640,640),(768,768)],
                         batch_size=2,
                         target_ratios={(512,512):0.3,(640,640):0.5,(768,768):0.2})
        cbs=ComposeCallback([ImageLoadCB(),ToDeviceCB(device="cpu")])
        data=BucketDataset(mani,cfg,cbs)
        import itertools
        print("batches/epoch:",len(data))
        for bi,b in enumerate(itertools.islice(data,len(data))):
            print(f"batch {bi}: res={b['bucket_key']} tensor={tuple(b['image'].shape)}")
    finally:
        shutil.rmtree(tmp); print("self-test complete.")
