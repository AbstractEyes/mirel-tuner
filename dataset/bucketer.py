from typing import List, Dict, Any, Iterable, Optional, Callable
from .hooks import DataHook, ComposeHook, BaseDataHook

class ImageBucketer:
    def __init__(
        self,
        manifest: List[Dict[str, Any]],
        cfg: BucketConfig,
        hook: DataHook | None = None,
    ):
        self.cfg = cfg
        self.manifest = manifest
        self.hook = hook or BaseDataHook()
        self._scan_and_bucket()

    # ------------------------------------------------------ #
    # internal: load / bucket                                #
    # ------------------------------------------------------ #
    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        # 1) disk load
        from PIL import Image
        sample = {"image": Image.open(meta["path"]).convert("RGB")}
        sample = self.hook.on_sample_load(sample, meta=meta, bucketer=self)
        # minor resize/center-crop to bucket resolution here …
        sample = self.hook.on_sample_ready(sample, meta=meta, bucketer=self)
        return sample

    # ------------------------------------------------------ #
    # iterator yields collated batch                        #
    # ------------------------------------------------------ #
    def __iter__(self):
        for bucket_key, idxs in self.bucket_map.items():
            batches = chunk(idxs, self.cfg.batch_size)
            for batch_indices in maybe_shuffle(batches, self.cfg.shuffle):
                raw_batch = [self._load_sample(self.manifest[i]) for i in batch_indices]
                yield self.hook.on_batch_collate(raw_batch, bucket=self, res=bucket_key)
