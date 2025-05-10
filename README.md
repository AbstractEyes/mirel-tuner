# LOBA-Trainer

A next-generation, slider-style LoRA/LoCoN/“lobotomy” trainer built on HuggingFace Diffusers and Accelerate.  
Dive into fully modular, pipeline-based fine-tuning for SD-XL (and beyond) with:

- **Associate**: runtime caches, model‐container interfaces  
- **Program**: training driver, pluggable optimizers, noise/​model/​layer schedulers, timestep hooks  

---

## 🚀 Core Directives

1. **Schema First**  
   - Follow & adhere to the core directory/schema exactly.  
   - _Do not_ modify unless explicitly instructed.

2. **Monkey-Patch Only on Demand**  
   - We patch sequentially & iteratively, _only_ when requested.

3. **HuggingFace Hub Is Primary Mule**  
   - `huggingface_hub` methods + storage power our download, versioning & offload.

4. **CUDA122 Workhorse**  
   - All pipeline loads, GPU offload/​onload, and inference/training loops assume `cuda:122` device.

---

## 📂 Repository Layout

```
loba_trainer/
├── associate/
│ ├── caches/ # on-disk & in-mem caches
│ ├── containers/ # model/pipeline container interfaces
│ └── model_cache.py # high-level get/put/unload
├── program/
│ ├── main.py # entry-point & CLI
│ ├── pipeline_wrapper.py # DiffusionPipeline loader/unloader (cuda122, offload, xformers, etc.)
│ ├── optimizers/ # 12 standard optimizer stubs (AdamW, RAdam, NovoGrad…)
│ ├── schedulers/
│ │ ├── noise/ # DDIM, PNDM, K-LMS, etc.
│ │ ├── model/ # cosine, linear_warmup, polynomial…
│ │ └── layer/ # block-wise_decay, attention_only…
│ └── timestep_schedulers/ # linear, exponential, custom timestep controllers
├── README.md
└── loba_trainer.zip
```
---

## 🔧 Installation

```
git clone https://github.com/you/loba_trainer.git
cd loba_trainer
pip install -r requirements.txt
Requirements are pinned for Colab-friendly CUDA122 compatibility:

diffusers>=0.29.0

transformers>=4.30.0

accelerate>=0.20.0

huggingface_hub>=0.17.0
```

---

## ⚙️ Quickstart
Cache & Load a Model

```
from associate.model_cache import ModelCache
cache = ModelCache(device="cuda:122")
pipe = cache.load("stabilityai/stable-diffusion-xl-base-1.0")
Run Training
```

```
python program/main.py \
  --pretrained_model stabilityai/stable-diffusion-xl-base-1.0 \
  --optimizer adamw \
  --lr 1e-4 \
  --scheduler cosine \
  --noise_scheduler ddim \
  --iterations 1000 \
  --batch_size 2 \
  --out_dir ./output
```

Unload When Done

```
cache.unload("stabilityai/stable-diffusion-xl-base-1.0")
```
## 🔌 Extending

```
Optimizers / program/optimizers/*.py
Drop in new .py files exporting a get_optimizer(params, lr, **cfg) factory.

Schedulers / program/schedulers/{noise,model,layer}/*.py
Each file should expose get_scheduler(optimizer, **cfg) or step_noise(...).

Timestep Hooks / program/timestep_schedulers/*.py
Register via the --timestep_scheduler CLI flag.

Pipeline Loader / program/pipeline_wrapper.py
Centralizes DiffusionPipeline.from_pretrained + offload, xFormers, gradient checkpoints, dtype settings.
```

## 🎯 Next Steps
“Hello World” Minimal Run – confirm pipeline load → dummy train loop

Masked Loss & Multi-Noise – add Huber/L1/L2, MIN_SNR, γ-noise, bucketed dataloader

Region-Control & Style-Shift – integrate binary/greyscale masks, style adapters

Cluster Deploy – Ulysses-ring offload, multi-node Accelerate, DeepSpeed support

## 📜 License
Apache 2.0
Contributions welcome! Let’s build a monument to flexible, robust SD-XL fine-tuning.