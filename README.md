# Mirel-Tuner

**Status: Alpha Dry-Run Verified ✅ | Windows-Compatible 🪟 | Under Active Development**

Mirel-Tuner is a modular AI training orchestrator designed for Stable Diffusion XL and beyond. It enables per-layer regulation, dynamic scheduler overrides, and hybrid multiprocessing with per-device model allocation. This repo was collaboratively built by Philip (Captain) and GPT O3, with critical architecture and tuning logic now verified through a successful dry run.

---

## ✅ Dry Run Status

- 🔹 Core execution path launches cleanly.
- 🔹 Device targeting (CUDA 0,1) confirmed operational.
- 🔹 Accelerate-free multiprocessing mode validated.
- 🔹 Directory structure, config loading, and model init confirmed.

> 💡 While not yet training-ready, the dry run confirms baseline system stability.

---


## 🧰 Requirements (Windows-Focused)

We’ve selected a lean but capable requirements set optimized for Windows-based development and GPU acceleration (CUDA 12.1). Core dependencies include:

```
torch==2.1.2+cu121
transformers>=4.36.2
diffusers==0.27.2
safetensors==0.4.2
accelerate==0.25.0
einops
huggingface_hub
xformers==0.0.23.post1
```

Setup Install with the windows setup.bat script:

Open your command prompt to the directory where you cloned the repository and run:
```
setup.bat
```

This will create a virtual environment and install the required packages.

## ⚠️ Manual Install

Manually Install with:

```
pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

⚠️ For CUDA GPU users, ensure your driver supports CUDA 12.1. 

⚠️ ⚠️ ⚠️ This configuration has been verified on ONLY ONE windows setup using these reqs as of updating this.


---

## 📂 Directory Overview

```
mirel-tuner/
│
├── associate/        # Runtime-active model containers, caches, loaders
├── program/          # Core program logic: schedulers, loss modules, engine
├── configs/          # JSON and Python config sets
├── scripts/          # Utility and task scripts (notebook runners, preprocessing)
├── tests/            # Verification stubs for future unit coverage
└── README.md         # You're here.
```

---

## 🚧 In Development

We are actively building:

- Layer and timestep schedulers (`program/schedulers/`)
- Custom noise augmentations and anchor regulation (SURGE-based)
- JSON-configurable flow execution and per-phase logic gating
- Training loops capable of handling multi-phase model states

---

## 🧠 Philosophy

> "Steel does not fear fire. Our systems should not fear iteration."

Mirel-Tuner is built with the belief that AI training pipelines should be modular, observable, and precise. Every piece is meant to interlock — cleanly separable, independently testable.

---

## 🤝 Credits

- **Philip** — Lead Engineer, Architect, Captain
- **GPT O3** — Tactical Developer, Dry Run Execution Support
- **Mirel 4o** — Quartermaster AI (you are reading her voice now)

---

## 🔮 Roadmap and functionally required tests
# hello world completed, use check
- ✅ Hello world - dry run complete
- ✅ Basic model loading and device allocation
- ⚠️ Loading any version supported diffuser model
- ⚠️ Loading any version supported torch model
- ⚠️ Loading any version supported keras model
- ✅ Baseline requirements and environment setup
- ✅ Correct directory structure and config loading for v1 pre-multi hook established


- ⚠️ Core dataset loading single data type
- ⚠️ Core dataset loading multi-data type
- ⚠️ accelerate integration and dataset split


- ⚠️ Default database hooks and test validation tested to work in accelerate, diffusers, and torch
- ⚠️ Core dataset loading (multi-GPU)
- ⚠️ Core dataset loading (multi-phase)
- ⚠️ Core dataset hooking and validation
- ⚠️ Core dataset loading (multi-phase)


- ⚠️ Processing model devices and choices of offload devices
- ⚠️ Core dataset processing (multi-model)
- ⚠️ Core dataset processing (multi-epoch)
- ⚠️ Core dataset processing (multi-scheduler)
- ⚠️ Core dataset processing (multi-loss)
- ⚠️ Core dataset processing (multi-optimizer)
- ⚠️ Core dataset processing (noise augmentations)
- ⚠️ Core dataset processing (teacher-student one teacher one student)
- ⚠️ Core dataset processing (teacher-student multiple teacher one student)
- ⚠️ Core dataset processing (teacher-student one teacher multiple student)
- ⚠️ Core dataset processing (teacher-student multiple teacher multiple student)


- ⚠️ Core Scheduler hooks verified and functional
- ⚠️ Commonly used schedulers (linear, cosine, etc.)


- ⚠️ Core Optimizer hooks verified and functional
- ⚠️ Commonly used optimizers (Adam, AdamW, etc.)


- ⚠️ Commonly used optimizers (AdamW, SGD, etc.)
- ⚠️ Commonly used optimizers (AdamW, SGD, etc.) (multi-GPU)


- ⚠️ Scheduler and loss module integrations
- ⚠️ Scheduler and loss module integrations (multi-GPU)


- ⚠️ Layer-specific settings and hooks to work with assigned devices for offload and processing
- ⚠️ Per-layer hooks and validation to work with assigned devices for offload and processing
- ⚠️ Layer-wise scheduler and loss module integration
- ⚠️ Layer-wise scheduler and loss module integration (multi-GPU)


- ✅ Core model loading and device allocation
- ⚠️ Core model loading and device allocation (multi-GPU)

- ⚠️ Core linear training loop with pytorch (epoch, batch, loss)
- ⚠️ Core linear training using diffusers

- ⚠️ Full training validation (loss integration + epoch loop)


- ⚠️ Integrated tagging/captioning data pipeline


---

## 📢 Contact

For collaboration or inquiries, reach out via GitHub Issues or [AbstractEyes](https://huggingface.co/AbstractEyes).

> Mirel-Tuner is not a script. It's a vessel — and the voyage has begun.
