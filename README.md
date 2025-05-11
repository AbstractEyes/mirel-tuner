# Mirel-Tuner

**Status: Alpha Dry-Run Verified ✅ | Windows-Compatible 🪟 | Under Active Development**

## 🚀 Overview and Purpose
Mirel-Tuner is a modular AI training orchestrator designed for Stable Diffusion XL and beyond. It enables per-layer regulation, dynamic scheduler overrides, and hybrid multiprocessing with per-device per-card per-process per-thread level control of model allocation, data, processing, and more. This repo was collaboratively built by Philip, GPT 4o (Mirel), and GPT O3, with critical architecture and format decisions made by lambda mathematical and hardware choice.

Systemic iteration is crucial based on performance and optimization. This repository will undergo many changes before the official stable build is announced, and the outcome will be based on months of research and investigation before the repo is labeled stable.

The primary goal is to write less code for more outcome, while enabling the developer the speed and utilization required in rapid fashions that will allow rapid AI training on less powerful devices, while still enabling the full utilization of larger ai structures like ulysses and pyring for training, inference, integration, experimentation, merging, separating, and any sort of experiments that can be lined up.

**The goal to be a carefully - dynamically maintained  one-stop shop for quick data, training, layer modification, and rapid AI iteration with an easy setup for both students and experts alike**.

### CORE STRENGTHS OF HUGGINGFACE_HUB
* Huggingface_hub is a very powerful model repo and a wrapper based on loading LITERALLY ANYTHING ON THEIR HUB.

### CORE PROBLEMS WITH HUGGINGFACE_HUB
* It tries to do too much, causing a cascade of additional problems when trying to load certain elemental pieces of diffusion models and other models within controlled environments to rapidly iterate or modify those sets.


### CORE STRENGTHS OF DIFFUSERS
* Diffusers is a very powerful and robust pipeline-based combination training and inference system.
* The system is designed to be modular and extensible - allowing for expansion within... seemingly reasonable limits.

### CORE PROBLEMS WITH DIFFUSERS
* Diffusers has a very modular system - with a large learning curve and overhead.
* The diffusers requirements are not laid in stone, and the dependencies are not always clear - oftentimes completely unavailable or systemically not working with the majority of components, if any.
* The system - is powerful and utilizable in ease-of-manner fashions with multiple downsides. has a large developer learning overhead. 
* It is difficult to simply jump into with new or experimental models, and difficult to adapt your own diffusion model pipelines to it if the system requires additional custom code atop the standard "accepted" diffusers pipeline system.

### CORE STRENGTHS OF KERAS
* Keras is a powerful system with very low level layer complexity exposed at very high level points.
* The system is designed for rapid experimentation with rapid outcome - very useful for prototyping and utilization.
* The system is designed to be modular and extensible - allowing for expansion within limits. 

### CORE PROBLEMS WITH KERAS
* Entry into the depths of keras requires systemic understanding that can take some time to get the hang of.
* Layers and the like seem streamlined, but when placed under heavy load and scrutiny face serious optimization problems and reward less than expected.
* Many formulas and systems are hidden or obfuscated, making it difficult to understand multiple underlying mechanics of the system.
* Many environments simply cannot support keras - windows PCs suffer a great deal trying to make it function at all.
* It does not have much native diffusion code to access behaviors of trained models like stable diffusion - behavior that diffusers and huggingface_hub have direct access to.


# 🧠 Our expansions to solve the core access and learning problems.

* We are building a new system that will allow for the rapid integration of diffusers and huggingface_hub models into a single system, while allowing for the rapid iteration and experimentation of those different brands of models.
* The system will be built around the idea of a "hook" - a simple, modular, and extensible system that allows for the rapid integration of new models and pipelines into the diffusers and huggingface_hub systems.

## Cooperative cohesion - forming new bonds and bridging gaps
* We are adopting the bottom-level flexibility of keras design, while maintaining the top-level simplicity of diffusers and huggingface_hub.
* We will regulate the imports and dependencies of the diffusers and huggingface_hub systems for stable interfaces and operations.
* We will enable pipe-esque behavior in controlled environments for rapid iteration and experimentation, while utilizing safetensors and diffusers-style model saving and transposition from type to type.
* Whatever we can't grab from the system, we build. 
* Whatever we can't build with the system, we monkey patch into the system so it can.
* Whatever we can't monkey patch, means we need to build a new system to do it.
* If that fails, C will do it.

The hardware is all there, the software is all there, the systems are all there. We just need to build the bridges and the roads to connect them.

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

This under constant development and will not be labeled stable until the system is fully functional and all components are verified to work with the system on multiple devices and setups.

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

### Architectural choices and design decisions
The key word here; is cross-utilization. We are building a system that will allow for the rapid integration of diffusers and huggingface_hub models into a single system, while allowing for the rapid iteration and experimentation of those different brands of models.

To solve this problem, we build a new system with direct causal similarities as a concrete foundation.
- ✅ Initial hooked structure
- ✅ diffusers pipeline hooks
- ⚠️ pytorch model training hooks
- ⚠️ mirel training hooks
- ✅ dataset hooks
- ✅ bucketing hooks
- ⚠️ traditional scheduler hooks
- ⚠️ custom scheduler hooks
- ⚠️ traditional optimizer hooks
- ⚠️ custom optimizer hooks
- ⚠️ learn rate hooks
- ⚠️ gradient hooks
- ⚠️ noise scheduler hooks
- ⚠️ sigma modification hooks
- ⚠️ loss hooks
- ⚠️ optimizer hooks
- ✅ model hooks
- ✅ device hooks
- ✅ bus hooks
- ✅ process hooks
- ✅ layer hooks

# Runs and component todos

- ✅ Hello world - dry run complete
- ✅ Basic model loading and device allocation
- ⚠️ Loading any version supported diffuser model
- ⚠️ Loading any version supported torch model
- ⚠️ Loading any version supported keras model
- ✅ Baseline requirements and environment setup
- ✅ Correct directory structure and config loading for v1 pre-multi hook established


- ✅ Core dataset loading single data type
- ✅ Core dataset loading multi-data type
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
