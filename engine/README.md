## Engine utilities

This contains PRIMARILY prepared hooks.

Simple, reusable, multithreaded, multiprocess, multigpu, and multidevice runnable.

The concept here is simple; parity.

We want full access to the various engine components, without having a large learning or developer scale overhead.

Want a new scheduler? Load up a plugin or modify the base script.
No need for 30 different libraries each doing the same thing, simply use the same base engine and modify it to your needs.

We crutch primarily on the torch, diffusers, and keras systems.


### Hook building
Want a new hook? Attach a new hook to the engine, call it.

Want a new core training routine? Attach a training hook to the engine, call it.  

When completed, everything will be fully dynamic and scalable based on modern python features and dynamic lambda programming standards.

This directory contains various utilities and helper functions for the engine. These are used to simplify and streamline the development process, making it easier to work with the engine's core components.

### Contents


#### Grouped
These can do everything from loading a model, to distilling an output from 15 models using 3500 computers.
The engineering is entirely dependent on how far you need to push the hooking, the limitations of your computation range, and the limitations of python.


* model_loader:
* * Has a hooks to load models for all hooks and schedule modifications.
* * model_cache for loading and offloading model through hardware and software mitigation
* * diffusers pipeline loader
* * * diffusers pipeline from_pretrained loader, local and remote
* * * supports all types of diffusers pipelines based on the current tested version.
* * * enables additional hooks to attach to the hooked structure of the trainer's engine components.
* * huggingface_hub from_pretrained loader, local and remote
* * * from_pretrained is a powerful model loading function that can load most models directly from huggingface_hub.
* * * this includes support for hooks to attach to the hooked structure of the trainer's engine components.

* dataset:
* * load_dataset
* * * built using standard Dataset flow primarily through the huggingface_hub dataset class.
* * * careful management of yield and direct hooking with callback are supported on every level
* * bucketing
* * * designed to function within the accelerate framework and environments built around requiring the unique capabilities of bucketing.
* * * supports direct bucketing with bucket scaling and hooks specifically designed to streamline bucketing in an optimal fashion.
* * * 


* lr_sched:
* * Has a hooks to schedule learning rate controlled by the traditional schedulers.
* * * These are more akin to the original full model training schedulers; where the entire model is trained at once and the scheduler maintains learn rate throughout based on full model adjustments.
* model_sched:
* * Has a hooks to schedule models for all hooks and schedule modifications.
* layer_sched:
* * Has a hooks to schedule layers for all hooks and schedule modifications.
* train_sched:
* * Has a hooks to control every level of training for all hooks and schedule modifications.
* * * Independent training routines live here to make things like loras, peft

#### Individual
* optim_sched:
* * Has a hooks to schedule models and layers for optimizers controlled by the traditional optimizers.
* lora_sched:
* * Has simplified hooks to schedule lora models for all hooks and schedule modifications.