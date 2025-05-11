#!/usr/bin/env python
import argparse
import json
import os

from associate.pipeline.pipeline_wrapper import PipelineWrapper


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("mirel-tuner")
    p.add_argument(
        "--config", "-c",
        default="configs/default_config.json",
        help="Path to JSON config (defaults to default_config.json)"
    )
    p.add_argument("--dry-run", action="store_true", help="Instantiate and exit")
    return p

def _load_and_merge_config(path: str) -> dict:
    # 1) load defaults

    default_path = os.path.join(os.path.dirname(__file__), "configs/default_config.json")
    with open(default_path, "r") as f:
        default_cfg = json.load(f)

    # 2) load user overrides (if any)
    if os.path.exists(path) and path != default_path:
        with open(path, "r") as f:
            user_cfg = json.load(f)
    else:
        user_cfg = {}

    # 3) merge (user overrides default)
    merged = {**default_cfg, **user_cfg}
    return merged

def main() -> None:
    args = _build_arg_parser().parse_args()

    # load & merge into a plain dict
    cfg = _load_and_merge_config(args.config)

    # build pipeline (uses cfg["model"], cfg["precision"], etc.)
    pipe_wrapper = PipelineWrapper(cfg)

    if args.dry_run:
        print("[ok] Pipeline instantiated; exiting (--dry-run).")
        return

    # launch CPU-only white-noise trainer
    from engine.trainer import Trainer
    trainer = Trainer(cfg, pipe_wrapper)
    trainer.train()

if __name__ == "__main__":
    main()
