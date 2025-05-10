#!/usr/bin/env python
import argparse, json
from associate.pipeline_wrapper import PipelineWrapper
from plugins import loader as _plugin_loader  # auto-installs deps & registers hooks

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("mirel-tuner")
    p.add_argument("--config", "-c", required=True, help="Path to JSON config")
    p.add_argument("--dry-run", action="store_true", help="Instantiate and exit")
    return p

def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = json.loads(open(args.config).read())
    pipe = PipelineWrapper(cfg)
    if args.dry_run:
        print("[ok] Pipeline instantiated; exiting (--dry-run).")
        return
    # → training loop placeholder
    print("TODO: Trainer not yet implemented.")

if __name__ == "__main__":
    main()
