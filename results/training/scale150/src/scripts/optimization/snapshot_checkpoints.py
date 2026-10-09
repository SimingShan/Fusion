#!/usr/bin/env python3
"""Preserve stable best checkpoints from already-running optimization jobs."""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    torch.set_num_threads(1)
    target = args.run / "checkpoints"
    target.mkdir(exist_ok=True)
    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        source = args.run / "best.pt"
        if source.exists():
            temporary = target / "copy.tmp.pt"
            before = source.stat()
            shutil.copyfile(source, temporary)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                try:
                    checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
                    epoch = int(checkpoint["epoch"])
                    path = target / f"epoch_{epoch:04d}.pt"
                    if not path.exists():
                        temporary.replace(path)
                        print(json.dumps({"saved": str(path), "epoch": epoch, "S": checkpoint.get("validation_score")}), flush=True)
                    del checkpoint
                except (RuntimeError, EOFError):
                    pass  # Retry if a writer's partial archive was observed.
            temporary.unlink(missing_ok=True)
        try:
            run = json.loads((args.run / "run.json").read_text())
            if run.get("status") == "completed":
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(15)


if __name__ == "__main__":
    main()
