#!/usr/bin/env python3
"""Prepare bounded, restartable per-shot features on a CPU compute node."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-shots", type=int, default=256, help="Training shot count; zero uses all available rows except holdout")
    parser.add_argument("--val-shots", type=int, default=32)
    parser.add_argument("--train-frames", type=int, default=64, help="Uniformly sampled frames per training shot; zero keeps all")
    parser.add_argument("--val-frames", type=int, default=32, help="Uniformly sampled frames per validation shot; zero keeps all")
    parser.add_argument("--seed", type=int, default=2026)
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument("--train-only", action="store_true")
    exclusive.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    for name in ("train_shots", "val_shots", "train_frames", "val_frames"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    manifest = prepare_dataset(**vars(args))
    print("Preparation complete: " + ", ".join(f"{split}={len(manifest[split])} shots" for split in ("train", "val", "test")), flush=True)


if __name__ == "__main__":
    main()
