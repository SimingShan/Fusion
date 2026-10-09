#!/usr/bin/env python3
"""Prepare full-data DIII-D caches on a CPU compute node; safely resumable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.advanced_data import prepare_advanced_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/advanced_v1"))
    parser.add_argument("--legacy-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--val-shots", type=int, default=128)
    parser.add_argument("--audit-shots", type=int, default=128)
    parser.add_argument("--train-frames", type=int, default=64, help="Native frames per training shot, uniformly spaced; 0 keeps all")
    parser.add_argument("--val-frames", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-shots", type=int, default=0, help="Cap new shots this invocation; 0 prepares all remaining shots")
    parser.add_argument("--no-consolidate", dest="consolidate", action="store_false")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "audit", "legacy_val", "test"))
    parser.add_argument("--plan-only", action="store_true", help="Report split/feature counts without writing or loading shot arrays")
    args = parser.parse_args()
    manifest = prepare_advanced_dataset(**vars(args))
    print(json.dumps({"output": str(args.output), "feature_count": len(manifest["feature_names"]),
                      "planned_shots": {key: len(values) for key, values in manifest["selection"].items()},
                      "prepared_shots": {key: len(manifest[key]) for key in manifest["selection"]},
                      "complete_splits": manifest.get("complete_splits", {})}, indent=2), flush=True)


if __name__ == "__main__":
    main()
