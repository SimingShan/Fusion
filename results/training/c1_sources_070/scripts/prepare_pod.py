#!/usr/bin/env python3
"""Fit train-only POD and project training/validation targets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.pod import fit_pod, project_split, manifest_hash, split_entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/advanced"))
    p.add_argument("--output", type=Path, default=Path("data/pod"))
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--basis-samples", type=int, default=16384)
    p.add_argument("--feature-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--reuse-basis", action="store_true")
    args = p.parse_args()
    if args.reuse_basis:
        meta = json.loads((args.output / "basis.json").read_text())
        if meta["source_manifest_sha256"] != manifest_hash(args.data_dir):
            raise ValueError("POD source manifest changed")
        with np.load(args.output / "normalization.npz") as saved:
            norm = dict(saved)
    else:
        norm = fit_pod(args.data_dir, args.output, args.rank, args.basis_samples, args.seed, args.feature_limit)
    for split in ("train", "val", "legacy_val"):
        try:
            selected = split_entries(args.data_dir, split)
        except KeyError:
            continue
        if selected:
            project_split(args.data_dir, args.output, split, norm)


if __name__ == "__main__":
    main()
