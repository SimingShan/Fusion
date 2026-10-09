#!/usr/bin/env python3
"""Collect finished ensemble members and score predeclared equal-weight blends on the native dev fold.

Members are discovered from results/training/ens070/*/native/val_predictions.npz (plus the five
checkpoints of candidate 20260921-070). Blend sets are predeclared here; the audit fold is never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE_MEMBERS = {
    "A": "results/optimization/native_validation_v3/scaled_li/val_predictions.npz",
    "B": "results/optimization/native_history128_li_v8/robust_history128_li/val_predictions.npz",
    "S040": "results/evaluations/20260920-040/native/val_predictions.npz",
    "S066": "results/evaluations/20260920-066/native/val_predictions.npz",
    "X067": "results/evaluations/20260920-067/native/val_predictions.npz",
}


def member_score(pred_path):
    score = Path(pred_path).with_name("val_score.json")
    if score.exists():
        return json.loads(score.read_text()).get("S")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-root", type=Path, default=ROOT / "results/training/ens070")
    parser.add_argument("--extra-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-single", type=float, default=0.9940, help="Drop members whose single native S is below this")
    parser.add_argument("--blends", nargs="*", default=None, help="Explicit comma-separated member sets; default = predeclared sets")
    args = parser.parse_args()
    members = dict(BASE_MEMBERS)
    for root in [args.ensemble_root, *args.extra_roots]:
        for pred in sorted(root.glob("*/native/val_predictions.npz")):
            name = pred.parents[1].name
            if name.endswith("_cold"):
                continue
            members[name] = str(pred.relative_to(ROOT)) if pred.is_relative_to(ROOT) else str(pred)
    singles = {k: member_score(ROOT / v) for k, v in members.items()}
    print(json.dumps({"singles": singles}, indent=1), flush=True)
    good = [k for k, s in singles.items() if s is not None and s >= args.min_single]
    new_b = [k for k in good if k.startswith("B_s")]
    new_a = [k for k in good if k.startswith("A_s")]
    base = [k for k in ("A", "B", "S040", "S066", "X067") if k in good]
    if args.blends is None:
        sets = {
            "base5": base,
            "base5_newB": base + new_b,
            "base5_newA": base + new_a,
            "all_good": good,
            "newAB_only": new_a + new_b,
            "all_good_top10": sorted(good, key=lambda k: -singles[k])[:10],
            "b_all": [k for k in ("B", "S040", "S066", "X067") if k in good] + new_b,
            "b_all_plus_A_top4": [k for k in ("B", "S040", "S066", "X067") if k in good] + new_b + ["A"] + sorted(new_a, key=lambda k: -singles[k])[:3],
            "base5_newB_newAtop4": base + new_b + sorted(new_a, key=lambda k: -singles[k])[:4],
        }
        blends = [",".join(v) for v in sets.values() if len(v) >= 2]
        blends = list(dict.fromkeys(blends))
    else:
        blends = args.blends
    cmd = [sys.executable, str(ROOT / "scripts/optimization/blend_members.py"), "--output", str(args.output),
           "--workers", str(args.workers), "--members", *[f"{k}={members[k]}" for k in good], "--blends", *blends]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
