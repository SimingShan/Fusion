#!/usr/bin/env python3
"""Collect re-refined ensemble members from one or more roots and score predeclared equal-weight blends.

Members are <root>/<name>/native/val_predictions.npz with a sibling val_score.json (single native S).
Names must be unique across roots (re-refined members supersede their originals; do not mix roots that
share names). The audit fold is never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-single", type=float, default=0.9940)
    parser.add_argument("--extra-blends", nargs="*", default=[], help="additional comma-separated member sets")
    args = parser.parse_args()
    members, singles, types = {}, {}, {}
    for root in args.roots:
        for pred in sorted(root.resolve().glob("*/native/val_predictions.npz")):
            name = pred.parents[1].name
            if name in members:
                raise SystemExit(f"duplicate member name {name} across roots")
            score = pred.with_name("val_score.json")
            if not score.exists():
                continue
            members[name] = str(pred.relative_to(ROOT))
            singles[name] = json.loads(score.read_text())["S"]
            types[name] = "A" if name.startswith("A_") or "scaled" in name else "B"
    print(json.dumps({"singles": singles, "types": types}, indent=1), flush=True)
    good = [k for k, s in singles.items() if s >= args.min_single]
    b = [k for k in good if types[k] == "B"]
    a = [k for k in good if types[k] == "A"]
    top_a = sorted(a, key=lambda k: -singles[k])[:4]
    sets = {"all_good": good, "b_only": b, "b_plus_topA4": b + top_a, "b_plus_allA": b + a,
            "top12_by_single": sorted(good, key=lambda k: -singles[k])[:12]}
    blends = list(dict.fromkeys([",".join(v) for v in sets.values() if len(v) >= 2] + list(args.extra_blends)))
    cmd = [sys.executable, str(ROOT / "scripts/optimization/blend_members.py"), "--output", str(args.output),
           "--workers", str(args.workers), "--members", *[f"{k}={members[k]}" for k in good], "--blends", *blends]
    (args.output).mkdir(parents=True, exist_ok=True)
    (args.output / "members.json").write_text(json.dumps({"members": members, "singles": singles, "types": types, "sets": sets}, indent=1))
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
