#!/usr/bin/env python3
"""Score robust psi-space blends (per-pixel median, trimmed mean) of the 31 re-refined members vs the plain mean."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.optimization.blend_members import write_predictions

def main():
    out = ROOT / "results/evaluations/robust_blend_084"; out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "results/evaluations/blend_gather_083/members.json").read_text())
    names = meta["sets"]["all_good"]; paths = [ROOT / meta["members"][n] for n in names]
    n_shots = 128
    zs = [np.load(p) for p in paths]
    variants = {"mean": [], "median": [], "trim10": []}
    q = {k: ([], []) for k in variants}
    for i in range(n_shots):
        stack = np.stack([z[f"shot_{i:04d}_psirz"].astype(np.float32) for z in zs])  # (K,T,65,65)
        qs = np.stack([z[f"shot_{i:04d}_q95"].astype(np.float32) for z in zs]); bs = np.stack([z[f"shot_{i:04d}_betaN"].astype(np.float32) for z in zs])
        K = stack.shape[0]; k = int(round(0.1 * K))
        srt = np.sort(stack, axis=0)
        variants["mean"].append(stack.mean(0)); variants["median"].append(np.median(stack, axis=0)); variants["trim10"].append(srt[k:K - k].mean(0))
        for key in variants:
            q[key][0].append(qs.mean(0)); q[key][1].append(bs.mean(0))
        if i % 16 == 0: print("shot", i, flush=True)
    report = {}
    for key, psi in variants.items():
        pred = out / f"{key}.npz"; write_predictions(pred, psi, q[key][0], q[key][1])
        score = out / f"{key}_score.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(ROOT / "data/advanced_val_native_v3"), "--split", "val",
                        "--pred", str(pred), "--output", str(score), "--workers", "8"], check=True, cwd=ROOT)
        s = json.loads(score.read_text()); report[key] = {k: s[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}; report[key]["cons"] = s["r2_cons_each"]
        print("ROBUST_BLEND", key, json.dumps(report[key]), flush=True); pred.unlink()
    (out / "report.json").write_text(json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
