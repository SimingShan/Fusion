#!/usr/bin/env python3
"""Score the re-refined POD ensemble mean blended with the Challenge-2 U-Net DIII-D predictions at small predeclared weights."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.optimization.blend_members import load_predictions, write_predictions

def main():
    out = ROOT / "results/evaluations/blend_cnn_081"; out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/advanced_val_native_v3/manifest.json").read_text()); n = len(manifest["val"])
    members = sorted(Path(ROOT / "results/training/ens078_liall_amp1_e60").glob("*/native/val_predictions.npz"))
    members += sorted(Path(ROOT / "results/training/ens078_liall_amp1").glob("*/native/val_predictions.npz"))
    good = []
    for p in members:
        s = json.loads(p.with_name("val_score.json").read_text())["S"]
        if s >= 0.9940: good.append(p)
    print(f"{len(good)} POD members", flush=True)
    acc = None
    for p in good:
        psi, q, b = load_predictions(p, n)
        if acc is None: acc = [psi, q, b]
        else:
            for i in range(n):
                acc[0][i] += psi[i]; acc[1][i] += q[i]; acc[2][i] += b[i]
    K = len(good)
    ens = [[a / K for a in acc[0]], [a / K for a in acc[1]], [a / K for a in acc[2]]]
    cnn = load_predictions(ROOT / "results/challenge2/c2_cnn_v3_main/val_predictions.npz", n)
    report = {"members": [str(p) for p in good]}
    for w in (0.0, 0.03, 0.06, 0.10):
        psi = [(1 - w) * e + w * c for e, c in zip(ens[0], cnn[0])]
        q = [(1 - w) * e + w * c for e, c in zip(ens[1], cnn[1])]
        b = [(1 - w) * e + w * c for e, c in zip(ens[2], cnn[2])]
        pred = out / f"blend_w{w:.2f}.npz"; write_predictions(pred, psi, q, b)
        score = out / f"blend_w{w:.2f}_score.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(ROOT / "data/advanced_val_native_v3"), "--split", "val",
                        "--pred", str(pred), "--output", str(score), "--workers", "8"], check=True, cwd=ROOT)
        s = json.loads(score.read_text()); report[f"w{w:.2f}"] = {k: s[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}; report[f"w{w:.2f}"]["cons"] = s["r2_cons_each"]
        print("CNN_BLEND", w, json.dumps(report[f"w{w:.2f}"]), flush=True); pred.unlink()
    (out / "report.json").write_text(json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
