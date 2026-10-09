#!/usr/bin/env python3
"""Score K-member flux/scalar averages of existing development predictions.

Loads native development predictions (128 shots, 28,447 frames) from several
models, reports pairwise residual correlations against the ground truth, writes
blended prediction files for a list of predeclared member subsets and scores
each with the unmodified official evaluator. Validation-only exploration; the
audit fold is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_predictions(path, n_shots):
    z = np.load(path)
    psi = [z[f"shot_{i:04d}_psirz"].astype(np.float32) for i in range(n_shots)]
    q95 = [z[f"shot_{i:04d}_q95"].astype(np.float32) for i in range(n_shots)]
    betaN = [z[f"shot_{i:04d}_betaN"].astype(np.float32) for i in range(n_shots)]
    return psi, q95, betaN


def write_predictions(path, psi, q95, betaN):
    arrays = {}
    for i, (p, q, b) in enumerate(zip(psi, q95, betaN)):
        arrays[f"shot_{i:04d}_psirz"] = p.astype(np.float32)
        arrays[f"shot_{i:04d}_q95"] = q.astype(np.float32)
        arrays[f"shot_{i:04d}_betaN"] = b.astype(np.float32)
    np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/advanced_val_native_v3")
    parser.add_argument("--members", nargs="+", required=True, help="name=path/to/val_predictions.npz")
    parser.add_argument("--blends", nargs="+", required=True, help="comma-separated member names per blend")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weights-fit", action="store_true", help="Also fit non-negative flux LSQ weights (validation tuning, exploration only)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    shots = manifest["val"]
    n = len(shots)
    members = {}
    for spec in args.members:
        name, path = spec.split("=", 1)
        members[name] = load_predictions(path, n)
        print("loaded", name, flush=True)
    truth = [np.load(args.data_dir / s["path"])["psi"].astype(np.float32) for s in shots]
    # Residual correlation on a deterministic frame subsample (every 8th frame).
    idx = [np.arange(0, t.shape[0], 8) for t in truth]
    names = list(members)
    res = {}
    for name in names:
        res[name] = np.concatenate([(members[name][0][i][k] - truth[i][k]).ravel() for i, k in enumerate(idx)])
    corr = np.corrcoef(np.stack([res[k] for k in names]))
    rms = {k: float(np.sqrt(np.mean(res[k] ** 2))) for k in names}
    report = {"members": names, "residual_rms": rms, "residual_correlation": corr.tolist(), "blends": {}}
    print(json.dumps({"rms": rms}), flush=True)
    print(np.round(corr, 3), flush=True)

    def score(tag, psi, q95, betaN):
        pred = args.output / f"blend_{hashlib.sha1(tag.encode()).hexdigest()[:12]}.npz"
        write_predictions(pred, psi, q95, betaN)
        out = args.output / f"blend_{hashlib.sha1(tag.encode()).hexdigest()[:12]}_score.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(args.data_dir),
                        "--split", "val", "--pred", str(pred), "--output", str(out), "--workers", str(args.workers)],
                       check=True, cwd=ROOT)
        s = json.loads(out.read_text())
        summary = {k: s[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
        summary["r2_cons_each"] = s.get("r2_cons_each")
        print(tag, json.dumps(summary), flush=True)
        pred.unlink()
        return summary

    for blend in args.blends:
        parts = blend.split(",")
        w = np.full(len(parts), 1.0 / len(parts))
        psi = [sum(wj * members[p][0][i] for wj, p in zip(w, parts)) for i in range(n)]
        q95 = [sum(wj * members[p][1][i] for wj, p in zip(w, parts)) for i in range(n)]
        betaN = [sum(wj * members[p][2][i] for wj, p in zip(w, parts)) for i in range(n)]
        tag = "eq_" + "_".join(parts)
        report["blends"][tag] = {"members": parts, "weights": w.tolist(), "score": score(tag, psi, q95, betaN)}
        if args.weights_fit and len(parts) > 1:
            # Non-negative least squares on the flux residual subsample.
            from scipy.optimize import nnls
            A = np.stack([np.concatenate([members[p][0][i][k].ravel() for i, k in enumerate(idx)]) for p in parts], 1)
            y = np.concatenate([truth[i][k].ravel() for i, k in enumerate(idx)])
            wf, _ = nnls(A.astype(np.float64), y.astype(np.float64))
            wf = wf / wf.sum()
            psi = [sum(wj * members[p][0][i] for wj, p in zip(wf, parts)) for i in range(n)]
            q95 = [sum(wj * members[p][1][i] for wj, p in zip(wf, parts)) for i in range(n)]
            betaN = [sum(wj * members[p][2][i] for wj, p in zip(wf, parts)) for i in range(n)]
            tag = "nnls_" + "_".join(parts)
            report["blends"][tag] = {"members": parts, "weights": wf.tolist(), "score": score(tag, psi, q95, betaN)}
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
