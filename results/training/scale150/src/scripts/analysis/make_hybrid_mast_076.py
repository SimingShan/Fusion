#!/usr/bin/env python3
"""Hybrid MAST payload: v4 (synthetic-augmented) flux shape with the v3 (DIII-D-only) per-frame flux level.

psi_hybrid[f] = psi_v4[f] - mean(psi_v4[f]) + mean(psi_v3[f]); q95/betaN from v4. Label-free construction
from two of our own models; used to test whether the flux-offset convention is what separates their scores.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    out = ROOT / "results/challenge2/c2_hybrid_076"; out.mkdir(parents=True, exist_ok=True)
    v3 = np.load(ROOT / "results/challenge2/c2_cnn_v3_main/mast/mast_public_test.npz")
    v4 = np.load(ROOT / "results/challenge2/c2_cnn_v4_synth_gdev/mast/mast_public_test.npz")
    arrays, stats = {}, []
    for i in range(1206):
        a3 = v3[f"shot_{i:04d}_psirz"].astype(np.float32); a4 = v4[f"shot_{i:04d}_psirz"].astype(np.float32)
        assert a3.shape == a4.shape
        m3 = a3.mean(axis=(1, 2), keepdims=True); m4 = a4.mean(axis=(1, 2), keepdims=True)
        arrays[f"shot_{i:04d}_psirz"] = (a4 - m4 + m3).astype(np.float32)
        arrays[f"shot_{i:04d}_q95"] = v4[f"shot_{i:04d}_q95"].astype(np.float32)
        arrays[f"shot_{i:04d}_betaN"] = v4[f"shot_{i:04d}_betaN"].astype(np.float32)
        stats.append(float((m3 - m4).mean()))
    npz = out / "mast_public_test.npz"
    np.savez(npz, **arrays)
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    from scripts.package_submission import validate_prediction
    try:
        v = validate_prediction(npz, "mast_public_test")
        status = v.get("status") if isinstance(v, dict) else str(v)
    except Exception as exc:
        status = f"validate_prediction error: {exc}"
    rep = {"construction": "psi_v4 - mean_grid(psi_v4) + mean_grid(psi_v3) per frame; q95/betaN from v4",
           "v3": "results/challenge2/c2_cnn_v3_main/mast/mast_public_test.npz", "v4": "results/challenge2/c2_cnn_v4_synth_gdev/mast/mast_public_test.npz",
           "checkpoint": str(ROOT / "results/challenge2/c2_cnn_v4_synth_gdev/best.pt"), "npz": str(npz), "npz_sha256": sha,
           "shots": 1206, "frames": int(sum(arrays[f"shot_{i:04d}_q95"].shape[0] for i in range(1206))),
           "offset_shift_mean_over_shots": float(np.mean(stats)), "offset_shift_std_over_shots": float(np.std(stats)),
           "validation": status, "mast_labels_read": False, "audit_labels_read": False,
           "label_role": "input-only construction from two of our own models; no MAST labels read"}
    (out / "hybrid_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
