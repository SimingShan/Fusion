#!/usr/bin/env python3
"""Average several MAST prediction files (flux fields and scalars) and optionally transfer the per-frame flux level
from a designated gauge model: psi = mean_k(psi_k) - mean_grid(mean_k psi_k) + mean_grid(psi_level) per frame.

Label-free construction from our own models; writes mast_public_test.npz plus a JSON report with per-frame statistics.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--members", nargs="+", required=True, help="mast_public_test.npz files to average (equal weights)")
    p.add_argument("--level-from", type=Path, default=None, help="npz whose per-frame grid mean replaces the blend's")
    p.add_argument("--scalars-from", nargs="*", default=None, help="npz files whose q95/betaN are averaged (default: members)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    zs = [np.load(m) for m in args.members]
    zl = np.load(args.level_from) if args.level_from else None
    zsc = [np.load(m) for m in args.scalars_from] if args.scalars_from else zs
    arrays, shift = {}, []
    for i in range(1206):
        key = f"shot_{i:04d}"
        psi = np.mean([z[f"{key}_psirz"].astype(np.float64) for z in zs], axis=0)
        if zl is not None:
            m_blend = psi.mean(axis=(1, 2), keepdims=True); m_level = zl[f"{key}_psirz"].astype(np.float64).mean(axis=(1, 2), keepdims=True)
            psi = psi - m_blend + m_level; shift.append(float((m_level - m_blend).mean()))
        arrays[f"{key}_psirz"] = psi.astype(np.float32)
        arrays[f"{key}_q95"] = np.mean([z[f"{key}_q95"].astype(np.float64) for z in zsc], axis=0).astype(np.float32)
        arrays[f"{key}_betaN"] = np.mean([z[f"{key}_betaN"].astype(np.float64) for z in zsc], axis=0).astype(np.float32)
    npz = args.output / "mast_public_test.npz"; np.savez(npz, **arrays)
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    rep = {"members": args.members, "level_from": str(args.level_from) if args.level_from else None, "scalars_from": args.scalars_from,
           "npz": str(npz), "npz_sha256": sha, "shots": 1206, "frames": int(sum(arrays[f"shot_{i:04d}_q95"].shape[0] for i in range(1206))),
           "level_shift_mean": float(np.mean(shift)) if shift else 0.0, "level_shift_std": float(np.std(shift)) if shift else 0.0,
           "mast_labels_read": False, "audit_labels_read": False, "label_role": "input-only construction from our own models"}
    (args.output / "blend_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
