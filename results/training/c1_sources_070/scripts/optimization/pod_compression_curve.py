#!/usr/bin/env python3
"""Measure train-fitted POD representation capacity, never predictive accuracy.

Projection coefficients and submitted scalar channels use validation labels.
Every resulting score is explicitly an oracle diagnostic and is unsuitable for
submission. Only val/legacy_val are accepted; independent audit labels stay closed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import manifest_hash, split_entries, validate_splits


def write_oracle(data_dir, split, basis, mean, output):
    temporary = output.with_suffix(".tmp.npz")
    squared_error, frames = 0., 0
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for index, entry in enumerate(split_entries(data_dir, split)):
            with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
                truth = shot["psi"].astype(np.float32)
                scalars = shot["scalars"].astype(np.float32)
                centered = truth.reshape(-1, 4225) - mean.reshape(-1)
                reconstructed = ((centered @ basis.T) @ basis).reshape(-1, 65, 65) + mean
                if not np.isfinite(reconstructed).all() or not np.isfinite(scalars).all():
                    raise ValueError("Oracle scorer requires finite reconstruction and scalar labels")
                squared_error += float(np.square(reconstructed.astype(np.float64) - truth).sum())
                frames += len(truth)
                for name, array in (("psirz", reconstructed), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                    with archive.open(f"shot_{index:04d}_{name}.npy", "w", force_zip64=True) as stream:
                        np.lib.format.write_array(stream, array.astype(np.float32), allow_pickle=False)
    os.replace(temporary, output)
    return {"frames": frames, "flux_rmse": float(np.sqrt(squared_error / (frames * 4225)))}


def make_figure(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    rank = [row["rank"] for row in records]
    score = [row["S"] for row in records]
    axes[0].plot(rank, score, "o-", color="#287D8E", label="Oracle composite S")
    axes[0].axhline(.995440, color="#A94442", ls=":", label="Historical public-test reference; different data")
    for x, y in zip(rank, score):
        axes[0].annotate(f"{y:.6f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("Oracle reconstruction composite S")
    axes[0].set_ylim(max(0, min(score) - .015), 1.012)
    for key, name, color in (("r2_psi", "Flux R²", "#287D8E"),
                             ("consistency", "Derived consistency", "#CF6272"),
                             ("boundary_agreement", "1 − D_LCFS", "#6F60A7")):
        axes[1].plot(rank, [row[key] for row in records], "o-", label=name, color=color)
    axes[1].set_ylabel("Oracle metric component")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(rank, [str(value) for value in rank])
        ax.set_xlabel("Number of retained training-derived POD modes")
        ax.grid(color="#DDE3E8", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("POD representation capacity — oracle diagnostic, not a learned-model result\nValidation labels determine coefficients and scalars; the basis uses training shots only", fontsize=13, fontweight="bold")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"compression_curve.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--pod-dir", type=Path, default=ROOT / "data/pod_legacy")
    parser.add_argument("--output", type=Path, default=ROOT / "results/optimization/pod_compression_curve")
    parser.add_argument("--split", choices=["val", "legacy_val"], default="val")
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ranks = sorted(set(args.ranks))
    if args.workers < 1 or not ranks or ranks[0] < 1:
        parser.error("workers and ranks must be positive")
    args.data_dir, args.pod_dir, args.output = args.data_dir.resolve(), args.pod_dir.resolve(), args.output.resolve()
    validate_splits(args.data_dir)
    metadata = json.loads((args.pod_dir / "basis.json").read_text())
    if metadata.get("basis_fit_split") != "train" or metadata.get("audit_labels_read") is not False:
        raise ValueError("POD metadata does not certify a train-only basis with audit labels untouched")
    if metadata["source_manifest_sha256"] != manifest_hash(args.data_dir):
        raise ValueError("Current data manifest differs from the POD basis source manifest")
    norm_path = args.pod_dir / "normalization.npz"
    norm_hash = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    with np.load(norm_path, allow_pickle=False) as archive:
        basis, mean = archive["basis"], archive["psi_mean"]
        eigenvalues = archive["eigenvalues"]
    if ranks[-1] > len(basis):
        raise ValueError("Requested rank exceeds the fitted basis")
    if not np.allclose(basis @ basis.T, np.eye(len(basis)), rtol=1e-5, atol=3e-6):
        raise ValueError("Basis is not orthonormal; ordinary POD projection is invalid")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for rank in ranks:
        print(f"Oracle compression rank {rank}/{len(basis)}", flush=True)
        prediction = args.output / f"oracle_rank_{rank:03d}_predictions.npz"
        summary = write_oracle(args.data_dir, args.split, basis[:rank], mean, prediction)
        score_path = args.output / f"oracle_rank_{rank:03d}_score.json"
        command = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(args.data_dir),
                   "--split", args.split, "--pred", str(prediction), "--output", str(score_path),
                   "--workers", str(args.workers)]
        with score_path.with_suffix(".log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        score = json.loads(score_path.read_text())
        score["evaluation"].update(type="oracle_POD_compression_diagnostic_uses_validation_labels",
                                   model_prediction=False, leaderboard_submission=False)
        score["compression_diagnostic"] = {
            "rank": rank, "source_rank": len(basis), "basis_fit_split": "train",
            "normalization_sha256": norm_hash, "basis_metadata": metadata,
            "uses_validation_labels_for_coefficients": True,
            "submitted_scalars_are_ground_truth": True, "audit_labels_read": False,
            "valid_submission_candidate": False,
        }
        score_path.write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
        variance_fraction = float(eigenvalues[:rank].sum() / eigenvalues.sum() * metadata["basis_variance_fraction"])
        record = {"rank": rank, "S": score["S"], "r2_psi": score["r2_psi"],
                  "dlcfs": score["dlcfs"], "boundary_agreement": 1 - score["dlcfs"],
                  "consistency": score["consistency"], "r2_cons_each": score["r2_cons_each"],
                  "r2_qb": score["r2_qb"], "training_variance_fraction": variance_fraction,
                  "score_path": str(score_path), **summary}
        records.append(record)
        payload = {"type": "oracle_POD_compression_diagnostic", "model_prediction": False,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "basis_metadata": metadata,
                   "normalization_sha256": norm_hash, "split": args.split, "records": records}
        (args.output / "compression_curve.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
    make_figure(records, args.output)
    lines = ["# POD compression capacity", "",
             "**Oracle diagnostic, not a learned prediction.** Coefficients and q95/βN are obtained from validation labels. The existing POD basis and mean were fitted only on training shots. This measures whether the representation can express accurate fields; it does not establish achievable prediction accuracy or leaderboard rank.", "",
             f"Basis source: `{args.pod_dir}`. Evaluated split: `{args.split}`; {len(split_entries(args.data_dir, args.split))} shots / {records[0]['frames']:,} native cached frames. Independent audit labels were not read.", "",
             "| Rank | Oracle S | Flux R² | Flux RMSE | D_LCFS | Consistency | Training variance retained |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['rank']} | {row['S']:.6f} | {row['r2_psi']:.6f} | {row['flux_rmse']:.8f} | {row['dlcfs']:.6f} | {row['consistency']:.6f} | {row['training_variance_fraction']:.9f} |")
    lines += ["", "[PNG curve](compression_curve.png) · [PDF curve](compression_curve.pdf)", "",
              "The plotted historical public-test reference is 0.995440, observed September 12 EDT / September 13 UTC, 2026. It uses different data. An oracle reconstruction above that value is not a competitive result.", "",
              "Refresh: `.venv/bin/python scripts/optimization/pod_compression_curve.py`."]
    (args.output / "COMPRESSION.md").write_text("\n".join(lines) + "\n")
    print(f"Saved oracle compression curve to {args.output}", flush=True)


if __name__ == "__main__":
    main()
