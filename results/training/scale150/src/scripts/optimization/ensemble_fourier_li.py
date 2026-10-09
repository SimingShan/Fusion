#!/usr/bin/env python3
"""Evaluate five prespecified Fourier/li model blends on two validation folds.

This experiment blends model predictions only, in native float32, with the same
global weight for psi, q95, and betaN. Both folds are model-selection data. It
does not read the audit split, fit per-frame corrections, or upload submissions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import split_entries, validate_splits
from scripts.evaluate import scorer_fingerprint

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FOLDS = ("val", "legacy_val")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot_file(source, destination):
    """Reject a concurrent source replacement or mutation during the copy."""
    before = source.stat()
    source_hash = digest(source)
    shutil.copyfile(source, destination)
    after = source.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after) or digest(source) != source_hash or digest(destination) != source_hash:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Source changed while snapshotting: {source}")
    return {"source_path": str(source.resolve()), "snapshot_path": str(destination.resolve()),
            "sha256": source_hash, "bytes": after.st_size}


def verify_prediction(path, entries):
    expected = {f"shot_{i:04d}_{field}" for i in range(len(entries)) for field in ("psirz", "q95", "betaN")}
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError(f"Wrong prediction key inventory: {path}")
        for i, entry in enumerate(entries):
            for field in ("psirz", "q95", "betaN"):
                value = archive[f"shot_{i:04d}_{field}"]
                shape = (entry["n_frames"], 65, 65) if field == "psirz" else (entry["n_frames"],)
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"Expected complete finite nativefloat32 {field}: {path}, shot{i}")


def verify_score(score, entries, split, fingerprint):
    evaluation = score["evaluation"]
    if evaluation["split"] != split or evaluation["scorer_sha256"] != fingerprint:
        raise ValueError("Source score uses a different fold or scorer")
    if evaluation["evaluated_frames"] != sum(entry["n_frames"] for entry in entries):
        raise ValueError("Source score did not evaluate all cached frames")
    identities = [entry["shot_index"] for entry in entries]
    if [shot["shot_index"] for shot in evaluation["shots"]] != identities:
        raise ValueError("Source score has different shot identities/order")
    if score["n_shots"] != len(entries):
        raise ValueError("Source score has different shot count")


def blend(fourier_path, li_path, output, weight):
    temporary = output.with_suffix(".tmp.npz")
    w = np.float32(weight)
    other = np.float32(1.0 - weight)
    with np.load(fourier_path, allow_pickle=False) as a, np.load(li_path, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Source prediction keys differ")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for key in sorted(a.files):
                value = w * a[key] + other * b[key]
                if value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("Blend lost nativefloat32 precision or finiteness")
                with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, value, allow_pickle=False)
    temporary.replace(output)


def score_summary(score):
    keys = ("S", "r2_psi", "r2_qb", "dlcfs", "consistency", "r2_each", "r2_cons_each", "cons_fail_each", "lcfs_fail_frac")
    return {key: score[key] for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_v1"))
    parser.add_argument("--fourier-dir", type=Path, default=Path("results/optimization/pod_fourier_full_v1"))
    parser.add_argument("--li-dir", type=Path, default=Path("results/optimization/pod_li_refined_cpu"))
    parser.add_argument("--output", type=Path, default=Path("results/optimization/fourier_li_ensemble_v1"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        parser.error("This bounded experiment uses one CPU worker")
    if args.output.exists():
        parser.error("Choose a new output directory; snapshots/results are immutable once written")
    validate_splits(args.data_dir)
    entries = {split: split_entries(args.data_dir, split) for split in FOLDS}
    for split in FOLDS:
        if not entries[split]:
            raise ValueError(f"No {split} entries")
    fingerprint = scorer_fingerprint()
    source_dirs = {"fourier": args.fourier_dir, "li": args.li_dir}
    # Require training completion *and* both final prediction/score pairs before
    # creating a snapshot. The Slurm dependency additionally covers final export.
    runs = {}
    for name, directory in source_dirs.items():
        run = json.loads((directory / "run.json").read_text())
        if run.get("status") != "completed":
            raise ValueError(f"{name} training is not complete; do not read changing best artifacts")
        for split in FOLDS:
            for suffix in ("predictions.npz", "score.json"):
                if not (directory / f"{split}_{suffix}").is_file():
                    raise ValueError(f"{name} {split} export is not complete")
        runs[name] = run
    args.output.mkdir(parents=True)
    sources, source_scores = {}, {}
    for name, directory in source_dirs.items():
        destination = args.output / "sources" / name
        destination.mkdir(parents=True)
        sources[name] = {"run": snapshot_file(directory / "run.json", destination / "run.json")}
        source_scores[name] = {}
        for split in FOLDS:
            source_prediction = directory / f"{split}_predictions.npz"
            prediction = destination / f"{split}_predictions.npz"
            sources[name][split] = snapshot_file(source_prediction, prediction)
            verify_prediction(prediction, entries[split])
            score_path = destination / f"{split}_score.json"
            sources[name][split]["score"] = snapshot_file(directory / f"{split}_score.json", score_path)
            score = json.loads(score_path.read_text())
            verify_score(score, entries[split], split, fingerprint)
            source_scores[name][split] = score
        # Ensure run metadata has also remained fixed across all source copies.
        if digest(directory / "run.json") != sources[name]["run"]["sha256"]:
            raise ValueError(f"{name} run changed while snapshotting")
    folds = {}
    for split, selected in entries.items():
        folds[split] = {"shots": len(selected), "cached_frames": sum(e["n_frames"] for e in selected),
                        "native_frames": sum(e["raw_frames"] for e in selected),
                        "shot_identities": [{key: e.get(key) for key in ("shot_index", "shot_id", "path", "n_frames", "raw_frames")} for e in selected]}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "type": "validation_tuned_prediction_ensemble",
        "formula": "prediction = weight_fourier * Fourier_prediction + (1 - weight_fourier) * li_refined_prediction",
        "weights_prespecified": list(WEIGHTS), "same_weight_for_all_outputs": True,
        "arithmetic_dtype": "float32", "rounding_or_quantization": "None beyond native float32 arithmetic",
        "validation_label_role": "Labels enter the unmodified official evaluator only. Both folds have been inspected and used during development; ensemble selection on them is not an independent generalization test.",
        "audit_labels_read": False, "leaderboard_submission": False, "source_snapshots": sources,
        "folds": folds, "scorer_sha256": fingerprint,
        "endpoint_evaluation": "Pure endpoint scores reused from completed, snapshotted official evaluations after exact fold/order/frame-count checks; interior blends newly evaluated.",
        "candidates": [],
    }
    summary_path = args.output / "summary.json"
    atomic_json(summary_path, report)
    for weight in WEIGHTS:
        directory = args.output / f"fourier_{weight:.2f}".replace(".", "p")
        directory.mkdir()
        candidate = {"weight_fourier": weight, "folds": {}}
        for split in FOLDS:
            prediction_path = directory / f"{split}_predictions.npz"
            score_path = directory / f"{split}_score.json"
            if weight in (0.0, 1.0):
                name = "fourier" if weight else "li"
                os.link(sources[name][split]["snapshot_path"], prediction_path)
                score = source_scores[name][split]
                shutil.copyfile(sources[name][split]["score"]["snapshot_path"], score_path)
                reused = True
            else:
                blend(Path(sources["fourier"][split]["snapshot_path"]),
                      Path(sources["li"][split]["snapshot_path"]), prediction_path, weight)
                with (directory / f"{split}_score.log").open("w") as log:
                    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
                                    "--data-dir", str(args.data_dir), "--split", split,
                                    "--pred", str(prediction_path), "--output", str(score_path),
                                    "--workers", "1"], stdout=log, stderr=subprocess.STDOUT, check=True)
                score = json.loads(score_path.read_text())
                verify_score(score, entries[split], split, fingerprint)
                reused = False
            candidate["folds"][split] = {**score_summary(score), "prediction_path": str(prediction_path.resolve()),
                                          "score_path": str(score_path.resolve()), "reused_endpoint_score": reused}
            print(json.dumps({"weight_fourier": weight, "split": split,
                              **{k: score[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency", "r2_cons_each")}}), flush=True)
        report["candidates"].append(candidate)
        atomic_json(summary_path, report)
    # Selection is explicit and deterministic. Legacy is reported separately,
    # never silently combined with the larger fresh validation fold.
    report["best_by_fresh_validation"] = max(report["candidates"], key=lambda c: c["folds"]["val"]["S"])
    report["best_by_legacy_validation"] = max(report["candidates"], key=lambda c: c["folds"]["legacy_val"]["S"])
    report["selection_rule"] = "Maximum fresh-fold official S among the five prespecified weights; ties retain the first weight in ascending order. Legacy optimum is separately disclosed."
    atomic_json(summary_path, report)
    lines = ["# Fourier and li-refined validation ensemble", "",
             "Both folds are development data. These scores do not establish independent audit performance or an official leaderboard result.", "",
             "The same prespecified Fourier fraction is applied to flux, q95, and betaN using native float32 arithmetic.", "",
             "| Fourier fraction | Fresh S | Legacy S | Fresh li R² | Fresh R-axis R² | Fresh Z-axis R² |",
             "|---:|---:|---:|---:|---:|---:|"]
    for c in report["candidates"]:
        fresh, legacy = c["folds"]["val"], c["folds"]["legacy_val"]
        derived = fresh["r2_cons_each"]
        lines.append(f"| {c['weight_fourier']:.2f} | {fresh['S']:.6f} | {legacy['S']:.6f} | {derived['li']:.6f} | {derived['R_axis']:.6f} | {derived['Z_axis']:.6f} |")
    selected = report["best_by_fresh_validation"]
    lines += ["", f"Selected by fresh validation: Fourier fraction {selected['weight_fourier']:.2f}.", "",
              "See summary.json for exact fold identities, frame counts, input artifact hashes, complete component metrics, endpoint reuse, and the separate legacy optimum. Source files were copied only after training completed and were checked for concurrent changes."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print("SELECTED " + json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
