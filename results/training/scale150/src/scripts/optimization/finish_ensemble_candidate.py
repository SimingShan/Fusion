#!/usr/bin/env python3
"""Finish a generic N-member gated POD-DeepONet ensemble as a frozen candidate.

The finisher produces exactly the file contract consumed by
``scripts/periodic_submission.py::candidate_evidence``:

    <root>/<candidate>/checkpoints/mNN.pt      frozen member checkpoints
    <root>/<candidate>/decision.json           frozen selection + development scores
    <root>/<candidate>/diii_d_public_test.npz  gated N-member public-test export
    <root>/<candidate>/export.json             export provenance and payload hash
    <root>/<candidate>/export_parity.json      frozen-inference parity on dev shots
    <root>/<candidate>/submission.zip          packaged archive (+ validation.json)
    <root>/<candidate>/finisher_summary.json   everything this script did

Members are POD-DeepONet ``best.pt`` checkpoints of two recipe families:

    A  data/advanced_v1 features (capped/legacy dir data/advanced_v1, native
       development dir data/advanced_val_native_v3, test cache data/advanced_test_v1)
    B  data/advanced_history_v2 features with X_extra history sidecars (legacy dir
       data/advanced_history_v2, native dir data/advanced_history_val_native_v3,
       test cache data/advanced_history_test_v2)

Blend policy (generalizes the shipped two-model gate): per frame, the
input-only quality gate of ``quality_gate.gate_from_inputs`` is evaluated on the
A-family feature row (useful-edge-Thomson fraction). When the fraction is at
most ``--gate-threshold`` only the B-family members are averaged (weights
renormalized); otherwise all members are averaged with the declared weights.
Blending happens in physical flux/scalar space with float32 arithmetic in
member order, so a two-member call reproduces ``export_gated_ensemble.py`` bit
for bit. If no B member exists the gate is a no-op and all members are used.

Development scores are computed with ``scripts/evaluate.py`` on exactly
``data/advanced_val_native_v3 --split val`` and ``data/advanced_v1 --split
legacy_val`` so that ``candidate_evidence`` accepts them against the baseline.
No audit split, test label or MAST label is ever read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import read_manifest
from scripts.codabench_submit import write_json
from scripts.train import prediction_entries
from scripts.optimization.export_gated_ensemble import input_arrays, load_expert, predict
from scripts.optimization.quality_gate import feature_columns, gate_from_inputs, sha256, write_array

FAMILIES = {
    "A": {"legacy": Path("data/advanced_v1"), "native": Path("data/advanced_val_native_v3"),
          "test": Path("data/advanced_test_v1")},
    "B": {"legacy": Path("data/advanced_history_v2"), "native": Path("data/advanced_history_val_native_v3"),
          "test": Path("data/advanced_history_test_v2")},
}
GATE_FAMILY = "A"          # the gate always reads the A-family feature row
FALLBACK_FAMILY = "B"      # members used alone when the gate activates
FOLDS = {"native": {"split": "val", "score": "val_score.json", "predictions": "val_predictions.npz"},
         "legacy": {"split": "legacy_val", "score": "legacy_val_score.json", "predictions": "legacy_val_predictions.npz"}}
FIELDS = ("psirz", "q95", "betaN")
PARITY_ATOL = {"psirz": 2e-6, "q95": 1e-5, "betaN": 1e-5}
PARITY_RTOL = 1e-5
NATIVE_GAIN = 0.0001
LEGACY_TOLERANCE = 0.0001
EXPECTED_TEST_SHOTS, EXPECTED_TEST_FRAMES = 874, 193359


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    """Repository-relative string for a path inside ROOT (absolute otherwise)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_script(script, *arguments, log=None):
    """Run a repository script with the current interpreter from ROOT."""
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    if log is None:
        return subprocess.run(command, cwd=ROOT, check=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        return subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)


def parse_members(values):
    members = []
    for value in values:
        family, _, path = value.partition("=")
        if family not in FAMILIES or not path:
            raise argparse.ArgumentTypeError(f"Member must be TYPE=path with TYPE in {sorted(FAMILIES)}: {value!r}")
        members.append({"type": family, "local_source": path})
    return members


def expected_keys(n_shots):
    return {f"shot_{i:04d}_{field}" for i in range(n_shots) for field in FIELDS}


# ----------------------------------------------------------------------------
# Blend policy shared by development folds, export and parity
# ----------------------------------------------------------------------------

def weighted_sum(arrays, weights):
    """Float32 weighted sum accumulated in member order (matches the shipped export)."""
    total = np.float32(weights[0]) * arrays[0]
    for array, weight in zip(arrays[1:], weights[1:]):
        total = total + np.float32(weight) * array
    return total


def fallback_selection(types, weights):
    """Indices and renormalized weights of the members used when the gate activates."""
    chosen = [i for i, family in enumerate(types) if family == FALLBACK_FAMILY]
    if not chosen:
        chosen = list(range(len(types)))
    mass = sum(weights[i] for i in chosen)
    return chosen, [weights[i] / mass for i in chosen]


def blend_frames(gated, psi_members, scalar_members, types, weights):
    """Gated N-member blend of per-member (frames,65,65) flux and (frames,2) scalars."""
    full_psi, full_scalars = weighted_sum(psi_members, weights), weighted_sum(scalar_members, weights)
    chosen, renormalized = fallback_selection(types, weights)
    if len(chosen) == 1 and renormalized[0] == 1.0:
        gate_psi, gate_scalars = psi_members[chosen[0]], scalar_members[chosen[0]]
    else:
        gate_psi = weighted_sum([psi_members[i] for i in chosen], renormalized)
        gate_scalars = weighted_sum([scalar_members[i] for i in chosen], renormalized)
    psi = np.where(gated[:, None, None], gate_psi, full_psi).astype(np.float32, copy=False)
    scalars = np.where(gated[:, None], gate_scalars, full_scalars).astype(np.float32, copy=False)
    return psi, scalars


def gate_frames(X_gate, columns, threshold):
    _, fraction, _, _ = gate_from_inputs(X_gate, columns)
    return fraction <= threshold


def infer_shot(experts, members, weights, family_dirs, entries_by_family, index, columns, threshold, batch_size, device):
    """Export code path for one shot: read every family's inputs, gate, predict, blend."""
    inputs = {}
    for family, directory in family_dirs.items():
        inputs[family] = input_arrays(directory, entries_by_family[family][index])
    reference_times, reference_frames = inputs[GATE_FAMILY][1], inputs[GATE_FAMILY][2]
    for family, (_, times, frames) in inputs.items():
        if not np.array_equal(times, reference_times) or not np.array_equal(frames, reference_frames):
            raise ValueError(f"Family {family} cache has different native query timestamps at shot {index}")
    gated = gate_frames(inputs[GATE_FAMILY][0], columns, threshold)
    psi_members, scalar_members = [], []
    for expert, member in zip(experts, members):
        psi, scalars = predict(expert, inputs[member["type"]][0], batch_size, device)
        psi_members.append(psi)
        scalar_members.append(scalars)
    psi, scalars = blend_frames(gated, psi_members, scalar_members, [m["type"] for m in members], weights)
    n = len(reference_times)
    if psi.shape != (n, 65, 65) or scalars.shape != (n, 2):
        raise ValueError("Ensemble prediction shape differs from the native output contract")
    return psi, scalars, gated


def fold_family_dirs(fold, members):
    """Data directories needed for a fold: the gate family plus every member family."""
    families = {GATE_FAMILY, *(m["type"] for m in members)}
    return {family: ROOT / FAMILIES[family][fold] for family in sorted(families)}


def fold_entries(fold, family_dirs):
    split = "test" if fold == "test" else FOLDS[fold]["split"]
    entries = {}
    for family, directory in family_dirs.items():
        entries[family] = prediction_entries(directory, "test") if fold == "test" else read_manifest(directory)[split]
    reference = entries[GATE_FAMILY]
    if not reference:
        raise ValueError(f"Fold {fold} is empty in {family_dirs[GATE_FAMILY]}")
    for family, items in entries.items():
        if len(items) != len(reference):
            raise ValueError(f"Family {family} cache has a different shot selection for fold {fold}")
        for first, second in zip(reference, items):
            for key in ("shot_index", "shot_id", "shard", "row", "n_frames", "raw_frames"):
                if first.get(key) != second.get(key):
                    raise ValueError(f"Family {family} cache shot identity disagrees on {key} for fold {fold}")
    return entries


# ----------------------------------------------------------------------------
# Development folds: per-member predictions, blend, official local scoring
# ----------------------------------------------------------------------------

def member_predictions(member, index, fold, provided, evaluation_dir, workers):
    """Reuse a provided per-member prediction archive or infer it with train_pod.py."""
    split, filename = FOLDS[fold]["split"], FOLDS[fold]["predictions"]
    directory = ROOT / FAMILIES[member["type"]][fold]
    n_shots = len(read_manifest(directory)[split])
    if provided:
        path = ROOT / provided
        record = {"path": relative(path), "sha256": sha256(path), "inferred": False}
    else:
        output = evaluation_dir / "members" / f"m{index:02d}" / fold
        path = output / filename
        if not path.exists():
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(member["source"], output / "best.pt")
            print(f"Inferring member m{index:02d} ({member['type']}) on {fold} fold with train_pod.py", flush=True)
            run_script("scripts/train_pod.py", "--data-dir", relative(directory), "--output", output,
                       "--predict-only", "--split", split, "--workers", workers, log=output / f"predict_{split}.log")
        record = {"path": relative(path), "sha256": sha256(path), "inferred": True}
        score = output / FOLDS[fold]["score"]
        if score.exists():
            record["member_S"] = json.loads(score.read_text())["S"]
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys(n_shots):
            raise ValueError(f"Prediction keys of {path} do not match the complete {fold} fold of {n_shots} shots")
    return path, record


def blend_fold(fold, members, weights, sources, output, columns, threshold):
    """Blend per-member development predictions with the gate read from the A cache."""
    family_dirs = fold_family_dirs(fold, members)
    entries = fold_entries(fold, family_dirs)[GATE_FAMILY]
    gate_dir = family_dirs[GATE_FAMILY]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    coverage = {"shots": len(entries), "frames": 0, "gated_frames": 0, "gated_shots": 0}
    archives = [np.load(path, allow_pickle=False) for path in sources]
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for i, entry in enumerate(entries):
                with np.load(gate_dir / entry["path"], allow_pickle=False) as shot:
                    X = shot["X"]   # inputs only; labels are never touched here
                if len(X) != entry["n_frames"]:
                    raise ValueError("Cache frame count disagrees with the split manifest")
                gated = gate_frames(X, columns, threshold)
                psi_members, scalar_members = [], []
                for source in archives:
                    psi = source[f"shot_{i:04d}_psirz"]
                    scalars = np.column_stack([source[f"shot_{i:04d}_q95"], source[f"shot_{i:04d}_betaN"]])
                    if psi.shape != (len(X), 65, 65) or scalars.shape != (len(X), 2):
                        raise ValueError(f"Member prediction shape mismatch at shot {i} of fold {fold}")
                    psi_members.append(psi.astype(np.float32, copy=False))
                    scalar_members.append(scalars.astype(np.float32, copy=False))
                psi, scalars = blend_frames(gated, psi_members, scalar_members, [m["type"] for m in members], weights)
                for field, values in (("psirz", psi), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                    write_array(archive, f"shot_{i:04d}_{field}", values)
                coverage["frames"] += len(X)
                coverage["gated_frames"] += int(gated.sum())
                coverage["gated_shots"] += int(gated.any())
    finally:
        for source in archives:
            source.close()
    os.replace(temporary, output)
    return coverage


def score_fold(fold, predictions, score_path, workers):
    """Official local evaluation with the exact data directory candidate_evidence expects."""
    data_dir = FAMILIES[GATE_FAMILY][fold]
    run_script("scripts/evaluate.py", "--data-dir", data_dir, "--split", FOLDS[fold]["split"], "--pred", predictions,
               "--output", score_path, "--workers", workers, log=score_path.with_suffix(".log"))
    return json.loads(score_path.read_text())


def breakdown(score):
    keys = ("S", "r2_psi", "r2_qb", "dlcfs", "consistency", "lcfs_fail_frac", "cons_fail_frac", "psi_nonfinite_frac", "n_shots_missing")
    return {key: score.get(key) for key in keys}


# ----------------------------------------------------------------------------
# Public-test export, parity and packaging
# ----------------------------------------------------------------------------

def export_test(candidate_dir, members, weights, threshold, batch_size, workers, device):
    started = time.monotonic()
    family_dirs = fold_family_dirs("test", members)
    entries = fold_entries("test", family_dirs)
    experts = [load_expert(m["frozen"], family_dirs[m["type"]], device) for m in members]
    columns = feature_columns(read_manifest(family_dirs[GATE_FAMILY])["feature_names"])
    hashes = [m["sha256"] for m in members]
    source_hashes = dict(zip(("primary", "secondary"), hashes))
    output = candidate_dir / "diii_d_public_test.npz"
    temporary = output.with_suffix(".tmp.npz")
    chosen, renormalized = fallback_selection([m["type"] for m in members], weights)
    report = {"status": "running", "created_utc": now(), "split": "test", "candidate": candidate_dir.name,
              "members": [{"index": i, "type": m["type"], "local_source": m["local_source"], "frozen_file": relative(m["frozen"]),
                           "sha256": m["sha256"], "epoch": expert[4], "data_directory": relative(family_dirs[m["type"]])}
                          for i, (m, expert) in enumerate(zip(members, experts))],
              "weights": list(weights), "primary_weight": list(weights), "gate_threshold": threshold,
              "gate_members": [{"index": i, "weight": w} for i, w in zip(chosen, renormalized)],
              "gate_rule": (f"members {chosen} alone (renormalized weights) when at most {threshold:.0%} of real edge channels have "
                            f"positive observed Te and ne; otherwise weighted average of all members"),
              "source_checkpoint_sha256": source_hashes,
              "family_data": {family: relative(path) for family, path in family_dirs.items()},
              "source_manifest_sha256": {family: sha256(path / "manifest.json") for family, path in family_dirs.items()},
              "targets_read": False, "audit_labels_read": False, "official_submission": False,
              "source_code_sha256": {relative(path): sha256(path) for path in (
                  Path(__file__).resolve(), ROOT / "scripts/optimization/export_gated_ensemble.py", ROOT / "scripts/train_pod.py",
                  ROOT / "fusion/pod.py", ROOT / "fusion/advanced_models.py", ROOT / "scripts/optimization/quality_gate.py")},
              "shots": len(entries[GATE_FAMILY]), "frames": 0, "gated_frames": 0, "gated_shots": 0}
    write_json(candidate_dir / "export.json", report)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i in range(report["shots"]):
            psi, scalars, gated = infer_shot(experts, members, weights, family_dirs, entries, i, columns, threshold, batch_size, device)
            for field, values in (("psirz", psi), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                write_array(archive, f"shot_{i:04d}_{field}", values)
            report["frames"] += len(gated)
            report["gated_frames"] += int(gated.sum())
            report["gated_shots"] += int(gated.any())
            if (i + 1) % 100 == 0 or i + 1 == report["shots"]:
                print(f"Ensemble export: {i + 1}/{report['shots']} shots, {report['frames']} frames", flush=True)
    if [sha256(m["frozen"]) for m in members] != hashes:
        raise ValueError("A frozen checkpoint changed during export")
    if report["shots"] != EXPECTED_TEST_SHOTS or report["frames"] != EXPECTED_TEST_FRAMES:
        raise ValueError(f"Export covered {report['shots']} shots / {report['frames']} frames; expected "
                         f"{EXPECTED_TEST_SHOTS} / {EXPECTED_TEST_FRAMES}")
    os.replace(temporary, output)
    report.update(status="complete", prediction=str(output.resolve()), prediction_sha256=sha256(output),
                  prediction_bytes=output.stat().st_size, elapsed_seconds=time.monotonic() - started)
    write_json(candidate_dir / "export.json", report)
    print(json.dumps({key: report[key] for key in ("status", "shots", "frames", "gated_frames", "gated_shots", "elapsed_seconds", "prediction_sha256")}), flush=True)
    return report


def export_parity(candidate_dir, members, weights, threshold, blended, batch_size, device):
    """Re-infer development shots through the export path and compare with the blended dev predictions."""
    records, passed = [], True
    for fold in ("native", "legacy"):
        family_dirs = fold_family_dirs(fold, members)
        entries = fold_entries(fold, family_dirs)
        experts = [load_expert(m["frozen"], family_dirs[m["type"]], device) for m in members]
        columns = feature_columns(read_manifest(family_dirs[GATE_FAMILY])["feature_names"])
        n = len(entries[GATE_FAMILY])
        selected = sorted(set(np.linspace(0, n - 1, 8).astype(int))) if fold == "native" else range(n)
        with np.load(blended[fold], allow_pickle=False) as expected:
            for i in selected:
                psi, scalars, gated = infer_shot(experts, members, weights, family_dirs, entries, i, columns, threshold, batch_size, device)
                errors, ok = {}, True
                for field, value in (("psirz", psi), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                    reference = expected[f"shot_{i:04d}_{field}"]
                    difference = np.abs(value.astype(np.float64) - reference.astype(np.float64))
                    errors[field] = float(difference.max())
                    ok &= bool(np.all(difference <= PARITY_ATOL[field] + PARITY_RTOL * np.abs(reference)))
                passed &= ok
                records.append({"split": FOLDS[fold]["split"], "shot": int(i), "frames": int(len(gated)),
                                "gated_frames": int(gated.sum()), "max_absolute_error": errors, "within_tolerance": ok})
    report = {"passed": bool(passed), "shots": len(records), "frames": sum(r["frames"] for r in records),
              "gated_frames": sum(r["gated_frames"] for r in records), "tolerance": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
              "max_absolute_error": {field: max(r["max_absolute_error"][field] for r in records) for field in FIELDS},
              "records": records, "audit_labels_read": False, "test_labels_read": False}
    write_json(candidate_dir / "export_parity.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "shots", "gated_frames", "max_absolute_error")}), flush=True)
    return report


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, help="Neutral identifier YYYYMMDD-NNN")
    parser.add_argument("--members", nargs="+", required=True, metavar="TYPE=PATH",
                        help="Ordered member checkpoints, e.g. A=results/optimization/pod_scaled_li_v2/best.pt")
    parser.add_argument("--weights", type=float, nargs="+", help="Member weights summing to one (default equal)")
    parser.add_argument("--gate-threshold", type=float, default=0.1)
    parser.add_argument("--dev-predictions", nargs="+", metavar="NPZ",
                        help="Per-member native val_predictions.npz in member order ('-' infers that member)")
    parser.add_argument("--legacy-predictions", nargs="+", metavar="NPZ",
                        help="Per-member legacy_val_predictions.npz in member order ('-' infers that member)")
    parser.add_argument("--baseline", type=Path, default=Path("results/candidates/20260919-001"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Score the development blend on both folds and write nothing under --output-root")
    parser.add_argument("--output-root", type=Path, default=Path("results/candidates"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("results/evaluations"))
    parser.add_argument("--weights-predeclared", action="store_true",
                        help="Record validation_tuned_weight=false (weights were not chosen on validation data)")
    parser.add_argument("--no-export-on-reject", action="store_true",
                        help="Skip test export/parity/packaging when a development gate fails")
    parser.add_argument("--force-accept", action="store_true",
                        help="Testing only: write decision=accept even when a development gate fails so that "
                             "candidate_evidence exercises its full contract; it still enforces the same thresholds itself")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{8}-[0-9]{3}", args.candidate):
        parser.error("Use a neutral numeric candidate identifier YYYYMMDD-NNN")
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch size must be positive")
    if not 0 <= args.gate_threshold <= 1:
        parser.error("gate threshold must lie in [0, 1]")
    try:
        members = parse_members(args.members)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    weights = args.weights if args.weights else [1.0 / len(members)] * len(members)
    if len(weights) != len(members) or any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-9:
        parser.error("weights must be nonnegative, one per member, and sum to one")
    for name in ("dev_predictions", "legacy_predictions"):
        given = getattr(args, name)
        if given and len(given) != len(members):
            parser.error(f"--{name.replace('_', '-')} needs one entry per member (use '-' to infer a member)")
        setattr(args, name, [None if x in ("-", "none", "") else x for x in (given or [None] * len(members))])
    started = time.monotonic()
    torch.set_num_threads(args.workers)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    candidate_dir = ROOT / args.output_root / args.candidate
    evaluation_dir = ROOT / args.evaluation_root / args.candidate
    if not args.evaluate_only and (candidate_dir / "decision.json").exists():
        raise ValueError("This candidate is already frozen; do not overwrite its evidence or checkpoints")
    baseline = json.loads((ROOT / args.baseline / "decision.json").read_text())

    # Resolve sources and check every member against its family feature contract.
    for index, member in enumerate(members):
        source = ROOT / member["local_source"]
        if not source.is_file():
            raise FileNotFoundError(f"Member checkpoint missing: {source}")
        member["source"] = source
        saved = torch.load(source, map_location="cpu", weights_only=False)
        names = read_manifest(ROOT / FAMILIES[member["type"]]["native"])["feature_names"]
        if saved.get("run", {}).get("feature_names") != names:
            raise ValueError(f"Member m{index:02d} feature contract differs from family {member['type']}: {source}")
        member.update(model=saved["model"], epoch=saved.get("epoch"), source_sha256=sha256(source))
        del saved
    summary = {"candidate": args.candidate, "started_utc": now(), "output_root": relative(ROOT / args.output_root),
               "evaluation_dir": relative(evaluation_dir), "evaluate_only": args.evaluate_only,
               "members": [{k: (relative(v) if isinstance(v, Path) else v) for k, v in m.items()} for m in members],
               "weights": weights, "gate_threshold": args.gate_threshold, "baseline": relative(ROOT / args.baseline),
               "audit_labels_read": False, "test_labels_read": False, "mast_labels_read": False}

    # (1) Freeze checkpoints.
    if not args.evaluate_only:
        (candidate_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(members):
            frozen = candidate_dir / "checkpoints" / f"m{index:02d}.pt"
            shutil.copy2(member["source"], frozen)
            member["frozen"], member["sha256"] = frozen, sha256(frozen)
            if member["sha256"] != member["source_sha256"]:
                raise ValueError(f"Frozen copy differs from its source: {frozen}")
        print(f"Froze {len(members)} checkpoints under {relative(candidate_dir / 'checkpoints')}", flush=True)
        summary["members"] = [{k: (relative(v) if isinstance(v, Path) else v) for k, v in m.items()} for m in members]
    else:
        for member in members:
            member["frozen"], member["sha256"] = member["source"], member["source_sha256"]

    # (2) Per-member development predictions and the gated blend.
    gate_columns = {fold: feature_columns(read_manifest(ROOT / FAMILIES[GATE_FAMILY][fold])["feature_names"]) for fold in FOLDS}
    blend_root = evaluation_dir / ("evaluate_only" if args.evaluate_only else "")
    blended, scores, summary["folds"] = {}, {}, {}
    for fold, provided in (("native", args.dev_predictions), ("legacy", args.legacy_predictions)):
        sources, records = [], []
        for index, member in enumerate(members):
            path, record = member_predictions(member, index, fold, provided[index], evaluation_dir, args.workers)
            sources.append(path)
            records.append(record)
        blended[fold] = blend_root / fold / FOLDS[fold]["predictions"]
        coverage = blend_fold(fold, members, weights, sources, blended[fold], gate_columns[fold], args.gate_threshold)
        print(f"{fold}: gate selects fallback members on {coverage['gated_frames']}/{coverage['frames']} frames "
              f"in {coverage['gated_shots']}/{coverage['shots']} shots", flush=True)
        # (3) Official local scoring with the mandatory data directories.
        score_path = blended[fold].with_name(FOLDS[fold]["score"])
        scores[fold] = score_fold(fold, blended[fold], score_path, args.workers)
        summary["folds"][fold] = {"member_predictions": records, "coverage": coverage,
                                  "blended_predictions": {"path": relative(blended[fold]), "sha256": sha256(blended[fold])},
                                  "score_path": relative(score_path), "score": breakdown(scores[fold])}
        print(f"{fold} S={scores[fold]['S']:.6f} " + json.dumps(breakdown(scores[fold])), flush=True)

    # (4) Development gates against the baseline candidate.
    # The baseline candidate's own development scores become this candidate's baseline entries.
    base_native, base_legacy = baseline["validation"]["native"]["S"], baseline["validation"]["legacy"]["S"]
    gates = {"native": {"S": scores["native"]["S"], "baseline_S": base_native, "required": base_native + NATIVE_GAIN,
                        "passed": scores["native"]["S"] >= base_native + NATIVE_GAIN},
             "legacy": {"S": scores["legacy"]["S"], "baseline_S": base_legacy, "required": base_legacy - LEGACY_TOLERANCE,
                        "passed": scores["legacy"]["S"] >= base_legacy - LEGACY_TOLERANCE}}
    gates_passed = gates["native"]["passed"] and gates["legacy"]["passed"]
    decision_value = "accept" if gates_passed or args.force_accept else "reject"
    summary.update(gates=gates, gates_passed=gates_passed, decision=decision_value, forced_accept=bool(args.force_accept and not gates_passed))
    print(f"Development gates: native {scores['native']['S']:.6f} vs {base_native + NATIVE_GAIN:.6f} "
          f"({'pass' if gates['native']['passed'] else 'FAIL'}); legacy {scores['legacy']['S']:.6f} vs "
          f"{base_legacy - LEGACY_TOLERANCE:.6f} ({'pass' if gates['legacy']['passed'] else 'FAIL'}) -> {decision_value}", flush=True)
    if args.evaluate_only:
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(blend_root / "summary.json", summary)
        print(json.dumps({"evaluate_only": True, "native_S": scores["native"]["S"], "legacy_S": scores["legacy"]["S"],
                          "gates_passed": gates_passed, "summary": relative(blend_root / "summary.json")}, indent=2), flush=True)
        return

    # (8) decision.json is frozen before any test inference happens.
    chosen, renormalized = fallback_selection([m["type"] for m in members], weights)
    rationale = (f"{len(members)}-member gated blend (weights {weights}, gate members {chosen}); native development "
                 f"S {scores['native']['S']:.6f} vs baseline {base_native:.6f} (gate {'passed' if gates['native']['passed'] else 'failed'}), "
                 f"legacy S {scores['legacy']['S']:.6f} vs baseline {base_legacy:.6f} (gate {'passed' if gates['legacy']['passed'] else 'failed'}). "
                 f"Weights frozen before test inference; no audit or test labels read.")
    if summary["forced_accept"]:
        rationale += " DECISION FORCED TO ACCEPT FOR CONTRACT TESTING; candidate_evidence still enforces the thresholds."
    decision = {"candidate": args.candidate, "public_name": f"Submission {args.candidate}", "frozen_utc": now(),
                "checkpoints": [{"local_source": m["local_source"], "frozen_file": relative(m["frozen"]), "sha256": m["sha256"]} for m in members],
                "members": [{"index": i, "type": m["type"], "weight": w} for i, (m, w) in enumerate(zip(members, weights))],
                "primary_weight": list(weights), "gate_threshold": args.gate_threshold,
                "validation_tuned_weight": not args.weights_predeclared,
                "audit_labels_read": False, "test_labels_read": False, "decision": decision_value, "rationale": rationale,
                "development_gates": gates,
                "validation": {"native": {"path": relative(blended["native"].with_name(FOLDS["native"]["score"])),
                                          "sha256": sha256(blended["native"].with_name(FOLDS["native"]["score"])), "S": scores["native"]["S"]},
                               "legacy": {"path": relative(blended["legacy"].with_name(FOLDS["legacy"]["score"])),
                                          "sha256": sha256(blended["legacy"].with_name(FOLDS["legacy"]["score"])), "S": scores["legacy"]["S"]},
                               "baseline_native": dict(baseline["validation"]["native"]),
                               "baseline_legacy": dict(baseline["validation"]["legacy"])}}
    write_json(candidate_dir / "decision.json", decision)
    summary["decision_json"] = relative(candidate_dir / "decision.json")
    if decision_value == "reject" and args.no_export_on_reject:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["export"] = "skipped (development gate failed)"
        write_json(candidate_dir / "finisher_summary.json", summary)
        print("Development gates failed; export skipped as requested.", flush=True)
        return

    # (5) Public-test export, (6) parity, (7) packaging.
    export = export_test(candidate_dir, members, weights, args.gate_threshold, args.batch_size, args.workers, device)
    summary["export"] = {key: export[key] for key in ("status", "shots", "frames", "gated_frames", "gated_shots",
                                                      "prediction_sha256", "prediction_bytes", "elapsed_seconds")}
    parity = export_parity(candidate_dir, members, weights, args.gate_threshold, blended, args.batch_size, device)
    summary["export_parity"] = {key: parity[key] for key in ("passed", "shots", "frames", "gated_frames", "max_absolute_error")}
    package_started = time.monotonic()
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/package_submission.py"), "--pred", str(candidate_dir / "diii_d_public_test.npz"),
                                "--output", str(candidate_dir / "submission.zip")], cwd=ROOT)
    validation = json.loads((candidate_dir / "validation.json").read_text()) if (candidate_dir / "validation.json").exists() else {}
    summary["packaging"] = {"returncode": completed.returncode, "status": validation.get("status"), "error": validation.get("error"),
                            "zip_sha256": validation.get("zip_sha256"), "input_sha256": validation.get("input_sha256"),
                            "validated_shots": validation.get("validated_shots"), "frames": validation.get("frames"),
                            "elapsed_seconds": time.monotonic() - package_started}

    # (9) Ask the periodic submitter what it thinks of the finished candidate.
    from scripts.periodic_submission import candidate_evidence
    try:
        evidence = candidate_evidence(candidate_dir)
        summary["candidate_evidence"] = {"outcome": "returned", "value": evidence}
        print(f"candidate_evidence -> {evidence!r}", flush=True)
    except Exception as exc:  # report, never hide, what the submitter would do
        summary["candidate_evidence"] = {"outcome": "raised", "exception": f"{type(exc).__name__}: {exc}"}
        print(f"candidate_evidence raised {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(candidate_dir / "finisher_summary.json", summary)
    print(json.dumps({"candidate": args.candidate, "decision": decision_value, "native_S": scores["native"]["S"],
                      "legacy_S": scores["legacy"]["S"], "prediction_sha256": export["prediction_sha256"],
                      "parity_passed": parity["passed"], "packaging": summary["packaging"]["status"],
                      "candidate_evidence": summary["candidate_evidence"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
