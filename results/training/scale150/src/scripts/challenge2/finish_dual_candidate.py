#!/usr/bin/env python3
"""Finish a generic DUAL (DIII-D + MAST) submission candidate from frozen parts.

The DIII-D half is copied byte-identically from an already finished (ideally
verified) DIII-D candidate directory; the MAST half is a complete
``mast_public_test.npz`` produced by ``scripts/challenge2/predict_mast_cnn.py``
from a Challenge-2 checkpoint. The result satisfies exactly the file contract
that ``scripts/periodic_submission.py::candidate_evidence`` checks on its dual
branch and that ``cycle()::preserves_best_diii_d`` requires:

    <root>/<candidate>/checkpoints/mNN.pt         frozen DIII-D member checkpoints (copied)
    <root>/<candidate>/checkpoints/mast_model.pt  frozen Challenge-2 checkpoint (appended last)
    <root>/<candidate>/decision.json              DIII-D decision + dual/MAST provenance
    <root>/<candidate>/export.json                DIII-D export + ``mast`` block
    <root>/<candidate>/export_parity.json         DIII-D parity (copied)
    <root>/<candidate>/diii_d_public_test.npz     hard link (or copy) of the DIII-D payload
    <root>/<candidate>/mast_public_test.npz       copy of the MAST payload
    <root>/<candidate>/mast_export.json           MAST export provenance and payload hash
    <root>/<candidate>/transfer_review.json       independent development review record
    <root>/<candidate>/submission.zip             scripts/package_dual_submission.py (+ validation.json)
    <root>/<candidate>/finisher_summary.json      everything this script did

Development-only evidence: the MAST payload is validated structurally against
the pinned raw manifest (keys, float32, finite, frame counts); the physics
report and the DIII-D dev-fold score of the Challenge-2 model are hash-frozen.
No audit split, test label or MAST label is ever read (none exist locally).
This script never submits and never touches submission_state.json.
"""
from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.codabench_submit import write_json  # noqa: E402
from scripts.periodic_submission import candidate_evidence, digest, preserves_best_diii_d  # noqa: E402
from package_submission import MACHINE_SHOTS, validate_prediction  # noqa: E402

MAST_CONFIG = "mast_public_test"
MAST_SHOTS = MACHINE_SHOTS[MAST_CONFIG]
MAST_FRAMES = 69075
DIII_D_CONFIG = "diii_d_public_test"
PLAUSIBILITY_KEYS = ("checkpoint", "checkpoint_epoch", "checkpoint_val_S", "feature_spec_hash", "shots", "frames",
                     "lcfs_success_rate_frames", "shots_with_all_lcfs_ok", "shots_with_no_lcfs", "plausible_fractions",
                     "o_point_local_max_fraction", "q95_pred", "betaN_pred", "label_role", "device", "predict_seconds")


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    """Repository-relative string for a path inside ROOT (absolute otherwise)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def frozen_record(path):
    return {"path": relative(path), "sha256": digest(path)}


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std())}


def validate_mast_npz(npz, raw_dir):
    """Structural validation of a complete MAST payload plus label-free statistics.

    Reuses ``package_submission.validate_prediction`` (keys, finite floating arrays, per-shot frame
    counts from the pinned manifest's ``efit_times``) and additionally requires float32 members,
    exactly 1206 shots / 69075 frames, and reports per-array statistics.
    """
    npz = Path(npz)
    if not npz.is_file():
        raise FileNotFoundError(f"MAST payload missing: {npz}")
    started = time.monotonic()
    report = validate_prediction(npz, Path(raw_dir), MAST_SHOTS, MAST_CONFIG)
    if report["validated_shots"] != MAST_SHOTS or report["frames"] != MAST_FRAMES:
        raise ValueError(f"MAST payload covers {report['validated_shots']} shots / {report['frames']} frames; "
                         f"expected {MAST_SHOTS} / {MAST_FRAMES}")
    if set(report["dtype_array_counts"]) != {"float32"}:
        raise ValueError(f"MAST payload must be float32 throughout: {report['dtype_array_counts']}")
    with zipfile.ZipFile(npz) as archive:
        if archive.comment:
            raise ValueError("MAST payload carries a zip comment")
        compression = sorted({info.compress_type for info in archive.infolist()})
    psi_mins, psi_maxs, psi_means, q95, betaN, psi_frames = [], [], [], [], [], 0
    axis_positive = 0
    with np.load(npz, allow_pickle=False) as arrays:
        for index in range(MAST_SHOTS):
            psi = arrays[f"shot_{index:04d}_psirz"]
            per_frame_max = psi.reshape(len(psi), -1).max(axis=1)
            interior_max = psi[:, 1:-1, 1:-1].reshape(len(psi), -1).max(axis=1)
            axis_positive += int((interior_max >= per_frame_max).sum())
            psi_frames += len(psi)
            psi_mins.append(float(psi.min())); psi_maxs.append(float(psi.max())); psi_means.append(float(psi.mean()))
            q95.append(arrays[f"shot_{index:04d}_q95"]); betaN.append(arrays[f"shot_{index:04d}_betaN"])
    q95, betaN = np.concatenate(q95), np.concatenate(betaN)
    frames = [s["frames"] for s in report["shots"]]
    statistics = {
        "shots": MAST_SHOTS, "frames": psi_frames, "frames_per_shot": {"min": int(min(frames)), "max": int(max(frames)),
                                                                        "mean": float(np.mean(frames))},
        "psirz": {"min": min(psi_mins), "max": max(psi_maxs), "mean_of_shot_means": float(np.mean(psi_means))},
        "psirz_interior_maximum_fraction": axis_positive / psi_frames,
        "q95": summarize_array(q95), "betaN": summarize_array(betaN),
        "npz_member_compression": compression, "bytes": npz.stat().st_size,
    }
    summary = {k: v for k, v in report.items() if k != "shots"}
    summary.update(statistics=statistics, elapsed_seconds=time.monotonic() - started, prediction=str(npz.resolve()),
                   prediction_sha256=digest(npz))
    return summary


def load_mast_checkpoint_metadata(path):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = {"epoch": saved.get("epoch"), "validation_score": saved.get("validation_score"),
            "model": saved.get("model"), "model_kwargs": saved.get("model_kwargs")}
    spec = saved.get("feature_spec")
    if spec is not None:
        try:
            from fusion import c2_fields_data as cd
            meta["feature_spec_hash"] = cd.feature_spec_hash(spec)
            meta["feature_spec_matches_current_code"] = spec == cd.feature_spec()
        except Exception as exc:  # metadata only; never blocks the finisher
            meta["feature_spec_hash_error"] = f"{type(exc).__name__}: {exc}"
    for key in ("feature_spec_hash",):
        if key in saved and "feature_spec_hash" not in meta:
            meta["feature_spec_hash"] = saved[key]
    run = saved.get("run") if isinstance(saved.get("run"), dict) else {}
    for key in ("mast_labels_read", "audit_labels_read", "started_utc", "source_sha256"):
        if key in run:
            meta.setdefault("run", {})[key] = run[key]
    return meta


def check_dev_score(path):
    score = read_json(path)
    evaluation = score.get("evaluation", {})
    if evaluation.get("type") != "local_held_out_validation" or evaluation.get("split") not in {"val", "legacy_val"}:
        raise ValueError("--dev-score must be an official local held-out validation (val or legacy_val) output")
    if evaluation.get("leaderboard_submission") not in (False, None) or evaluation.get("machine") not in ("DIII-D", None):
        raise ValueError("--dev-score must be a DIII-D development evaluation, not a leaderboard score")
    S = float(score["S"])
    if not (0 <= S <= 1):
        raise ValueError("Development S is outside its valid range")
    return {"S": S, "split": evaluation.get("split"), "sampled": evaluation.get("sampled"),
            "evaluated_shots": evaluation.get("evaluated_shots"), "evaluated_frames": evaluation.get("evaluated_frames"),
            "scorer_version": evaluation.get("scorer_version"), "scorer_sha256": evaluation.get("scorer_sha256"),
            "r2_psi": score.get("r2_psi"), "r2_qb": score.get("r2_qb"), "dlcfs": score.get("dlcfs"),
            "consistency": score.get("consistency"), "n_shots_missing": score.get("n_shots_missing"),
            "psi_nonfinite_frac": score.get("psi_nonfinite_frac")}


def same_file(recorded, actual):
    recorded = Path(recorded)
    recorded = recorded if recorded.is_absolute() else ROOT / recorded
    return recorded.resolve() == Path(actual).resolve()


def check_physics_report(path, mast_npz, mast_checkpoint):
    report = read_json(path)
    if not isinstance(report, dict):
        raise ValueError("--physics-report must be a JSON object")
    for key in ("mast_labels_read", "audit_labels_read", "test_labels_read"):
        if report.get(key) not in (False, None):
            raise ValueError(f"Physics report declares {key}={report[key]!r}")
    if "npz" in report and not same_file(report["npz"], mast_npz):
        raise ValueError(f"Physics report describes a different payload: {report['npz']}")
    if "checkpoint" in report and not same_file(report["checkpoint"], mast_checkpoint):
        raise ValueError(f"Physics report describes a different checkpoint: {report['checkpoint']}")
    for key, expected in (("shots", MAST_SHOTS), ("frames", MAST_FRAMES)):
        if key in report and report[key] != expected:
            raise ValueError(f"Physics report {key}={report[key]} but a complete MAST payload has {expected}")
    return {k: report[k] for k in PLAUSIBILITY_KEYS if k in report}


def replace_paths(value, mapping):
    """Recursively replace string values that name an old frozen file (relative or absolute)."""
    if isinstance(value, dict):
        return {k: replace_paths(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v, mapping) for v in value]
    if isinstance(value, str) and value in mapping:
        return mapping[value]
    return value


def link_or_copy(source, target, allow_link=True):
    if allow_link:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def run_packaging(candidate_dir, log_path):
    command = [sys.executable, str(ROOT / "scripts/package_dual_submission.py"),
               "--diii-d", str(candidate_dir / f"{DIII_D_CONFIG}.npz"), "--mast", str(candidate_dir / f"{MAST_CONFIG}.npz"),
               "--output", str(candidate_dir / "submission.zip")]
    with log_path.open("w") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    validation = read_json(candidate_dir / "validation.json") if (candidate_dir / "validation.json").exists() else {}
    return {"command": command, "returncode": completed.returncode, "log": relative(log_path),
            "status": validation.get("status"), "error": validation.get("error"), "zip_sha256": validation.get("zip_sha256"),
            "zip_bytes": validation.get("zip_bytes"),
            "payloads": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in validation.get("payloads", {}).items()},
            "machines": {k: {kk: v.get(kk) for kk in ("status", "validated_shots", "frames")}
                         for k, v in validation.get("machines", {}).items()},
            "elapsed_seconds": validation.get("elapsed_seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", help="Neutral identifier YYYYMMDD-NNN of the new dual candidate")
    parser.add_argument("--diii-d-candidate", help="Finished DIII-D candidate: a name under --diii-d-root or a directory path")
    parser.add_argument("--diii-d-root", type=Path, default=Path("results/candidates"))
    parser.add_argument("--mast-npz", type=Path, required=True, help="mast_public_test.npz from predict_mast_cnn.py")
    parser.add_argument("--mast-checkpoint", type=Path, help="best.pt of the Challenge-2 model that produced --mast-npz")
    parser.add_argument("--physics-report", type=Path, help="JSON development report (mast_plausibility.json or run.json)")
    parser.add_argument("--dev-score", type=Path, help="Official evaluator val_score.json of the C2 model on the DIII-D dev fold")
    parser.add_argument("--rationale", help="Independent development rationale recorded in transfer_review.json")
    parser.add_argument("--output-root", type=Path, default=Path("results/candidates"))
    parser.add_argument("--state", type=Path, default=Path("results/candidates/submission_state.json"),
                        help="Read-only submission state used for the preserves_best_diii_d check")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--copy-diii-d", action="store_true", help="Copy the DIII-D payload instead of hard-linking it")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Validate --mast-npz and print its statistics; write nothing under --output-root")
    args = parser.parse_args()
    args.raw_dir = (ROOT / args.raw_dir) if not args.raw_dir.is_absolute() else args.raw_dir

    if args.evaluate_only:
        summary = validate_mast_npz(args.mast_npz, args.raw_dir)
        print(json.dumps(summary, indent=2))
        return 0

    missing = [name for name in ("candidate", "diii_d_candidate", "mast_checkpoint", "physics_report", "dev_score", "rationale")
               if getattr(args, name) is None]
    if missing:
        parser.error("Required unless --evaluate-only: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if not re.fullmatch(r"[0-9]{8}-[0-9]{3}", args.candidate):
        parser.error("--candidate must be a neutral identifier YYYYMMDD-NNN")
    if not args.rationale.strip():
        parser.error("--rationale must not be empty")
    started = time.monotonic()
    started_utc = now()
    source_dir = Path(args.diii_d_candidate)
    if not source_dir.is_dir():
        source_dir = (ROOT / args.diii_d_root / args.diii_d_candidate) if not args.diii_d_root.is_absolute() \
            else args.diii_d_root / args.diii_d_candidate
    source_dir = source_dir.resolve()
    output_root = (ROOT / args.output_root) if not args.output_root.is_absolute() else args.output_root
    candidate_dir = (output_root / args.candidate).resolve()
    if candidate_dir.exists():
        parser.error(f"Use a fresh candidate directory; {candidate_dir} exists")
    if source_dir == candidate_dir:
        parser.error("The DIII-D source candidate must differ from the new candidate")

    # ---- DIII-D source: decision, export, parity, frozen checkpoints and payload ----
    for name in ("decision.json", "export.json", "export_parity.json", f"{DIII_D_CONFIG}.npz"):
        if not (source_dir / name).exists():
            raise FileNotFoundError(f"DIII-D source lacks {name}: {source_dir}")
    original = read_json(source_dir / "decision.json")
    d3export = read_json(source_dir / "export.json")
    d3parity = read_json(source_dir / "export_parity.json")
    if original.get("decision") != "accept" or d3export.get("status") != "complete" or d3parity.get("passed") is not True:
        raise ValueError("The DIII-D source must be an accepted, completely exported, parity-checked candidate")
    if original.get("test_labels_read") is not False or original.get("audit_labels_read") is not False:
        raise ValueError("The DIII-D source lacks independent validation provenance")
    d3payload = source_dir / f"{DIII_D_CONFIG}.npz"
    d3sha = digest(d3payload)
    if d3sha != d3export["prediction_sha256"]:
        raise ValueError("The DIII-D payload differs from its frozen export hash")
    d3_validation = read_json(source_dir / "validation.json") if (source_dir / "validation.json").exists() else {}
    if d3_validation:
        packaged = d3_validation.get("input_sha256") or d3_validation.get("payloads", {}).get(DIII_D_CONFIG, {}).get("sha256")
        if packaged != d3sha:
            raise ValueError("The DIII-D payload differs from the one validated for its own submission")

    # ---- MAST inputs: payload, checkpoint, report, dev score ----
    mast_npz = args.mast_npz.resolve()
    mast_validation = validate_mast_npz(mast_npz, args.raw_dir)
    mast_sha = mast_validation["prediction_sha256"]
    print(f"MAST payload validated: {MAST_SHOTS} shots, {mast_validation['frames']} frames, SHA256 {mast_sha}", flush=True)
    mast_ckpt = args.mast_checkpoint.resolve()
    if mast_ckpt.is_dir():
        mast_ckpt = mast_ckpt / "best.pt"
    mast_ckpt_meta = load_mast_checkpoint_metadata(mast_ckpt)
    report_path = args.physics_report.resolve()
    plausibility = check_physics_report(report_path, mast_npz, mast_ckpt)
    if "feature_spec_hash" in plausibility and mast_ckpt_meta.get("feature_spec_hash") not in (None, plausibility["feature_spec_hash"]):
        raise ValueError("Physics report and checkpoint carry different feature specs")
    dev_score_path = args.dev_score.resolve()
    dev_score = check_dev_score(dev_score_path)

    state_path = (ROOT / args.state) if not args.state.is_absolute() else args.state
    state = read_json(state_path) if state_path.exists() else None
    best_submission_id = (state or {}).get("historical_best", {}).get("d3d_S", {}).get("submission_id")

    # ---- Assemble the candidate ----
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "checkpoints").mkdir()
    summary = {"candidate": args.candidate, "started_utc": started_utc, "output_root": relative(output_root),
               "evaluate_only": False, "diii_d_candidate": relative(source_dir), "diii_d_prediction_sha256": d3sha,
               "mast_npz_source": relative(mast_npz), "mast_prediction_sha256": mast_sha,
               "audit_labels_read": False, "test_labels_read": False, "mast_labels_read": False}
    try:
        records, path_map = [], {}
        for old in original["checkpoints"]:
            source = ROOT / old["frozen_file"]
            if digest(source) != old["sha256"]:
                raise ValueError(f"Frozen DIII-D checkpoint changed: {source}")
            target = candidate_dir / "checkpoints" / source.name
            if target.exists():
                raise ValueError(f"Duplicate checkpoint basename in the DIII-D source: {source.name}")
            shutil.copy2(source, target)
            sha = digest(target)
            if sha != old["sha256"]:
                raise ValueError(f"Checkpoint copy differs: {target}")
            record = {k: v for k, v in old.items() if k not in ("frozen_file", "sha256", "local_source")}
            record.update(local_source=relative(source), frozen_file=relative(target), sha256=sha)
            records.append(record)
            for form in (old["frozen_file"], str(source), relative(source)):
                path_map[form] = relative(target)
        mast_target = candidate_dir / "checkpoints" / "mast_model.pt"
        shutil.copy2(mast_ckpt, mast_target)
        mast_ckpt_sha = digest(mast_target)
        if mast_ckpt_sha != digest(mast_ckpt):
            raise ValueError("MAST checkpoint copy differs from its source")
        mast_record = {"role": "mast_transfer_model", "local_source": relative(mast_ckpt), "frozen_file": relative(mast_target),
                       "sha256": mast_ckpt_sha}
        summary["checkpoints"] = {"diii_d": len(records), "mast": mast_record}

        d3target = candidate_dir / f"{DIII_D_CONFIG}.npz"
        transfer = link_or_copy(d3payload, d3target, allow_link=not args.copy_diii_d)
        if digest(d3target) != d3sha:
            raise ValueError("The staged DIII-D payload is not byte-identical to its source")
        mast_payload = candidate_dir / f"{MAST_CONFIG}.npz"
        shutil.copy2(mast_npz, mast_payload)
        if digest(mast_payload) != mast_sha:
            raise ValueError("The staged MAST payload is not byte-identical to its source")
        summary["payloads"] = {DIII_D_CONFIG: {"transfer": transfer, "bytes": d3target.stat().st_size, "sha256": d3sha},
                               MAST_CONFIG: {"transfer": "copy", "bytes": mast_payload.stat().st_size, "sha256": mast_sha}}
        print(f"Staged payloads: DIII-D via {transfer} ({d3sha}); MAST copied ({mast_sha})", flush=True)

        # Frozen copies of the development evidence live inside the candidate.
        report_copy = candidate_dir / "mast_development_report.json"
        shutil.copy2(report_path, report_copy)
        dev_copy = candidate_dir / "mast_model_dev_val_score.json"
        shutil.copy2(dev_score_path, dev_copy)
        report_frozen = frozen_record(report_copy) | {"source_path": relative(report_path), "source_sha256": digest(report_path)}
        dev_frozen = frozen_record(dev_copy) | {"source_path": relative(dev_score_path), "source_sha256": digest(dev_score_path)}
        if report_frozen["sha256"] != report_frozen["source_sha256"] or dev_frozen["sha256"] != dev_frozen["source_sha256"]:
            raise ValueError("Development evidence copy differs from its source")

        mast_model = {"checkpoint": mast_record, "feature_spec_hash": mast_ckpt_meta.get("feature_spec_hash"),
                      "feature_spec_matches_current_code": mast_ckpt_meta.get("feature_spec_matches_current_code"),
                      "epoch": mast_ckpt_meta.get("epoch"), "checkpoint_validation_score": mast_ckpt_meta.get("validation_score"),
                      "model": mast_ckpt_meta.get("model"), "model_kwargs": mast_ckpt_meta.get("model_kwargs"),
                      "dev_S": dev_score["S"], "dev_score": dev_frozen | dev_score,
                      "inference_script": frozen_record(ROOT / "scripts/challenge2/predict_mast_cnn.py"),
                      "mast_accuracy_established": False}
        if "run" in mast_ckpt_meta:
            mast_model["training_run"] = mast_ckpt_meta["run"]

        # (1) decision.json
        decision = copy.deepcopy(original)
        decision.update(candidate=args.candidate, public_name="Submission " + args.candidate, frozen_utc=now(),
                        checkpoints=records + [mast_record], dual=True, rationale=args.rationale,
                        diii_d_source={"candidate": source_dir.name, "path": relative(source_dir),
                                       "decision_rationale": original.get("rationale"), "prediction_sha256": d3sha,
                                       "public_name": original.get("public_name"), "frozen_utc": original.get("frozen_utc")},
                        mast_model=mast_model, mast_labels_read=False)
        decision["validation"]["baseline_native"] = copy.deepcopy(decision["validation"]["native"])
        decision["validation"]["baseline_legacy"] = copy.deepcopy(decision["validation"]["legacy"])
        if "development_gates" in decision:
            decision["diii_d_source"]["development_gates"] = decision.pop("development_gates")
        write_json(candidate_dir / "decision.json", decision)

        # (2) export.json
        exported = replace_paths(copy.deepcopy(d3export), path_map | {str(d3payload): str(d3target), relative(d3payload): str(d3target)})
        exported.update(candidate=args.candidate, prediction=str(d3target),
                        diii_d_source={"candidate": source_dir.name, "export_created_utc": d3export.get("created_utc"),
                                       "prediction": str(d3payload)},
                        mast={"status": "complete", "config": MAST_CONFIG, "prediction": str(mast_payload), "prediction_sha256": mast_sha,
                              "prediction_bytes": mast_payload.stat().st_size, "shots": MAST_SHOTS, "frames": mast_validation["frames"],
                              "source_checkpoint": mast_record, "source_checkpoint_sha256": mast_ckpt_sha,
                              "feature_spec_hash": mast_model["feature_spec_hash"], "mast_labels_read": False,
                              "audit_labels_read": False, "official_submission": False})
        write_json(candidate_dir / "export.json", exported)

        # (3) export_parity.json
        parity = copy.deepcopy(d3parity)
        parity.update(unchanged_verified_diii_d_payload=True, diii_d_source_candidate=source_dir.name,
                      baseline_submission_id=best_submission_id)
        write_json(candidate_dir / "export_parity.json", parity)

        # (4) mast_export.json
        mast_export = {"status": "complete", "scope": "full", "created_utc": now(), "candidate": args.candidate,
                       "prediction": str(mast_payload), "prediction_sha256": mast_sha, "prediction_bytes": mast_payload.stat().st_size,
                       "prediction_source": relative(mast_npz), "shots": MAST_SHOTS, "frames": mast_validation["frames"],
                       "scalar_valid": mast_validation["frames"], "runtime_limited_frames": 0,
                       "source_checkpoint": mast_record, "source_checkpoint_sha256": mast_ckpt_sha,
                       "feature_spec_hash": mast_model["feature_spec_hash"], "inference_script": mast_model["inference_script"],
                       "psi_convention": "MAST axis maximum (+phi_hat)",
                       "structural_validation": {k: v for k, v in mast_validation.items() if k not in ("prediction", "prediction_sha256")},
                       "plausibility": plausibility, "development_physics_report": report_frozen,
                       "mast_labels_read": False, "audit_labels_read": False, "test_labels_read": False,
                       "eligible_for_submission": False,
                       "limitations": "Structural validity and input-only plausibility are distinct from unverified MAST accuracy"}
        write_json(candidate_dir / "mast_export.json", mast_export)

        # (5) transfer_review.json
        review = {"reviewed_utc": now(), "created_utc": now(), "decision": "accept", "candidate": args.candidate,
                  "purpose": "Dual submission: verified DIII-D payload plus a Challenge-2 MAST transfer payload",
                  "mast_labels_read": False, "audit_labels_read": False, "test_labels_read": False,
                  "mast_accuracy_established": False,
                  "development_physics_report": report_frozen,
                  "dev_score": dev_frozen | {"S": dev_score["S"], "evaluation": {k: v for k, v in dev_score.items() if k != "S"}},
                  "mast_model": {"checkpoint": mast_record, "feature_spec_hash": mast_model["feature_spec_hash"],
                                 "epoch": mast_model["epoch"]},
                  "rationale": args.rationale,
                  "baseline_diii_d_prediction_sha256": d3sha, "baseline_submission_id": best_submission_id,
                  "diii_d_candidate": source_dir.name, "mast_prediction_sha256": mast_sha}
        write_json(candidate_dir / "transfer_review.json", review)

        # (6) package both machines
        print("Packaging both machines with scripts/package_dual_submission.py ...", flush=True)
        summary["packaging"] = run_packaging(candidate_dir, candidate_dir / "package.log")
        if summary["packaging"]["returncode"] != 0 or summary["packaging"]["status"] != "passed":
            raise RuntimeError(f"Dual packaging failed: {summary['packaging'].get('error')}")
        with zipfile.ZipFile(candidate_dir / "submission.zip") as archive:
            summary["zip_members"] = [{"name": i.filename, "bytes": i.file_size, "compress_type": i.compress_type}
                                      for i in archive.infolist()]
            summary["zip_comment_empty"] = not archive.comment

        # (7) queue admission checks (read-only)
        evidence = None
        try:
            evidence = candidate_evidence(candidate_dir)
            summary["candidate_evidence"] = {"outcome": "returned", "value": evidence}
        except Exception as exc:
            summary["candidate_evidence"] = {"outcome": "raised", "exception": f"{type(exc).__name__}: {exc}"}
        if state is None:
            summary["preserves_best_diii_d"] = {"outcome": "skipped", "reason": f"no state at {relative(state_path)}"}
        else:
            try:
                preserved = preserves_best_diii_d(state_path.parent, state, d3sha)
                summary["preserves_best_diii_d"] = {"outcome": "returned", "value": bool(preserved),
                                                    "best_submission_id": best_submission_id}
            except Exception as exc:
                summary["preserves_best_diii_d"] = {"outcome": "raised", "exception": f"{type(exc).__name__}: {exc}"}
        ok = (summary["candidate_evidence"]["outcome"] == "returned" and isinstance(evidence, dict) and evidence.get("dual") is True
              and summary["preserves_best_diii_d"].get("value") is True)
        summary["ready_for_periodic_submission"] = bool(ok)
        summary["status"] = "finished"
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(candidate_dir / "finisher_summary.json", summary)
        print(f"Finisher failed: {exc}", file=sys.stderr)
        return 1
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(candidate_dir / "finisher_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("candidate", "candidate_evidence", "preserves_best_diii_d", "zip_members",
                                              "ready_for_periodic_submission", "elapsed_seconds")}, indent=2), flush=True)
    return 0 if summary["ready_for_periodic_submission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
