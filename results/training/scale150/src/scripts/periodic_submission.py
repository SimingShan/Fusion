#!/usr/bin/env python3
"""Submit frozen, validated improvements with persistent cadence and receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json

BASE = "https://www.codabench.org"
SCORES = {"d3d_S", "mast_S", "g_ratio", "d3d_r2_psi", "d3d_r2_qb",
          "d3d_dlcfs", "d3d_consistency", "mast_r2_psi", "mast_r2_qb",
          "mast_dlcfs", "mast_consistency"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def submission_priority(run, private):
    # Reconcile every prior POST before considering another submission. Among
    # new candidates, inspect the strongest declared development result first;
    # its hashes, actual scores and readiness are still checked below.
    ambiguous = private.get("submission_post_started") or (private.get("dataset_post_started") and not private.get("dataset_key"))
    if private.get("submission_id") or ambiguous:
        return (0, 0., run.name)
    decision = read_json(run / "decision.json", {})
    score = float(decision.get("validation", {}).get("native", {}).get("S", 0))
    return (1, -score if math.isfinite(score) else 0., run.name)


def candidate_evidence(run):
    decision = read_json(run / "decision.json")
    validation = read_json(run / "validation.json")
    exported = read_json(run / "export.json")
    if not decision or not validation or not exported:
        return None
    if decision.get("decision") != "accept" or validation.get("status") != "passed" or exported.get("status") != "complete":
        return None
    if decision.get("test_labels_read") is not False or decision.get("audit_labels_read") is not False:
        raise ValueError("Candidate lacks independent validation provenance")
    if not re.fullmatch(r"Submission [0-9][0-9-]{2,39}", decision["public_name"]):
        raise ValueError("Public candidate identifier is not neutral")
    if Path(validation["submission_zip"]).resolve() != (run / "submission.zip").resolve():
        raise ValueError("Validation references a different archive")
    dual = set(validation.get("machines", {})) == {"diii_d_public_test", "mast_public_test"}
    values, evaluations = {}, {}
    for key in ("native", "legacy", "baseline_native", "baseline_legacy"):
        source = decision["validation"][key]
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError("Frozen evaluation changed")
        score = read_json(path)
        evaluation = score["evaluation"]
        if evaluation.get("type") != "local_held_out_validation" or evaluation.get("split") not in {"val", "legacy_val"}:
            raise ValueError("Only development evaluation may select candidates")
        if any(score.get(k, 0) for k in ("n_shots_missing", "psi_nonfinite_frac", "lcfs_fail_frac", "cons_fail_frac")):
            raise ValueError("Candidate has incomplete or failed predictions")
        values[key], evaluations[key] = float(score["S"]), evaluation
        if not math.isfinite(values[key]) or not 0 <= values[key] <= 1:
            raise ValueError("Development score is outside its valid finite range")
    for key in ("native", "legacy"):
        a, b = evaluations[key], evaluations["baseline_" + key]
        for field in ("split", "evaluated_shots", "evaluated_frames", "scorer_sha256", "data_directory"):
            if a.get(field) != b.get(field):
                raise ValueError("Candidate and baseline were evaluated differently")
    if evaluations["native"].get("sampled") is not False:
        raise ValueError("Full native-frame validation is required")
    minimum_gain = 0 if dual else 0.0001
    if values["native"] < values["baseline_native"] + minimum_gain or values["legacy"] < values["baseline_legacy"] - 0.0001:
        raise ValueError("Development improvement does not meet the submission threshold")
    parity = read_json(run / "export_parity.json")
    if not parity or parity.get("passed") is not True:
        return None
    for checkpoint in decision["checkpoints"]:
        if digest(ROOT / checkpoint["frozen_file"]) != checkpoint["sha256"]:
            raise ValueError("Frozen checkpoint changed")
    if exported["primary_weight"] != decision["primary_weight"]:
        raise ValueError("Export differs from the frozen selection")
    if exported["gate_threshold"] != decision["gate_threshold"]:
        raise ValueError("Export gate differs from the frozen selection")
    input_hash = validation["payloads"]["diii_d_public_test"]["sha256"] if dual else validation["input_sha256"]
    if exported["prediction_sha256"] != input_hash:
        raise ValueError("Packaged predictions differ from the frozen export")
    expected_sources = dict(zip(("primary", "secondary"), (x["sha256"] for x in decision["checkpoints"])))
    if exported["source_checkpoint_sha256"] != expected_sources:
        raise ValueError("Export used different checkpoints")
    archive = run / "submission.zip"
    if digest(archive) != validation["zip_sha256"]:
        raise ValueError("Archive changed after validation")
    expected_members = ["diii_d_public_test.npz"]
    if dual:
        review, mast = read_json(run / "transfer_review.json"), read_json(run / "mast_export.json")
        if not review or not mast or review.get("decision") != "accept" or mast.get("status") != "complete":
            return None
        if review.get("mast_labels_read") is not False or review.get("audit_labels_read") is not False:
            raise ValueError("Transfer review lacks label independence")
        if not review.get("development_physics_report") or not review.get("rationale"):
            raise ValueError("Transfer review lacks development evidence")
        physical_report = review["development_physics_report"]
        if digest(ROOT / physical_report["path"]) != physical_report["sha256"]:
            raise ValueError("Transfer development evidence changed")
        if review.get("baseline_diii_d_prediction_sha256") != input_hash:
            raise ValueError("Transfer candidate must preserve the validated DIII-D predictions")
        if mast.get("prediction_sha256") != validation["payloads"]["mast_public_test"]["sha256"]:
            raise ValueError("Transfer export differs from the validated archive")
        for machine, count in (("diii_d_public_test", 874), ("mast_public_test", 1206)):
            report = validation["machines"][machine]
            if report.get("status") != "passed" or report.get("validated_shots") != count:
                raise ValueError("Incomplete machine coverage")
        expected_members.append("mast_public_test.npz")
    with zipfile.ZipFile(archive) as z:
        if z.comment or sorted(z.namelist()) != expected_members:
            raise ValueError("Unexpected content in the public archive")
    return {"public_name": decision["public_name"], "sha256": validation["zip_sha256"],
            "diii_d_prediction_sha256": input_hash,
            "native": values["native"], "legacy": values["legacy"], "dual": dual}


def preserves_best_diii_d(root, state, prediction_hash):
    """A transfer attempt must keep the best verified DIII-D payload intact."""
    best = state.get("historical_best", {}).get("d3d_S", {})
    sid = best.get("submission_id")
    matches = [(key, value) for key, value in state.get("entries", {}).items()
               if value.get("verified_success") and value.get("submission_id") == sid]
    if sid is None or len(matches) != 1:
        raise ValueError("A unique verified DIII-D reference is required")
    key, entry = matches[0]
    validation = read_json(root / key / "validation.json", {})
    if validation.get("status") != "passed" or validation.get("zip_sha256") != entry.get("sha256"):
        raise ValueError("The verified DIII-D archive reference changed")
    reference = validation.get("payloads", {}).get("diii_d_public_test", {}).get("sha256") or validation.get("input_sha256")
    exported = read_json(root / key / "export.json", {})
    if not reference or exported.get("status") != "complete" or exported.get("prediction_sha256") != reference:
        raise ValueError("The verified DIII-D prediction reference changed")
    if entry.get("diii_d_prediction_sha256", reference) != reference:
        raise ValueError("Persisted DIII-D payload differs from the verified reference")
    return prediction_hash == reference


def api(session, path):
    response = session.get(BASE + path, timeout=(20, 60))
    if not response.ok:
        raise RuntimeError(f"Official service returned HTTP {response.status_code}")
    return response.json()


def monitor(session, run, sid):
    result = api(session, f"/api/submissions/{sid}/")
    children = [api(session, f"/api/submissions/{child['id'] if isinstance(child, dict) else child}/")
                for child in result.get("children", [])]
    scores = {x["column_key"]: float(x["score"]) for x in result.get("scores", []) if x["column_key"] in SCORES}
    receipt = {"submission_id": sid, "checked_utc": now(), "status": result.get("status"),
               "scores": scores, "artifact_private": result.get("is_public") is False,
               "on_leaderboard": result.get("on_leaderboard"),
               "children": [{"id": c["id"], "status": c.get("status")} for c in children]}
    if result.get("status") == "Finished":
        receipt["verified_success"] = bool(scores and "d3d_S" in scores and
                                              all(c.get("status") == "Finished" for c in children))
        try:
            board = api(session, "/api/leaderboards/19740/")
            precision = "unrounded API scores"
        except RuntimeError:
            board = api(session, "/api/phases/29580/get_leaderboard/?page_size=all")
            precision = "rounded display scores; rank interval accounts for ties"
        rows = board.get("submissions", [])
        own = next((r for r in rows if r["id"] == sid), None)
        ranks = {}
        if own:
            for metric in ("d3d_S", "g_ratio"):
                def value(row):
                    return next((float(x["score"]) for x in row["scores"] if x["column_key"] == metric), 0.)
                v = value(own)
                ranks[metric] = [1 + sum(value(r) > v for r in rows), sum(value(r) >= v for r in rows)]
        receipt["rank_intervals"], receipt["rank_precision"] = ranks, precision
    write_json(run / "official_receipt.json", receipt)
    print(json.dumps({"time": now(), "submission_id": sid, "status": receipt["status"], "scores": scores}), flush=True)
    return receipt


def cycle(root, state_path, min_interval):
    root.mkdir(parents=True, exist_ok=True)
    state = read_json(state_path, {"entries": {}, "created_utc": now()})
    private_root = Path.home() / ".config/fusion"
    session = requests.Session()
    session.headers.update({"Authorization": "Token " + (private_root / "codabench_token").read_text().strip()})
    active = False
    runs = [run for run in root.iterdir() if run.is_dir() and re.fullmatch(r"[0-9]{8}-[0-9]{3}", run.name)]
    privates = {run.name: read_json(private_root / "submissions" / f"17739_{run.name}.json", {}) for run in runs}
    runs.sort(key=lambda run: submission_priority(run, privates[run.name]))
    for run in runs:
        key = run.name
        private = privates[key]
        entry = state["entries"].get(key, {})
        if private.get("submission_id"):
            sid = private["submission_id"]
            if entry.get("verified_success") and entry.get("submission_id") == sid:
                continue
            receipt = monitor(session, run, sid)
            state["entries"][key] = entry | receipt | {"sha256": private["sha256"]}
            if receipt.get("verified_success"):
                for metric in ("d3d_S", "mast_S", "g_ratio"):
                    value = receipt["scores"].get(metric, 0)
                    best = state.setdefault("historical_best", {}).get(metric, {})
                    if value > best.get("score", -1):
                        state["historical_best"][metric] = {"score": value, "submission_id": sid, "checked_utc": now()}
            active |= receipt["status"] not in {"Finished", "Failed", "Cancelled"}
            write_json(state_path, state)
            continue
        if private.get("submission_post_started") or private.get("dataset_post_started") and not private.get("dataset_key"):
            state["entries"][key] = entry | {"needs_reconciliation": True}
            active = True
            write_json(state_path, state)
            continue
        if entry.get("rejected"):
            continue
        elapsed = time.time() - state.get("last_attempt_unix", 0)
        if active or elapsed < min_interval:
            continue
        try:
            evidence = candidate_evidence(run)
        except (ValueError, KeyError):
            state["entries"][key] = entry | {"rejected": True, "reason": "Candidate evidence check failed", "checked_utc": now()}
            write_json(state_path, state)
            continue
        if not evidence:
            continue
        previous = [v for v in state["entries"].values() if v.get("verified_success")]
        if any(k != key and v.get("sha256") == evidence["sha256"] for k, v in state["entries"].items()):
            continue
        if evidence["dual"]:
            if not preserves_best_diii_d(root, state, evidence["diii_d_prediction_sha256"]):
                state["entries"][key] = entry | {"needs_diii_d_update": True,
                    "reason": "Transfer candidate must preserve the best verified DIII-D payload", "checked_utc": now()}
                write_json(state_path, state)
                continue
        elif previous and evidence["native"] < max(v.get("native", 0) for v in previous) + 0.0001:
            continue
        competition = api(session, "/api/competitions/17739/")
        phase = next(p for p in competition["phases"] if p["id"] == 29580)
        if competition.get("participant_status") != "approved" or phase.get("status") != "Current":
            continue
        if any(phase.get(limit) and phase.get(used, 0) >= phase[limit] for used, limit in (
                ("used_submissions_per_day", "max_submissions_per_day"),
                ("used_submissions_per_person", "max_submissions_per_person"))):
            continue
        state["last_attempt_unix"] = time.time()
        state["entries"][key] = entry | evidence | {"attempt_started_utc": now()}
        write_json(state_path, state)
        with (run / "upload.log").open("a") as log:
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/codabench_submit.py"),
                "--run-dir", str(run), "--name", evidence["public_name"],
                "--expected-sha256", evidence["sha256"]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({"time": now(), "candidate": key, "upload_exit_code": completed.returncode}), flush=True)
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=1800)
    parser.add_argument("--minimum-submission-hours", type=float, default=6)
    args = parser.parse_args()
    if args.interval < 30 or args.minimum_submission_hours < 6:
        parser.error("Poll at least 30 seconds apart; submit at least six hours apart")
    root = ROOT / "results/candidates"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                cycle(root, root / "submission_state.json", args.minimum_submission_hours * 3600)
            except (requests.RequestException, RuntimeError, OSError, ValueError, KeyError) as exc:
                print(json.dumps({"time": now(), "event": "cycle_deferred", "error_type": type(exc).__name__}), flush=True)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
