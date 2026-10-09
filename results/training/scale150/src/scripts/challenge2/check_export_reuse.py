#!/usr/bin/env python3
"""Exercise numerical-change rejection and actual frozen cache adoption."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.reuse_valid_export import prove_default_branch, digest, read, write
from scripts.challenge2.distribute_frozen_export import load_exporter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parent, run = a.parent.resolve(), a.run.resolve()
    if a.output.exists():
        raise ValueError("Preserve previous checks")
    old, new = parent / "source_snapshot", run / "source_snapshot"
    checks = {"original_contract": bool(prove_default_branch(old, new, {}))}
    cases = [
        ("changed_default_math_rejected", "evaluate_profile_prior.py", "y[8]) * ctx[\"pressure_scale\"]", "y[8]) * 1.01 * ctx[\"pressure_scale\"]"),
        ("changed_sequence_state_rejected", "export_mast_predictions.py", "previous=current", "previous=None"),
        ("changed_validity_rejected", "evaluate_profile_prior.py", 'info[key] > 0', 'info[key] > -1'),
        ("changed_unchanged_return_rejected", "evaluate_profile_prior.py", 'return original\n        failure', 'return (original[0] * 1.01, original[1], original[2])\n        failure'),
    ]
    for label, name, before, after in cases:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "scripts/challenge2"
            folder.mkdir(parents=True)
            for source in ("evaluate_profile_prior.py", "export_mast_predictions.py"):
                shutil.copyfile(new / "scripts/challenge2" / source, folder / source)
            target = folder / name
            text = target.read_text()
            if before not in text:
                raise ValueError("The intended corruption was not applied")
            target.write_text(text.replace(before, after, 1))
            try:
                prove_default_branch(old, root, {})
            except ValueError:
                checks[label] = True
            else:
                raise ValueError(f"A numerical corruption was accepted: {label}")
    try:
        prove_default_branch(old, new, {"pressure_factor_model": None})
    except ValueError:
        checks["active_optional_estimator_rejected"] = True
    else:
        raise ValueError("A changed prior contract was accepted")
    subprocess.run([sys.executable, str(run / "reuse/reuse_valid_export.py"), "--run", str(run), "--indices", "0", "1"], check=True)
    exporter = load_exporter(run)
    frames = 0
    for index in (0, 1):
        before = parent / "shots" / f"{index:04d}.npz"
        after = run / "shots" / before.name
        receipt = read(after.with_suffix(".json"))
        summary = exporter.predict_shot(index)
        if digest(before) != digest(after) or summary["observed_pressure_retry_frames"] != 0:
            raise ValueError("Cached export altered a valid prediction")
        if receipt["reuse"]["prediction_recomputed"] is not False:
            raise ValueError("Reuse provenance is inaccurate")
        frames += summary["frames"]
    checks["copied_predictions_byte_identical"] = True
    checks["frozen_exporter_resumes_copied_receipts"] = True
    report = {"passed": all(checks.values()), "checks": checks, "native_frames": frames,
              "proof_sha256": digest(run / "reuse_proof.json"), "test_labels_read": False, "audit_labels_read": False}
    write(a.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
