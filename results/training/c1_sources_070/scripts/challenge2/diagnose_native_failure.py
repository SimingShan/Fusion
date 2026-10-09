#!/usr/bin/env python3
"""Replay a frozen complete-sequence failure and probe declared input-only alternatives."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve previous diagnostics")
    a.output.mkdir(parents=True)
    run = a.run.resolve()
    source = run / "source_snapshot/scripts/challenge2/export_mast_predictions.py"
    spec = importlib.util.spec_from_file_location("_frozen_failure_replay", source)
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)
    exporter.initialize(run)
    work, plan = exporter.WORK, exporter.WORK["plan"]
    item = plan["shots"][a.index]
    path = Path(plan["raw_root"]) / item["path"]
    if path.stat().st_size != item["size"]:
        raise ValueError("Pinned input size changed")
    row = work["read_row"](path, labels=False)
    if not 0 <= a.frame < len(row["efit_times"]):
        raise ValueError("Invalid native diagnostic frame")
    solver = work["solver_type"](row["efit_grid_R"], row["efit_grid_Z"])
    evaluator = sys.modules["scripts.challenge2.evaluate_profile_prior"]
    previous = None
    records = []
    options = dict(max_seconds=plan["max_seconds"], max_evaluations=plan["max_evaluations"],
                   measured_pressure=True, pressure_factor=plan["pressure_factor"],
                   solver_method=plan["solver_method"], constrain_pressure_current=plan["constrain_pressure_current"])
    for frame in range(a.frame+1):
        field, current, info = work["infer"](row, frame, work["prior"], solver, work["envelope"], previous, **options)
        valid = evaluator.valid_prediction((field, current, info))
        records.append(exporter.clean({"frame": frame, "valid": valid, **{key: info.get(key) for key in
            ("status", "converged", "scalar_valid", "scalar_failure", "nonlinear_profile_relative_residual",
             "pressure_observation_retry_used", "original_prediction_failure")}}))
        exporter.write_json(a.output / "progress.json", {"frames": len(records), "records": records})
        if not valid:
            if frame != a.frame:
                raise ValueError("A different native failure occurred during replay")
            np.savez_compressed(a.output / "failure_state.npz", field=field, current=current,
                                previous=previous if previous is not None else np.empty(0))
            exporter.write_json(a.output / "original_failure.json", exporter.clean(info))
            print(json.dumps(records[-1]), flush=True)
            break
        if frame == a.frame:
            raise ValueError("The known failure did not reproduce")
        if info["converged"]:
            previous = current
        elif plan["allow_approximate_seed"] and info.get("nonlinear_profile_relative_residual") is not None:
            if info["nonlinear_profile_relative_residual"] < .02 and np.min(current) >= -1e-6:
                previous = current
    # These alternatives are selected before looking at any target or score.
    # None alters a flux pixel or supplies a replacement scalar value.
    probes = [
        ("fresh", None, {}),
        ("fresh_observed", None, {"observed_pressure_only": True}),
        ("bounded", previous, {"constrain_pressure_current": True}),
        ("fresh_bounded", None, {"constrain_pressure_current": True}),
        ("prior", previous, {"measured_pressure": False}),
    ]
    results = []
    for name, initial, override in probes:
        try:
            field, current, info = evaluator.infer(row, a.frame, work["prior"], solver, work["envelope"],
                                                  initial, **(options | override))
            valid = evaluator.valid_prediction((field, current, info))
            np.savez_compressed(a.output / f"{name}.npz", field=field, current=current)
            exporter.write_json(a.output / f"{name}.json", exporter.clean(info))
            result = {"probe": name, "valid": valid, **{key: info.get(key) for key in
                ("status", "converged", "scalar_failure", "nonlinear_profile_relative_residual", "q95", "betaN")}}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            result = {"probe": name, "valid": False, "failure": f"{type(exc).__name__}: {exc}"}
        results.append(exporter.clean(result))
        print(json.dumps(results[-1]), flush=True)
    exporter.write_json(a.output / "diagnostic.json", {"run": str(run), "index": a.index,
        "raw_path": item["path"], "frame": a.frame, "native_frames": len(row["efit_times"]),
        "export_plan_sha256": exporter.digest(run / "export_plan.json"),
        "replayed_frames": len(records), "failure_reproduced": True, "probes": results,
        "test_labels_read": False, "audit_labels_read": False, "eligible_for_submission": False,
        "diagnostic_source_sha256": exporter.digest(__file__)})
    (a.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
