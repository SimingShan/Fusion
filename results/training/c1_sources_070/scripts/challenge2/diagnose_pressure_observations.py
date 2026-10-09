#!/usr/bin/env python3
"""Input-only diagnosis of pressure extrapolation on a failed native frame."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.mast_profiles import read_row, FreeBoundaryGS
from fusion.measured_pressure import measurements
from scripts.challenge2.evaluate_profile_prior import infer
from scripts.challenge2.prepare_profile_prior import clean
from scripts.codabench_submit import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--reference-failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    args.output.mkdir(parents=True)
    plan = json.loads((args.base_run / "export_plan.json").read_text())
    row = read_row(Path(plan["raw_root"])/plan["shots"][args.index]["path"], labels=False)
    prior = joblib.load(args.base_run / "checkpoints/prior.joblib")
    for key in ("model", "coil_model"):
        prior[key].set_params(n_jobs=1)
    solver = FreeBoundaryGS(row["efit_grid_R"], row["efit_grid_Z"])
    with np.load(ROOT / "starter/fusion_scoring/masks/mast_envelope.npz", allow_pickle=False) as z:
        envelope = z["mask_coarse"].astype(bool)
    observations = measurements(row, float(row["efit_times"][0]))
    records = []
    report = {"shot_index": args.index, "frame": 0, "time_ms": float(row["efit_times"][0]),
              "pressure_channels": len(observations["pressure"]),
              "observed_pressure_channels": int(observations["observed"].sum()),
              "diagnostic_only": True, "mast_labels_read": False, "audit_labels_read": False,
              "base_plan_sha256": hashlib.sha256((args.base_run/"export_plan.json").read_bytes()).hexdigest(),
              "records": records}
    for name, measured, observed_only in (("control", True, False), ("observed_only", True, True), ("prior_only", False, False)):
        field, current, info = infer(row, 0, prior, solver, envelope, None, 300, 600,
                                     measured, 2., observed_only)
        if name == "control":
            with np.load(args.reference_failure / "original.npz", allow_pickle=False) as reference:
                np.testing.assert_array_equal(field, reference["psi"])
                np.testing.assert_array_equal(current, reference["current"])
            report["unchanged_default_prediction_verified"] = True
        np.savez_compressed(args.output / f"{name}.npz", psi=field, current=current)
        records.append(clean({"name": name, **info}))
        write_json(args.output / "diagnostic.json", report)
        print(json.dumps({"name": name, **{key: clean(info.get(key)) for key in
            ("converged", "scalar_valid", "scalar_failure", "status", "nonlinear_profile_relative_residual",
             "pressure_axis_Pa", "pressure_channels", "q95", "betaN")}}), flush=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
