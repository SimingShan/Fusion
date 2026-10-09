#!/usr/bin/env python3
"""Reproduce an invalid frozen native prediction using input-only diagnostics."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    args.output.mkdir(parents=True)
    run = args.run.resolve()
    source = run / "source_snapshot/scripts/challenge2/export_mast_predictions.py"
    specification = importlib.util.spec_from_file_location("_diagnostic_frozen_export", source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    module.initialize(run)
    state, plan = module.WORK, module.WORK["plan"]
    entry = plan["shots"][args.index]
    row = state["read_row"](Path(plan["raw_root"]) / entry["path"], labels=False)
    solver = state["solver_type"](row["efit_grid_R"], row["efit_grid_Z"])
    if args.frame != 0:
        raise ValueError("This first diagnostic isolates failure at sequence initialization")
    records = []

    def evaluate(frame, previous, label):
        field, current, info = state["infer"](row, frame, state["prior"], solver, state["envelope"], previous,
            plan["max_seconds"], plan["max_evaluations"], True, plan["pressure_factor"])
        np.savez_compressed(args.output / f"{label}.npz", psi=field, current=current)
        record = module.clean({"label": label, "frame": frame, "time_ms": float(row["efit_times"][frame]), **info})
        records.append(record)
        module.write_json(args.output / "diagnostic.json", {"shot_index": args.index,
            "export_plan_sha256": module.digest(run / "export_plan.json"), "records": records,
            "test_labels_read": False, "audit_labels_read": False, "diagnostic_only": True})
        print(json.dumps({key: record.get(key) for key in ("label", "frame", "converged", "scalar_valid",
             "scalar_failure", "status", "nonlinear_profile_relative_residual", "current_negative_mass_fraction", "q95", "betaN")}), flush=True)
        return current, info

    current, info = evaluate(0, None, "original")
    if info.get("scalar_valid"):
        raise ValueError("The recorded invalid prediction did not reproduce")
    # A bounded input-only probe. These future inputs are already available in
    # the whole-shot inference task; none are target geometry or target fields.
    for offset in (1, 2, 4, 8):
        if offset >= len(row["efit_times"]):
            continue
        try:
            anchor, anchor_info = evaluate(offset, None, f"anchor_{offset}")
            if anchor_info.get("scalar_valid"):
                _, retried = evaluate(0, anchor, f"retry_{offset}")
                if retried.get("scalar_valid"):
                    break
        except ValueError as exc:
            print(json.dumps({"anchor": offset, "failure": str(exc)}), flush=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
