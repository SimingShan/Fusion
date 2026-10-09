#!/usr/bin/env python3
"""Refit the DIII-D coil/gauge calibration with a training-only affine model.

The dimensionless closure prior is unchanged. The replacement is used only on
DIII-D. No extrapolation of this calibration to MAST is asserted or performed.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.mast_profiles import MU0
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=1.)
    a = p.parse_args()
    if (a.output / "prior.joblib").exists():
        raise ValueError("Use a fresh output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.data / "targets.npz", allow_pickle=False) as z:
        d = dict(z)
    prior = joblib.load(a.prior)
    source_hash = hashlib.sha256((a.data / "targets.npz").read_bytes()).hexdigest()
    if source_hash != prior["training_source_sha256"]:
        raise ValueError("Closure and calibration training sources differ")
    train = np.zeros(len(d["fold"]), bool)
    train[prior["training_rows"]] = True
    if np.any(d["fold"][train] != "train"):
        raise ValueError("Training mask includes held-out shots")
    outside = ~d["envelope"]
    design = np.column_stack((d["cs_unit"][outside], np.ones(outside.sum())))
    # Compute auxiliary evaluation targets separately from fitting rows.
    def targets(mask):
        residual = d["phi"][mask] - d["oracle_plasma"][mask] - d["pf_flux"][mask]
        fitted = np.linalg.lstsq(design, residual[:, outside].T, rcond=None)[0].T
        return fitted / np.column_stack((d["Ip_A"][mask], MU0*d["Ip_A"][mask]*d["L"][mask]))
    y = targets(train)
    _, inverse, count = np.unique(d["shot_index"][train], return_inverse=True, return_counts=True)
    weights = 1/count[inverse]
    model = make_pipeline(StandardScaler(), Ridge(alpha=a.alpha))
    model.fit(d["X"][train], y, ridge__sample_weight=weights)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "training_frames": int(train.sum()),
              "training_shots": len(count), "scope": "DIII-D coil/gauge only",
              "audit_labels_read": False, "mast_labels_read": False, "closure_prior_unchanged": True,
              "alpha": a.alpha, "validation": {}}
    for fold in ("val", "legacy_val"):
        mask = (d["fold"] == fold) & d["fit_valid"]
        truth = targets(mask)
        predicted = model.predict(d["X"][mask])
        old = prior["coil_model"].predict(d["X"][mask])
        scale = np.column_stack((d["Ip_A"][mask], MU0*d["Ip_A"][mask]*d["L"][mask]))
        report["validation"][fold] = {
            "frames": int(mask.sum()),
            "old_calibration_RMSE_physical": np.sqrt(np.mean(((old-truth)*scale)**2, axis=0)).tolist(),
            "new_calibration_RMSE_physical": np.sqrt(np.mean(((predicted-truth)*scale)**2, axis=0)).tolist(),
            "role": "Held-out auxiliary diagnostic, not a reconstruction score"}
    prior["coil_model"] = model
    prior["calibration_refit"] = report
    prior["calibration_parent_sha256"] = hashlib.sha256(a.prior.read_bytes()).hexdigest()
    joblib.dump(prior, a.output / "prior.joblib")
    write_json(a.output / "training.json", report)
    (a.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

