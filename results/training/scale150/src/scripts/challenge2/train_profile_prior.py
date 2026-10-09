#!/usr/bin/env python3
"""Fit a training-only transferable closure prior and diagnose held-out errors."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.mast_profiles import MU0
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.data / "targets.npz", allow_pickle=False) as f:
        d = dict(f)
    provenance = json.loads((a.data / "preparation.json").read_text())
    train = (d["fold"] == "train") & d["fit_valid"] & (d["paxis_star"] > 1e-6)
    quality = np.array([r.get("source_relative_RMSE", 100) < .1 and
                       .85 < r.get("input_Ip_ratio", 0) < 1.15 for r in provenance["records"]])
    train &= quality
    # Only the training mask is allowed into fitting, normalization, and gauge
    # calibration. Development labels are opened only for the final diagnostics.
    field_scale = MU0 * d["Ip_A"] * d["L"]
    outside = ~d["envelope"]
    cs_unit = d["cs_unit"]
    design = np.column_stack((cs_unit[outside], np.ones(outside.sum())))
    residual = d["phi"][train] - d["oracle_plasma"][train] - d["pf_flux"][train]
    calibration = np.linalg.lstsq(design, residual[:, outside].T, rcond=None)[0].T
    calibration[:, 0] /= d["Ip_A"][train]
    calibration[:, 1] /= field_scale[train]
    Y = np.column_stack((d["A_shape"], d["C_shape"], np.log(np.maximum(d["paxis_star"], 1e-8)), d["geometry"]))
    yy = Y[train]
    center = yy.mean(axis=0)
    scale = np.maximum(yy.std(axis=0), 1e-4)
    model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=3, max_features=.8,
                               random_state=2040, n_jobs=4)
    _, inverse, count = np.unique(d["shot_index"][train], return_inverse=True, return_counts=True)
    weights = 1 / count[inverse]
    model.fit(d["X"][train], (yy-center)/scale, sample_weight=weights)
    coil_model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=3, max_features=.8,
                                    random_state=2041, n_jobs=4)
    coil_model.fit(d["X"][train], calibration, sample_weight=weights)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "training_frames": int(train.sum()),
        "training_shots": len(count), "audit_labels_read": False, "mast_labels_read": False,
        "input_target_poison_parity": provenance["input_target_poison_parity"],
        "gauge_scope": "DIII-D calibration only; no assumption of transferable MAST gauge", "validation": {}}
    for fold in ("val", "legacy_val"):
        mask = (d["fold"] == fold) & d["fit_valid"] & (d["paxis_star"] > 1e-6)
        predicted = model.predict(d["X"][mask]) * scale + center
        report["validation"][fold] = {"frames": int(mask.sum()),
            "profile_geometry_r2_each": r2_score(Y[mask], predicted, multioutput="raw_values").tolist(),
            "pressure_log_rmse": float(np.sqrt(np.mean((Y[mask, 8]-predicted[:, 8])**2))),
            "purpose": "Auxiliary target diagnostic, not an official reconstruction score"}
    joblib.dump({"model": model, "coil_model": coil_model, "center": center, "scale": scale,
        "feature_names": d["feature_names"].tolist(), "training_rows": np.flatnonzero(train),
        "closure_basis": provenance.get("closure_basis", "original"),
        "training_source_sha256": hashlib.sha256((a.data / "targets.npz").read_bytes()).hexdigest(),
        "training_shots": np.unique(d["shot_index"][train]).tolist(), "report": report}, a.output / "prior.joblib")
    write_json(a.output / "training.json", report)
    print(json.dumps({"training_frames": report["training_frames"], "training_shots": report["training_shots"],
                      "development_pressure_log_rmse": report["validation"]["val"]["pressure_log_rmse"]}), flush=True)


if __name__ == "__main__":
    main()
