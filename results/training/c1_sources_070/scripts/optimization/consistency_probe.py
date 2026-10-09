#!/usr/bin/env python3
"""Check whether training-only residual prediction improves derived quantities.

This is an auxiliary development diagnostic. These seven quantities are never
submitted as independent outputs. Any useful correction still needs to produce
a flux field and pass the unchanged full composite scorer.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import load_arrays, read_manifest, split_entries, validate_splits
from scripts.optimization.train_pair_policy import PairPolicy
from scripts.train_pod import reconstruct, torch_normalization
from scripts.codabench_submit import write_json
from starter import local_score as official
from derive import _shape_from_contour, _clamp_to_nan, derive_frame

GRID = None


def initialize_geometry():
    global GRID
    with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz", allow_pickle=False) as z:
        GRID = (z["grid_R"], z["grid_Z"], z["mask_coarse"].astype(bool))


def geometry(field):
    R, Z, mask = GRID
    field = np.asarray(field, dtype=np.float64)
    contour, sign = official.extract_lcfs_with_sign(field, R, Z, official.MACHINE, mask,
                        mask.astype(float), n_iter=official.N_ITER, axis_sign=-1.)
    values = derive_frame(field, R, Z, official.MACHINE, mask, mask.astype(float),
                         contour=contour, with_li=True, axis_sign=sign)
    return [values[key] for key in official.CONS_SCALARS]


def geometry_targets(directory, rows):
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("R", "Z", "contours", "li", "axis_indices", "axis_flux")}
    result = np.empty((len(rows), 7), dtype=np.float64)
    for index, row in enumerate(rows):
        center = int(arrays["axis_indices"][row, 0]); iz, ir = divmod(center, len(arrays["R"]))
        flux = arrays["axis_flux"][row]
        axis = []
        for left, right, coordinate, location in ((1, 2, arrays["R"], ir), (3, 4, arrays["Z"], iz)):
            denominator = flux[left] - 2.*flux[0] + flux[right]
            shift = 0. if denominator == 0 else np.clip(.5*(flux[left]-flux[right])/denominator, -1., 1.)
            axis.append(float(coordinate[location] + shift*(coordinate[1]-coordinate[0])))
        values = _clamp_to_nan({"R_axis": axis[0], "Z_axis": axis[1],
            **_shape_from_contour(arrays["contours"][row]), "li": float(arrays["li"][row])})
        result[index] = [values[key] for key in official.CONS_SCALARS]
    return result


@torch.no_grad()
def features(model, raw, pool, output):
    feature_parts, derived = [], []
    norm = torch_normalization(model.bnorm, "cpu")
    for first in range(0, len(raw), 1024):
        x = torch.as_tensor(model.inputs(raw[first:first+1024]))
        xa, xb = x[:, :model.na], x[:, model.na:-1]
        ha, hb = model.primary.encode(xa), model.secondary.encode(xb)
        pa, pb = model.primary._outputs(xa, ha), model.secondary._outputs(xb, hb)
        weight = model.weight*(1-x[:, -1:])
        paired = {key: weight*pa[key]+(1-weight)*pb[key] for key in pa}
        if first == 0:
            for key, value in model(x).items():
                torch.testing.assert_close(paired[key], value, atol=1e-6, rtol=1e-6)
        fields = reconstruct(model, paired, norm).numpy()
        feature_parts.append(torch.cat((xb, ha, hb, paired["coefficients"], paired["scalars"]), dim=1).numpy())
        derived.extend(pool.map(geometry, fields, chunksize=32))
        if first % 8192 == 0:
            print(json.dumps({"output": str(output), "prepared_frames": min(first+1024, len(raw)), "total": len(raw)}), flush=True)
    result = {"X": np.concatenate(feature_parts), "baseline": np.asarray(derived)}
    np.savez(output, **result)
    return result


def diagnostic(truth, prediction):
    valid = np.isfinite(truth)
    means = np.nanmean(truth, axis=0)
    values = np.where(np.isfinite(prediction), prediction, means)
    sse = np.where(valid, (values-truth)**2, 0.).sum(0)
    total = np.where(valid, (truth-means)**2, 0.).sum(0)
    r2 = 1.-sse/total
    return {"mean_clipped_r2": float(np.maximum(r2, 0).mean()),
            "r2_each": dict(zip(official.CONS_SCALARS, r2.tolist())),
            "prediction_failures": (~np.isfinite(prediction)&valid).sum(0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    output.mkdir(parents=True)
    torch.set_num_threads(args.workers)
    data = ROOT / "data/advanced_history_v2"
    targets = ROOT / "data/geometry_history_v4"
    validate_splits(data)
    manifest = read_manifest(data)
    metadata = json.loads((targets / "metadata.json").read_text())
    selection_hash = hashlib.sha256(json.dumps(split_entries(data, "train"), sort_keys=True).encode()).hexdigest()
    if (metadata["source_training_entries_sha256"] != selection_hash
            or metadata["source_split"] != "train" or not metadata["geometry"]["complete"]):
        raise ValueError("Auxiliary references are not aligned training targets")
    paths = [ROOT / "results/candidates/20260919-001/checkpoints" / name for name in ("a.pt", "b.pt")]
    saved = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    model = PairPolicy(*saved, manifest["feature_names"]).eval()
    rows = np.flatnonzero(np.load(targets / "valid.npy", allow_pickle=False))
    indices = np.load(targets / "global_frame_indices.npy", allow_pickle=False)[rows]
    truth_train = geometry_targets(targets, rows)
    # Reconstructing references from stored contours/stencils must reproduce the
    # original full-field functional before they can supervise anything.
    initialize_geometry()
    training_fields = load_arrays(data, "train", ("psi",))["psi"]
    selected = np.linspace(0, len(rows)-1, 16).astype(int)
    check = np.asarray([geometry(training_fields[indices[i]]) for i in selected])
    np.testing.assert_allclose(truth_train[selected], check, atol=1e-8, rtol=1e-8, equal_nan=True)
    write_json(output / "target_check.json", {"passed": True, "frames": 16, "audit_labels_read": False})
    del training_fields
    sources = [Path(__file__), ROOT / "scripts/optimization/train_pair_policy.py",
               ROOT / "fusion/pod.py", ROOT / "fusion/advanced_models.py", ROOT / "scripts/train_pod.py",
               ROOT / "scripts/optimization/quality_gate.py", ROOT / "starter/local_score.py"]
    sources += sorted((ROOT / "starter/fusion_scoring").glob("*.py"))
    hashes = {}
    for path in sources:
        relative = path.resolve().relative_to(ROOT.resolve())
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        contents = path.read_bytes(); destination.write_bytes(contents)
        hashes[str(relative)] = hashlib.sha256(contents).hexdigest()
    run = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
           "train_frames": len(rows), "validation_fold": "fresh capped development",
           "declared_alphas": [1., 100., 10000.], "audit_labels_read": False, "test_labels_read": False,
           "auxiliary_diagnostic_only": True, "source_sha256": hashes,
           "checkpoints_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
           "source_training_entries_sha256": selection_hash}
    write_json(output / "run.json", run)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_geometry) as pool:
        raw = load_arrays(data, "train", ("X",))["X"]
        train = features(model, raw[indices], pool, output / "train_features.npz")
        del raw
        validation = load_arrays(data, "val", ("X", "psi"))
        valid = features(model, validation["X"], pool, output / "val_features.npz")
        truth_val = np.asarray(list(pool.map(geometry, validation["psi"], chunksize=32)))
    np.savez(output / "targets.npz", train=truth_train, val=truth_val,
             train_global_indices=indices, train_reference_rows=rows)
    baseline_report = diagnostic(truth_val, valid["baseline"])
    exact = json.loads((ROOT / "results/optimization/quality_gate_history128_li_tuned_v9/gated_primary50/val_score.json").read_text())
    for key in official.CONS_SCALARS:
        if abs(baseline_report["r2_each"][key]-exact["r2_cons_each"][key]) > 5e-5:
            raise ValueError("Diagnostic baseline differs from the official development reference")
    # Only reference-valid training rows enter fitting. Prediction failures on
    # development remain in the diagnostic and receive the scorer's penalty.
    fitting = np.isfinite(truth_train).all(1) & np.isfinite(train["baseline"]).all(1)
    means = truth_train[fitting].mean(0)
    scale = np.maximum(truth_train[fitting].std(0), 1e-8)
    baseline_train = np.where(np.isfinite(train["baseline"]), train["baseline"], means)
    baseline_val = np.where(np.isfinite(valid["baseline"]), valid["baseline"], means)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.column_stack((train["X"], baseline_train))[fitting])
    V = scaler.transform(np.column_stack((valid["X"], baseline_val)))
    target = (truth_train[fitting]-baseline_train[fitting])/scale
    trials = []
    for alpha in run["declared_alphas"]:
        regressor = Ridge(alpha=alpha).fit(X, target)
        predicted = baseline_val + regressor.predict(V)*scale
        report = diagnostic(truth_val, predicted)
        trials.append({"alpha": alpha, **report})
        joblib.dump({"regressor": regressor, "scaler": scaler, "reference_mean": means,
                     "reference_scale": scale, "source_run": run}, output / f"probe_{int(alpha)}.joblib")
        np.save(output / f"val_quantities_{int(alpha)}.npy", predicted)
    report = {"baseline": baseline_report, "trials": trials, "train_frames": int(fitting.sum()),
              "val_frames": len(truth_val), "audit_labels_read": False,
              "auxiliary_diagnostic_only": True, "eligible_for_submission": False,
              "limitation": "Improved auxiliary quantities alone do not change submitted flux or establish a composite-score gain"}
    write_json(output / "comparison.json", report)
    run.update(status="completed")
    write_json(output / "run.json", run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
