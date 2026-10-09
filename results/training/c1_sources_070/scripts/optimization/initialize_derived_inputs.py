#!/usr/bin/env python3
"""Add input interactions with an exactly unchanged initial prediction function."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.advanced_models import build_advanced_model
from fusion.input_derived import feature_schema
from fusion.pod import read_manifest, manifest_hash, validate_splits, load_arrays
from scripts.codabench_submit import write_json
from scripts.train_pod import reconstruct, torch_normalization


@torch.no_grad()
def analytic_check(module):
    raw = module.input_mean[None].clone()
    p = module.pressure_columns
    raw[:, p[:, 0]] = np.log(3.)  # 2 keV
    raw[:, p[:, 1]] = np.log(4.)  # 3e19 m^-3
    raw[:, p[:, 2]] = 1.
    raw[:, p[:, 3]] = 1.
    raw[:, module.ip] = 1000.  # kA
    if module.ipmask >= 0:
        raw[:, module.ipmask] = 1.
    x = (raw-module.input_mean)/module.input_std
    physical = module.physical(x)
    count = len(p)
    torch.testing.assert_close(physical[:, :2*count], torch.full_like(physical[:, :2*count], np.log(7.)), atol=2e-5, rtol=2e-5)
    raw[:, module.ip] = 2000.
    doubled = module.physical((raw-module.input_mean)/module.input_std)
    torch.testing.assert_close(doubled[:, :count], physical[:, :count], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(doubled[:, count:2*count], torch.full_like(doubled[:, count:2*count], np.log(2.5)), atol=2e-5, rtol=2e-5)
    raw[:, p[:, 2]] = 0.
    absent = module.physical((raw-module.input_mean)/module.input_std)
    assert torch.count_nonzero(absent[:, :2*count]) == 0
    assert torch.isfinite(physical).all() and torch.isfinite(doubled).all() and torch.isfinite(absent).all()
    return {"passed": True, "pressure_product_units": True, "inverse_current_square_scaling": True,
            "missing_measurements_zero_pressure_features": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/candidates/20260919-001/checkpoints/b.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_history_v2"))
    parser.add_argument("--pod-dir", type=Path, default=Path("data/pod_history128_robust_v4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh initialization directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.workers)
    torch.manual_seed(2070)
    validate_splits(args.data_dir)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    if source["model"] != "pod_deeponet" or source["model_kwargs"].get("trunk_rank", 0):
        raise ValueError("Only the established fixed-output branch is supported")
    norm = source["normalization"]
    manifest = read_manifest(args.data_dir)
    basis = json.loads((args.pod_dir / "basis.json").read_text())
    if (source["run"]["feature_names"] != manifest["feature_names"]
            or basis["source_manifest_sha256"] != manifest_hash(args.data_dir)):
        raise ValueError("Training input contract differs from the existing checkpoint")
    with np.load(args.pod_dir / "normalization.npz", allow_pickle=False) as z:
        if set(z.files) != set(norm) or any(not np.array_equal(z[key], norm[key]) for key in norm):
            raise ValueError("Training normalization changed")
    names = manifest["feature_names"]
    used = set(norm["feature_indices"].tolist())
    absent = [i for i, name in enumerate(names) if name.startswith("magnetics_") and "observed" in name and i not in used]
    raw = load_arrays(args.data_dir, "train", ("X",))["X"]
    low, high = np.full(len(absent), np.inf), np.full(len(absent), -np.inf)
    for first in range(0, len(raw), 8192):
        values = raw[first:first+8192, absent]
        low = np.minimum(low, values.min(0)); high = np.maximum(high, values.max(0))
    if np.any(low != high) or not np.isin(low, [0., 1.]).all():
        raise ValueError("Unrepresented observation masks are not verified constants")
    constants = {names[index]: float(value) for index, value in zip(absent, low)}
    del raw
    schema = feature_schema(names, norm, constants)
    kwargs = {**copy.deepcopy(source["model_kwargs"]), "physical_schema": schema}
    old = build_advanced_model(source["model"], **source["model_kwargs"]).eval()
    old.load_state_dict(source["state_dict"])
    new = build_advanced_model("pod_derived_inputs", **kwargs).eval()
    missing, unexpected = new.load_state_dict(source["state_dict"], strict=False)
    if unexpected or any(not name.startswith("derived_") for name in missing):
        raise ValueError("The extension changed a pretrained parameter")
    check = analytic_check(new.derived_inputs)
    train = np.load(args.pod_dir / "train_X.npy", mmap_mode="r", allow_pickle=False)
    count, mean, m2 = 0, np.zeros(new.derived_inputs.width), np.zeros(new.derived_inputs.width)
    with torch.no_grad():
        for first in range(0, len(train), 8192):
            values = new.derived_inputs.physical(torch.from_numpy(np.array(train[first:first+8192]))).numpy().astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite input interaction")
            n = len(values); batch_mean = values.mean(0); batch_m2 = np.square(values-batch_mean).sum(0)
            delta = batch_mean-mean; total = count+n
            mean += delta*n/total; m2 += batch_m2 + delta**2*count*n/total; count = total
        std = np.sqrt(np.maximum(m2/count, 0.))
        std[std < 1e-6] = 1.
        new.derived_inputs.mean.copy_(torch.from_numpy(mean))
        new.derived_inputs.std.copy_(torch.from_numpy(std))
    valid = np.load(args.pod_dir / "val_X.npy", mmap_mode="r", allow_pickle=False)
    chosen = np.unique(np.linspace(0, len(valid)-1, 512).astype(int))
    x = torch.from_numpy(np.array(valid[chosen]))
    with torch.no_grad():
        before, after = old(x), new(x)
        for key in before:
            torch.testing.assert_close(after[key], before[key], atol=0., rtol=0.)
        tnorm = torch_normalization(norm, "cpu")
        original_field = reconstruct(old, before, tnorm)
        new_field = reconstruct(new, after, tnorm)
        torch.testing.assert_close(new_field, original_field, atol=0., rtol=0.)
    new(x[:8])["coefficients"].square().mean().backward()
    for name in ("derived_stem", "derived_coefficients"):
        gradient = getattr(new, name).weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or not torch.any(gradient != 0):
            raise ValueError("New input paths must receive finite nonzero gradients")
    new.zero_grad(set_to_none=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "analytic_check": check, "prediction_equivalence_frames": len(x),
              "field_max_absolute_difference": float((new_field-original_field).abs().max()),
              "input_features_added_internally": new.derived_inputs.width, "training_statistics_frames": count,
              "old_parameters": sum(p.numel() for p in old.parameters()),
              "new_parameters": sum(p.numel() for p in new.parameters()),
              "parent_checkpoint_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "source_manifest_sha256": manifest_hash(args.data_dir), "basis_and_normalization_unchanged": True,
              "targets_read": False, "audit_labels_read": False, "test_labels_read": False,
              "constant_observation_masks_verified_on_training": constants}
    run = {"feature_names": names, "initialization": report}
    torch.save({"model": "pod_derived_inputs", "model_kwargs": kwargs, "state_dict": new.state_dict(),
                "normalization": norm, "run": run, "epoch": 0}, args.output / "best.pt")
    reloaded = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    restored = build_advanced_model(reloaded["model"], **reloaded["model_kwargs"]).eval()
    restored.load_state_dict(reloaded["state_dict"])
    with torch.no_grad():
        for key, value in restored(x).items():
            torch.testing.assert_close(value, before[key], atol=0., rtol=0.)
    report["serialized_prediction_equivalence"] = True
    paths = [Path(__file__), ROOT / "fusion/advanced_models.py", ROOT / "fusion/input_derived.py", ROOT / "fusion/pod.py"]
    hashes = {}
    for path in paths:
        relative = path.resolve().relative_to(ROOT.resolve()); target = args.output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True); content = path.read_bytes(); target.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    report["source_sha256"] = hashes
    write_json(args.output / "initialization.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("constant_observation_masks_verified_on_training", "source_sha256")}), flush=True)


if __name__ == "__main__":
    main()
