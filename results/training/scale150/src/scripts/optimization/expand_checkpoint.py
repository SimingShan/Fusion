#!/usr/bin/env python3
"""Prepare a larger initialized predictor and verify field/scalar preservation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.advanced_models import build_advanced_model
from fusion.network_expansion import expand_model, expand_linear
from scripts.train_pod import reconstruct, torch_normalization
from scripts.codabench_submit import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pod-dir", type=Path, default=Path("data/pod_history128_robust_v4"))
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=2091)
    p.add_argument("--noise-fraction", type=float, default=.01)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve initialized checkpoint evidence")
    a.output.mkdir(parents=True)
    torch.set_num_threads(a.workers)
    torch.manual_seed(a.seed)
    checkpoint_hash = digest(a.checkpoint)
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if saved["model"] != "pod_deeponet":
        raise ValueError("Unsupported fitted predictor")
    old = build_advanced_model(saved["model"], **saved["model_kwargs"]).eval()
    old.load_state_dict(saved["state_dict"])
    new = expand_model(old, a.width, a.depth, a.seed, a.noise_fraction).eval()
    norm = saved["normalization"]
    with np.load(a.pod_dir / "normalization.npz", allow_pickle=False) as values:
        if norm.keys() != set(values.files) or any(not np.array_equal(v, values[k]) for k, v in norm.items()):
            raise ValueError("Prepared inputs do not use the fitted normalization")
    tnorm = torch_normalization(norm, "cpu")
    samples = []
    input_hashes = {}
    for split in ("train", "val"):
        path = a.pod_dir / f"{split}_X.npy"
        X = np.load(path, mmap_mode="r", allow_pickle=False)
        selected = np.unique(np.linspace(0, len(X)-1, 512).astype(int))
        values = np.array(X[selected])
        samples.append(torch.from_numpy(values))
        input_hashes[split] = {"rows": len(values), "indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                              "selected_input_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
    samples.append(torch.randn(512, old.input_dim).clamp(-4, 4))
    worst = {"coefficients": 0., "normalized_scalars": 0., "field": 0., "scalars": 0.}
    with torch.no_grad():
        for X in torch.cat(samples).split(128):
            before, after = old(X), new(X)
            field_before, field_after = reconstruct(old, before, tnorm), reconstruct(new, after, tnorm)
            scalar_before = before["scalars"]*tnorm["scalar_std"]+tnorm["scalar_mean"]
            scalar_after = after["scalars"]*tnorm["scalar_std"]+tnorm["scalar_mean"]
            for key, x, y in (("coefficients", before["coefficients"], after["coefficients"]),
                              ("normalized_scalars", before["scalars"], after["scalars"]),
                              ("field", field_before, field_after), ("scalars", scalar_before, scalar_after)):
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError("Nonfinite initialized prediction")
                # Tiny modes have their own scale and can amplify harmless
                # float32 reduction differences. Preserve the actual exported
                # field/scalar contract; record all internal errors separately.
                if key in ("field", "scalars"):
                    torch.testing.assert_close(x, y, atol=2e-6 if key == "field" else 1e-5, rtol=1e-5)
                worst[key] = max(worst[key], float((x-y).abs().max()))
    # Independently check the replication identity using float64 matrix algebra.
    gen = torch.Generator().manual_seed(901)
    source, target = nn.Linear(7, 5).double(), nn.Linear(14, 15).double()
    expand_linear(source, target, gen, .01)
    X = torch.randn(19, 7, dtype=torch.float64, generator=gen)
    torch.testing.assert_close(target(X.repeat(1, 2)), source(X).repeat(1, 3), atol=1e-14, rtol=1e-14)
    with torch.no_grad():
        for block in new.blocks[len(old.blocks):]:
            X = torch.randn(17, a.width)
            torch.testing.assert_close(block(X), X, atol=0, rtol=0)
    # An additional block must receive a finite nonzero training signal.
    probe = copy.deepcopy(new).train()
    loss = (probe(samples[0][:32])["coefficients"]-0.1).square().mean()
    loss.backward()
    extra = probe.blocks[-1].layers[3].weight.grad
    if extra is None or not torch.isfinite(extra).all() or extra.abs().max() <= 0:
        raise ValueError("Added capacity cannot receive a training signal")
    result = copy.deepcopy(saved)
    result.update(state_dict=new.state_dict(), model_kwargs={**saved["model_kwargs"], "width": a.width, "depth": a.depth},
                  epoch=0, initialization_parent_sha256=checkpoint_hash)
    result.pop("validation_score", None)
    result["run"] = {**saved["run"], "model_kwargs": result["model_kwargs"],
                     "parameters": sum(p.numel() for p in new.parameters()),
                     "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                     "initialization_parent_arguments": saved["run"].get("arguments"),
                     "status": "expanded_initialization_only", "eligible_for_submission": False,
                     "initialization_parent": str(a.checkpoint), "initialization_parent_sha256": checkpoint_hash,
                     "capacity_expansion": {"width": a.width, "depth": a.depth, "seed": a.seed, "noise_fraction": a.noise_fraction}}
    path = a.output / "initial.pt"
    torch.save(result, path)
    if digest(a.checkpoint) != checkpoint_hash:
        raise ValueError("Fitted parent changed while preparing initialization")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "original_parameters": sum(p.numel() for p in old.parameters()),
              "expanded_parameters": sum(p.numel() for p in new.parameters()), "checked_inputs": sum(len(x) for x in samples),
              "maximum_absolute_errors": worst, "float64_replication_identity_passed": True,
              "physical_prediction_tolerances": {"field_atol": 2e-6, "scalar_atol": 1e-5, "rtol": 1e-5},
              "normalized_output_errors_recorded_separately": True,
              "added_blocks_exact_identity": True, "added_capacity_receives_gradients": True,
              "parent_sha256": checkpoint_hash, "initialized_checkpoint_sha256": digest(path),
              "input_checks": input_hashes,
              "source_sha256": {name: digest(ROOT / name) for name in (
                  "fusion/network_expansion.py", "fusion/advanced_models.py", "scripts/train_pod.py",
                  str(Path(__file__).relative_to(ROOT)))},
              "target_labels_read": False, "audit_labels_read": False, "eligible_for_submission": False}
    write_json(a.output / "initialization_check.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "input_checks")}), flush=True)


if __name__ == "__main__":
    main()
