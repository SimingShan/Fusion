#!/usr/bin/env python3
"""Test nonlinear predictability of retained-field consistency residuals.

This uses already prepared, shot-separated training/development features. An
auxiliary gain only motivates a later flux-field experiment; these quantities
are not independently submittable outputs and no composite gain is claimed.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.optimization.consistency_probe import diagnostic
from scripts.codabench_submit import write_json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("results/evaluations/c1_probe_019"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    a.output.mkdir(parents=True)
    torch.set_num_threads(a.workers)
    torch.manual_seed(2085)
    np.random.seed(2085)
    previous = json.loads((a.source / "run.json").read_text())
    if previous["status"] != "completed" or previous["audit_labels_read"] or previous["test_labels_read"]:
        raise ValueError("Require completed, permitted cached features")
    target_check = json.loads((a.source / "target_check.json").read_text())
    if not target_check["passed"]:
        raise ValueError("Cached geometry references did not pass their independent check")
    for path, expected in previous["checkpoints_sha256"].items():
        if digest(path) != expected:
            raise ValueError("The retained field predictor changed")
    hashes = {name: digest(a.source / name) for name in
              ("train_features.npz", "val_features.npz", "targets.npz", "target_check.json", "run.json")}
    with np.load(a.source / "targets.npz", allow_pickle=False) as values:
        truth_train, truth_val = values["train"], values["val"]
    with np.load(a.source / "train_features.npz", allow_pickle=False) as values:
        raw, baseline_train = values["X"], values["baseline"]
    fitting = np.isfinite(truth_train).all(1) & np.isfinite(baseline_train).all(1)
    means = truth_train[fitting].mean(0)
    scale = np.maximum(truth_train[fitting].std(0), 1e-8)
    scaler = StandardScaler()
    combined = np.column_stack((raw, baseline_train.astype(np.float32)))[fitting]
    X = torch.from_numpy(scaler.fit_transform(combined).astype(np.float32))
    del combined, raw
    Y = torch.from_numpy(((truth_train[fitting]-baseline_train[fitting])/scale).astype(np.float32))
    with np.load(a.source / "val_features.npz", allow_pickle=False) as values:
        raw, baseline_val = values["X"], values["baseline"]
    valid_baseline = np.where(np.isfinite(baseline_val), baseline_val, means)
    V = torch.from_numpy(scaler.transform(np.column_stack((raw, valid_baseline.astype(np.float32)))).astype(np.float32))
    del raw
    if not all(torch.isfinite(x).all() for x in (X, Y, V)):
        raise ValueError("Nonfinite diagnostic inputs")
    model = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 7))
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40, eta_min=1e-5)
    baseline = diagnostic(truth_val, baseline_val)
    best, best_epoch = baseline, 0
    best_state = copy.deepcopy(model.state_dict())
    best_prediction = baseline_val.copy()
    history = []
    started = time.monotonic()
    run = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
           "training_frames": len(X), "development_frames": len(V), "source_feature_sha256": hashes,
           "source_checkpoints_sha256": previous["checkpoints_sha256"], "epochs": 40, "seconds": 900,
           "widths": [128, 128], "learning_rate": 5e-4, "weight_decay": .01, "seed": 2085,
           "minimum_auxiliary_gain_to_investigate": .0005, "audit_labels_read": False, "test_labels_read": False,
           "auxiliary_diagnostic_only": True, "eligible_for_submission": False}
    write_json(a.output / "run.json", run)
    (a.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for epoch in range(1, 41):
        model.train()
        order = torch.randperm(len(X))
        total = 0.
        for indices in order.split(512):
            opt.zero_grad(set_to_none=True)
            loss = (model(X[indices])-Y[indices]).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach())*len(indices)
        schedule.step()
        model.eval()
        with torch.no_grad():
            prediction = valid_baseline + model(V).numpy()*scale
        score = diagnostic(truth_val, prediction)
        if score["mean_clipped_r2"] > best["mean_clipped_r2"]:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_prediction = prediction.copy()
        row = {"epoch": epoch, "training_loss": total/len(X), "mean_clipped_r2": score["mean_clipped_r2"],
               "elapsed_seconds": time.monotonic()-started}
        history.append(row)
        write_json(a.output / "history.json", history)
        print(json.dumps(row), flush=True)
        if time.monotonic()-started >= 900:
            break
    gain = best["mean_clipped_r2"]-baseline["mean_clipped_r2"]
    torch.save({"state_dict": best_state, "input_mean": scaler.mean_, "input_scale": scaler.scale_,
                "target_mean": means, "target_scale": scale, "widths": [128, 128],
                "best_epoch": best_epoch, "source": run}, a.output / "probe.pt")
    np.save(a.output / "val_quantities.npy", best_prediction)
    report = {"baseline": baseline, "best": best, "best_epoch": best_epoch, "auxiliary_gain": gain,
              "investigate_flux_correction": gain >= .0005, "eligible_for_submission": False,
              "audit_labels_read": False, "test_labels_read": False,
              "limitation": "Auxiliary predictability alone does not change the flux field or establish a composite-score gain"}
    write_json(a.output / "comparison.json", report)
    run.update(status="completed", elapsed_seconds=time.monotonic()-started)
    write_json(a.output / "run.json", run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
