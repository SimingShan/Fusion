#!/usr/bin/env python3
"""Measure actual training-step cost using permitted training rows only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve prior runtime evidence")
    torch.set_num_threads(a.workers)
    torch.manual_seed(2090)
    np.random.seed(2090)
    source = a.source.resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("_capacity_training", source / "scripts/train_pod.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    directory = ROOT / "data/pod_history128_robust_v4"
    with np.load(directory / "normalization.npz", allow_pickle=False) as saved:
        norm = dict(saved)
    tnorm = train.torch_normalization(norm, "cpu")
    arrays = {key: np.load(directory / f"train_{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("X", "coefficients", "scalars")}
    n = len(arrays["X"])
    rng = np.random.default_rng(2090)
    results = []
    for width, depth in ((256, 3), (512, 4)):
        model = train.build_advanced_model("pod_deeponet", input_dim=len(norm["feature_indices"]),
            n_coefficients=len(norm["basis"]), width=width, depth=depth, dropout=0.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        Q = train.loss_matrix(tnorm, 1., .25, "all")
        durations = []
        for index in range(24):
            selected = rng.integers(0, n, size=2048)
            started = time.monotonic()
            batch = {key: torch.from_numpy(np.array(value[selected])) for key, value in arrays.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = train.batch_loss(model, batch, tnorm, Q, .25)
            if not torch.isfinite(loss):
                raise ValueError("Benchmark training loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if index >= 4:
                durations.append(time.monotonic()-started)
        median, p95 = np.median(durations), np.quantile(durations, .95)
        record = {"width": width, "depth": depth, "parameters": sum(p.numel() for p in model.parameters()),
                  "batch_size": 2048, "workers": a.workers, "measured_steps": len(durations),
                  "step_median_seconds": float(median), "step_p95_seconds": float(p95),
                  "compact_epoch_estimated_seconds": float(math.ceil(n/2048)*median),
                  "compact_600_epochs_p95_hours": float(math.ceil(n/2048)*p95*600/3600),
                  "native_1200_epochs_p95_hours": float(math.ceil(1496407/2048)*p95*1200/3600)}
        results.append(record)
        print(json.dumps(record), flush=True)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Training cost measurement only",
              "training_frames_available": n, "records": results,
              "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                  for name in ("scripts/train_pod.py", "fusion/advanced_models.py", "fusion/pod.py")},
              "validation_labels_read": False, "audit_labels_read": False, "test_labels_read": False,
              "model_weights_retained": False, "eligible_for_submission": False,
              "limitation": "Timing excludes full-array loading, validation, export and auxiliary refinement; node performance may differ"}
    write_json(a.output, result)


if __name__ == "__main__":
    main()
