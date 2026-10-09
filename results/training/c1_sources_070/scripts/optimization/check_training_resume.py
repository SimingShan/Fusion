#!/usr/bin/env python3
"""Compare uninterrupted and serialized training, including all RNG states."""
from pathlib import Path
import argparse
import copy
import json
import random
import sys
import tempfile

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.training_resume import save_training_state, restore_training_state
from scripts.codabench_submit import write_json


def initialize(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = nn.Sequential(nn.Linear(10, 24), nn.GELU(), nn.Dropout(.15), nn.Linear(24, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6)
    return model, optimizer, scheduler


def train(model, optimizer, scheduler, history, first, last, X, Y):
    model.train()
    for epoch in range(first, last+1):
        for indices in torch.randperm(len(X)).split(13):
            factor = np.random.uniform(.95, 1.05)+random.uniform(-.01, .01)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(X[indices]*factor)-Y[indices]).square().mean()
            loss.backward(); optimizer.step()
        scheduler.step()
        history.append({"epoch": epoch, "elapsed_seconds": float(epoch), "loss": float(loss.detach())})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve completed checks")
    torch.set_num_threads(2)
    generator = np.random.default_rng(902)
    X = torch.from_numpy(generator.normal(size=(97, 10)).astype(np.float32))
    Y = torch.from_numpy(generator.normal(size=(97, 3)).astype(np.float32))
    norm = {"feature_indices": np.arange(10), "x_mean": np.zeros(10), "x_std": np.ones(10)}
    run = {"model": "synthetic_check", "model_kwargs": {"width": 24}, "arguments": {"epochs": 6},
           "source_sha256": {"test": "fixed"}, "feature_names": list(map(str, range(10))), "basis": {},
           "initial_checkpoint_sha256": None, "status": "paused", "started_utc": "synthetic",
           "elapsed_seconds": 3.}
    whole = initialize(2080); full_history = [{"epoch": 0}]
    train(*whole, full_history, 1, 6, X, Y)
    expected_rng = (torch.rand(5), np.random.random(5), random.random())
    before = initialize(2080); partial_history = [{"epoch": 0}]
    train(*before, partial_history, 1, 3, X, Y)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "state.pt"
        save_training_state(path, *before, norm, run, partial_history, .7, .2)
        after = initialize(991)
        restored = restore_training_state(path, *after, norm, run)
        train(*after, restored["history"], 4, 6, X, Y)
        for key, value in whole[0].state_dict().items():
            torch.testing.assert_close(after[0].state_dict()[key], value, atol=0, rtol=0)
        torch.testing.assert_close(after[1].state_dict(), whole[1].state_dict(), atol=0, rtol=0)
        assert after[2].state_dict() == whole[2].state_dict()
        assert restored["history"] == full_history
        torch.testing.assert_close(torch.rand(5), expected_rng[0], atol=0, rtol=0)
        np.testing.assert_array_equal(np.random.random(5), expected_rng[1])
        assert random.random() == expected_rng[2]
        rejected = []
        for name in ("source_sha256", "arguments", "feature_names"):
            changed = copy.deepcopy(run); changed[name] = {"changed": True}
            try:
                restore_training_state(path, *after, norm, changed)
            except ValueError:
                rejected.append(name)
            else:
                raise ValueError("A changed continuation contract was accepted")
        changed_norm = dict(norm, x_mean=np.ones(10))
        try:
            restore_training_state(path, *after, changed_norm, run)
        except ValueError:
            rejected.append("normalization")
        else:
            raise ValueError("Changed normalization was accepted")
    result = {"passed": True, "model_weights_bit_identical": True, "optimizer_state_bit_identical": True,
              "scheduler_state_identical": True, "training_history_identical": True,
              "all_three_rng_states_identical": True, "rejected_contract_changes": rejected,
              "synthetic_only": True, "target_data_read": False}
    write_json(a.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
