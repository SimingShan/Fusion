"""Exact epoch-boundary continuation for bounded CPU training allocations."""
from pathlib import Path
import os
import random

import numpy as np
import torch


def save_training_state(path, model, optimizer, scheduler, norm, run, history, best_score, best_loss):
    path = Path(path)
    state = {"version": 1, "torch_version": str(torch.__version__), "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "normalization": norm, "run": run, "history": history,
             "best_score": best_score, "best_loss": best_loss,
             "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(path, model, optimizer, scheduler, norm, run):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["version"] != 1 or state["torch_version"] != str(torch.__version__):
        raise ValueError("Training-state format or numerical dependency changed")
    previous = state["run"]
    for key in ("model", "model_kwargs", "arguments", "source_sha256", "feature_names", "basis", "initial_checkpoint_sha256"):
        if previous.get(key) != run.get(key):
            raise ValueError(f"Training continuation contract changed: {key}")
    if state["normalization"].keys() != norm.keys() or any(not np.array_equal(value, norm[key]) for key, value in state["normalization"].items()):
        raise ValueError("Training continuation normalization changed")
    history = state["history"]
    if not history or [x["epoch"] for x in history] != list(range(history[-1]["epoch"]+1)):
        raise ValueError("Training history is not complete and consecutive")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state
