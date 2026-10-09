#!/usr/bin/env python3
"""Derive one fixed predictor from the last scheduled states of a completed run."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.advanced_models import build_advanced_model
from scripts.codabench_submit import write_json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tail-count", type=int, default=3)
    a = p.parse_args()
    if a.tail_count < 2 or a.output.exists():
        p.error("Use at least two states and a fresh output directory")
    parent = json.loads((a.training / "run.json").read_text())
    if parent.get("status") != "completed" or parent.get("audit_labels_read") is not False:
        raise ValueError("Require a completed non-audit training run")
    paths = sorted((a.training / "checkpoints").glob("epoch_*.pt"))
    if len(paths) < a.tail_count:
        raise ValueError("Insufficient scheduled checkpoints")
    paths = paths[-a.tail_count:]
    hashes = [digest(path) for path in paths]
    members = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    first = members[0]
    for member in members:
        if (member["model"] != first["model"] or member["model_kwargs"] != first["model_kwargs"]
                or member["run"]["feature_names"] != first["run"]["feature_names"]
                or member["run"]["arguments"] != first["run"]["arguments"]
                or member["normalization"].keys() != first["normalization"].keys()
                or member["state_dict"].keys() != first["state_dict"].keys()
                or any(not np.array_equal(v, member["normalization"][k]) for k, v in first["normalization"].items())):
            raise ValueError("Checkpoint architectures, normalization or training contracts differ")
    state = {}
    for key, original in first["state_dict"].items():
        values = [x["state_dict"][key] for x in members]
        if any(x.shape != original.shape or x.dtype != original.dtype for x in values):
            raise ValueError("Incompatible state tensor")
        if original.is_floating_point():
            if any(not torch.isfinite(x).all() for x in values):
                raise ValueError("Nonfinite model parameter")
            state[key] = torch.stack([x.double() for x in values]).mean(0).to(original.dtype)
        else:
            if any(not torch.equal(original, x) for x in values):
                raise ValueError("Nonfloating model state changed")
            state[key] = original.clone()
    model = build_advanced_model(first["model"], **first["model_kwargs"])
    model.load_state_dict(state, strict=True)
    if hashes != [digest(path) for path in paths]:
        raise ValueError("A source checkpoint changed")
    a.output.mkdir(parents=True)
    run = copy.deepcopy(parent)
    run.pop("best_validation_score", None)
    run.update(status="completed", assembly_only=True, audit_labels_read=False, test_labels_read=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_selection=f"Uniform parameter mean of the last {len(members)} scheduled states; no new score used in assembly",
        assembly_source_sha256=digest(__file__), source_run_sha256=digest(a.training / "run.json"),
        members=[{"path": str(path.resolve()), "sha256": sha, "epoch": member["epoch"]}
                 for path, sha, member in zip(paths, hashes, members)],
        member_weights=[1/len(members)]*len(members), validation_score_established=False)
    saved = {"model": first["model"], "model_kwargs": first["model_kwargs"], "state_dict": state,
             "normalization": first["normalization"], "epoch": members[-1]["epoch"], "run": run}
    torch.save(saved, a.output / "best.pt")
    write_json(a.output / "run.json", run)
    shutil.copy2(__file__, a.output / Path(__file__).name)
    print(json.dumps({"assembled": True, "epochs": [x["epoch"] for x in members],
        "source_hashes": hashes, "output_sha256": digest(a.output / "best.pt"), "validation_required": True}), flush=True)


if __name__ == "__main__":
    main()
