#!/usr/bin/env python3
"""Fine-tune one pair member against the actual input-gated deployed output.

The primary is frozen and each normalization is applied directly to raw inputs.
Only the secondary is serialized, for the independent existing export pipeline.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.advanced_models import build_advanced_model
from fusion.pod import load_arrays, read_manifest, manifest_hash, split_entries, standardize_x, validate_splits
from fusion.geometry_loss import GeometryTargetStore
from scripts.train_pod import torch_normalization, reconstruct, loss_matrix, batch_loss, validation, score_predictions, write_predictions
from scripts.optimization.quality_gate import feature_columns, gate_from_inputs, write_array
from scripts.codabench_submit import write_json


class PairPolicy(nn.Module):
    def __init__(self, primary, secondary, names, weight=.5):
        super().__init__()
        self.primary = build_advanced_model(primary["model"], **primary["model_kwargs"])
        self.secondary = build_advanced_model(secondary["model"], **secondary["model_kwargs"])
        self.primary.load_state_dict(primary["state_dict"])
        self.secondary.load_state_dict(secondary["state_dict"])
        self.primary.requires_grad_(False)
        self.anorm, self.bnorm = primary["normalization"], secondary["normalization"]
        for key in ("basis", "coefficient_std", "psi_mean", "psi_std", "scalar_mean", "scalar_std"):
            if not np.array_equal(self.anorm[key], self.bnorm[key]):
                raise ValueError(f"Output coordinates must match exactly: {key}")
        self.na, self.nb = len(self.anorm["feature_indices"]), len(self.bnorm["feature_indices"])
        self.names, self.weight = names, float(weight)
        self.gate_columns = feature_columns(names)
        if not 0 < self.weight < 1:
            raise ValueError("The primary weight must be strictly between zero and one")
        self.plans = []
        for system in ("core", "edge"):
            parts = []
            for offset, norm in ((0, self.anorm), (self.na, self.bnorm)):
                names = [self.names[i] for i in norm["feature_indices"]]
                columns = np.array([i for i, name in enumerate(names)
                                    if name.startswith((f"thomson_{system}_Te_", f"thomson_{system}_ne_"))], dtype=np.int64)
                missing = np.clip(-norm["x_mean"][columns]/norm["x_std"][columns], -12, 12).astype(np.float32)
                parts.append((torch.from_numpy(columns+offset), torch.from_numpy(missing)))
            self.plans.append(parts)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.primary.eval()
        return self

    def inputs(self, raw):
        gate = gate_from_inputs(raw, self.gate_columns)[0]
        return np.column_stack((standardize_x(raw, self.anorm), standardize_x(raw, self.bnorm), gate)).astype(np.float32)

    def forward(self, x):
        with torch.no_grad():
            a = self.primary(x[:, :self.na])
        b = self.secondary(x[:, self.na:self.na+self.nb])
        if set(a) != {"coefficients", "scalars"} or set(b) != set(a):
            raise ValueError("This experiment supports only common linear output coordinates")
        weight = self.weight*(1-x[:, -1:])
        return {k: weight*a[k] + (1-weight)*b[k] for k in a}

    def augment(self, x, probability, draws=None):
        result = x.clone()
        for index, parts in enumerate(self.plans):
            selected = (torch.rand((len(x), 1), device=x.device) < probability
                        if draws is None else draws[:, index:index+1])
            for columns, missing in parts:
                result[:, columns] = torch.where(selected, missing[None], result[:, columns])
            if index == 1:
                result[:, -1:] = torch.where(selected, 1., result[:, -1:])
        return result


def raw_shot(data, entry):
    with np.load(data / entry["path"], allow_pickle=False) as z:
        x = z["X"]
    if entry.get("X_extra"):
        extra = np.load(data / entry["X_extra"], allow_pickle=False)
        if len(extra) != len(x):
            raise ValueError("Input sidecar is misaligned")
        x = np.column_stack((x, extra))
    return x


def check_contract(model, data, device):
    raw = np.concatenate([raw_shot(data, e) for e in split_entries(data, "val")[:4]])
    x = torch.as_tensor(model.inputs(raw), device=device)
    norm = torch_normalization(model.bnorm, device)
    model.eval()
    with torch.no_grad():
        a = model.primary(x[:, :model.na]); b = model.secondary(x[:, model.na:-1])
        gate = x[:, -1:, None].bool()
        expected = torch.where(gate, reconstruct(model.secondary, b, norm),
                               model.weight*reconstruct(model.primary, a, norm)+(1-model.weight)*reconstruct(model.secondary, b, norm))
        actual = reconstruct(model, model(x), norm)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        draws = torch.tensor([[False, False], [True, False], [False, True], [True, True]], device=device).repeat((len(x)+3)//4, 1)[:len(x)]
        altered = raw.copy()
        for i, system in enumerate(("core", "edge")):
            columns = [j for j, n in enumerate(model.names) if n.startswith((f"thomson_{system}_Te_", f"thomson_{system}_ne_"))]
            altered[np.ix_(draws[:, i].cpu().numpy(), columns)] = 0
        expected_x = torch.as_tensor(model.inputs(altered), device=device)
        augmented = model.augment(x, 0, draws)
        torch.testing.assert_close(augmented, expected_x, atol=1e-6, rtol=1e-6)
    model.train()
    model(augmented[:8])["coefficients"].square().mean().backward()
    assert all(p.grad is None for p in model.primary.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.secondary.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.secondary.parameters())
    model.zero_grad(set_to_none=True)
    return {"passed": True, "frames": len(x), "field_max_absolute_difference": float((actual-expected).abs().max()),
            "paired_input_dropout_matches_raw": True, "frozen_primary_gradient_absent": True, "audit_labels_read": False}


@torch.no_grad()
def write_policy(model, data, split, output, batch_size, device):
    model.eval()
    norm = torch_normalization(model.bnorm, device)
    temporary = output.with_suffix(".tmp.npz")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for i, entry in enumerate(split_entries(data, split)):
            x = model.inputs(raw_shot(data, entry))
            fields, scalars = [], []
            for first in range(0, len(x), batch_size):
                pred = model(torch.as_tensor(x[first:first+batch_size], device=device))
                fields.append(reconstruct(model, pred, norm).cpu().numpy())
                scalars.append(pred["scalars"].cpu().numpy()*model.bnorm["scalar_std"]+model.bnorm["scalar_mean"])
            field, scalar = np.concatenate(fields), np.concatenate(scalars)
            if len(field) != entry["n_frames"]:
                raise ValueError("Incomplete prediction")
            for key, value in (("psirz", field), ("q95", scalar[:, 0]), ("betaN", scalar[:, 1])):
                write_array(z, f"shot_{i:04d}_{key}", value)
    temporary.replace(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/advanced_history_v2"))
    p.add_argument("--pod-dir", type=Path, default=Path("data/pod_history128_robust_v4"))
    p.add_argument("--primary", type=Path, default=Path("results/candidates/20260919-001/checkpoints/a.pt"))
    p.add_argument("--secondary", type=Path, default=Path("results/candidates/20260919-001/checkpoints/b.pt"))
    p.add_argument("--li-dir", type=Path, default=Path("data/geometry_history_v4"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seconds", type=int, default=2400)
    p.add_argument("--score-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--coefficient-power", type=float, default=1.)
    p.add_argument("--derived-inputs-only", action="store_true")
    p.add_argument("--li-weight", type=float, default=.02)
    p.add_argument("--geometry-weight", type=float, default=.01)
    p.add_argument("--exact-axis-weight", type=float, default=0.)
    p.add_argument("--thomson-dropout", type=float, default=.1)
    p.add_argument("--seed", type=int, default=2051)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if not a.check_only and (a.output / "run.json").exists():
        raise ValueError("Use a fresh experiment directory")
    if min(a.epochs, a.seconds, a.score_every, a.batch_size, a.workers) < 1 or not 0 <= a.thomson_dropout < 1:
        p.error("Invalid training settings")
    if not np.isfinite(a.coefficient_power) or not 0 <= a.coefficient_power <= 2:
        p.error("Coefficient power must be finite and between zero and two")
    if not np.isfinite(a.exact_axis_weight) or a.exact_axis_weight < 0:
        p.error("Axis supervision weight must be finite and nonnegative")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.workers)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cpu")
    validate_splits(a.data_dir)
    manifest = read_manifest(a.data_dir)
    primary, secondary = [torch.load(x, map_location="cpu", weights_only=False) for x in (a.primary, a.secondary)]
    model = PairPolicy(primary, secondary, manifest["feature_names"]).to(device)
    check = check_contract(model, a.data_dir, device)
    write_json(a.output / "contract_check.json", check)
    print(json.dumps(check), flush=True)
    if a.check_only:
        return
    basis = json.loads((a.pod_dir / "basis.json").read_text())
    if basis["source_manifest_sha256"] != manifest_hash(a.data_dir):
        raise ValueError("Compact training targets refer to a different manifest")
    with np.load(a.pod_dir / "normalization.npz", allow_pickle=False) as z:
        if set(z.files) != set(model.bnorm) or any(not np.array_equal(z[k], model.bnorm[k]) for k in z.files):
            raise ValueError("Compact target normalization differs from the checkpoint")
    arrays = {}
    for split in ("train", "val"):
        raw = load_arrays(a.data_dir, split, ("X",))["X"]
        x = np.empty((len(raw), model.na+model.nb+1), dtype=np.float32)
        compact_x = np.load(a.pod_dir / f"{split}_X.npy", mmap_mode="r", allow_pickle=False)
        for first in range(0, len(raw), 4096):
            sl = slice(first, first+4096)
            x[sl] = model.inputs(raw[sl])
            np.testing.assert_allclose(x[sl, model.na:-1], compact_x[sl], atol=1e-6, rtol=1e-6)
        arrays[split] = {"X": torch.from_numpy(x)}
        for key in ("coefficients", "scalars"):
            values = np.load(a.pod_dir / f"{split}_{key}.npy", mmap_mode="r", allow_pickle=False)
            if len(values) != len(x):
                raise ValueError("Training target and input rows differ")
            arrays[split][key] = torch.from_numpy(np.array(values))
    targets = GeometryTargetStore(a.li_dir)
    selection_hash = hashlib.sha256(json.dumps(split_entries(a.data_dir, "train"), sort_keys=True).encode()).hexdigest()
    if selection_hash != targets.metadata["source_training_entries_sha256"]:
        raise ValueError("Auxiliary training targets are misaligned")
    li, geometry = targets.loss_module(device), targets.geometry_loss_module(device)
    exact_axis = targets.exact_axis_loss_module(device) if a.exact_axis_weight else None
    norm = torch_normalization(model.bnorm, device)
    Q = loss_matrix(norm, a.coefficient_power, .25, "envelope")
    if a.derived_inputs_only:
        if secondary["model"] != "pod_derived_inputs":
            raise ValueError("A derived-input checkpoint is required for restricted training")
        for name, parameter in model.secondary.named_parameters():
            parameter.requires_grad_(name.startswith(("derived_stem.", "derived_coefficients.", "derived_scalars.")))
    trainable = [parameter for parameter in model.secondary.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs, eta_min=a.lr*.03)
    source_paths = [Path(__file__), ROOT / "scripts/train_pod.py", ROOT / "fusion/pod.py", ROOT / "fusion/advanced_models.py",
                    ROOT / "fusion/geometry_loss.py", ROOT / "fusion/li_loss.py", ROOT / "fusion/input_derived.py",
                    ROOT / "scripts/optimization/quality_gate.py"]
    hashes = {}
    for path in source_paths:
        relative = path.resolve().relative_to(ROOT.resolve())
        destination = a.output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        contents = path.read_bytes(); destination.write_bytes(contents)
        hashes[str(relative)] = hashlib.sha256(contents).hexdigest()
    run = {"status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
           "feature_names": manifest["feature_names"], "audit_labels_read": False, "test_labels_read": False,
           "train_frames": len(arrays["train"]["X"]), "primary_weight": .5, "gate_threshold": .1,
           "trainable_parameters": sum(parameter.numel() for parameter in trainable),
           "checkpoint_selection": "maximum actual paired composite on capped fresh development shots",
           "source_sha256": hashes, "started_utc": datetime.now(timezone.utc).isoformat(),
           "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.primary, a.secondary)}}
    write_json(a.output / "run.json", run)
    started = time.monotonic()
    history, best = [], -float("inf")
    training, valid = arrays["train"], arrays["val"]
    for epoch in range(a.epochs+1):
        train_loss = 0.
        if epoch:
            model.train()
            for indices in torch.randperm(len(training["X"])).split(a.batch_size):
                batch = {k: v[indices] for k, v in training.items()}
                if a.thomson_dropout:
                    batch["X"] = model.augment(batch["X"], a.thomson_dropout)
                positions, auxiliary = targets.batch(indices, device)
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(model, batch, norm, Q, .25, li_module=li, li_positions=positions,
                                  li_targets=auxiliary, li_weight=a.li_weight,
                                  geometry_module=geometry, geometry_weight=a.geometry_weight,
                                  exact_axis_module=exact_axis, exact_axis_weight=a.exact_axis_weight)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.secondary.parameters(), 1.)
                optimizer.step()
                train_loss += float(loss.detach())*len(indices)
            scheduler.step()
        val_loss = validation(model, valid, norm, Q, .25, a.batch_size)
        last = epoch == a.epochs or time.monotonic()-started >= a.seconds
        record = {"epoch": epoch, "train_loss": train_loss/len(training["X"]), "val_loss": val_loss,
                  "elapsed_seconds": time.monotonic()-started, "lr": scheduler.get_last_lr()[0]}
        if epoch % a.score_every == 0 or last:
            pred_path, score_path = a.output / "candidate_policy_val_predictions.npz", a.output / "candidate_policy_val_score.json"
            write_policy(model, a.data_dir, "val", pred_path, a.batch_size, device)
            metrics = score_predictions(a.data_dir, "val", pred_path, score_path, a.workers)
            record["S"] = metrics["S"]
            checkpoint = {"model": secondary["model"], "model_kwargs": secondary["model_kwargs"],
                          "normalization": model.bnorm, "state_dict": model.secondary.state_dict(),
                          "epoch": epoch, "run": run, "validation_score": metrics["S"]}
            snapshots = a.output / "checkpoints"; snapshots.mkdir(exist_ok=True)
            torch.save(checkpoint, snapshots / f"epoch_{epoch:04d}.pt")
            if metrics["S"] > best:
                best = metrics["S"]
                torch.save(checkpoint, a.output / "best.pt")
                pred_path.replace(a.output / "policy_val_predictions.npz")
                score_path.replace(a.output / "policy_val_score.json")
        history.append(record)
        write_json(a.output / "history.json", history)
        if epoch % 5 == 0 or last:
            print(json.dumps(record), flush=True)
        if last:
            break
    saved = torch.load(a.output / "best.pt", map_location="cpu", weights_only=False)
    model.secondary.load_state_dict(saved["state_dict"])
    # Independent staging expects standalone member predictions on this fold.
    write_predictions(model.secondary, model.bnorm, a.data_dir, "legacy_val", a.output / "legacy_val_predictions.npz", a.batch_size, device)
    score_predictions(a.data_dir, "legacy_val", a.output / "legacy_val_predictions.npz", a.output / "legacy_val_score.json", a.workers)
    run.update(status="completed", completed_epochs=epoch, best_validation_score=best,
               best_epoch=saved["epoch"], elapsed_seconds=time.monotonic()-started)
    write_json(a.output / "run.json", run)


if __name__ == "__main__":
    main()
