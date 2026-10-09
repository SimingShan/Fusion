#!/usr/bin/env python3
"""Reuse exact training li targets and add local axis/LCFS supervision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.geometry_loss import FixedGeometryLoss
from fusion.li_loss import contour_to_sampling_grid
from fusion.pod import load_arrays, split_entries, validate_splits
from scripts.optimization.prepare_scaled_features import _link_or_copy
from starter import local_score as official
from derive import magnetic_axis


def entries_hash(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def verify_identical_targets(base_dir, data_dir, source_metadata):
    base, new = split_entries(base_dir, "train"), split_entries(data_dir, "train")
    if entries_hash(base) != source_metadata["source_training_entries_sha256"]:
        raise ValueError("Source li target indexing differs from source training entries")
    if len(base) != len(new):
        raise ValueError("Target view changes training shot count")
    for a, b in zip(base, new):
        for key in ("shot_index", "shot_id", "shard", "row", "n_frames", "raw_frames"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Training target identity differs: {key}")
        if (base_dir / a["path"]).resolve() != (data_dir / b["path"]).resolve():
            raise ValueError("Target view must reference the exact same immutable shot cache")
    if sum(e["n_frames"] for e in new) != source_metadata["total_training_frames"]:
        raise ValueError("Target view changes flattened frame indexing")
    return new


def axis_reference(psi, R, Z, mask):
    ra, za, iz, ir = magnetic_axis(psi, R, Z, official.MACHINE, mask)
    if not 0 < ir < len(R)-1 or not 0 < iz < len(Z)-1:
        raise ValueError("Valid training contour has boundary magnetic axis")
    indices = np.array([iz*len(R)+ir, iz*len(R)+ir-1, iz*len(R)+ir+1,
                        (iz-1)*len(R)+ir, (iz+1)*len(R)+ir], dtype=np.int64)
    values = psi.ravel()[indices].astype(np.float64)
    response = np.zeros((2, 5), dtype=np.float64)
    for k, (left, right, step) in enumerate(((1, 2, R[1]-R[0]), (3, 4, Z[1]-Z[0]))):
        denominator = values[left] - 2*values[0] + values[right]
        numerator = values[left] - values[right]
        # Fixed reference degeneracy guard, with no prediction-dependent clamp.
        if abs(denominator) <= 1e-8 or abs(0.5*numerator/denominator) >= 1:
            continue
        response[k, 0] = step*numerator/denominator**2
        response[k, left] = 0.5*step*(denominator-numerator)/denominator**2
        response[k, right] = 0.5*step*(-denominator-numerator)/denominator**2
    return indices, values, response, np.array([ra, za])


def prepare(data_dir, source_li, output, limit=0):
    started = time.monotonic()
    data_dir, source_li, output = [Path(p).resolve() for p in (data_dir, source_li, output)]
    if source_li == output:
        raise ValueError("Geometry preparation requires a separate target directory")
    validate_splits(data_dir)
    source = json.loads((source_li / "metadata.json").read_text())
    base_dir = Path(source["source_data_dir"])
    if not source.get("complete") or source["source_split"] != "train":
        raise ValueError("Complete training-only source targets are required")
    entries = verify_identical_targets(base_dir, data_dir, source)
    indices = np.load(source_li / "global_frame_indices.npy", mmap_mode="r")
    valid = np.load(source_li / "valid.npy", mmap_mode="r")
    contours = np.load(source_li / "contours.npy", mmap_mode="r")
    segments = np.load(source_li / "segment_lengths.npy", mmap_mode="r")
    R, Z = [np.load(source_li / f"{key}.npy") for key in ("R", "Z")]
    with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"]
    psi = load_arrays(base_dir, "train", keys=("psi",))["psi"]
    rows = np.flatnonzero(valid)
    if limit:
        rows = rows[:limit]
    if output.exists() and (output / "metadata.json").exists():
        raise ValueError("Output already has metadata; use a unique target directory")
    output.mkdir(parents=True, exist_ok=True)
    shapes = {"axis_indices": (len(indices), 5), "axis_flux": (len(indices), 5),
              "axis_response": (len(indices), 2, 5), "boundary_flux": (len(indices), contours.shape[1]),
              "boundary_gradient_floor": (len(indices), contours.shape[1])}
    arrays = {key: np.lib.format.open_memmap(output / f"{key}.partial.npy", mode="w+", shape=shape,
                                            dtype=np.int64 if key == "axis_indices" else np.float64)
              for key, shape in shapes.items()}
    axes = []
    for first in range(0, len(rows), 256):
        selected = rows[first:first+256]
        fields = np.asarray(psi[indices[selected]], dtype=np.float64)
        axis = [axis_reference(p, R, Z, mask) for p in fields]
        for key, position in (("axis_indices", 0), ("axis_flux", 1), ("axis_response", 2)):
            arrays[key][selected] = np.stack([a[position] for a in axis])
        axes.extend(a[3] for a in axis)
        grid = torch.as_tensor(contour_to_sampling_grid(contours[selected], R, Z))[:, :, None]
        field = torch.as_tensor(fields)
        gz, gr = torch.gradient(field, spacing=(torch.as_tensor(Z), torch.as_tensor(R)), dim=(1, 2))
        gradient = (gr.square() + gz.square()).sqrt()
        boundary_flux = F.grid_sample(field[:, None], grid, align_corners=True, padding_mode="border")[:, 0, :, 0].numpy()
        boundary_gradient = F.grid_sample(gradient[:, None], grid, align_corners=True, padding_mode="border")[:, 0, :, 0].numpy()
        # X-points have vanishing gradients; a reference-only relative floor
        # avoids an unbounded displacement linearization near a saddle.
        gradient_floor = np.maximum(boundary_gradient, 0.1*boundary_gradient.mean(1, keepdims=True))
        arrays["boundary_flux"][selected] = boundary_flux
        arrays["boundary_gradient_floor"][selected] = np.maximum(gradient_floor, 1e-8)
        if first % 5120 == 0:
            print(f"Geometry targets: {min(first+256, len(rows))}/{len(rows)} training frames", flush=True)
    variance = np.maximum(np.var(np.stack(axes), axis=0), 1e-6)
    chosen = rows[:min(20, len(rows))]
    truth = torch.as_tensor(np.asarray(psi[indices[chosen]], dtype=np.float64))
    targets = {key: torch.as_tensor(np.array(value[chosen])) for key, value in arrays.items()}
    targets.update(sampling_grid=torch.as_tensor(contour_to_sampling_grid(contours[chosen], R, Z)),
                   segment_lengths=torch.as_tensor(np.array(segments[chosen])))
    module = FixedGeometryLoss(variance)
    exact_parts = {key: float(value) for key, value in module.parts(truth, targets).items()}
    shifted = module(truth + 0.123, targets)
    perturbation = torch.as_tensor(np.broadcast_to(R[None, :]**2, (len(Z), len(R))).copy())
    perturbed = (truth + 1e-4*perturbation).requires_grad_(True)
    loss = module(perturbed, targets)
    loss.backward()
    if max(exact_parts.values()) > 1e-20 or float(shifted) > 1e-18:
        raise ValueError(f"Geometry target/offset parity failed: {exact_parts}, {shifted}")
    if not torch.isfinite(perturbed.grad).all() or not torch.any(perturbed.grad != 0):
        raise ValueError("Geometry gradient must be finite and nonzero away from target")
    for key, value in arrays.items():
        value.flush()
        os.replace(output / f"{key}.partial.npy", output / f"{key}.npy")
    if not limit:
        for path in source_li.glob("*.npy"):
            _link_or_copy(path, output / path.name)
    metadata = {**source, "source_data_dir": str(data_dir), "source_training_entries_sha256": entries_hash(entries),
                "complete": not bool(limit), "reused_targets": {"source": str(source_li),
                 "source_metadata_sha256": hashlib.sha256((source_li / "metadata.json").read_bytes()).hexdigest(),
                 "same_immutable_target_paths_verified": len(entries), "targets_reselected": False},
                "geometry": {"complete": not bool(limit), "prepared_frames": len(rows),
                 "axis_variance": variance.tolist(), "boundary_length_scale": 0.1,
                 "method": "First-order axis displacement from exact parabolic derivative; offset-invariant LCFS normal displacement",
                 "boundary_gradient_floor": "0.1 times each target contour's mean gradient; minimum 1e-8",
                 "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "self_check": {"target_loss_parts": exact_parts, "offset_loss": float(shifted),
                                "finite_nonzero_gradient": True, "frames": len(chosen)},
                 "validation_labels_read": False, "audit_labels_read": False,
                 "completed_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.monotonic()-started}}
    temporary = output / "metadata.partial.json"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    temporary.replace(output / "metadata.json")
    print(json.dumps(metadata["geometry"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_history_v2"))
    parser.add_argument("--source-li", type=Path, default=Path("data/li_full_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/geometry_history_v4"))
    parser.add_argument("--limit", type=int, default=0, help="Smoke check only; output cannot be trained")
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(**vars(args))


if __name__ == "__main__":
    main()
