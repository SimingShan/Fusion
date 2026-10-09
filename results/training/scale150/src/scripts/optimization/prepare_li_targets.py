#!/usr/bin/env python3
"""Precompute fixed-contour li supervision from training labels only.

Select up to eight existing cached frames per training shot, without temporal
interpolation. The contour is the exact 512-point polygon returned by the
official scorer, saved in float64; there is no additional contour resampling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.li_loss import FixedContourLiLoss, contour_to_sampling_grid
from fusion.pod import split_entries, validate_splits
from starter import local_score as official
from derive import _points_in_polygon, internal_inductance, PLAUSIBLE


def fingerprint():
    digest = hashlib.sha256()
    paths = [ROOT / "starter/local_score.py"]
    paths += sorted((ROOT / "starter/fusion_scoring").glob("*.py"))
    paths += [ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz"]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reference(psi, R, Z, mask):
    if not np.isfinite(psi).all():
        raise ValueError("Nonfinite training flux encountered")
    C = official.extract_lcfs(psi, R, Z, official.MACHINE, mask, mask.astype(np.float64),
                              n_iter=official.N_ITER, n_points=official.N_POINTS)
    if C is None:
        return None
    li = internal_inductance(psi, R, Z, official.MACHINE, C)
    lower, upper = PLAUSIBLE["li"]
    if not np.isfinite(li) or not lower <= li <= upper:
        return None
    RR, ZZ = np.meshgrid(R, Z)
    inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    segment_lengths = np.linalg.norm(np.roll(C, -1, axis=0) - C, axis=1)
    return dict(contour=C, inside=inside, segment_lengths=segment_lengths, li=float(li))


def self_check(data_dir, entries, R, Z, mask, count=20):
    fields, refs, identity = [], [], []
    for entry in entries:
        with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
            if not np.array_equal(shot["R"], R) or not np.array_equal(shot["Z"], Z):
                raise ValueError("Training grid differs from official envelope grid")
            frame = len(shot["psi"]) // 2
            psi = shot["psi"][frame]
            value = reference(psi, R, Z, mask)
            if value is None:
                continue
            fields.append(psi)
            refs.append(value)
            identity.append({"shot_index": entry["shot_index"], "cache_frame_index": frame,
                             "native_frame_index": int(shot["frame_indices"][frame])})
        if len(fields) == count:
            break
    if len(fields) != count:
        raise ValueError(f"Needed {count} valid training self-check frames; found {len(fields)}")
    targets = {
        "inside": torch.as_tensor(np.stack([r["inside"] for r in refs])),
        "sampling_grid": torch.as_tensor(contour_to_sampling_grid(np.stack([r["contour"] for r in refs]), R, Z)),
        "segment_lengths": torch.as_tensor(np.stack([r["segment_lengths"] for r in refs])),
        "li": torch.as_tensor([r["li"] for r in refs], dtype=torch.float64),
    }
    model = FixedContourLiLoss(R, Z, max(float(targets["li"].var(unbiased=False)), 1e-8))
    exact = torch.as_tensor(np.stack(fields), dtype=torch.float64)
    output64 = model.predict(exact, targets)
    output32 = model.predict(exact.float(), targets)
    max64 = float((output64 - targets["li"]).abs().max())
    max32 = float((output32.double() - targets["li"]).abs().max())
    if max64 > 1e-9 or max32 > 1e-4:
        raise ValueError(f"Torch/official li parity failed: float64={max64}, float32={max32}")
    shifted = model.predict(1.3 * exact + 0.123, targets)
    if not torch.allclose(output64, shifted, rtol=1e-10, atol=1e-10):
        raise ValueError("Expected flux-scale/offset invariance failed")
    perturbation = torch.as_tensor(np.broadcast_to(R[None, :] ** 2, (len(Z), len(R))).copy())
    perturbed = (exact + 1e-4 * perturbation).requires_grad_(True)
    loss = model(perturbed, targets)
    loss.backward()
    if not torch.isfinite(perturbed.grad).all() or not torch.any(perturbed.grad != 0):
        raise ValueError("Perturbed real-field li gradient check failed")
    return {"passed": True, "training_frames": count, "identities": identity,
            "reference": "official internal_inductance with official512-point target LCFS",
            "max_absolute_error_float64": max64, "max_absolute_error_float32": max32,
            "scale_offset_invariance_passed": True, "finite_nonzero_gradient_passed": True,
            "validation_labels_read": False, "audit_labels_read": False,
            "scorer_sha256": fingerprint()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/li_full_v1"))
    parser.add_argument("--frames-per-shot", type=int, default=8)
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames_per_shot <= 8:
        parser.error("frames-per-shot must lie between1 and8")
    torch.set_num_threads(1)
    validate_splits(args.data_dir)
    entries = split_entries(args.data_dir, "train")
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz") as saved:
        R, Z, mask = saved["grid_R"], saved["grid_Z"], saved["mask_coarse"].astype(bool)
    audit = self_check(args.data_dir, entries, R, Z, mask)
    atomic_json(args.output / "self_check.json", audit)
    print("SELF_CHECK " + json.dumps({key: value for key, value in audit.items() if key != "identities"}), flush=True)
    if args.self_check_only:
        return
    selections, global_indices, shot_indices, cache_indices = [], [], [], []
    offset = 0
    for entry in entries:
        chosen = np.unique(np.linspace(0, entry["n_frames"] - 1,
                                        min(args.frames_per_shot, entry["n_frames"])).astype(np.int64))
        selections.append(chosen)
        global_indices.extend((chosen + offset).tolist())
        shot_indices.extend([entry["shot_index"]] * len(chosen))
        cache_indices.extend(chosen.tolist())
        offset += entry["n_frames"]
    n = len(global_indices)
    plan_hash = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    settings = {"version": 1, "source_split": "train", "source_data_dir": str(args.data_dir.resolve()),
                "source_training_entries_sha256": plan_hash, "scorer_version": official.SCORING_VERSION,
                "scorer_sha256": fingerprint(), "frames_per_shot": args.frames_per_shot,
                "training_shots": len(entries), "total_training_frames": offset,
                "selected_training_frames": n, "contour_points": official.N_POINTS,
                "selection": "Uniform existing cached frames within each training shot; no target interpolation.",
                "inside_storage": "np.packbits flattened Z,R grid, bitorder=little; uint8",
                "contour_storage": "Exact official512-point contour, float64 physical R,Z; no further resampling.",
                "supervision_limit": "Fixed GT contour is a differentiable training surrogate; predicted contour motion is not included. Official validation remains necessary.",
                "validation_labels_read": False, "audit_labels_read": False}
    shapes = {"global_frame_indices": ((n,), "int64"), "shot_indices": ((n,), "int64"),
              "cache_frame_indices": ((n,), "int32"), "valid": ((n,), "bool"),
              "inside_packed": ((n, (len(R) * len(Z) + 7) // 8), "uint8"),
              "contours": ((n, official.N_POINTS, 2), "float64"),
              "segment_lengths": ((n, official.N_POINTS), "float64"), "li": ((n,), "float64")}
    metadata_path = args.output / "metadata.json"
    first_shot, next_row = 0, 0
    if args.resume and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for key, value in settings.items():
            if metadata.get(key) != value:
                raise ValueError(f"li target resume settings differ: {key}")
        if metadata.get("complete"):
            print("Training li targets already complete", flush=True)
            return
        first_shot, next_row = metadata["completed_shots"], metadata["completed_rows"]
        arrays = {name: np.lib.format.open_memmap(args.output / f"{name}.npy", mode="r+") for name in shapes}
    else:
        if metadata_path.exists():
            raise ValueError("Output exists; use --resume or choose a new directory")
        arrays = {name: np.lib.format.open_memmap(args.output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
                  for name, (shape, dtype) in shapes.items()}
        arrays["global_frame_indices"][:] = global_indices
        arrays["shot_indices"][:] = shot_indices
        arrays["cache_frame_indices"][:] = cache_indices
        arrays["valid"][:] = False
        arrays["li"][:] = np.nan
        np.save(args.output / "R.npy", R)
        np.save(args.output / "Z.npy", Z)
        metadata = {**settings, "complete": False, "completed_shots": 0, "completed_rows": 0,
                    "started_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(metadata_path, metadata)
    started = time.monotonic()
    for shot_position in range(first_shot, len(entries)):
        entry, chosen = entries[shot_position], selections[shot_position]
        with np.load(args.data_dir / entry["path"], allow_pickle=False) as shot:
            if not np.array_equal(shot["R"], R) or not np.array_equal(shot["Z"], Z):
                raise ValueError("Training grid differs from official envelope grid")
            fields = shot["psi"]
            for frame in chosen:
                arrays["valid"][next_row] = False
                arrays["li"][next_row] = np.nan
                value = reference(fields[frame], R, Z, mask)
                if value is not None:
                    arrays["inside_packed"][next_row] = np.packbits(value["inside"].ravel(), bitorder="little")
                    arrays["contours"][next_row] = value["contour"]
                    arrays["segment_lengths"][next_row] = value["segment_lengths"]
                    arrays["li"][next_row] = value["li"]
                    arrays["valid"][next_row] = True
                next_row += 1
        if (shot_position + 1) % 100 == 0 or shot_position + 1 == len(entries):
            for array in arrays.values():
                array.flush()
            metadata.update(completed_shots=shot_position + 1, completed_rows=next_row,
                            valid_references=int(arrays["valid"][:next_row].sum()),
                            updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(metadata_path, metadata)
            print(json.dumps({key: metadata[key] for key in ("completed_shots", "completed_rows", "valid_references")}
                             | {"run_elapsed_seconds": time.monotonic() - started}), flush=True)
    valid_li = arrays["li"][arrays["valid"]]
    if len(valid_li) < 2 or not np.isfinite(valid_li).all():
        raise ValueError("Insufficient finite training li references")
    metadata.update(complete=True, li_mean=float(valid_li.mean()), li_variance=float(max(valid_li.var(), 1e-8)),
                    valid_references=len(valid_li), invalid_references=n - len(valid_li),
                    finished_utc=datetime.now(timezone.utc).isoformat(), self_check=audit)
    atomic_json(metadata_path, metadata)
    print("COMPLETE " + json.dumps({key: value for key, value in metadata.items() if key != "self_check"}), flush=True)


if __name__ == "__main__":
    main()
