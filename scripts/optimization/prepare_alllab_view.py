#!/usr/bin/env python3
"""Build the all-labeled B-family training view (train + native-dev val + legacy_val).

Creates three NEW directories and never writes into an existing one:

  data/advanced_history_alllab_v1   manifest whose train split = original 6,753 train entries
                                    followed by the 128 val and 32 legacy_val entries (same shot
                                    files, shot_index renumbered +1,000,000 so validate_splits
                                    accepts them alongside the unchanged val / legacy_val splits);
                                    history/ and consolidated/ (val, legacy_val only) are symlinks,
                                    scoring_cache/ is a copy.
  data/pod_history128_alllab_v1     train_* = concat(train_*, val_*, legacy_val_*) of
                                    pod_history128_robust_v4; normalization.npz and
                                    basis_sample_indices.npy copied unchanged; basis.json copied with
                                    only source_manifest_sha256/data_dir re-pointed (+ provenance).
  data/geometry_history_alllab_v1   geometry_history_all_v5 rows followed by rows for every appended
                                    dev frame, computed with the exact generator code of
                                    results/training/c1_sources_070 (prepare_li_targets.reference,
                                    prepare_geometry_targets.axis_reference + boundary sampling).

The audit split is never read: its manifest entries are dropped without being inspected.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[2])
GEN = ROOT / "results/training/c1_sources_070"  # exact code that produced li_full_all_v2 / geometry_history_all_v5
sys.path.insert(0, str(GEN))
from fusion.pod import read_manifest, split_entries, validate_splits, manifest_hash, standardize_x  # noqa: E402
from fusion.li_loss import FixedContourLiLoss, contour_to_sampling_grid  # noqa: E402
from fusion.geometry_loss import FixedGeometryLoss  # noqa: E402
from scripts.optimization.prepare_li_targets import reference, fingerprint  # noqa: E402
from scripts.optimization.prepare_geometry_targets import axis_reference, entries_hash  # noqa: E402

SRC_DATA = ROOT / "data/advanced_history_v2"
NATIVE_DATA = ROOT / "data/advanced_history_val_native_v3"
SRC_POD = ROOT / "data/pod_history128_robust_v4"
SRC_GEOM = ROOT / "data/geometry_history_all_v5"
OUT_DATA = ROOT / "data/advanced_history_alllab_v1"
OUT_POD = ROOT / "data/pod_history128_alllab_v1"
OUT_GEOM = ROOT / "data/geometry_history_alllab_v1"
OFFSET = 1_000_000
APPENDED_SPLITS = ("val", "legacy_val")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def shot_x(data_dir, entry):
    with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
        x = shot["X"]
    if entry.get("X_extra"):
        x = np.concatenate((x, np.load(data_dir / entry["X_extra"], allow_pickle=False)), axis=1)
    return x


# ----------------------------------------------------------------------------- data dir
def build_data_dir():
    m = read_manifest(SRC_DATA)
    out = {}
    for key, value in m.items():
        if key in ("audit", "previously_used_indices"):
            continue  # dropped unread: audit entries / ids never enter the new view
        out[key] = copy.deepcopy(value)
    out["selection"] = {k: v for k, v in out["selection"].items() if k != "audit"}
    out["complete_splits"] = {k: v for k, v in out["complete_splits"].items() if k != "audit"}
    appended = []
    for split in APPENDED_SPLITS:
        for e in m[split]:
            e2 = dict(e)
            e2["shot_index"] = OFFSET + e["shot_index"]
            e2["alllab_source"] = {"split": split, "shot_index": e["shot_index"]}
            appended.append(e2)
    out["train"] = m["train"] + appended
    out["selection"]["train"] = list(out["selection"]["train"]) + [e["alllab_source"]["shot_index"] for e in appended]
    out["consolidated"] = {k: v for k, v in out["consolidated"].items() if k != "train"}
    out["alllab"] = {
        "version": 1, "created_utc": now(),
        "source_manifest": str(SRC_DATA / "manifest.json"), "source_manifest_sha256": manifest_hash(SRC_DATA),
        "train_order": "original train entries (6753) then val entries (128, original val order) then legacy_val entries (32)",
        "appended_shot_index_rule": f"shot_index = {OFFSET} + original shot_index; original in entry['alllab_source']",
        "val_and_legacy_val_in_sample": True,
        "train_shots": len(out["train"]), "train_frames": sum(e["n_frames"] for e in out["train"]),
        "audit_split": "omitted, never read", "consolidated_train": "absent (per-shot fallback); val/legacy_val symlinked",
        "native_dev_source": str(NATIVE_DATA)}
    OUT_DATA.mkdir(parents=True, exist_ok=False)
    (OUT_DATA / "history").mkdir()
    for split in ("train", "val", "legacy_val"):
        os.symlink(SRC_DATA / "history" / split, OUT_DATA / "history" / split)
    (OUT_DATA / "consolidated").mkdir()
    for path in sorted((SRC_DATA / "consolidated").glob("*.npy")):
        if path.name.startswith(("val_", "legacy_val_")):
            os.symlink(path, OUT_DATA / "consolidated" / path.name)
    shutil.copytree(SRC_DATA / "scoring_cache", OUT_DATA / "scoring_cache")
    (OUT_DATA / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    validate_splits(OUT_DATA)
    print(f"data dir: {len(out['train'])} train shots, {out['alllab']['train_frames']} frames", flush=True)
    return m, out, appended


# ----------------------------------------------------------------------------- checks
def check_alignment(m, appended, norm):
    """Recompute pod rows from shot files: proves row order = manifest entry order, frame order = file order."""
    report = {}
    def compare(entries, split, data_dir, label):
        X = np.load(SRC_POD / f"{split}_X.npy", mmap_mode="r")
        C = np.load(SRC_POD / f"{split}_coefficients.npy", mmap_mode="r")
        S = np.load(SRC_POD / f"{split}_scalars.npy", mmap_mode="r")
        offsets = np.concatenate(([0], np.cumsum([e["n_frames"] for e in split_entries(SRC_DATA, split)])))
        if offsets[-1] != len(X):
            raise ValueError(f"{split}: manifest frames {offsets[-1]} != pod rows {len(X)}")
        position = {e["shot_index"]: i for i, e in enumerate(split_entries(SRC_DATA, split))}
        mx = mc = ms = 0.0
        for e in entries:
            i = position[e["shot_index"]]
            sl = slice(offsets[i], offsets[i + 1])
            with np.load(data_dir / e["path"], allow_pickle=False) as shot:
                psi, scal = shot["psi"], shot["scalars"]
            x = standardize_x(shot_x(data_dir, e), norm)
            c = (psi.reshape(-1, 4225) - norm["psi_mean"].reshape(-1)) @ norm["basis"].T / norm["coefficient_std"]
            s = (scal - norm["scalar_mean"]) / norm["scalar_std"]
            mx = max(mx, float(np.nanmax(np.abs(x - X[sl]))))
            mc = max(mc, float(np.nanmax(np.abs(c - C[sl]))))
            ms = max(ms, float(np.nanmax(np.abs(s - S[sl]))))
        report[label] = {"shots": len(entries), "max_abs_X": mx, "max_abs_coefficients": mc, "max_abs_scalars": ms}
        if mx > 1e-3 or mc > 1e-3 or ms > 1e-4:
            raise ValueError(f"Pod rows do not match shot files for {label}: {report[label]}")
    tr = m["train"]
    compare([tr[0], tr[1], tr[len(tr) // 2], tr[-1]], "train", SRC_DATA, "train_sample")
    compare(m["val"], "val", SRC_DATA, "val_all")
    compare(m["legacy_val"], "legacy_val", SRC_DATA, "legacy_val_all")
    # native-dev provenance: subsampled val frames are exact copies of native_v3 frames
    native = {e["shot_index"]: e for e in split_entries(NATIVE_DATA, "val")}
    worst = 0.0
    for e in m["val"]:
        n = native[e["shot_index"]]
        if n["shot_id"] != e["shot_id"]:
            raise ValueError("native-dev shot id mismatch")
        with np.load(SRC_DATA / e["path"]) as a, np.load(NATIVE_DATA / n["path"]) as b:
            pos = np.searchsorted(b["frame_indices"], a["frame_indices"])
            if not np.array_equal(b["frame_indices"][pos], a["frame_indices"]):
                raise ValueError("val frame not present in native-dev shot")
            for key in ("psi", "scalars", "X", "times"):
                if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])[pos], equal_nan=True):
                    raise ValueError(f"val {key} differs from native-dev for {e['shot_id']}")
        xa = np.load(SRC_DATA / e["X_extra"])
        xb = np.load(NATIVE_DATA / n["X_extra"])[pos]
        worst = max(worst, float(np.nanmax(np.abs(xa - xb))) if xa.size else 0.0)
    report["native_dev_provenance"] = {"shots": len(m["val"]), "psi_scalars_X_times_identical": True,
                                       "history_sidecar_max_abs_diff": worst}
    print(json.dumps(report), flush=True)
    return report


# ----------------------------------------------------------------------------- pod dir
def build_pod_dir(new_manifest_hash, alignment):
    OUT_POD.mkdir(parents=True, exist_ok=False)
    for name in ("normalization.npz", "basis_sample_indices.npy"):
        shutil.copy2(SRC_POD / name, OUT_POD / name)
    for split in APPENDED_SPLITS:
        for key in ("X", "coefficients", "scalars"):
            shutil.copy2(SRC_POD / f"{split}_{key}.npy", OUT_POD / f"{split}_{key}.npy")
    counts = {}
    for key in ("X", "coefficients", "scalars"):
        parts = [np.load(SRC_POD / f"{s}_{key}.npy", mmap_mode="r") for s in ("train",) + APPENDED_SPLITS]
        n = sum(len(p) for p in parts)
        out = np.lib.format.open_memmap(OUT_POD / f"train_{key}.npy", mode="w+", dtype=parts[0].dtype,
                                        shape=(n,) + parts[0].shape[1:])
        row = 0
        for p in parts:
            for first in range(0, len(p), 65536):
                block = np.asarray(p[first:first + 65536])
                out[row:row + len(block)] = block
                row += len(block)
        out.flush()
        del out
        counts[key] = n
    basis = json.loads((SRC_POD / "basis.json").read_text())
    basis["alllab"] = {"original_source_manifest_sha256": basis["source_manifest_sha256"],
                       "original_data_dir": basis["data_dir"], "source_pod": str(SRC_POD),
                       "train_rows": counts["X"], "train_row_order": "train (426684) | val (4068) | legacy_val (1024)",
                       "normalization_and_basis": "copied unchanged; fit on the original 6753 train shots only",
                       "val_legacy_val_in_sample": True, "alignment_check": alignment, "created_utc": now()}
    basis["source_manifest_sha256"] = new_manifest_hash
    basis["data_dir"] = str(OUT_DATA)
    (OUT_POD / "basis.json").write_text(json.dumps(basis, indent=2) + "\n")
    for name in ("normalization.npz", "basis_sample_indices.npy"):
        if sha(SRC_POD / name) != sha(OUT_POD / name):
            raise ValueError(f"{name} copy differs")
    print(json.dumps({"pod_train_rows": counts}), flush=True)
    return counts


# ----------------------------------------------------------------------------- geometry store
def build_geometry(new_train, appended, n_old):
    meta_old = json.loads((SRC_GEOM / "metadata.json").read_text())
    if entries_hash(new_train[:len(new_train) - len(appended)]) != meta_old["source_training_entries_sha256"]:
        raise ValueError("Existing geometry store is not aligned to the original train entries")
    if meta_old["total_training_frames"] != n_old or meta_old["scorer_sha256"] != fingerprint():
        raise ValueError("Geometry store frame count or scorer fingerprint differs")
    gen_sha = sha(GEN / "scripts/optimization/prepare_geometry_targets.py")
    if gen_sha != meta_old["geometry"]["source_code_sha256"]:
        raise ValueError("Geometry generator code differs from the one that built v5")
    R, Z = np.load(SRC_GEOM / "R.npy"), np.load(SRC_GEOM / "Z.npy")
    with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask_bool = saved["mask_coarse"].astype(bool)   # as in prepare_li_targets
        mask_raw = saved["mask_coarse"]                 # as in prepare_geometry_targets
        if not np.array_equal(saved["grid_R"], R) or not np.array_equal(saved["grid_Z"], Z):
            raise ValueError("grid mismatch")
    names = ("global_frame_indices", "shot_indices", "cache_frame_indices", "valid", "inside_packed", "contours",
             "segment_lengths", "li", "axis_indices", "axis_flux", "axis_response", "boundary_flux",
             "boundary_gradient_floor")
    old = {k: np.load(SRC_GEOM / f"{k}.npy", mmap_mode="r") for k in names}
    if len(old["li"]) != n_old or not np.array_equal(np.asarray(old["global_frame_indices"]), np.arange(n_old)):
        raise ValueError("Existing store is not one row per training frame in order")
    # plan new rows exactly like prepare_li_targets --frames-per-shot 64
    plan, offset = [], n_old
    for e in appended:
        chosen = np.unique(np.linspace(0, e["n_frames"] - 1, min(64, e["n_frames"])).astype(np.int64))
        plan.append((e, chosen, offset))
        offset += e["n_frames"]
    n_new = sum(len(c) for _, c, _ in plan)
    n = n_old + n_new
    OUT_GEOM.mkdir(parents=True, exist_ok=False)
    arrays = {k: np.lib.format.open_memmap(OUT_GEOM / f"{k}.npy", mode="w+", dtype=old[k].dtype,
                                           shape=(n,) + old[k].shape[1:]) for k in names}
    for k in names:
        for first in range(0, n_old, 32768):
            last = min(first + 32768, n_old)
            arrays[k][first:last] = old[k][first:last]
    print(f"Copied {n_old} existing geometry rows", flush=True)
    np.save(OUT_GEOM / "R.npy", R)
    np.save(OUT_GEOM / "Z.npy", Z)
    row = n_old
    new_axes, new_fields = [], {}
    for e, chosen, base in plan:
        with np.load(SRC_DATA / e["path"], allow_pickle=False) as shot:
            if not np.array_equal(shot["R"], R) or not np.array_equal(shot["Z"], Z):
                raise ValueError("grid mismatch")
            fields = shot["psi"]
        for frame in chosen:
            arrays["global_frame_indices"][row] = base + frame
            arrays["shot_indices"][row] = e["shot_index"]
            arrays["cache_frame_indices"][row] = frame
            arrays["valid"][row] = False
            arrays["li"][row] = np.nan
            value = reference(fields[frame], R, Z, mask_bool)
            if value is not None:
                arrays["inside_packed"][row] = np.packbits(value["inside"].ravel(), bitorder="little")
                arrays["contours"][row] = value["contour"]
                arrays["segment_lengths"][row] = value["segment_lengths"]
                arrays["li"][row] = value["li"]
                arrays["valid"][row] = True
                new_fields[row] = np.asarray(fields[frame], dtype=np.float64)
            row += 1
    rows = np.array(sorted(new_fields), dtype=np.int64)
    # geometry, same batch code as prepare_geometry_targets.prepare
    for first in range(0, len(rows), 256):
        selected = rows[first:first + 256]
        fields = np.stack([new_fields[r] for r in selected])
        axis = [axis_reference(p, R, Z, mask_raw) for p in fields]
        for key, position in (("axis_indices", 0), ("axis_flux", 1), ("axis_response", 2)):
            arrays[key][selected] = np.stack([a[position] for a in axis])
        new_axes.extend(a[3] for a in axis)
        grid = torch.as_tensor(contour_to_sampling_grid(arrays["contours"][selected], R, Z))[:, :, None]
        field = torch.as_tensor(fields)
        gz, gr = torch.gradient(field, spacing=(torch.as_tensor(Z), torch.as_tensor(R)), dim=(1, 2))
        gradient = (gr.square() + gz.square()).sqrt()
        boundary_flux = F.grid_sample(field[:, None], grid, align_corners=True, padding_mode="border")[:, 0, :, 0].numpy()
        boundary_gradient = F.grid_sample(gradient[:, None], grid, align_corners=True, padding_mode="border")[:, 0, :, 0].numpy()
        gradient_floor = np.maximum(boundary_gradient, 0.1 * boundary_gradient.mean(1, keepdims=True))
        arrays["boundary_flux"][selected] = boundary_flux
        arrays["boundary_gradient_floor"][selected] = np.maximum(gradient_floor, 1e-8)
    for a in arrays.values():
        a.flush()
    # self checks on new rows
    chosen = rows[:20]
    truth = torch.as_tensor(np.stack([new_fields[r] for r in chosen]))
    li_targets = {"inside": torch.as_tensor(np.unpackbits(arrays["inside_packed"][chosen], axis=1, count=len(R) * len(Z),
                                                          bitorder="little").reshape(-1, len(Z), len(R))),
                  "sampling_grid": torch.as_tensor(contour_to_sampling_grid(arrays["contours"][chosen], R, Z)),
                  "segment_lengths": torch.as_tensor(np.array(arrays["segment_lengths"][chosen])),
                  "li": torch.as_tensor(np.array(arrays["li"][chosen]))}
    li_err = float((FixedContourLiLoss(R, Z, meta_old["li_variance"]).predict(truth, li_targets) - li_targets["li"]).abs().max())
    gtargets = {k: torch.as_tensor(np.array(arrays[k][chosen])) for k in
                ("axis_indices", "axis_flux", "axis_response", "boundary_flux", "boundary_gradient_floor")}
    gtargets.update(sampling_grid=li_targets["sampling_grid"], segment_lengths=li_targets["segment_lengths"])
    module = FixedGeometryLoss(np.asarray(meta_old["geometry"]["axis_variance"]))
    parts = {k: float(v) for k, v in module.parts(truth, gtargets).items()}
    if li_err > 1e-9 or max(parts.values()) > 1e-20:
        raise ValueError(f"New-row self check failed: li {li_err}, geometry {parts}")
    for k in names:  # spot-check copy of old rows
        idx = [0, n_old // 2, n_old - 1]
        if not np.array_equal(np.asarray(arrays[k][idx]), np.asarray(old[k][idx]), equal_nan=True):
            raise ValueError(f"old row copy differs: {k}")
    valid_all = np.asarray(arrays["valid"])
    li_all = np.asarray(arrays["li"])[valid_all]
    new_valid = int(valid_all[n_old:].sum())
    meta = copy.deepcopy(meta_old)
    meta.update({
        "source_data_dir": str(OUT_DATA), "source_training_entries_sha256": entries_hash(new_train),
        "training_shots": len(new_train), "total_training_frames": n, "selected_training_frames": n,
        "completed_shots": len(new_train), "completed_rows": n, "valid_references": int(valid_all.sum()),
        "invalid_references": int(n - valid_all.sum()), "updated_utc": now(), "finished_utc": now(),
        "validation_labels_read": True,
    })
    meta["geometry"] = dict(meta["geometry"], prepared_frames=int(valid_all.sum()), completed_utc=now(),
                            validation_labels_read=True)
    meta["alllab"] = {
        "source_store": str(SRC_GEOM), "source_store_metadata_sha256": sha(SRC_GEOM / "metadata.json"),
        "rows": "rows 0..426683 copied unchanged from source store; rows 426684.. are the appended val then legacy_val "
                "frames in manifest order (all cached frames, frames_per_shot=64 rule)",
        "appended_rows": n_new, "appended_valid": new_valid, "appended_invalid": n_new - new_valid,
        "generator_code": str(GEN), "generator_sha256": {
            "prepare_li_targets.py": sha(GEN / "scripts/optimization/prepare_li_targets.py"),
            "prepare_geometry_targets.py": gen_sha},
        "normalization_constants": "li_mean, li_variance and geometry.axis_variance kept FROZEN from the source store "
                                   "(train-only), so the B-family loss scale is unchanged",
        "recomputed_li_mean_all_valid": float(li_all.mean()),
        "recomputed_li_variance_all_valid": float(max(li_all.var(), 1e-8)),
        "appended_axis_variance": np.var(np.stack(new_axes), axis=0).tolist() if new_axes else None,
        "self_check_new_rows": {"frames": len(chosen), "li_max_abs_error_float64": li_err, "geometry_target_parts": parts},
        "val_legacy_val_labels_used_as_training_targets": True, "audit_labels_read": False, "created_utc": now()}
    (OUT_GEOM / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"geometry_rows": n, "appended_rows": n_new, "appended_valid": new_valid,
                      "li_err": li_err, "parts": parts}), flush=True)
    return meta


def main():
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    started = time.monotonic()
    with np.load(SRC_POD / "normalization.npz") as saved:
        norm = dict(saved)
    m = read_manifest(SRC_DATA)
    if json.loads((SRC_POD / "basis.json").read_text())["source_manifest_sha256"] != manifest_hash(SRC_DATA):
        raise ValueError("source pod not aligned to source data dir")
    if (OUT_DATA / "manifest.json").exists() and (OUT_POD / "basis.json").exists():
        # Resume: data/pod views already built and verified by an earlier run; rebuild geometry only.
        out = read_manifest(OUT_DATA)
        appended = out["train"][len(m["train"]):]
        assert json.loads((OUT_POD / "basis.json").read_text())["source_manifest_sha256"] == manifest_hash(OUT_DATA)
        counts = {"X": len(np.load(OUT_POD / "train_X.npy", mmap_mode="r"))}
    else:
        alignment = check_alignment(m, None, norm)
        m, out, appended = build_data_dir()
        counts = build_pod_dir(manifest_hash(OUT_DATA), alignment)
    n_old = sum(e["n_frames"] for e in m["train"])
    if counts["X"] != out["alllab"]["train_frames"]:
        raise ValueError("pod rows != manifest train frames")
    build_geometry(out["train"], appended, n_old)
    # Final check with the trainer's own store class
    sys.path.insert(0, str(ROOT / "results/training/muon128/src"))
    from fusion.geometry_loss import GeometryTargetStore  # noqa: E402  (GEN copy; same loader)
    store = GeometryTargetStore(OUT_GEOM)
    selected_hash = hashlib.sha256(json.dumps(split_entries(OUT_DATA, "train"), sort_keys=True).encode()).hexdigest()
    assert selected_hash == store.metadata["source_training_entries_sha256"]
    assert len(store.lookup) == counts["X"]
    print(json.dumps({"done": True, "seconds": time.monotonic() - started,
                      "li_supervised_frames": int((store.lookup >= 0).sum())}), flush=True)


if __name__ == "__main__":
    main()
