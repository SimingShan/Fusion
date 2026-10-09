#!/usr/bin/env python3
"""View C: append linear vacuum-response (coil Green's function) features to the history view.

For every cached frame the DIII-D coil flux field and the Gaussian plasma-blob
field are evaluated with ``fusion.physics_fields`` from the frame's own input
currents (the first 21 ``magnetics_*`` columns of X, kA), projected onto the
frozen training POD basis of the source POD directory (in the psi convention,
i.e. ``-phi`` is projected, nothing subtracted) and stored, together with four
scalar summaries, as 260 extra features.  The sidecar of each shot keeps the
relative name of the history sidecar (``history/<split>/shot_XXXXX.npy``) and
holds ``[210 history | 260 cview]`` columns, so the manifest split entries are
byte-identical to the history view (fixed-contour target stores keyed on the
training entries remain valid) and no trainer code changes are needed.

No target flux map is read by this script; consolidated target arrays are
hard-linked from the base view.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.physics_fields import (TABLE_DIR, envelope_centroid, external_field, load_calibration,  # noqa: E402
                                   load_envelope, plasma_blob_table, shot_ip_sign)
from fusion.pod import manifest_hash, read_manifest, standardize_x, validate_splits  # noqa: E402
from scripts.optimization.prepare_scaled_features import fit_statistics, _link_or_copy  # noqa: E402

HISTORY_FEATURES = 210
POD_RANK = 128
CVIEW_FEATURES = 2 * POD_RANK + 4
BLOB_SIGMA = 0.25
TABLE_HASH = "86ed077ed771c8e4"
COIL_NAMES = ["magnetics_ECOILA"] + [f"magnetics_F{i}{s}" for i in range(1, 10) for s in "AB"]
SCALAR_NAMES = ["cview_phi_ext_centroid", "cview_coil_field_rms", "cview_blob_amplitude", "cview_ecoila_gauge"]


def cview_feature_names():
    return ([f"cview_coil_pod_{i:03d}" for i in range(POD_RANK)]
            + [f"cview_blob_pod_{i:03d}" for i in range(POD_RANK)] + SCALAR_NAMES)


class CviewEngine:
    """Coil/blob response tables + POD projection for one machine grid."""

    def __init__(self, source_pod, base_feature_names, table_path=None, calibration=None):
        table_path = Path(table_path) if table_path else TABLE_DIR / f"diii_d_{TABLE_HASH}.npz"
        with np.load(table_path, allow_pickle=False) as data:
            self.tables = {k: np.asarray(data[k], np.float64) for k in data.files if k != "__meta__"}
        if sorted(self.tables) != sorted(COIL_NAMES):
            raise ValueError("Response table coil set differs from the expected 19 DIII-D signals")
        calibration = load_calibration() if calibration is None else calibration
        self.turns = float(calibration["ecoila_effective_turns"])
        self.multipliers = {"magnetics_ECOILA": self.turns}
        self.ecoila_unit_mean = float(self.tables["magnetics_ECOILA"].mean())
        self.grid_R, self.grid_Z, _ = load_envelope("DIII-D")
        self.centre = envelope_centroid("DIII-D")
        self.blob = plasma_blob_table((self.grid_R, self.grid_Z), *self.centre, BLOB_SIGMA)
        self.ir = int(np.argmin(np.abs(self.grid_R - self.centre[0])))
        self.iz = int(np.argmin(np.abs(self.grid_Z - self.centre[1])))
        with np.load(Path(source_pod) / "normalization.npz", allow_pickle=False) as saved:
            self.basis = np.asarray(saved["basis"], np.float64)
            self.coefficient_std = np.asarray(saved["coefficient_std"], np.float64)
        if self.basis.shape != (POD_RANK, 4225):
            raise ValueError("Unexpected POD basis shape")
        names = list(base_feature_names)
        self.columns = {k: names.index(k) for k in COIL_NAMES}
        self.ip_column = names.index("magnetics_plasma_current")

    def fields(self, X, R=None, Z=None):
        """Signed coil field and blob field (Wb/rad, axis-max phi convention) for cached frames."""
        if R is not None and (not np.allclose(R, self.grid_R) or not np.allclose(Z, self.grid_Z)):
            raise ValueError("Shot grid differs from the DIII-D envelope grid")
        X = np.asarray(X, np.float64)
        Ip = X[:, self.ip_column]
        sign = shot_ip_sign(Ip)
        currents = {k: X[:, i] for k, i in self.columns.items()}
        zero_blob = np.zeros_like(self.blob)
        phi_coils = external_field(self.tables, currents, Ip, zero_blob, coil_multipliers=self.multipliers, ip_sign=sign)
        phi_blob = np.abs(Ip)[:, None, None] * self.blob
        return phi_coils, phi_blob, Ip, sign

    def features(self, X, R=None, Z=None):
        phi_coils, phi_blob, Ip, sign = self.fields(X, R, Z)
        n = len(X)
        # psi convention (axis minimum): project -phi; nothing is subtracted.
        coil_coefficients = (-phi_coils.reshape(n, -1) @ self.basis.T) / self.coefficient_std
        blob_coefficients = (-phi_blob.reshape(n, -1) @ self.basis.T) / self.coefficient_std
        scalars = np.stack([
            phi_coils[:, self.iz, self.ir] + phi_blob[:, self.iz, self.ir],
            np.sqrt(np.mean(np.square(phi_coils.reshape(n, -1)), axis=1)),
            phi_blob[:, self.iz, self.ir],
            sign * X[:, self.columns["magnetics_ECOILA"]].astype(np.float64) * self.turns * self.ecoila_unit_mean,
        ], axis=1)
        out = np.concatenate([coil_coefficients, blob_coefficients, scalars], axis=1).astype(np.float32)
        if out.shape != (n, CVIEW_FEATURES) or not np.isfinite(out).all():
            raise ValueError("Invalid view-C features")
        return out

    def configuration(self):
        return {"table_hash": TABLE_HASH, "ecoila_effective_turns": self.turns, "blob_sigma_m": BLOB_SIGMA,
                "blob_centre_RZ": list(self.centre), "centroid_node_izir": [self.iz, self.ir],
                "phi_convention": "phi_ext = sign(Ip)*sum_c G_c I_c + |Ip| G_blob (axis-max); projected field = -phi (psi convention), nothing subtracted",
                "projection": "coefficients = (-phi @ basis.T) / coefficient_std of the source POD",
                "ip_sign": "shot_ip_sign over the cached frames' magnetics_plasma_current column",
                "scalars": SCALAR_NAMES}


_ENGINE = None


def _engine(task):
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CviewEngine(task["source_pod"], task["base_feature_names"])
    return _ENGINE


def _json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _task(task):
    entry = task["entry"]
    extra_path = Path(task["output"]) / entry["X_extra"]
    width = HISTORY_FEATURES + CVIEW_FEATURES
    if extra_path.exists():
        array = np.load(extra_path, mmap_mode="r", allow_pickle=False)
        if array.shape == (entry["n_frames"], width) and array.dtype == np.float32:
            return task["split"], entry["shot_index"]
    history = np.load(Path(task["base_view"]) / entry["X_extra"], allow_pickle=False)
    with np.load(entry["path"], allow_pickle=False) as original:
        # Only inputs are read: X (currents), R, Z. psi/scalars are never touched.
        X, R, Z = original["X"], original["R"], original["Z"]
    if history.shape != (len(X), HISTORY_FEATURES) or len(X) != entry["n_frames"]:
        raise ValueError("History sidecar / base cache frame counts disagree")
    values = np.concatenate([history.astype(np.float32), _engine(task).features(X, R, Z)], axis=1)
    extra_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = extra_path.with_name(extra_path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, extra_path)
    return task["split"], entry["shot_index"]


def _consolidate(base_view, output, manifest, split):
    entries = manifest[split]
    base_path = base_view / "consolidated" / f"{split}_X.npy"
    base_X = np.load(base_path, mmap_mode="r", allow_pickle=False)
    n = sum(entry["n_frames"] for entry in entries)
    width = len(manifest["feature_names"])
    if base_X.shape != (n, width - CVIEW_FEATURES):
        raise ValueError("Base-view consolidated features disagree with the preserved shot/frame selection")
    root = output / "consolidated"
    root.mkdir(exist_ok=True)
    temporary = root / f"{split}_X.partial.npy"
    X = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(n, width))
    offset = 0
    for entry in entries:
        end = offset + entry["n_frames"]
        extra = np.load(output / entry["X_extra"], allow_pickle=False)
        X[offset:end, :-CVIEW_FEATURES] = base_X[offset:end]
        if not np.array_equal(base_X[offset:end, -HISTORY_FEATURES:], extra[:, :HISTORY_FEATURES]):
            raise ValueError("History columns of the sidecar differ from the base view")
        X[offset:end, -CVIEW_FEATURES:] = extra[:, HISTORY_FEATURES:]
        offset = end
    X.flush()
    del X
    os.replace(temporary, root / f"{split}_X.npy")
    for original in (base_view / "consolidated").glob(f"{split}_*"):
        if original.name != f"{split}_X.npy" and ".partial." not in original.name:
            _link_or_copy(original, root / original.name)
    return {"n_frames": n, "n_shots": len(entries), "feature_count": width,
            "X": f"consolidated/{split}_X.npy", "targets_reused_unchanged": True}


def _prepare_pod(output, source_pod, pod_output, manifest, batch_size):
    """Same recipe as prepare_robust_features.prepare, with the 260 new raw features (log scale 1)."""
    source_meta = json.loads((source_pod / "basis.json").read_text())
    names = manifest["feature_names"]
    with np.load(source_pod / "normalization.npz", allow_pickle=False) as saved:
        norm = {key: saved[key].copy() for key in saved.files}
    base_width = len(names) - CVIEW_FEATURES
    scales = np.concatenate((norm.get("raw_log_scale", np.ones(base_width)), np.ones(CVIEW_FEATURES)))
    if len(scales) != len(names):
        raise ValueError("Source log-scale definition differs from base features")
    X = np.load(output / "consolidated/train_X.npy", mmap_mode="r", allow_pickle=False)
    if len(X) != source_meta["training_frames"]:
        raise ValueError("View C changed the training frame selection")
    mean, std = fit_statistics(X, scales, batch_size)
    masks = np.asarray(["observed" in name or name.endswith("_coverage") for name in names])
    dropout_masks = np.asarray([name.startswith(("thomson_core_Te_", "thomson_core_ne_", "thomson_edge_Te_", "thomson_edge_ne_")) for name in names]) & masks
    active = np.flatnonzero((std > 1e-6) | (dropout_masks & (mean > 0)))
    mean[masks], std[masks] = 0, 1
    norm.update(raw_log_scale=scales, feature_indices=active,
                x_mean=mean[active].astype(np.float32), x_std=std[active].astype(np.float32))
    pod_output.mkdir(parents=True, exist_ok=True)
    temporary = pod_output / "normalization.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **norm)
    os.replace(temporary, pod_output / "normalization.npz")
    for split in ("train", "val", "legacy_val"):
        raw = X if split == "train" else np.load(output / "consolidated" / f"{split}_X.npy", mmap_mode="r", allow_pickle=False)
        temporary = pod_output / f"{split}_X.partial.npy"
        normalized = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(raw), len(active)))
        for first in range(0, len(raw), batch_size):
            normalized[first:first + batch_size] = standardize_x(raw[first:first + batch_size], norm)
        normalized.flush()
        del normalized
        os.replace(temporary, pod_output / f"{split}_X.npy")
        for key in ("coefficients", "scalars"):
            _link_or_copy(source_pod / f"{split}_{key}.npy", pod_output / f"{split}_{key}.npy")
        print(f"View C POD features normalized: {split} {len(raw)} frames", flush=True)
    if (source_pod / "basis_sample_indices.npy").exists():
        _link_or_copy(source_pod / "basis_sample_indices.npy", pod_output / "basis_sample_indices.npy")
    metadata = {**source_meta, "source_manifest_sha256": manifest_hash(output), "data_dir": str(output),
                "input_features": len(names), "active_features": len(active),
                "feature_transform": {"name": "physical_scaled_Thomson_plus_current_history_plus_vacuum_response",
                                      "fit_split": "train", "complete": True},
                "cview": {**manifest["feature_augmentation"]["cview"], "added_raw_features": CVIEW_FEATURES,
                          "source_pod": str(source_pod), "source_basis_metadata": source_meta,
                          "source_normalization_sha256": hashlib.sha256((source_pod / "normalization.npz").read_bytes()).hexdigest(),
                          "basis_refitted": False, "target_arrays_reused_unchanged": True,
                          "active_availability_features": int(masks[active].sum()),
                          "audit_features_prepared": False, "audit_labels_read": False}}
    _json(pod_output / "basis.json", metadata)
    return metadata


def prepare(args):
    started = time.monotonic()
    base_view, output = args.base_view.resolve(), args.output.resolve()
    if base_view == output:
        raise ValueError("View C requires a separate view directory")
    base = read_manifest(base_view)
    augmentation = base.get("feature_augmentation")
    if not augmentation or augmentation.get("extra_feature_count") != HISTORY_FEATURES or "cview" in augmentation:
        raise ValueError("Supply the history view (210 history sidecar features) as the base view")
    base_names = base["feature_names"][:augmentation["base_feature_count"]]
    engine = CviewEngine(args.source_pod.resolve(), base_names)
    configuration = {**engine.configuration(), "base_view": str(base_view), "base_view_manifest_sha256": manifest_hash(base_view),
                     "source_pod": str(args.source_pod.resolve()),
                     "source_normalization_sha256": hashlib.sha256((args.source_pod / "normalization.npz").read_bytes()).hexdigest(),
                     "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(output)
        if manifest["feature_augmentation"]["cview"] != configuration:
            raise ValueError("Existing view-C cache has a different source/configuration")
    else:
        manifest = json.loads(json.dumps(base))
        manifest["feature_names"] = base["feature_names"] + cview_feature_names()
        manifest["feature_augmentation"] = {**augmentation, "cview": configuration,
                                             "extra_feature_count": HISTORY_FEATURES + CVIEW_FEATURES,
                                             "history_feature_count": HISTORY_FEATURES, "cview_feature_count": CVIEW_FEATURES,
                                             "sidecar_layout": "history/<split>/shot_XXXXX.npy = [210 history | 260 cview] float32",
                                             "complete_splits": {}, "target_values_read": False}
        manifest["consolidated"] = {}
        manifest["complete_splits"] = {split: False for split in ("train", "val", "legacy_val", "audit", "test")}
        # Split entries (paths, X_extra names) are kept byte-identical to the base view.
        _json(manifest_path, manifest)
    for split in ("train", "val", "legacy_val", "audit", "test"):
        if manifest.get(split) != base.get(split):
            raise ValueError(f"Split entries of {split} differ from the base view")
    tasks = [{"output": str(output), "base_view": str(base_view), "source_pod": str(args.source_pod.resolve()),
              "base_feature_names": base_names, "split": split, "entry": entry}
             for split in args.splits for entry in manifest[split]
             if not (output / entry["X_extra"]).exists()]
    if args.limit_shots:
        tasks = tasks[:args.limit_shots]
    print(f"View C feature tasks: {len(tasks)}, workers={args.workers}, added_features={CVIEW_FEATURES}", flush=True)
    completed = 0
    if args.workers == 1:
        for task in tasks:
            _task(task)
            completed += 1
    else:
        iterator = iter(tasks)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {}
            for _ in range(min(len(tasks), args.workers * 2)):
                task = next(iterator)
                pending[executor.submit(_task, task)] = task
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    future.result()
                    completed += 1
                    if completed % 256 == 0:
                        print(f"Prepared view C: {completed}/{len(tasks)} new shots ({time.monotonic()-started:.1f}s)", flush=True)
                    task = next(iterator, None)
                    if task is not None:
                        pending[executor.submit(_task, task)] = task
    for split in args.splits:
        complete = bool(manifest[split]) and all((output / entry["X_extra"]).exists() for entry in manifest[split])
        manifest["feature_augmentation"]["complete_splits"][split] = complete
        manifest["complete_splits"][split] = complete
        if complete and not args.limit_shots:
            manifest["consolidated"][split] = _consolidate(base_view, output, manifest, split)
    if (base_view / "scoring_cache").is_dir():
        (output / "scoring_cache").mkdir(exist_ok=True)
        for cached in (base_view / "scoring_cache").iterdir():
            if cached.is_file():
                _link_or_copy(cached, output / "scoring_cache" / cached.name)
    _json(manifest_path, manifest)
    if not args.no_pod:
        if not all(manifest["feature_augmentation"]["complete_splits"].get(split) for split in ("train", "val", "legacy_val")):
            raise ValueError("Prepare complete train/val/legacy sidecars before POD normalization")
        validate_splits(output)
        _prepare_pod(output, args.source_pod.resolve(), args.pod_output.resolve(), manifest, args.batch_size)
    print(json.dumps({"output": str(output), "complete_splits": manifest["feature_augmentation"]["complete_splits"],
                      "elapsed_seconds": time.monotonic() - started}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-view", type=Path, default=Path("data/advanced_history_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/advanced_cview_v1"))
    parser.add_argument("--source-pod", type=Path, default=Path("data/pod_history128_robust_v4"))
    parser.add_argument("--pod-output", type=Path, default=Path("data/pod_cview_v1"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "legacy_val", "test"), default=["legacy_val", "val", "train"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-shots", type=int, default=0)
    parser.add_argument("--no-pod", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.limit_shots < 0:
        parser.error("Workers/batch size must be positive and shot limit nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
