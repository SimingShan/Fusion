#!/usr/bin/env python3
"""Build data views + POD-input directory that append the 126 vessel-current proxy features (asinh-transformed, train-median scaled)
to the members' raw inputs, so the unchanged trainer (scale150/src/scripts/train_pod.py) can train and predict with them.
  train : data/advanced_history_vessel_v1  (copy of advanced_history_v2: manifest + names, per-shot X_extra = [history | vessel],
          consolidated/ symlinks except {train,val,legacy_val}_X = [old raw X | vessel])
          data/pod_history128_vessel_v1    (normalization with 126 extra active features; {split}_X = [old pod X | standardised vessel];
          coefficients/scalars/basis indices symlinked; basis.json re-hashed to the new manifest)
  native: data/advanced_history_val_native_vessel_v1 (per-shot sidecars + consolidated val_X)
  test  : data/advanced_history_test_vessel_v1 (per-shot sidecars + consolidated test_X)
Raw values written into X are already transformed (asinh(v / s)), raw_log_scale = 1 for the new columns (no further transform).
Inputs only; no labels are read beyond what the copied consolidated label arrays already were.   usage: make_vessel_view.py <train|native|test>"""
import hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); D = ROOT / "data"; F = ROOT / "results/analysis/vessel_196"
which = sys.argv[1]; t0 = time.monotonic()
NAMES = ["vessel_" + n for n in json.loads((F / "feats/feature_names.json").read_text())]
SCALE_PATH = F / "vessel_scale.json"


def raw_feats(sub, n_entries):
    return [np.load(F / sub / f"shot_{i:05d}.npy").astype(np.float64) for i in range(n_entries)]


def transform(parts, scale):
    return [np.arcsinh(p / scale).astype(np.float32) for p in parts]


def write_view(src, dst, splits_feats, scale, consolidated_splits):
    """splits_feats: {split: feats_subdir}. Copies the manifest with extended names and new sidecars; consolidated X for the given splits."""
    src, dst = D / src, D / dst; dst.mkdir(parents=True, exist_ok=True); man = json.loads((src / "manifest.json").read_text())
    assert len(man["feature_names"]) == 1348, len(man["feature_names"])
    man["feature_names"] = man["feature_names"] + NAMES
    man["vessel_augmentation"] = {"features": len(NAMES), "transform": "asinh(v / train_median_abs)", "scale_file": str(SCALE_PATH),
                                  "source_view": str(src), "builder": "results/analysis/vessel_196/{build_features,make_vessel_view}.py", "target_values_read": False}
    out_V = {}
    for split, sub in splits_feats.items():
        entries = man[split]; V = transform(raw_feats(sub, len(entries)), scale); out_V[split] = V
        (dst / "vessel_extra" / split).mkdir(parents=True, exist_ok=True)
        for i, (e, v) in enumerate(zip(entries, V)):
            old = np.load(src / e["X_extra"]) if e.get("X_extra") else np.zeros((len(v), 0), np.float32)
            assert len(old) == len(v) == e["n_frames"], (split, i, len(old), len(v), e["n_frames"])
            rel = f"vessel_extra/{split}/shot_{i:05d}.npy"; np.save(dst / rel, np.concatenate([old.astype(np.float32), v], 1)); e["X_extra"] = rel
            if not str(e["path"]).startswith("/"):
                e["path"] = str((src / e["path"]).resolve())
        print(f"{dst.name}: {split} sidecars written ({len(entries)} shots) [{time.monotonic()-t0:.0f}s]", flush=True)
    (dst / "manifest.json").write_text(json.dumps(man))
    cdst = dst / "consolidated"; cdst.mkdir(exist_ok=True); csrc = src / "consolidated"
    for f in sorted(csrc.glob("*.npy")) if csrc.exists() else []:
        split = f.name.rsplit("_", 1)[0]
        if f.name.endswith("_X.npy") and split in consolidated_splits:
            continue
        link = cdst / f.name
        if not link.exists():
            os.symlink(f.resolve(), link)
    for split in consolidated_splits:
        old = np.load(csrc / f"{split}_X.npy", mmap_mode="r"); V = np.concatenate(out_V[split]); assert len(old) == len(V), (split, old.shape, V.shape)
        out = np.lib.format.open_memmap(cdst / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(len(old), old.shape[1] + V.shape[1]))
        for s in range(0, len(old), 65536):
            out[s:s + 65536] = np.concatenate([old[s:s + 65536], V[s:s + 65536]], 1)
        out.flush(); del out
        print(f"{dst.name}: consolidated {split}_X {len(old)} x {old.shape[1] + V.shape[1]} [{time.monotonic()-t0:.0f}s]", flush=True)
    return dst, out_V


if which == "train":
    man_tr = json.loads((D / "advanced_history_v2/manifest.json").read_text())
    Vtr = np.concatenate(raw_feats("feats/train", len(man_tr["train"])))
    scale = np.median(np.abs(Vtr), 0) + 1e-9; SCALE_PATH.write_text(json.dumps({"names": NAMES, "scale": scale.tolist()}))
    dst, out_V = write_view("advanced_history_v2", "advanced_history_vessel_v1", {"train": "feats/train", "val": "feats_v2view/val", "legacy_val": "feats/legacy_val"},
                            scale, ("train", "val", "legacy_val"))
    # POD-input directory
    psrc, pdst = D / "pod_history128_robust_v4", D / "pod_history128_vessel_v1"; pdst.mkdir(exist_ok=True)
    norm = dict(np.load(psrc / "normalization.npz", allow_pickle=True)); Vt = np.concatenate(out_V["train"]).astype(np.float64)
    vm, vs = Vt.mean(0), Vt.std(0); keep = vs > 1e-6
    new_idx = np.arange(1348, 1348 + len(NAMES))[keep]
    norm["feature_indices"] = np.concatenate([norm["feature_indices"], new_idx])
    norm["x_mean"] = np.concatenate([norm["x_mean"], vm[keep].astype(np.float32)]); norm["x_std"] = np.concatenate([norm["x_std"], vs[keep].astype(np.float32)])
    norm["raw_log_scale"] = np.concatenate([norm["raw_log_scale"], np.ones(len(NAMES))])
    np.savez(pdst / "normalization.npz", **norm)
    meta = json.loads((psrc / "basis.json").read_text()); meta.update({"source_manifest_sha256": hashlib.sha256((dst / "manifest.json").read_bytes()).hexdigest(),
                                                                        "data_dir": str(dst), "input_features": 1348 + len(NAMES), "active_features": int(len(norm["feature_indices"])),
                                                                        "vessel_features_active": int(keep.sum()), "derived_from": str(psrc)})
    (pdst / "basis.json").write_text(json.dumps(meta, indent=2) + "\n")
    for f in psrc.glob("*.npy"):
        if not f.name.endswith("_X.npy") and not (pdst / f.name).exists():
            os.symlink(f.resolve(), pdst / f.name)
    for split in ("train", "val", "legacy_val"):
        old = np.load(psrc / f"{split}_X.npy", mmap_mode="r"); V = np.concatenate(out_V[split]).astype(np.float64)[:, keep]
        Vs = np.clip((V - vm[keep]) / vs[keep], -12, 12).astype(np.float32); assert len(old) == len(Vs)
        out = np.lib.format.open_memmap(pdst / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(len(old), old.shape[1] + Vs.shape[1]))
        for s in range(0, len(old), 65536):
            out[s:s + 65536] = np.concatenate([old[s:s + 65536], Vs[s:s + 65536]], 1)
        out.flush(); del out
        print(f"pod_history128_vessel_v1: {split}_X {len(old)} x {old.shape[1] + Vs.shape[1]} [{time.monotonic()-t0:.0f}s]", flush=True)
elif which == "native":
    scale = np.array(json.loads(SCALE_PATH.read_text())["scale"])
    write_view("advanced_history_val_native_v3", "advanced_history_val_native_vessel_v1", {"val": "feats/val"}, scale, ("val",))
elif which == "test":
    scale = np.array(json.loads(SCALE_PATH.read_text())["scale"])
    write_view("advanced_history_test_v2", "advanced_history_test_vessel_v1", {"test": "feats/test"}, scale, ("test",))
print("done", flush=True)
