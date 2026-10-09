#!/usr/bin/env python3
"""A-family (no-history) views + POD-input dir with the 126 vessel-current features appended, for vessel versions of the A members.
  data/advanced_vessel_v1           : advanced_v1 manifest + vessel names; TRAIN entries unchanged (li-store hash; training reads pod X);
                                      val/legacy_val entries get vessel sidecars (X_extra, 126 columns); consolidated: links for train/val/legacy_val
                                      files except {val,legacy_val}_X (rebuilt); NO audit files are linked or read
  data/pod_scaled_vessel_v1         : pod_scaled_v2 normalization + 126 vessel columns (train mean/std), {split}_X = [pod X | standardised vessel]
  data/advanced_val_native_vessel_v1: A native dev view (advanced_val_native_v3) + vessel sidecars + consolidated val_X
  data/advanced_test_vessel_v1      : A test view (advanced_test_v1) + vessel sidecars + consolidated test_X
Feature rows follow each source view's manifest order; the vessel features were built per shot for the history views, whose shot and frame
order is identical to the A views (checked: frame counts per entry must match). Same asinh scale as the B vessel view."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); D = ROOT / "data"; F = ROOT / "results/analysis/vessel_196"; t0 = time.monotonic()
sc = json.loads((F / "vessel_scale.json").read_text()); NAMES, scale = sc["names"], np.array(sc["scale"])


def feats(sub, entries):
    out = []
    for i, e in enumerate(entries):
        v = np.load(F / sub / f"shot_{i:05d}.npy"); assert len(v) == e["n_frames"], (sub, i, len(v), e["n_frames"]); out.append(np.arcsinh(v / scale).astype(np.float32))
    return out


def build_view(src, dst, splits, consolidated_x, keep_train_entries=True):
    """splits: {split: feats_subdir}; per-shot sidecars for every split except train (train entries untouched)."""
    src, dst = D / src, D / dst; dst.mkdir(parents=True, exist_ok=True); man = json.loads((src / "manifest.json").read_text())
    n0 = len(man["feature_names"]); man["feature_names"] = man["feature_names"] + NAMES
    man["vessel_augmentation"] = {"features": len(NAMES), "transform": "asinh(v / train_median_abs)", "source_view": str(src), "base_feature_count": n0,
                                  "train_entries_note": "train entries unchanged (li-store hash); training reads pod X", "audit_files_linked": False, "target_values_read": False}
    V = {}
    for split, sub in splits.items():
        V[split] = feats(sub, man[split])
        if split == "train" and keep_train_entries:
            continue
        (dst / "vessel_extra" / split).mkdir(parents=True, exist_ok=True)
        for i, (e, v) in enumerate(zip(man[split], V[split])):
            assert not e.get("X_extra"), "A views carry no sidecars"
            rel = f"vessel_extra/{split}/shot_{i:05d}.npy"; np.save(dst / rel, v); e["X_extra"] = rel
            if not str(e["path"]).startswith("/"):
                e["path"] = str((src / e["path"]).resolve())
        print(f"{dst.name}: {split} sidecars ({len(man[split])} shots) [{time.monotonic()-t0:.0f}s]", flush=True)
    (dst / "manifest.json").write_text(json.dumps(man))
    (dst / "consolidated").mkdir(exist_ok=True)
    for f in (src / "consolidated").glob("*.npy") if (src / "consolidated").exists() else []:
        if f.name.startswith("audit") or f.name in {f"{s}_X.npy" for s in consolidated_x} or (dst / "consolidated" / f.name).exists():
            continue
        os.symlink(f.resolve(), dst / "consolidated" / f.name)
    for split in consolidated_x:
        old = np.load(src / "consolidated" / f"{split}_X.npy", mmap_mode="r"); Vv = np.concatenate(V[split]); assert len(old) == len(Vv), (split, old.shape, Vv.shape)
        out = np.lib.format.open_memmap(dst / "consolidated" / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(len(old), old.shape[1] + Vv.shape[1]))
        for s in range(0, len(old), 65536):
            out[s:s + 65536] = np.concatenate([old[s:s + 65536], Vv[s:s + 65536]], 1)
        out.flush(); del out
    return dst, V, n0


which = sys.argv[1]
if which == "train":
    dst, V, n0 = build_view("advanced_v1", "advanced_vessel_v1", {"train": "feats/train", "val": "feats_v2view/val", "legacy_val": "feats/legacy_val"}, ("train", "val", "legacy_val"))
    psrc, pdst = D / "pod_scaled_v2", D / "pod_scaled_vessel_v1"; pdst.mkdir(exist_ok=True)
    norm = dict(np.load(psrc / "normalization.npz", allow_pickle=True)); Vt = np.concatenate(V["train"]).astype(np.float64); vm, vs = Vt.mean(0), Vt.std(0); keep = vs > 1e-6
    norm["feature_indices"] = np.concatenate([norm["feature_indices"], np.arange(n0, n0 + len(NAMES))[keep]])
    norm["x_mean"] = np.concatenate([norm["x_mean"], vm[keep].astype(np.float32)]); norm["x_std"] = np.concatenate([norm["x_std"], vs[keep].astype(np.float32)])
    norm["raw_log_scale"] = np.concatenate([norm["raw_log_scale"], np.ones(len(NAMES))]); np.savez(pdst / "normalization.npz", **norm)
    meta = json.loads((psrc / "basis.json").read_text())
    meta.update({"source_manifest_sha256": hashlib.sha256((dst / "manifest.json").read_bytes()).hexdigest(), "data_dir": str(dst), "input_features": n0 + len(NAMES),
                 "active_features": int(len(norm["feature_indices"])), "vessel_features_active": int(keep.sum()), "derived_from": str(psrc)})
    (pdst / "basis.json").write_text(json.dumps(meta, indent=2) + "\n")
    for f in psrc.iterdir():
        if f.suffix == ".npy" and not f.name.endswith("_X.npy") and not f.name.startswith("audit") and not (pdst / f.name).exists():
            os.symlink(f.resolve(), pdst / f.name)
    for split in ("train", "val", "legacy_val"):
        if not (psrc / f"{split}_X.npy").exists():
            continue
        old = np.load(psrc / f"{split}_X.npy", mmap_mode="r"); Vv = np.concatenate(V[split]).astype(np.float64)[:, keep]; assert len(old) == len(Vv), (split, old.shape, Vv.shape)
        Vs = np.clip((Vv - vm[keep]) / vs[keep], -12, 12).astype(np.float32)
        out = np.lib.format.open_memmap(pdst / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(len(old), old.shape[1] + Vs.shape[1]))
        for s in range(0, len(old), 65536):
            out[s:s + 65536] = np.concatenate([old[s:s + 65536], Vs[s:s + 65536]], 1)
        out.flush(); del out
        print(f"pod {split}_X {len(old)} x {old.shape[1] + Vs.shape[1]} [{time.monotonic()-t0:.0f}s]", flush=True)
elif which == "native":
    build_view("advanced_val_native_v3", "advanced_val_native_vessel_v1", {"val": "feats/val"}, ("val",) if (D / "advanced_val_native_v3/consolidated/val_X.npy").exists() else ())
elif which == "test":
    build_view("advanced_test_v1", "advanced_test_vessel_v1", {"test": "feats/test"}, ("test",) if (D / "advanced_test_v1/consolidated/test_X.npy").exists() else ())
print("done", flush=True)
