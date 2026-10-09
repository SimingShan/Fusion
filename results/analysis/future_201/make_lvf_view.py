#!/usr/bin/env python3
"""All-labelled + vessel + FUTURE actuator view for LVF members (adapted from make_lv_view.py: source = the all-labelled vessel view).
Original docstring:
All-labelled + vessel-current view for L members:
  data/advanced_history_alllab_vessel_v1 : alllab manifest + 126 vessel names; TRAIN entries unchanged (li-store hash; training reads the
      POD-input arrays, never per-shot train X); history -> symlink to the alllab history; val/legacy_val entries get vessel sidecars and
      consolidated val_X / legacy_val_X are rebuilt (other consolidated files symlinked)
  data/pod_history128_alllab_vessel_v1  : alllab normalization + 126 vessel columns (mean/std over the alllab train rows), {split}_X =
      [alllab pod X | standardised vessel], targets symlinked, basis.json re-hashed
Vessel values use the same asinh scale as the B vessel view (results/analysis/vessel_196/vessel_scale.json)."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); D = ROOT / "data"; F = ROOT / "results/analysis/future_201"; t0 = time.monotonic()
sc = json.loads((F / "future_scale.json").read_text()); NAMES, scale = sc["names"], np.array(sc["scale"])
src, dst, psrc, pdst = D / "advanced_history_alllab_vessel_v1", D / "advanced_history_alllab_vf_v1", D / "pod_history128_alllab_vessel_v1", D / "pod_history128_alllab_vf_v1"
v2 = json.loads((D / "advanced_history_v2/manifest.json").read_text())
index = {}
for split, sub in (("train", "feats/train"), ("val", "feats_v2view/val"), ("legacy_val", "feats/legacy_val")):
    for i, e in enumerate(v2[split]):
        index[e["path"]] = F / sub / f"shot_{i:05d}.npy"


def feats(entries):
    out = []
    for e in entries:
        v = np.load(index[e["path"]]); assert len(v) == e["n_frames"], (e["path"], len(v), e["n_frames"]); out.append(np.arcsinh(v / scale).astype(np.float32))
    return out


man = json.loads((src / "manifest.json").read_text()); assert len(man["feature_names"]) == 1474; N0 = 1474
man["feature_names"] = man["feature_names"] + NAMES
man["future_augmentation"] = {"features": len(NAMES), "transform": "asinh(v / train_median_abs)", "scale_file": str(F / "future_scale.json"), "source_view": str(src),
                              "train_entries_note": "train entries unchanged (li-store hash); training reads pod X", "target_values_read": False}
dst.mkdir(parents=True, exist_ok=True)
if not (dst / "history").exists():
    os.symlink((src / "history").resolve(), dst / "history")
V = {}
for split in ("train", "val", "legacy_val"):
    V[split] = feats(man[split])
    if split != "train":
        (dst / "fut_extra" / split).mkdir(parents=True, exist_ok=True)
        for i, (e, v) in enumerate(zip(man[split], V[split])):
            old = np.load(src / e["X_extra"]); rel = f"fut_extra/{split}/shot_{i:05d}.npy"
            np.save(dst / rel, np.concatenate([old.astype(np.float32), v], 1)); e["X_extra"] = rel
    print(f"{split}: {len(V[split])} shots [{time.monotonic()-t0:.0f}s]", flush=True)
(dst / "manifest.json").write_text(json.dumps(man))
(dst / "consolidated").mkdir(exist_ok=True)
for f in (src / "consolidated").glob("*.npy"):
    if f.name in ("val_X.npy", "legacy_val_X.npy") or (dst / "consolidated" / f.name).exists():
        continue
    os.symlink(f.resolve(), dst / "consolidated" / f.name)
for split in ("val", "legacy_val"):
    old = np.load(src / "consolidated" / f"{split}_X.npy", mmap_mode="r"); Vv = np.concatenate(V[split]); assert len(old) == len(Vv)
    np.save(dst / "consolidated" / f"{split}_X.npy", np.concatenate([np.asarray(old), Vv], 1).astype(np.float32))
pdst.mkdir(exist_ok=True); norm = dict(np.load(psrc / "normalization.npz", allow_pickle=True)); Vt = np.concatenate(V["train"]).astype(np.float64)
vm, vs = Vt.mean(0), Vt.std(0); keep = vs > 1e-6
norm["feature_indices"] = np.concatenate([norm["feature_indices"], np.arange(N0, N0 + len(NAMES))[keep]])
norm["x_mean"] = np.concatenate([norm["x_mean"], vm[keep].astype(np.float32)]); norm["x_std"] = np.concatenate([norm["x_std"], vs[keep].astype(np.float32)])
norm["raw_log_scale"] = np.concatenate([norm["raw_log_scale"], np.ones(len(NAMES))]); np.savez(pdst / "normalization.npz", **norm)
meta = json.loads((psrc / "basis.json").read_text())
meta.update({"source_manifest_sha256": hashlib.sha256((dst / "manifest.json").read_bytes()).hexdigest(), "data_dir": str(dst), "input_features": N0 + len(NAMES),
             "active_features": int(len(norm["feature_indices"])), "future_features_active": int(keep.sum()), "derived_from": str(psrc)})
(pdst / "basis.json").write_text(json.dumps(meta, indent=2) + "\n")
for f in psrc.iterdir():
    if f.suffix in (".npy", ".txt") and not f.name.endswith("_X.npy") and not (pdst / f.name).exists():
        os.symlink(f.resolve(), pdst / f.name)
for split in ("train", "val", "legacy_val"):
    old = np.load(psrc / f"{split}_X.npy", mmap_mode="r"); Vv = np.concatenate(V[split]).astype(np.float64)[:, keep]; assert len(old) == len(Vv), (split, old.shape, Vv.shape)
    Vs = np.clip((Vv - vm[keep]) / vs[keep], -12, 12).astype(np.float32)
    out = np.lib.format.open_memmap(pdst / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(len(old), old.shape[1] + Vs.shape[1]))
    for s in range(0, len(old), 65536):
        out[s:s + 65536] = np.concatenate([old[s:s + 65536], Vs[s:s + 65536]], 1)
    out.flush(); del out
    print(f"pod {split}_X {len(old)} x {old.shape[1] + Vs.shape[1]} [{time.monotonic()-t0:.0f}s]", flush=True)
sys.path.insert(0, str(ROOT))
from fusion.pod import split_entries, manifest_hash
li = json.loads((D / "geometry_history_alllab_v1/metadata.json").read_text())
h = hashlib.sha256(json.dumps(split_entries(dst, "train"), sort_keys=True).encode()).hexdigest()
print("li-store hash matches:", h == li["source_training_entries_sha256"], "| basis hash matches:", json.loads((pdst / "basis.json").read_text())["source_manifest_sha256"] == manifest_hash(dst), flush=True)
print("done", flush=True)
