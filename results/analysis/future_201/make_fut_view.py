#!/usr/bin/env python3
"""Views for VF members (B recipe + vessel + FUTURE actuator response): the B vessel views with the 126 future features appended
(asinh(v / train median |v|), like the vessel block; raw_log_scale 1; standardised in the pod X).
  train : data/advanced_history_vf_v1 (sidecars for val/legacy_val, consolidated X for train/val/legacy_val; train entries unchanged ->
          the geometry_history_all_v5 li-store hash still matches) + data/pod_history128_vf_v1 ({split}_X = [vessel pod X | standardised future])
  native: data/advanced_history_val_native_vf_v1      test: data/advanced_history_test_vf_v1
usage: make_fut_view.py <train|native|test>"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); D = ROOT / "data"; F = ROOT / "results/analysis/future_201"; t0 = time.monotonic()
NAMES = ["future_" + n for n in json.loads((F / "feats/feature_names.json").read_text())]; SCALE = F / "future_scale.json"


def feats(sub, n):
    return [np.load(F / sub / f"shot_{i:05d}.npy").astype(np.float64) for i in range(n)]


def build(src, dst, subs, consolidated, scale):
    src, dst = D / src, D / dst; dst.mkdir(parents=True, exist_ok=True); man = json.loads((src / "manifest.json").read_text()); n0 = len(man["feature_names"])
    man["feature_names"] = man["feature_names"] + NAMES
    man["future_augmentation"] = {"features": len(NAMES), "transform": "asinh(v / train_median_abs)", "scale_file": str(SCALE), "source_view": str(src), "inputs_only": True}
    T = {}
    for split, sub in subs.items():
        T[split] = [np.arcsinh(v / scale).astype(np.float32) for v in feats(sub, len(man[split]))]
        if split == "train":
            continue
        (dst / "fut_extra" / split).mkdir(parents=True, exist_ok=True)
        for i, (e, tf) in enumerate(zip(man[split], T[split])):
            old = np.load(src / e["X_extra"]) if e.get("X_extra") else np.zeros((len(tf), 0), np.float32)
            assert len(old) == len(tf) == e["n_frames"], (split, i, len(old), len(tf))
            rel = f"fut_extra/{split}/shot_{i:05d}.npy"; np.save(dst / rel, np.concatenate([old.astype(np.float32), tf], 1)); e["X_extra"] = rel
            if not str(e["path"]).startswith("/"):
                e["path"] = str((src / e["path"]).resolve())
        print(f"{dst.name}: {split} sidecars [{time.monotonic()-t0:.0f}s]", flush=True)
    if "train" in subs:
        hist = src / "history"
        if hist.exists() and not (dst / "history").exists():
            os.symlink(hist.resolve(), dst / "history")      # train entries stay byte-identical (li-store hash); training reads pod X
    (dst / "manifest.json").write_text(json.dumps(man)); (dst / "consolidated").mkdir(exist_ok=True)
    for f in (src / "consolidated").glob("*.npy") if (src / "consolidated").exists() else []:
        if f.name in {f"{s}_X.npy" for s in consolidated} or (dst / "consolidated" / f.name).exists():
            continue
        os.symlink(f.resolve(), dst / "consolidated" / f.name)
    for split in consolidated:
        old = np.load(src / "consolidated" / f"{split}_X.npy", mmap_mode="r"); Tt = np.concatenate(T[split]); assert len(old) == len(Tt), (split, len(old), len(Tt))
        out = np.lib.format.open_memmap(dst / "consolidated" / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(len(old), old.shape[1] + Tt.shape[1]))
        for s in range(0, len(old), 65536):
            out[s:s + 65536] = np.concatenate([old[s:s + 65536], Tt[s:s + 65536]], 1)
        out.flush(); del out
    return dst, T, n0


which = sys.argv[1]
if which == "train":
    n_tr = len(json.loads((D / "advanced_history_v2/manifest.json").read_text())["train"])
    raw = np.concatenate(feats("feats/train", n_tr)); scale = np.median(np.abs(raw), 0) + 1e-9; SCALE.write_text(json.dumps({"names": NAMES, "scale": scale.tolist()}))
    dst, T, n0 = build("advanced_history_vessel_v1", "advanced_history_vf_v1", {"train": "feats/train", "val": "feats_v2view/val", "legacy_val": "feats/legacy_val"},
                       ("train", "val", "legacy_val"), scale)
    psrc, pdst = D / "pod_history128_vessel_v1", D / "pod_history128_vf_v1"; pdst.mkdir(exist_ok=True)
    norm = dict(np.load(psrc / "normalization.npz", allow_pickle=True)); Tt = np.concatenate(T["train"]).astype(np.float64); m, s = Tt.mean(0), Tt.std(0); keep = s > 1e-6
    norm["feature_indices"] = np.concatenate([norm["feature_indices"], np.arange(n0, n0 + len(NAMES))[keep]])
    norm["x_mean"] = np.concatenate([norm["x_mean"], m[keep].astype(np.float32)]); norm["x_std"] = np.concatenate([norm["x_std"], s[keep].astype(np.float32)])
    norm["raw_log_scale"] = np.concatenate([norm["raw_log_scale"], np.ones(len(NAMES))]); np.savez(pdst / "normalization.npz", **norm)
    meta = json.loads((psrc / "basis.json").read_text()); meta.update({"source_manifest_sha256": hashlib.sha256((dst / "manifest.json").read_bytes()).hexdigest(), "data_dir": str(dst),
                                                                        "input_features": n0 + len(NAMES), "active_features": int(len(norm["feature_indices"])), "derived_from": str(psrc)})
    (pdst / "basis.json").write_text(json.dumps(meta, indent=2) + "\n")
    for f in psrc.iterdir():
        if f.suffix == ".npy" and not f.name.endswith("_X.npy") and not (pdst / f.name).exists():
            os.symlink(f.resolve(), pdst / f.name)
    for split in ("train", "val", "legacy_val"):
        old = np.load(psrc / f"{split}_X.npy", mmap_mode="r"); Ts = np.clip((np.concatenate(T[split]).astype(np.float64)[:, keep] - m[keep]) / s[keep], -12, 12).astype(np.float32)
        assert len(old) == len(Ts), (split, len(old), len(Ts))
        out = np.lib.format.open_memmap(pdst / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(len(old), old.shape[1] + Ts.shape[1]))
        for st in range(0, len(old), 65536):
            out[st:st + 65536] = np.concatenate([old[st:st + 65536], Ts[st:st + 65536]], 1)
        out.flush(); del out
        print(f"pod {split}_X {len(old)} x {old.shape[1] + Ts.shape[1]} [{time.monotonic()-t0:.0f}s]", flush=True)
    import sys as _s; _s.path.insert(0, str(ROOT))
    from fusion.pod import split_entries, manifest_hash
    li = json.loads((D / "geometry_history_all_v5/metadata.json").read_text())
    h = hashlib.sha256(json.dumps(split_entries(dst, "train"), sort_keys=True).encode()).hexdigest()
    print("li-store hash matches:", h == li["source_training_entries_sha256"], "| basis hash matches:", json.loads((pdst / "basis.json").read_text())["source_manifest_sha256"] == manifest_hash(dst), flush=True)
elif which == "native":
    build("advanced_history_val_native_vessel_v1", "advanced_history_val_native_vf_v1", {"val": "feats/val"}, ("val",), np.array(json.loads(SCALE.read_text())["scale"]))
elif which == "test":
    build("advanced_history_test_vessel_v1", "advanced_history_test_vf_v1", {"test": "feats/test"}, ("test",), np.array(json.loads(SCALE.read_text())["scale"]))
print("done", flush=True)
