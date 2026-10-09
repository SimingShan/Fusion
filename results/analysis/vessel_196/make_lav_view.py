#!/usr/bin/env python3
"""All-labelled A-family + vessel view for A-type all-labelled members (LAV):
  data/advanced_alllab_vessel_v1 : feature names = the A-vessel view's (1138 + 126); TRAIN entries copied VERBATIM from
      advanced_history_alllab_v1 (same shots/frames/order; keeps the geometry_history_alllab_v1 li-store hash; training reads pod X,
      never per-shot train X); val/legacy_val entries = advanced_vessel_v1's (A-vessel sidecars); history -> alllab history (unused)
  data/pod_scaled_alllab_vessel_v1 : A-vessel normalization; train rows = [A-vessel pod train | val | legacy_val] X and the matching A
      (pod_scaled_v2) coefficients/scalars, in the alllab train order (train shots, then val, then legacy_val); val/legacy links.
The A and B families share the same 128-mode POD basis (checked)."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); D = ROOT / "data"; t0 = time.monotonic()
hall, av, dst = D / "advanced_history_alllab_v1", D / "advanced_vessel_v1", D / "advanced_alllab_vessel_v1"
mh, ma = json.loads((hall / "manifest.json").read_text()), json.loads((av / "manifest.json").read_text())
assert len(mh["train"]) == 6913 and [e["path"].split("/")[-2] for e in mh["train"][6753:6754]] == ["val"]
man = dict(ma); man["train"] = mh["train"]; man["val"] = ma["val"]; man["legacy_val"] = ma["legacy_val"]
for k in ("audit", "test"):
    man[k] = []
man["alllab_vessel_note"] = {"train_entries_from": str(hall), "val_legacy_entries_from": str(av), "features": "A-vessel (no history)", "target_values_read": False}
dst.mkdir(parents=True, exist_ok=True)
for sub in ("history",):
    if not (dst / sub).exists():
        os.symlink((hall / sub).resolve(), dst / sub)
for split in ("val", "legacy_val"):
    for e in man[split]:
        if e.get("X_extra") and not str(e["X_extra"]).startswith("/"):
            e["X_extra"] = str((av / e["X_extra"]).resolve())
(dst / "manifest.json").write_text(json.dumps(man))
(dst / "consolidated").mkdir(exist_ok=True)
for f in (av / "consolidated").glob("*.npy"):
    if (f.name.startswith("val_") or f.name.startswith("legacy_val_")) and not (dst / "consolidated" / f.name).exists():
        os.symlink(f.resolve(), dst / "consolidated" / f.name)
psv, p2, pdst = D / "pod_scaled_vessel_v1", D / "pod_scaled_v2", D / "pod_scaled_alllab_vessel_v1"; pdst.mkdir(exist_ok=True)
os.system(f"cp {psv}/normalization.npz {pdst}/normalization.npz")
meta = json.loads((psv / "basis.json").read_text()); meta.update({"source_manifest_sha256": hashlib.sha256((dst / "manifest.json").read_bytes()).hexdigest(), "data_dir": str(dst),
                                                                    "alllab": "train rows = train + val + legacy_val (alllab order)", "derived_from": str(psv)})
(pdst / "basis.json").write_text(json.dumps(meta, indent=2) + "\n")
for k, srcdir in (("X", psv), ("coefficients", p2), ("scalars", p2)):
    parts = [np.load(srcdir / f"{s}_{k}.npy", mmap_mode="r") for s in ("train", "val", "legacy_val")]
    n = sum(len(p) for p in parts); out = np.lib.format.open_memmap(pdst / f"train_{k}.npy", mode="w+", dtype=np.float32, shape=(n,) + parts[0].shape[1:]); i = 0
    for p in parts:
        for s in range(0, len(p), 65536):
            blk = np.asarray(p[s:s + 65536]); out[i:i + len(blk)] = blk; i += len(blk)
    out.flush(); del out
    for s in ("val", "legacy_val"):
        if not (pdst / f"{s}_{k}.npy").exists():
            os.symlink((srcdir / f"{s}_{k}.npy").resolve(), pdst / f"{s}_{k}.npy")
    print(f"pod train_{k}: {n} rows [{time.monotonic()-t0:.0f}s]", flush=True)
if not (pdst / "basis_sample_indices.npy").exists():
    os.symlink((p2 / "basis_sample_indices.npy").resolve(), pdst / "basis_sample_indices.npy")
# checks
import sys; sys.path.insert(0, str(ROOT))
from fusion.pod import split_entries, manifest_hash
li = json.loads((D / "geometry_history_alllab_v1/metadata.json").read_text())
h = hashlib.sha256(json.dumps(split_entries(dst, "train"), sort_keys=True).encode()).hexdigest()
print("li-store hash matches:", h == li["source_training_entries_sha256"], "| basis hash matches:", json.loads((pdst / "basis.json").read_text())["source_manifest_sha256"] == manifest_hash(dst))
print("train rows: pod X", np.load(pdst / "train_X.npy", mmap_mode="r").shape, "coefficients", np.load(pdst / "train_coefficients.npy", mmap_mode="r").shape, "expected frames", sum(e["n_frames"] for e in man["train"]))
print("done", flush=True)
