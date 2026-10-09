#!/usr/bin/env python3
"""Transfer-range audit of every Challenge-2 CNN input (conditioning entries and channels), DIII-D vs MAST.

Computes the featurizer's named per-frame entries (fusion.c2_fields_data.ENTRY_NAMES, i.e. the field
conditioning, the scalar-head conditioning and the removed diagnostics), every input channel (pixel
statistics) and the coordinate diagnostics (x_R, x_Z, R/L) on
  (a) N random DIII-D training shots x K frames   (statistics on the frames training keeps, |Ip| >= --min-ip-kA,
      and on all frames),
  (b) the native dev fold (128 shots) x K frames,
  (c) N random MAST public-test shots x K frames (all frames = what inference sees; |Ip| >= min-ip also reported).
Prints a table and flags any entry whose MAST q01-q99 range falls outside the DIII-D TRAIN q001-q999 range
(training-kept frames). Every FIELD conditioning entry and every channel must pass or be listed as a documented
exception. Inputs only: no labels of any kind are read (DIII-D caches are opened with with_labels=False).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion import c2_fields_data as cd  # noqa: E402

Q_MAST = (0.01, 0.99)
Q_D3D = (0.001, 0.999)


def stats(values: np.ndarray) -> dict:
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    q = np.quantile(v, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "q001": float(q[0]), "q01": float(q[1]), "q05": float(q[2]), "median": float(q[3]), "q95": float(q[4]),
            "q99": float(q[5]), "q999": float(q[6])}


@torch.no_grad()
def collect(store: cd.FrameStore, featurizer: cd.FrameFeaturizer, batch_size: int = 256) -> tuple[dict, dict]:
    store_t = store.to_torch(torch.device("cpu"), with_labels=False)
    entries = {k: [] for k in cd.ENTRY_NAMES}
    channels = {k: [] for k in cd.CHANNEL_NAMES}
    for first in range(0, store.n, batch_size):
        idx = torch.arange(first, min(first + batch_size, store.n))
        feats = featurizer(cd.batch_from_store(store_t, idx))
        for k in cd.ENTRY_NAMES:
            entries[k].append(feats["entries"][k].numpy())
        ch = feats["channels"].numpy()
        for j, k in enumerate(cd.CHANNEL_NAMES):
            channels[k].append(ch[:, j].reshape(len(idx), -1))
    entries = {k: np.concatenate(v) for k, v in entries.items()}
    channels = {k: np.concatenate(v) for k, v in channels.items()}   # (n, 4225)
    env = featurizer.setup.envelope.ravel()
    channels.update({k + "@env": v[:, env] for k, v in list(channels.items())})   # envelope pixels only
    return entries, channels


def summarise(entries, channels, coords, keep=None) -> dict:
    sel = np.ones(len(next(iter(entries.values()))), bool) if keep is None else keep
    out = {"frames": int(sel.sum()), "entries": {k: stats(v[sel]) for k, v in entries.items()},
           "channels": {k: stats(v[sel]) for k, v in channels.items()},
           "coordinates": {k: stats(v) for k, v in coords.items()}}
    return out


def flag(name, d3d: dict, mast: dict) -> dict:
    ok_low = mast["q01"] >= d3d["q001"] - 1e-9
    ok_high = mast["q99"] <= d3d["q999"] + 1e-9
    return {"name": name, "pass": bool(ok_low and ok_high),
            "d3d_train_q001_q999": [d3d["q001"], d3d["q999"]], "mast_q01_q99": [mast["q01"], mast["q99"]],
            "below": None if ok_low else d3d["q001"] - mast["q01"], "above": None if ok_high else mast["q99"] - d3d["q999"]}


def row(name, group, d3d, dev, mast, verdict, mark):
    def rng(s):
        return f"{s['mean']:8.3f} {s['std']:7.3f} [{s['min']:8.3f},{s['max']:8.3f}] q01/99 [{s['q01']:8.3f},{s['q99']:8.3f}]"
    return (f"{mark} {group:6s} {name:16s} | D3D-train {rng(d3d)} q001/999 [{d3d['q001']:8.3f},{d3d['q999']:8.3f}]\n"
            f"{'':26s}| D3D-dev   {rng(dev)}\n"
            f"{'':26s}| MAST      {rng(mast)}  -> {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/advanced_v1")
    p.add_argument("--val-dir", type=Path, default=ROOT / "data/advanced_val_native_v3")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    p.add_argument("--n-train-shots", type=int, default=200)
    p.add_argument("--n-mast-shots", type=int, default=200)
    p.add_argument("--n-dev-shots", type=int, default=0, help="0 = all 128 dev shots")
    p.add_argument("--frames-per-shot", type=int, default=8)
    p.add_argument("--min-ip-kA", type=float, default=100.0, help="training frame filter (train_c2_cnn.py --min-ip-kA)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=ROOT / "results/challenge2/c2_cnn_conditioning_ranges.json")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.workers)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()

    d3d = cd.build_machine_setup("DIII-D")
    feat_d3d = cd.FrameFeaturizer(d3d, torch.device("cpu"))
    feature_names = cd.read_manifest(args.data_dir)["feature_names"]
    index = cd.D3DFeatureIndex(feature_names, d3d.table_names)
    train_entries = cd.split_entries(args.data_dir, "train")
    train_pos = sorted(rng.choice(len(train_entries), size=min(args.n_train_shots, len(train_entries)), replace=False).tolist())
    shots, _ = cd.load_d3d_split(args.data_dir, "train", index, args.frames_per_shot, positions=train_pos,
                                 with_labels=False, verbose=False)
    store = cd.FrameStore(shots, K=cd.D3D_K, with_labels=False)
    tr_entries, tr_channels = collect(store, feat_d3d)
    tr_keep = store.frame_mask(args.min_ip_kA)
    coords_d3d = feat_d3d.coordinate_maps()
    print(f"DIII-D train: {len(shots)} shots, {store.n} frames ({int(tr_keep.sum())} with |Ip|>={args.min_ip_kA:g} kA) "
          f"{time.monotonic() - started:.0f} s", flush=True)

    dev_entries_list = cd.split_entries(args.val_dir, "val")
    dev_pos = list(range(len(dev_entries_list)))
    if args.n_dev_shots:
        dev_pos = sorted(rng.choice(len(dev_pos), size=min(args.n_dev_shots, len(dev_pos)), replace=False).tolist())
    dev_index = cd.D3DFeatureIndex(cd.read_manifest(args.val_dir)["feature_names"], d3d.table_names)
    dev_shots, _ = cd.load_d3d_split(args.val_dir, "val", dev_index, args.frames_per_shot, positions=dev_pos,
                                     with_labels=False, verbose=False)
    dev_store = cd.FrameStore(dev_shots, K=cd.D3D_K, with_labels=False)
    dv_entries, dv_channels = collect(dev_store, feat_d3d)
    print(f"DIII-D dev: {len(dev_shots)} shots, {dev_store.n} frames {time.monotonic() - started:.0f} s", flush=True)

    mast = cd.build_machine_setup("MAST")
    feat_mast = cd.FrameFeaturizer(mast, torch.device("cpu"))
    paths = cd.mast_parquet_paths(args.raw_dir)
    mast_pos = sorted(rng.choice(len(paths), size=min(args.n_mast_shots, len(paths)), replace=False).tolist())
    mast_shots = []
    for count, i in enumerate(mast_pos):
        full = cd.mast_shot_inputs(paths[i], mast)
        sel = cd.uniform_frames(full.n_frames, args.frames_per_shot)
        mast_shots.append(cd.mast_shot_inputs(paths[i], mast, frames=full.frames[sel]))
        if (count + 1) % 50 == 0:
            print(f"MAST: {count + 1}/{len(mast_pos)} shots {time.monotonic() - started:.0f} s", flush=True)
    mast_store = cd.FrameStore(mast_shots, with_labels=False)
    ms_entries, ms_channels = collect(mast_store, feat_mast)
    ms_keep = mast_store.frame_mask(args.min_ip_kA)
    coords_mast = feat_mast.coordinate_maps()
    print(f"MAST: {len(mast_shots)} shots, {mast_store.n} frames ({int(ms_keep.sum())} with |Ip|>={args.min_ip_kA:g} kA) "
          f"{time.monotonic() - started:.0f} s", flush=True)

    summary = {"d3d_train_kept": summarise(tr_entries, tr_channels, coords_d3d, tr_keep),
               "d3d_train_all": summarise(tr_entries, tr_channels, coords_d3d),
               "d3d_dev_all": summarise(dv_entries, dv_channels, coords_d3d),
               "mast_all": summarise(ms_entries, ms_channels, coords_mast),
               "mast_ip_kept": summarise(ms_entries, ms_channels, coords_mast, ms_keep)}
    ref, dev, tgt = summary["d3d_train_kept"], summary["d3d_dev_all"], summary["mast_all"]
    flags, lines = [], []
    groups = ([(k, "FIELD") for k in cd.FIELD_COND_NAMES]
              + [(k, "SCALAR") for k in cd.SCALAR_COND_NAMES if k not in cd.FIELD_COND_NAMES]
              + [(k, "REMOVED") for k in cd.ENTRY_NAMES if k not in cd.SCALAR_COND_NAMES])
    for name, group in groups:
        f = flag(name, ref["entries"][name], tgt["entries"][name])
        f["group"] = group
        f["required"] = group == "FIELD"
        flags.append(f)
        verdict = "PASS" if f["pass"] else ("FAIL" if f["required"] else "EXTRAPOLATES (heads only / removed)")
        lines.append(row(name, group, ref["entries"][name], dev["entries"][name], tgt["entries"][name], verdict,
                         "  " if f["pass"] else "!!"))
    for name in cd.CHANNEL_NAMES:
        for suffix, group in (("", "CHAN"), ("@env", "CHAN@E")):
            key = name + suffix
            f = flag(key, ref["channels"][key], tgt["channels"][key])
            f["group"], f["required"] = ("CHANNEL", True) if not suffix else ("CHANNEL_ENVELOPE", False)
            flags.append(f)
            lines.append(row(key, group, ref["channels"][key], dev["channels"][key], tgt["channels"][key],
                             "PASS" if f["pass"] else "FAIL", "  " if f["pass"] else "!!"))
    for name in cd.COORDINATE_DIAGNOSTICS:
        f = flag(name, ref["coordinates"][name], tgt["coordinates"][name])
        f["group"], f["required"] = "COORD-DIAG", False
        flags.append(f)
        lines.append(row(name, "DIAG", ref["coordinates"][name], dev["coordinates"][name], tgt["coordinates"][name],
                         "PASS (not used)" if f["pass"] else "EXTRAPOLATES (not used)", "  " if f["pass"] else "!!"))
    print("\nRange audit: MAST q01-q99 must lie inside DIII-D TRAIN (|Ip|>=min) q001-q999. '!!' = outside.\n")
    print("\n".join(lines))
    documented = {"dR_over_L", "dZ_over_L", "phi_ext", "phi_coils", "phi_plasma"}
    required_fail = [f["name"] for f in flags if f["required"] and not f["pass"]]
    undocumented = [n for n in required_fail if n not in documented]
    print(f"\nrequired entries failing: {required_fail or 'none'}; of which undocumented: {undocumented or 'none'}")
    # Same test on |Ip|-kept MAST frames (what a low-Ip filter at inference would see), informational.
    kept_flags = {k: flag(k, ref["entries"][k], summary["mast_ip_kept"]["entries"][k])["pass"] for k in cd.ENTRY_NAMES}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "feature_spec": cd.feature_spec(),
              "feature_spec_hash": cd.feature_spec_hash(), "criterion": "MAST q01-q99 inside DIII-D train (kept frames) q001-q999",
              "sampling": {"train_shots": len(shots), "train_positions": train_pos, "dev_shots": len(dev_shots),
                           "mast_shots": len(mast_shots), "mast_positions": mast_pos, "frames_per_shot": args.frames_per_shot,
                           "min_ip_kA": args.min_ip_kA, "seed": args.seed},
              "flags": flags, "required_failures": required_fail, "undocumented_failures": undocumented,
              "mast_ip_kept_entry_pass": kept_flags,
              "exceptions": {"bt_over_bp": "scalar-head conditioning only (q95 needs Bt); MAST ~1.2 vs DIII-D 4-7",
                             "log_bt_over_bp": "scalar-head conditioning only",
                             "pe_mean/pe_peak": "scalar-head conditioning only (betaN needs pressure); MAST p_e/P0 below DIII-D q001",
                             "tf_sign_rel": "removed from all inputs (machine switch: -1 on every MAST shot)",
                             "pe_valid_frac": "removed (fraction of padded chord capacity, machine artefact)",
                             "time_s": "removed (absolute seconds; MAST inside DIII-D ramp-up window); replaced by time_frac",
                             "x_R/x_Z/R_over_L": "diagnostics only, replaced by grid-normalised u_R/u_Z",
                             "dR_over_L/dZ_over_L": "constant per machine (0.017/0.031 DIII-D, 0.036/0.074 MAST): fails the numeric "
                                                    "criterion by construction; kept per task instruction (physical scale) - flagged",
                             "phi_ext/phi_coils/phi_plasma": "physics field channels; pixel tails extrapolate modestly on MAST "
                                                             "(see @env rows = envelope pixels); no in-range substitute exists"},
              "labels_read": False, "statistics": summary, "seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output} in {time.monotonic() - started:.0f} s")
    return 1 if undocumented else 0


if __name__ == "__main__":
    sys.exit(main())
