#!/usr/bin/env python3
"""FUTURE actuator response inputs (inputs only, raw 0.05 ms magnetics): the plasma-control system reacts to the plasma's present position
and shape (sensed by the magnetic diagnostics this challenge withholds) with a delay of ms to tens of ms, so the coil currents AFTER a frame
carry information about that frame's state that the present and past currents do not (above all during end-of-shot vertical drifts, where
the blend's error lives).  Mirror of results/analysis/vessel_196/build_features.py in time: per current (21), the ANTI-causal exponentially
filtered derivative at tau = 2, 5, 10, 20, 50 ms (filter run backwards in time) plus the least-squares slope over the NEXT 2 ms
-> 126 features per frame.  Ip uses the official starter.data_fixes.fix_d3d_ip_times axis.
usage: build_features.py <view_dir> <split> <out_dir> [shard k/n]   -> <out_dir>/<split>/shot_XXXXX.npy (frames, 126) + feature_names.json"""
import json, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); RAW = ROOT / "data/raw/data"; sys.path.insert(0, str(ROOT))
from starter.data_fixes import fix_d3d_ip_times  # noqa: E402

CURRENTS = ["magnetics_ECOILA", "magnetics_bcoil"] + [f"magnetics_F{i}{s}" for i in range(1, 10) for s in "AB"] + ["magnetics_plasma_current"]
TAUS = [2.0, 5.0, 10.0, 20.0, 50.0]; NAMES = [f"{c}_{kind}" for c in CURRENTS for kind in [f"fut_dfilt_tau{t:g}ms" for t in TAUS] + ["fut_slope2ms"]]


def anticausal_filtered_derivative(t, v, tau, tq):
    """Exponential filter of dv/dt run BACKWARDS in time (uniform raw step): y_i = a*y_{i+1} + (1-a)*rate_i, evaluated at tq."""
    dt = float(np.median(np.diff(t))); a = np.exp(-dt / tau); rate = np.gradient(v, dt)
    y = lfilter([1 - a], [1, -a], rate[::-1])[::-1]
    return np.interp(tq, t, y, left=0.0, right=0.0)


def future_slope2ms(t, v, tq):
    out = np.zeros(len(tq)); dt = float(np.median(np.diff(t))); w = max(2, int(round(2.0 / dt))); idx = np.searchsorted(t, tq)
    for j, i in enumerate(idx):
        lo, hi = max(0, i), min(len(t), i + w + 1)
        if hi - lo >= 3:
            x = t[lo:hi] - t[lo]; out[j] = np.polyfit(x, v[lo:hi], 1)[0]
    return out


if __name__ == "__main__":
    view, split, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); k, n = (int(x) for x in sys.argv[4].split("/")) if len(sys.argv) > 4 else (0, 1)
    man = json.loads((view / "manifest.json").read_text()); entries = man[split]
    (out / split).mkdir(parents=True, exist_ok=True); (out / "feature_names.json").write_text(json.dumps(NAMES))
    t0 = time.monotonic(); done = 0
    for i, e in enumerate(entries):
        if i % n != k:
            continue
        target = out / split / f"shot_{i:05d}.npy"
        if target.exists():
            continue
        with np.load(e["path"] if str(e["path"]).startswith("/") else view / e["path"]) as f:
            tq = np.asarray(f["times"], dtype=np.float64)
        shard = RAW / e["shard"].replace("data/", "", 1) if e["shard"].startswith("data/") else RAW / e["shard"]
        cols = ["magnetics_time", "magnetics_plasma_current_times"] + CURRENTS
        table = pq.read_table(shard, columns=cols)
        tb = {c: table.column(c).combine_chunks().flatten().to_numpy(zero_copy_only=False) for c in cols}
        mt = np.asarray(tb["magnetics_time"], dtype=np.float64); ipt = np.asarray(tb["magnetics_plasma_current_times"], dtype=np.float64)
        ipv = np.asarray(tb["magnetics_plasma_current"], dtype=np.float64)
        if len(ipv) >= 10 and len(mt):
            ipt = np.asarray(fix_d3d_ip_times({"magnetics_plasma_current_times": ipt, "magnetics_plasma_current": ipv, "magnetics_time": mt[:1]}), dtype=np.float64)
        feats = np.zeros((len(tq), len(NAMES)), dtype=np.float32); col = 0
        for c in CURRENTS:
            v = np.asarray(tb[c], dtype=np.float64); tt = ipt if c == "magnetics_plasma_current" else mt
            if len(v) != len(tt) or len(v) < 10:
                col += len(TAUS) + 1; continue
            win = (tt >= tq.min() - 1.0) & (tt <= tq.max() + 500.0)
            if win.sum() >= 10:
                tt, v = tt[win], v[win]
            ok = np.isfinite(v)
            if not ok.all():
                v = np.interp(tt, tt[ok], v[ok]) if ok.sum() > 2 else np.zeros_like(v)
            for tau in TAUS:
                feats[:, col] = anticausal_filtered_derivative(tt, v, tau, tq); col += 1
            feats[:, col] = future_slope2ms(tt, v, tq); col += 1
        np.save(target, feats); done += 1
        if done % 100 == 0:
            print(f"shard {k}/{n}: {done} shots [{time.monotonic()-t0:.0f}s]", flush=True)
    print(f"shard {k}/{n} done: {done} shots written [{time.monotonic()-t0:.0f}s]", flush=True)
