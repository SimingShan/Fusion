#!/usr/bin/env python3
"""Vessel-eddy-current proxy inputs from the RAW 0.05 ms magnetics (coil currents + plasma current), evaluated at each cached frame time.
Physics: vessel currents obey L dI_v/dt + R I_v = -M dI_c/dt, i.e. I_v(t) = -(M/L) * int exp(-(t-s)/tau) dI_c/ds ds, with tau of order
ms to tens of ms.  Our cached inputs sample the currents at t, t+-20 ms, t-100 ms ... so the fast part of that convolution is absent.
Features per current (21 of them): the exponentially filtered derivative at tau = 2, 5, 10, 20, 50 ms (causal, evaluated at the frame
time), plus the least-squares slope over the last 2 ms -> 21 x 6 = 126 features per frame.  Inputs only (no labels are read).
usage: build_features.py <view_dir> <split> <out_dir> [shard k/n]   -> <out_dir>/<split>/shot_XXXXX.npy (frames, 126) aligned to the
view's shot files (their 'times'), + feature_names.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); RAW = ROOT / "data/raw/data"
view, split, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); k, n = (int(x) for x in sys.argv[4].split("/")) if len(sys.argv) > 4 else (0, 1)
man = json.loads((view / "manifest.json").read_text()); entries = man[split]
CURRENTS = ["magnetics_ECOILA", "magnetics_bcoil"] + [f"magnetics_F{i}{s}" for i in range(1, 10) for s in "AB"] + ["magnetics_plasma_current"]
TAUS = [2.0, 5.0, 10.0, 20.0, 50.0]; NAMES = [f"{c}_{kind}" for c in CURRENTS for kind in [f"dfilt_tau{t:g}ms" for t in TAUS] + ["slope2ms"]]
(out / split).mkdir(parents=True, exist_ok=True); (out / "feature_names.json").write_text(json.dumps(NAMES))


def filtered_derivative(t, v, tau, tq):
    """Causal exponential filter of dv/dt with time constant tau (ms), evaluated at query times tq: y' = (dv/dt - y)/tau."""
    dt = np.diff(t); dv = np.diff(v); rate = dv / np.maximum(dt, 1e-6); tm = t[1:]
    a = np.exp(-dt / tau); y = np.empty_like(rate); acc = 0.0
    for i in range(len(rate)):          # O(N) recursion; N ~ 480k per shot -> vectorised below is not trivial, loop in numpy via cumulative trick
        acc = a[i] * acc + (1 - a[i]) * rate[i]; y[i] = acc
    return np.interp(tq, tm, y, left=0.0, right=0.0)


def filtered_derivative_fast(t, v, tau, tq):
    """Same filter, vectorised with the uniform-step approximation (raw step 0.05 ms is uniform): y_i = a*y_{i-1} + (1-a)*rate_i."""
    dt = float(np.median(np.diff(t))); a = np.exp(-dt / tau); rate = np.gradient(v, dt)
    # y = (1-a) * sum_j a^(i-j) rate_j  -> use scipy.signal.lfilter
    from scipy.signal import lfilter
    y = lfilter([1 - a], [1, -a], rate)
    return np.interp(tq, t, y, left=0.0, right=0.0)


def slope2ms(t, v, tq):
    out = np.zeros(len(tq)); dt = float(np.median(np.diff(t))); w = max(2, int(round(2.0 / dt)))
    idx = np.searchsorted(t, tq)
    for j, i in enumerate(idx):
        lo, hi = max(0, i - w), min(len(t), i + 1)
        if hi - lo >= 3:
            x = t[lo:hi] - t[hi - 1]; out[j] = np.polyfit(x, v[lo:hi], 1)[0]
    return out


t0 = time.monotonic(); done = 0
for i, e in enumerate(entries):
    if i % n != k:
        continue
    target = out / split / f"shot_{i:05d}.npy"
    if target.exists():
        continue
    with np.load(e["path"] if e["path"].startswith("/") else view / e["path"]) as f:
        tq = np.asarray(f["times"], dtype=np.float64)
    shard = RAW / e["shard"].replace("data/", "", 1) if e["shard"].startswith("data/") else RAW / e["shard"]
    cols = ["magnetics_time", "magnetics_plasma_current_times"] + CURRENTS
    table = pq.read_table(shard, columns=cols)
    tb = {c: table.column(c).combine_chunks().flatten().to_numpy(zero_copy_only=False) for c in cols}   # one row per file -> that row's values
    mt = np.asarray(tb["magnetics_time"], dtype=np.float64); ipt = np.asarray(tb["magnetics_plasma_current_times"], dtype=np.float64)
    feats = np.zeros((len(tq), len(NAMES)), dtype=np.float32); col = 0
    for c in CURRENTS:
        v = np.asarray(tb[c], dtype=np.float64); tt = ipt if c == "magnetics_plasma_current" else mt
        if len(v) != len(tt) or len(v) < 10:
            col += len(TAUS) + 1; continue
        win = (tt >= tq.min() - 500.0) & (tt <= tq.max() + 1.0)
        if win.sum() >= 10:
            tt, v = tt[win], v[win]
        ok = np.isfinite(v)
        if not ok.all():
            v = np.interp(tt, tt[ok], v[ok]) if ok.sum() > 2 else np.zeros_like(v)
        for tau in TAUS:
            feats[:, col] = filtered_derivative_fast(tt, v, tau, tq); col += 1
        feats[:, col] = slope2ms(tt, v, tq); col += 1
    np.save(target, feats); done += 1
    if done % 50 == 0:
        print(f"shard {k}/{n}: {done} shots [{time.monotonic()-t0:.0f}s]", flush=True)
print(f"shard {k}/{n} done: {done} shots written [{time.monotonic()-t0:.0f}s]", flush=True)
