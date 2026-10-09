#!/usr/bin/env python3
"""Label-free comparison of the v3 (DIII-D-only) and v4 (synthetic-augmented) MAST predictions.

Per frame: grid mean, axis value (max), boundary mean, peak-to-peak for both models, their differences, and the
input-side quantities I_sol, dI_sol/dt, Ip, time, plus the solenoid/PF/plasma decomposition of phi_ext from
fusion.physics_fields. Regress the v4-v3 mean difference on these to see what drives the gauge disagreement.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion import physics_fields as pf
from fusion import c2_fields_data as cd

def main():
    out = ROOT / "results/analysis/mast_gauge_v3_vs_v4"; out.mkdir(parents=True, exist_ok=True)
    v3 = np.load(ROOT / "results/challenge2/c2_cnn_v3_main/mast/mast_public_test.npz")
    v4 = np.load(ROOT / "results/challenge2/c2_cnn_v4_synth_gdev/mast/mast_public_test.npz")
    paths = cd.mast_parquet_paths()
    setup = cd.build_machine_setup("MAST")
    rows = []
    step = max(1, len(paths) // 150)
    for i in range(0, len(paths), step):
        inp = cd.mast_shot_inputs(paths[i], setup) if hasattr(cd, "mast_shot_inputs") else None
        if inp is None:
            continue
        a3 = v3[f"shot_{i:04d}_psirz"]; a4 = v4[f"shot_{i:04d}_psirz"]
        T = a3.shape[0]
        names = list(setup.table_names)
        cmat = np.asarray(inp.currents, dtype=float)  # (T, C) kA in table order
        sol_idx = [k for k, n in enumerate(names) if "sol" in n]
        pf_idx = [k for k, n in enumerate(names) if "sol" not in n]
        sol = cmat[:, sol_idx].sum(1) if sol_idx else np.zeros(T)
        t = np.asarray(inp.times_ms, dtype=float)
        dsol = np.gradient(sol, t / 1000.0) if T > 1 else np.zeros(T)
        ip = np.asarray(getattr(inp, "ip_kA", getattr(inp, "Ip_kA", np.zeros(T))), dtype=float)
        table = np.asarray(setup.table_matrix)  # (C, 4225) Wb/rad per kA
        sol_flux = (cmat[:, sol_idx] @ table[sol_idx]).mean(1) if sol_idx else np.zeros(T)
        pf_flux = (cmat[:, pf_idx] @ table[pf_idx]).mean(1)
        for k in range(T):
            rows.append(dict(shot=i, frame=k, t_ms=float(t[k]), Ip=float(ip[k]), sol=float(sol[k]), dsol=float(dsol[k]),
                             sol_flux_mean=float(sol_flux[k]), pf_flux_mean=float(pf_flux[k]),
                             m3=float(a3[k].mean()), m4=float(a4[k].mean()), ax3=float(a3[k].max()), ax4=float(a4[k].max()),
                             ptp3=float(np.ptp(a3[k])), ptp4=float(np.ptp(a4[k])),
                             rms_diff=float(np.sqrt(np.mean((a4[k] - a3[k]) ** 2))),
                             rms_diff_demeaned=float(np.sqrt(np.mean(((a4[k] - a3[k]) - (a4[k] - a3[k]).mean()) ** 2)))))
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "frames.csv", index=False)
    d = df["m4"] - df["m3"]
    rep = {"n_frames": len(df), "n_shots": int(df.shot.nunique()),
           "mean_diff_v4_minus_v3": {"mean": float(d.mean()), "std": float(d.std()), "p05": float(d.quantile(.05)), "p95": float(d.quantile(.95))},
           "axis_diff_v4_minus_v3": {"mean": float((df.ax4 - df.ax3).mean()), "std": float((df.ax4 - df.ax3).std())},
           "ptp": {"v3_median": float(df.ptp3.median()), "v4_median": float(df.ptp4.median())},
           "rms_diff_median": float(df.rms_diff.median()), "rms_diff_demeaned_median": float(df.rms_diff_demeaned.median()),
           "corr_meandiff": {k: float(np.corrcoef(d, df[k])[0, 1]) for k in ("sol", "dsol", "Ip", "t_ms", "sol_flux_mean", "pf_flux_mean")},
           "sol_flux_mean_stats": {"p05": float(df.sol_flux_mean.quantile(.05)), "median": float(df.sol_flux_mean.median()), "p95": float(df.sol_flux_mean.quantile(.95))},
           "pf_flux_mean_stats": {"p05": float(df.pf_flux_mean.quantile(.05)), "median": float(df.pf_flux_mean.median()), "p95": float(df.pf_flux_mean.quantile(.95))}}
    X = np.column_stack([np.ones(len(df)), df.sol, df.dsol, df.Ip / 1000.0])
    coef, *_ = np.linalg.lstsq(X, d.values, rcond=None)
    pred = X @ coef; rep["ols_meandiff_on_[1,sol_kA,dsol_kA_per_s,Ip_MA]"] = {"coef": coef.tolist(), "r2": float(1 - ((d - pred) ** 2).sum() / ((d - d.mean()) ** 2).sum())}
    (out / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
