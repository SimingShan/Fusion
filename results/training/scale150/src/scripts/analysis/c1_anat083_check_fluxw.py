#!/usr/bin/env python3
"""Checker job: seam-free flux-weighted oracle counterfactuals for the 31-member blend (blend_gather_083 all_good).

w is a smooth function of the PREDICTED normalised flux s_p = (psi_p - psi_axis_p)/(psi_bdry_p - psi_axis_p), so
psi' = psi_p + w(s_p) * (psi_gt - psi_p) has no spatial seam. Scored with official.score_shot / finalize_machine
exactly as scripts/evaluate.py. Output: results/analysis/c1_error_anatomy_083_check/fluxw_ceilings.json
"""
import sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import c1_error_anatomy_083 as A
from c1_error_anatomy_083 import official, evaluate, N_SCALARS, N_CONS, PSI_SIGN, OUT_DIR
CHK = ROOT / "results/analysis/c1_error_anatomy_083_check"
VAR = ["blend", "core_s03", "core_s05", "core_s07", "interior_s09", "edge_band", "axis_scale_oracle"]


def smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def worker(vi, task):
    df = pd.read_parquet(OUT_DIR / "frames.parquet", columns=["validation_index", "frame", "psi_axis_pred", "psi_bdry_pred",
                                                               "psi_axis_gt", "psi_bdry_gt"])
    sub = df[df.validation_index == vi].sort_values("frame")
    gt, R, Z, coarse, fine, frames, ref, X, times = A.load_gt_and_ref(task)
    with np.load(OUT_DIR / "blend" / f"shot_{vi:04d}.npz") as z:
        blend = z["psirz"]; bq = z["q95"]; bb = z["betaN"]
    gpsi = gt["psi"].astype(np.float64)
    st = {v: blend.copy() for v in VAR}
    for k in range(blend.shape[0]):
        pp = blend[k]; pg = gpsi[k]; r = sub.iloc[k]
        ax_p, bd_p = r.psi_axis_pred, r.psi_bdry_pred
        if not (np.isfinite(ax_p) and np.isfinite(bd_p)) or ax_p == bd_p:
            continue
        s = (pp - ax_p) / (bd_p - ax_p)
        d = pg - pp
        st["core_s03"][k] = pp + (1 - smoothstep((s - 0.3) / 0.3)) * d
        st["core_s05"][k] = pp + (1 - smoothstep((s - 0.5) / 0.3)) * d
        st["core_s07"][k] = pp + (1 - smoothstep((s - 0.7) / 0.3)) * d
        st["interior_s09"][k] = pp + (1 - smoothstep((s - 0.9) / 0.3)) * d
        st["edge_band"][k] = pp + smoothstep((s - 0.5) / 0.2) * (1 - smoothstep((s - 0.95) / 0.15)) * d
        # pure gauge-free core reshaping: raise/lower the well by a smooth bump that vanishes for s>0.6
        # (oracle: match GT axis value only) -> tests whether axis level alone moves li/R_axis
        if np.isfinite(r.psi_axis_gt):
            st["axis_scale_oracle"][k] = pp + (1 - smoothstep(s / 0.6)) * (r.psi_axis_gt - ax_p)
    parts = {v: official.score_shot(gt, ref, {"psirz": st[v], "q95": bq, "betaN": bb}, R, Z, coarse, fine, PSI_SIGN, means_g)
             for v in VAR}
    return vi, parts


means_g = None


def init(m):
    global means_g; means_g = m


def main():
    tasks = A.make_tasks()
    with ProcessPoolExecutor(16) as ex:
        stats_list = list(ex.map(A.stats_worker, tasks))
    stats = {k: sum(s[k] for s in stats_list) for k in stats_list[0]}
    means = (stats["psi_sum"] / stats["psi_n"],
             np.divide(stats["scal_sum"], stats["scal_n"], out=np.zeros(N_SCALARS), where=stats["scal_n"] > 0),
             np.divide(stats["cons_sum"], stats["cons_n"], out=np.zeros(N_CONS), where=stats["cons_n"] > 0))
    accs = {v: official.Accum() for v in VAR}
    for a in accs.values():
        a.psi_sign = PSI_SIGN
    t0 = time.time()
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i]["n_frames"])
    with ProcessPoolExecutor(16, initializer=init, initargs=(means,)) as ex:
        futs = [ex.submit(worker, i, tasks[i]) for i in order]
        for n, f in enumerate(as_completed(futs), 1):
            vi, parts = f.result()
            for v in VAR:
                accs[v].add(parts[v])
            print(f"{n}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    res = {v: official.finalize_machine(accs[v], stats) for v in VAR}
    b = res["blend"]["S"]
    for v in VAR:
        res[v]["delta_S_vs_blend"] = res[v]["S"] - b
        r = res[v]
        print(f"{v:18s} S={r['S']:.6f} dS={r['S'] - b:+.6f} r2psi={r['r2_psi']:.6f} qb={r['r2_qb']:.6f} D={r['dlcfs']:.6f} "
              f"cons={r['consistency']:.6f} " + " ".join(f"{k}={x:.4f}" for k, x in r["r2_cons_each"].items()), flush=True)
    CHK.mkdir(parents=True, exist_ok=True)
    (CHK / "fluxw_ceilings.json").write_text(json.dumps(evaluate.json_safe(res), indent=2) + "\n")


if __name__ == "__main__":
    main()
