#!/usr/bin/env python3
"""Independent checks of stored synthetic MAST equilibria (data/c2_synth_mast_*).

For a random sample of records: (1) rebuild the vacuum field from the STORED currents with the CNN's MAST coil tables,
rebuild j = J(psi) from the stored profile parameters and lambda, and report the linear GS residual of psi - psi_vac
against -mu0 R j and the Ip integral error (self-consistency of the record as a fixed-coil free-boundary solution);
(2) compare the synthetic Thomson p_e with the REAL Thomson p_e of the same frame (max ratio, in-LCFS RMSE) as a
closure-plausibility diagnostic; (3) run the independent spline saddle check (fusion.mast_structure.check_separatrix)
for the diverted/limited classification; (4) summarise the official derived scalars of the sample.
Writes <output>/checks.json. No MAST labels are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion import c2_fields_data as cd  # noqa: E402
from fusion import c2_synthetic as cs  # noqa: E402
from fusion import physics_fields as pf  # noqa: E402
from fusion.mast_physics import MU0  # noqa: E402
from fusion.mast_structure import check_separatrix  # noqa: E402


def quantiles(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return {"count": 0}
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"count": int(len(values)), "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6]), "mean": float(values.mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/c2_synth_mast_v1"))
    p.add_argument("--output", type=Path, default=Path("results/challenge2/c2_synth_mast_v1"))
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    setup = cd.build_machine_setup("MAST")
    solver = cs.SyntheticMASTSolver(setup)
    row = pf.read_shot_row(cd.reference_parquet("MAST"))
    rng = np.random.default_rng(args.seed)
    shards = manifest["shards"]
    per_shard = max(1, args.samples // len(shards))
    paths = cd.mast_parquet_paths()
    shot_cache = {}
    out = []
    for sh in shards:
        data = cs.read_shard(args.data_dir / sh["path"])
        n = len(data["psi"])
        for i in rng.choice(n, size=min(per_shard, n), replace=False):
            psi = data["psi"][i].astype(np.float64)
            currents = data["currents"][i].astype(np.float64)
            params = cs.ProfileParams(*[float(v) for v in data["params"][i]])
            ip_A = float(data["ip_kA"][i]) * 1000.0
            vac = solver.vacuum(currents)
            j, lam, surface = solver.profile_current(psi, params, ip_A, n_iter=22)
            plasma = psi - vac
            rec = {"shard": sh["path"], "index": int(i), "shot_index": int(data["shot_index"][i]), "frame_index": int(data["frame_index"][i]),
                   "gs_relative_residual": solver.solver.residual(plasma, j), "lambda_rel_diff": float(abs(lam - data["lam"][i]) / data["lam"][i]),
                   "ip_integral_rel_error": float(abs(j.sum() * solver.dA - ip_A) / ip_A),
                   "kappa": float(data["kappa"][i]), "li": float(data["li"][i]), "q95": float(data["q95"][i]), "betaN": float(data["betaN"][i]),
                   "diverted_flag": bool(data["diverted"][i] > 0.5), "dI5_kAt": float(data["dI5_kAt"][i]), "dI6_kAt": float(data["dI6_kAt"][i])}
            # boundary check: plasma psi on the grid boundary equals the free-space Green integral (already imposed by the
            # solver); report the max |psi_plasma| on the boundary relative to the axis-edge difference
            C, (iz, ir), axis_val, edge, inside = surface
            border = np.concatenate([plasma[0], plasma[-1], plasma[:, 0], plasma[:, -1]])
            rec["boundary_plasma_flux_rel"] = float(np.abs(border).max() / (axis_val - edge))
            try:
                structure = check_separatrix(psi, solver.R, solver.Z, C, edge, axis_val - edge, row)
                rec["separatrix_supported"] = bool(structure["separatrix_supported"])
                rec["n_saddles"] = len(structure["saddles"])
            except Exception as exc:  # noqa: BLE001
                rec["separatrix_supported"] = None
                rec["separatrix_error"] = repr(exc)
            # real Thomson comparison
            si = rec["shot_index"]
            if si not in shot_cache:
                shot_cache[si] = cd.mast_shot_inputs(paths[si], setup)
            shot = shot_cache[si]
            local = int(np.flatnonzero(shot.frames == rec["frame_index"])[0])
            real_pe, mask = shot.pe[local].astype(float), shot.pe_mask[local]
            syn_pe = data["pe"][i][:len(real_pe)].astype(float)
            valid = mask & (real_pe > 0)
            if valid.sum() > 10:
                rec["pe_max_ratio_syn_over_real"] = float(syn_pe[valid].max() / real_pe[valid].max())
                rec["pe_rmse_over_real_rms"] = float(np.sqrt(np.mean((syn_pe[valid] - real_pe[valid]) ** 2)) / np.sqrt(np.mean(real_pe[valid] ** 2)))
                r = shot.chord_R[:len(real_pe)]
                hot = valid & (real_pe > 0.1 * real_pe[valid].max())
                rec["real_hot_chord_R_range"] = [float(r[hot].min()), float(r[hot].max())] if hot.any() else None
                rec["synthetic_pe_support_R_range"] = [float(r[syn_pe > 0].min()), float(r[syn_pe > 0].max())] if (syn_pe > 0).any() else None
                rec["real_ip_kA"] = float(shot.Ip_kA[local])
                rec["stored_ip_kA"] = float(data["ip_kA"][i])
                rec["currents_max_abs_change_kAt"] = float(np.abs(currents - shot.currents[local]).max())
            out.append(rec)
    summary = {"samples": len(out), "data_dir": str(args.data_dir),
               "gs_relative_residual": quantiles([r["gs_relative_residual"] for r in out]),
               "ip_integral_rel_error": quantiles([r["ip_integral_rel_error"] for r in out]),
               "lambda_rel_diff": quantiles([r["lambda_rel_diff"] for r in out]),
               "boundary_plasma_flux_rel": quantiles([r["boundary_plasma_flux_rel"] for r in out]),
               "separatrix_supported_fraction": float(np.mean([r["separatrix_supported"] for r in out if r.get("separatrix_supported") is not None])),
               "diverted_flag_fraction": float(np.mean([r["diverted_flag"] for r in out])),
               "separatrix_supported_given_diverted_flag": float(np.mean([r["separatrix_supported"] for r in out if r["diverted_flag"] and r.get("separatrix_supported") is not None])),
               "pe_max_ratio_syn_over_real": quantiles([r.get("pe_max_ratio_syn_over_real") for r in out]),
               "pe_rmse_over_real_rms": quantiles([r.get("pe_rmse_over_real_rms") for r in out]),
               "currents_max_abs_change_kAt": quantiles([r.get("currents_max_abs_change_kAt") for r in out]),
               "ip_match": bool(all(abs(r.get("real_ip_kA", 0) - r.get("stored_ip_kA", 0)) < 1e-3 for r in out if "real_ip_kA" in r)),
               "records": out}
    (args.output / "checks.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, default=float, indent=1))


if __name__ == "__main__":
    main()
