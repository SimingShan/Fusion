#!/usr/bin/env python3
"""Independent checks of stored synthetic MAST equilibria, extended for the v9 shards (data/c2_synth_mast_v9).

v9 copy of check_synthetic_mast_v3.py: the profile is reconstructed WITH the v9 edge term (edge_amp / edge_w record scalars),
the stored current centroid, closure beta_p (beta_p_achieved) and edge-current ratio are recomputed from the stored psi, and
the reference set for the X-point comparison defaults to v3. Everything else is the v3 checker verbatim.

Extends scripts/challenge2/check_synthetic_mast.py (unchanged) with the v3 unobserved-perturbation bookkeeping:
(1) GS self-consistency: the plasma flux is psi - psi_vac(currents_solve) - psi_passive(passive_currents_kA, geometry
    table from the manifest); the linear GS residual against -mu0 R J(psi) and the Ip integral are reported WITH the
    passive flux included, and, for contrast, with the passive flux omitted and with the RECORD currents instead of
    the solve currents (these two must be worse: they are the unobserved perturbations the network has to average over).
(2) Record inputs: ``currents`` must equal the REAL frame currents + the P5/P6 control corrections only (max deviation
    from real inputs on coils other than P5/P6 must be ~0), i.e. no passive current or coil noise leaks into the inputs.
(3) X-point classification (fusion.c2_synthetic.find_xpoints) on the sample and on a reference set (v1) for the
    double-null fraction comparison; independent spline saddle check (fusion.mast_structure.check_separatrix).
(4) The unchanged reader API: cs.load_synthetic_shots(data_dir, split, table_names) on both splits.
(5) Real-Thomson plausibility of p_e as in the v1 checker.
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
from fusion.mast_structure import check_separatrix  # noqa: E402


def quantiles(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return {"count": 0}
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"count": int(len(values)), "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6]), "mean": float(values.mean())}


def dn_fractions(classes):
    classes = np.asarray(classes, int)
    names = {0: "single_null_lower", 1: "single_null_upper", 2: "near_double_null", 3: "connected_double_null", -1: "unknown"}
    return {names[k]: float(np.mean(classes == k)) for k in names} | {"count": int(len(classes))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/c2_synth_mast_v9"))
    p.add_argument("--output", type=Path, default=Path("results/challenge2/c2_synth_mast_v9"))
    p.add_argument("--reference-dir", type=Path, default=Path("data/c2_synth_mast_v3"), help="earlier set for the X-point classification comparison")
    p.add_argument("--samples", type=int, default=240)
    p.add_argument("--reference-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    setup = cd.build_machine_setup("MAST")
    solver = cs.SyntheticMASTSolver(setup)
    row = pf.read_shot_row(cd.reference_parquet("MAST"))
    rng = np.random.default_rng(args.seed)
    v3 = manifest.get("v3", {})
    geometry = v3.get("passive_geometry")
    green = cs.passive_green_table(solver.R, solver.Z, geometry) if geometry else None
    shards = manifest["shards"]
    per_shard = max(1, args.samples // len(shards))
    paths = cd.mast_parquet_paths()
    i5 = [solver.i5u, solver.i5l]
    i6 = [solver.i6u, solver.i6l]
    other = [k for k in range(len(setup.table_names)) if k not in i5 + i6]
    shot_cache = {}
    out = []
    for sh in shards:
        data = cs.read_shard(args.data_dir / sh["path"])
        n = len(data["psi"])
        for i in rng.choice(n, size=min(per_shard, n), replace=False):
            psi = data["psi"][i].astype(np.float64)
            currents = data["currents"][i].astype(np.float64)
            params = cs.ProfileParams(*[float(v) for v in data["params"][i]])
            if "edge_amp" in data:      # v9 edge term
                params = cs.ProfileParams(*params.as_array(), float(data["edge_amp"][i]), float(data["edge_w"][i]))
            ip_A = float(data["ip_kA"][i]) * 1000.0
            j, lam, surface = solver.profile_current(psi, params, ip_A, n_iter=22)
            C, (iz, ir), axis_val, edge, inside = surface
            rec = {"shard": sh["path"], "index": int(i), "shot_index": int(data["shot_index"][i]), "frame_index": int(data["frame_index"][i]),
                   "lambda_rel_diff": float(abs(lam - data["lam"][i]) / data["lam"][i]),
                   "ip_integral_rel_error": float(abs(j.sum() * solver.dA - ip_A) / ip_A),
                   "kappa": float(data["kappa"][i]), "li": float(data["li"][i]), "q95": float(data["q95"][i]), "betaN": float(data["betaN"][i]),
                   "diverted_flag": bool(data["diverted"][i] > 0.5), "dI5_kAt": float(data["dI5_kAt"][i]), "dI6_kAt": float(data["dI6_kAt"][i])}
            if "centroid_R" in data:    # v9 bookkeeping recomputed from the stored psi
                rec["centroid_R_recomputed_minus_stored"] = float(solver.centroid_r(j, solver.RR) - data["centroid_R"][i])
                rec["beta_p_closure_recomputed_minus_stored_achieved"] = float(solver.beta_p_of(psi, params, lam, surface, ip_A) - data["beta_p_achieved"][i])
                rec["beta_p_closure_recomputed_minus_record_beta_p"] = float(solver.beta_p_of(psi, params, lam, surface, ip_A) - data["beta_p"][i])
                rec["j_edge_over_axis_recomputed_minus_stored"] = float(cs.edge_current_ratio(j, psi, surface)["j_edge_over_axis"] - data["j_edge_over_axis"][i])
                rec["edge_amp"], rec["j_edge_over_axis"] = float(data["edge_amp"][i]), float(data["j_edge_over_axis"][i])
                # r_target = FINAL pinned/balanced centroid target; the seed radius is the separate scalar r_seed (iteration 3 fix)
                rec["stored_centroid_R"], rec["stored_r_target"] = float(data["centroid_R"][i]), float(data["r_target"][i])
                rec["stored_r_seed"] = float(data["r_seed"][i]) if "r_seed" in data else np.nan
                rec["beta0_at_bound"], rec["position_mode_balance"] = float(data["beta0_at_bound"][i]), float(data["position_mode_balance"][i])
                rec["position_mode_chord"] = float(data["position_mode_chord"][i]) if "position_mode_chord" in data else 0.0
                if rec["position_mode_chord"] > 0.5:
                    # chord mode: the observable (core centre / LCFS midplane centre of the equilibrium's own chord pressure) is
                    # recomputed from the stored psi on the stored chord geometry and compared with the stored value and target
                    state = {"psi": psi, "params": params, "lam": lam, "surface": surface, "ip_A": ip_A}
                    match = manifest.get("v9", {}).get("options", {}).get("chord_match", "axis")
                    try:
                        r_syn, _ = cs.chord_position_observable(solver, state, data["chord_R"][i], data["chord_Z"][i], match=match)
                    except ValueError:
                        r_syn = np.nan
                    rec["chord_R_syn_recomputed_minus_stored"] = float(r_syn - data["chord_R_syn"][i])
                    rec["chord_target_minus_syn_recomputed"] = float(data["chord_R_target"][i] - r_syn)
                    rec["chord_steps"] = float(data["chord_steps"][i])
                    rec["beta_p_over_ref"] = float(data["beta_p"][i] / data["beta_p_ref"][i]) if "beta_p_ref" in data and data["beta_p_ref"][i] > 0 else np.nan
            vac_record = solver.vacuum(currents)
            solve_currents = data["currents_solve"][i].astype(np.float64) if "currents_solve" in data else currents
            vac_solve = solver.vacuum(solve_currents)
            passive = np.zeros_like(psi)
            if green is not None and "passive_currents_kA" in data:
                passive = np.tensordot(data["passive_currents_kA"][i].astype(np.float64), green, axes=(0, 0))
            rec["gs_relative_residual"] = solver.solver.residual(psi - vac_solve - passive, j)           # the true solve setup
            rec["gs_residual_without_passive"] = solver.solver.residual(psi - vac_solve, j)
            rec["gs_residual_record_currents_no_passive"] = solver.solver.residual(psi - vac_record, j)
            border = np.concatenate([(psi - vac_solve - passive)[0], (psi - vac_solve - passive)[-1], (psi - vac_solve - passive)[:, 0], (psi - vac_solve - passive)[:, -1]])
            rec["boundary_plasma_flux_rel"] = float(np.abs(border).max() / (axis_val - edge))
            # boundary condition check: plasma flux on the grid boundary vs the free-space Green integral of j
            green_boundary = solver.solver.boundary_green @ j.ravel()[solver.solver.source_indices]
            plasma_b = (psi - vac_solve - passive).ravel()[solver.solver.boundary]
            rec["boundary_green_rel_error"] = float(np.abs(plasma_b - green_boundary).max() / max(np.abs(green_boundary).max(), 1e-30))
            rec["psi_passive_axis"] = float(passive[iz, ir])
            rec["psi_passive_rms"] = float(np.sqrt(np.mean(passive ** 2)))
            m = cs.double_null_metrics(psi, solver.R, solver.Z, surface, solver.envelope)
            rec["dn_asym"] = m["dn_asym"]
            rec["dr_sep_m"] = m["dr_sep_m"]
            rec["dn_class"] = cs.classify_double_null(m["dn_asym"])
            rec["stored_dn_class"] = int(data["dn_class"][i]) if "dn_class" in data else None
            rec["dn_mode"] = int(data["dn_mode"][i]) if "dn_mode" in data else None
            lim = np.nanmax([m["xpt_up_psi"], m["xpt_low_psi"]])
            rec["limiting_saddle_vs_edge_rel"] = float((lim - edge) / (axis_val - edge)) if np.isfinite(lim) else None
            try:
                structure = check_separatrix(psi, solver.R, solver.Z, C, edge, axis_val - edge, row)
                rec["separatrix_supported"] = bool(structure["separatrix_supported"])
                rec["n_saddles"] = len(structure["saddles"])
            except Exception as exc:  # noqa: BLE001
                rec["separatrix_supported"] = None
                rec["separatrix_error"] = repr(exc)
            si = rec["shot_index"]
            if si not in shot_cache:
                shot_cache[si] = cd.mast_shot_inputs(paths[si], setup)
            shot = shot_cache[si]
            local = int(np.flatnonzero(shot.frames == rec["frame_index"])[0])
            real = shot.currents[local].astype(np.float64)
            rec["inputs_other_coils_max_abs_dev_from_real_kAt"] = float(np.abs(currents[other] - real[other]).max())
            rec["inputs_p5_dev_minus_dI5_kAt"] = float(np.abs(currents[i5] - real[i5] - rec["dI5_kAt"]).max())
            rec["inputs_p6_dev_minus_dI6_kAt"] = float(max(abs(currents[solver.i6u] - real[solver.i6u] - rec["dI6_kAt"]),
                                                            abs(currents[solver.i6l] - real[solver.i6l] + rec["dI6_kAt"])))
            rec["solve_currents_rel_dev_from_record"] = float(np.max(np.abs(solve_currents - currents) / np.maximum(np.abs(currents), 1e-3)))
            rec["real_ip_kA"], rec["stored_ip_kA"] = float(shot.Ip_kA[local]), float(data["ip_kA"][i])
            real_pe, mask = shot.pe[local].astype(float), shot.pe_mask[local]
            syn_pe = data["pe"][i][:len(real_pe)].astype(float)
            valid = mask & (real_pe > 0)
            if valid.sum() > 10:
                rec["pe_max_ratio_syn_over_real"] = float(syn_pe[valid].max() / real_pe[valid].max())
                rec["pe_rmse_over_real_rms"] = float(np.sqrt(np.mean((syn_pe[valid] - real_pe[valid]) ** 2)) / np.sqrt(np.mean(real_pe[valid] ** 2)))
            out.append(rec)
    # reference-set X-point classification (v1 has no stored dn fields)
    reference = None
    if args.reference_dir is not None and (args.reference_dir / "manifest.json").exists():
        ref_manifest = json.loads((args.reference_dir / "manifest.json").read_text())
        per = max(1, args.reference_samples // len(ref_manifest["shards"]))
        classes, asyms, drs = [], [], []
        for sh in ref_manifest["shards"]:
            data = cs.read_shard(args.reference_dir / sh["path"])
            for i in rng.choice(len(data["psi"]), size=min(per, len(data["psi"])), replace=False):
                psi = data["psi"][i].astype(np.float64)
                surf = cs.plasma_surface(psi, solver.R, solver.Z, solver.envelope, 22)
                if surf is None:
                    classes.append(-1); continue
                m = cs.double_null_metrics(psi, solver.R, solver.Z, surf, solver.envelope)
                classes.append(cs.classify_double_null(m["dn_asym"])); asyms.append(m["dn_asym"]); drs.append(m["dr_sep_m"])
        reference = {"data_dir": str(args.reference_dir), "double_null_fractions": dn_fractions(classes),
                     "dn_asym": quantiles(asyms), "abs_dn_asym": quantiles(np.abs(asyms)), "dr_sep_mm": quantiles(1000 * np.asarray(drs))}
    # reader API
    api = {}
    for split in ("train", "val"):
        try:
            shots = cs.load_synthetic_shots(args.data_dir, split, setup.table_names, max_frames=2000)
            api[split] = {"shots": len(shots), "frames": int(sum(s.n_frames for s in shots)), "currents_shape": list(shots[0].currents.shape),
                          "phi_shape": list(shots[0].phi.shape), "scalars_shape": list(shots[0].scalars.shape),
                          "meta_keys": sorted(shots[0].meta.keys())}
        except Exception as exc:  # noqa: BLE001
            api[split] = {"error": repr(exc)}
    summary = {"samples": len(out), "data_dir": str(args.data_dir), "passive_geometry_version": v3.get("options", {}).get("passive_geometry"),
               "gs_relative_residual_with_passive_and_solve_currents": quantiles([r["gs_relative_residual"] for r in out]),
               "gs_residual_without_passive": quantiles([r["gs_residual_without_passive"] for r in out]),
               "gs_residual_record_currents_no_passive": quantiles([r["gs_residual_record_currents_no_passive"] for r in out]),
               "boundary_green_rel_error": quantiles([r["boundary_green_rel_error"] for r in out]),
               "ip_integral_rel_error": quantiles([r["ip_integral_rel_error"] for r in out]),
               "lambda_rel_diff": quantiles([r["lambda_rel_diff"] for r in out]),
               "boundary_plasma_flux_rel": quantiles([r["boundary_plasma_flux_rel"] for r in out]),
               "psi_passive_axis": quantiles([r["psi_passive_axis"] for r in out]), "psi_passive_rms": quantiles([r["psi_passive_rms"] for r in out]),
               "inputs_other_coils_max_abs_dev_from_real_kAt": quantiles([r["inputs_other_coils_max_abs_dev_from_real_kAt"] for r in out]),
               "inputs_p5_dev_minus_dI5_kAt": quantiles([r["inputs_p5_dev_minus_dI5_kAt"] for r in out]),
               "inputs_p6_dev_minus_dI6_kAt": quantiles([r["inputs_p6_dev_minus_dI6_kAt"] for r in out]),
               "solve_currents_rel_dev_from_record": quantiles([r["solve_currents_rel_dev_from_record"] for r in out]),
               "ip_match": bool(all(abs(r["real_ip_kA"] - r["stored_ip_kA"]) < 1e-3 for r in out)),
               "separatrix_supported_fraction": float(np.mean([r["separatrix_supported"] for r in out if r.get("separatrix_supported") is not None])),
               "diverted_flag_fraction": float(np.mean([r["diverted_flag"] for r in out])),
               "double_null_fractions": dn_fractions([r["dn_class"] for r in out]),
               "double_null_fractions_dn_mode": dn_fractions([r["dn_class"] for r in out if r["dn_mode"] == 1]) if any(r["dn_mode"] for r in out) else None,
               "double_null_fractions_standard_mode": dn_fractions([r["dn_class"] for r in out if r["dn_mode"] == 0]) if any(r["dn_mode"] is not None for r in out) else None,
               "stored_vs_recomputed_dn_class_agreement": float(np.mean([r["dn_class"] == r["stored_dn_class"] for r in out if r["stored_dn_class"] is not None])) if any(r["stored_dn_class"] is not None for r in out) else None,
               "dn_asym": quantiles([r["dn_asym"] for r in out]), "abs_dn_asym": quantiles([abs(r["dn_asym"]) for r in out]),
               "dr_sep_mm": quantiles([1000 * r["dr_sep_m"] for r in out]),
               "limiting_saddle_vs_edge_rel": quantiles([r["limiting_saddle_vs_edge_rel"] for r in out]),
               "pe_max_ratio_syn_over_real": quantiles([r.get("pe_max_ratio_syn_over_real") for r in out]),
               "v9": {"centroid_R_recomputed_minus_stored": quantiles([r.get("centroid_R_recomputed_minus_stored") for r in out]),
                      "beta_p_closure_recomputed_minus_stored_achieved": quantiles([r.get("beta_p_closure_recomputed_minus_stored_achieved") for r in out]),
                      "beta_p_closure_recomputed_minus_record_beta_p": quantiles([r.get("beta_p_closure_recomputed_minus_record_beta_p") for r in out]),
                      "j_edge_over_axis_recomputed_minus_stored": quantiles([r.get("j_edge_over_axis_recomputed_minus_stored") for r in out]),
                      "j_edge_over_axis": quantiles([r.get("j_edge_over_axis") for r in out]), "edge_amp": quantiles([r.get("edge_amp") for r in out]),
                      "centroid_R": quantiles([r.get("stored_centroid_R") for r in out]),
                      "centroid_R_minus_r_target_final": quantiles([r.get("stored_centroid_R", np.nan) - r.get("stored_r_target", np.nan) for r in out]),
                      "centroid_R_minus_seed": quantiles([r.get("stored_centroid_R", np.nan) - r.get("stored_r_seed", np.nan) for r in out]),
                      "chord_R_syn_recomputed_minus_stored": quantiles([r.get("chord_R_syn_recomputed_minus_stored") for r in out if "chord_R_syn_recomputed_minus_stored" in r]) if any("chord_R_syn_recomputed_minus_stored" in r for r in out) else None,
                      "chord_target_minus_syn_recomputed": quantiles([r.get("chord_target_minus_syn_recomputed") for r in out if "chord_target_minus_syn_recomputed" in r]) if any("chord_target_minus_syn_recomputed" in r for r in out) else None,
                      "abs_chord_target_minus_syn_recomputed": quantiles([abs(r.get("chord_target_minus_syn_recomputed")) for r in out if "chord_target_minus_syn_recomputed" in r]) if any("chord_target_minus_syn_recomputed" in r for r in out) else None,
                      "chord_steps": quantiles([r.get("chord_steps") for r in out if "chord_steps" in r]) if any("chord_steps" in r for r in out) else None,
                      "beta_p_over_ref": quantiles([r.get("beta_p_over_ref") for r in out if "beta_p_over_ref" in r]) if any("beta_p_over_ref" in r for r in out) else None,
                      "position_mode_chord_fraction": float(np.nanmean([r.get("position_mode_chord", np.nan) for r in out])) if any("position_mode_chord" in r for r in out) else None,
                      "beta0_at_bound_fraction": float(np.nanmean([r.get("beta0_at_bound", np.nan) for r in out])) if any("beta0_at_bound" in r for r in out) else None,
                      "position_mode_balance_fraction": float(np.nanmean([r.get("position_mode_balance", np.nan) for r in out])) if any("position_mode_balance" in r for r in out) else None},
               "pe_rmse_over_real_rms": quantiles([r.get("pe_rmse_over_real_rms") for r in out]),
               "reference_set": reference, "reader_api": api, "records": out}
    (args.output / "checks.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, default=float, indent=1))


if __name__ == "__main__":
    main()
