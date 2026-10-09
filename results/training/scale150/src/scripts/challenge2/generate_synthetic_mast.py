#!/usr/bin/env python3
"""Generate synthetic free-boundary MAST equilibria on real public-test inputs (fusion.c2_synthetic).

Frames are sampled uniformly over the 1206 official MAST shots and uniformly within each shot. For every
frame the REAL coil/solenoid currents, Ip and TF current are used; a FreeGS-style profile
(beta0, alpha_p, gamma_p, alpha_f, gamma_f) is sampled, the current-centroid radius r_target and height
z_target are sampled, and the outer loop of ``SyntheticMASTSolver.solve_frame`` adjusts beta0 until the
REAL vacuum field holds the plasma at r_target (radial force balance); only the small P6 vertical-control
correction (and the sub-kA-turn residual P5 correction) is added to the stored coil currents.

Outputs (``--output``): shards ``shard_XXXX.npz`` (train/val split by SOURCE SHOT), ``manifest.json``,
``report.json`` (achieved distributions of kappa, li, volume, R/Z axis, q95, betaN, beta_p, psi peak-to-peak,
LCFS extents, control efforts, convergence and throughput) and ``failures.json``.

Example (bigmem, 32 workers, ~20k equilibria):
    python scripts/challenge2/generate_synthetic_mast.py --output data/c2_synth_mast_v1 --target 20000 --workers 32
No MAST labels exist locally and none are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion import c2_fields_data as cd  # noqa: E402
from fusion import c2_synthetic as cs  # noqa: E402

_WORKER = {}


# v9 argparse keys: left out of report/manifest 'arguments' when no v9 option is on (v1-v3 manifests stay byte-identical)
V9_ARG_KEYS = {"position_mode", "proxy_ratio_range", "beta_p_jitter", "beta_p_bounds", "beta_p_tol", "proxy_threshold", "proxy_a_range",
               "r_init_range", "r_init_offset", "edge_current", "edge_amp_range", "edge_w_range", "reject_unbalanced", "chord_match",
               "chord_steps", "chord_tol", "chord_accept_tol", "chord_seed_offset", "beta_p_consistency"}


def _init_worker(config):
    setup = cd.build_machine_setup("MAST")
    _WORKER["setup"] = setup
    _WORKER["solver"] = cs.SyntheticMASTSolver(setup, lcfs_iter=config["lcfs_iter"])
    _WORKER["config"] = config
    _WORKER["passive_green"] = None
    if config.get("vessel_current_kA", 0.0) > 0:
        geometry = cs.PASSIVE_GEOMETRIES[config["passive_geometry"]]
        _WORKER["passive_green"] = cs.passive_green_table(setup.grid_R, setup.grid_Z, geometry)
        _WORKER["passive_geometry"] = geometry


def _boundary(a):
    return np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])


def quantiles(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return {"count": 0}
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"count": int(len(values)), "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6]), "mean": float(values.mean()), "std": float(values.std())}


def process_shot(task):
    """Solve the sampled frames of one shot; returns records and per-attempt statistics."""
    setup, solver, config = _WORKER["setup"], _WORKER["solver"], _WORKER["config"]
    shot_index, path, n_sample, seed = task
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    out = {"shot_index": shot_index, "path": str(path), "records": [], "attempts": 0, "accepted": 0, "reasons": {},
           "solve_seconds": [], "errors": []}
    try:
        shot = cd.mast_shot_inputs(Path(path), setup)
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"load:{exc!r}")
        return out
    T = shot.n_frames
    frames = np.sort(rng.choice(T, size=min(n_sample, T), replace=False))
    ranges = config["ranges"]
    prev_j = None
    green = _WORKER.get("passive_green")
    sigma_v = config.get("vessel_current_kA", 0.0)
    coil_noise = config.get("coil_noise", 0.0)
    dn_fraction = config.get("double_null_fraction", 0.0)
    pe_ratio_range = config.get("pe_ratio_range", (2.0, 2.0))
    # ---- v9 options (all default-off) ------------------------------------------------------------------------- #
    position_balance = config.get("position_mode", "prior") == "balance"
    position_chord = config.get("position_mode", "prior") == "chord"
    edge_current = bool(config.get("edge_current", False))
    v9 = position_balance or position_chord or edge_current or bool(config.get("reject_unbalanced", False)) or config.get("beta_p_consistency", 0.0) > 0
    rules = dict(cs.ACCEPT, reject_beta0_bounds=True) if config.get("reject_unbalanced", False) else dict(cs.ACCEPT)
    if config.get("beta_p_consistency", 0.0) > 0:
        rules["beta_p_consistency"] = float(config["beta_p_consistency"])
    if position_chord:
        rules["chord_tol"] = float(config.get("chord_accept_tol", 0.0))
        chord_match = config.get("chord_match", "axis")
        chord_offset = config.get("chord_seed_offset")
        if chord_offset is None:
            chord_offset = -0.11 if chord_match == "axis" else 0.0     # MAST synthetic (v9 pilot): R_axis - centroid_R = 0.11 +- 0.02 m
    for local in frames:
        if not shot.Ip_observed[local] or shot.Ip_kA[local] < config["min_ip_kA"] or abs(shot.tf_kA[local]) <= 0:
            out["reasons"]["bad_inputs"] = out["reasons"].get("bad_inputs", 0) + 1
            continue
        proxy = None
        if position_balance or position_chord:
            # label-free beta_p estimate from the REAL MAST midplane Thomson chord of this frame (Pa); a frame whose
            # chord does not resolve the plasma (few points, no hot extent) is skipped
            try:
                proxy = cs.thomson_beta_p_proxy(shot.chord_R, shot.chord_Z, shot.pe[local], shot.pe_mask[local], float(shot.Ip_kA[local]) * 1000.0,
                                                layout="midplane", threshold=config.get("proxy_threshold", 0.1))
                if not proxy["extent_ok"] or not (config["proxy_a_range"][0] <= proxy["a_est"] <= config["proxy_a_range"][1]):
                    raise ValueError("proxy_extent")
            except ValueError as exc:
                out["reasons"][f"proxy_failed:{str(exc)[:24]}"] = out["reasons"].get(f"proxy_failed:{str(exc)[:24]}", 0) + 1
                continue
        for attempt in range(config["max_attempts"]):
            out["attempts"] += 1
            params = cs.ProfileParams.sample(rng, ranges)
            r_target = float(rng.uniform(*config["r_target_range"]))
            z_target = float(np.clip(rng.normal(0.0, config["z_std"]), -config["z_max"], config["z_max"]))
            v9_kw, v9_extra = {}, {}
            if edge_current:
                edge_amp, edge_w = cs.sample_edge_params(rng, tuple(config["edge_amp_range"]), tuple(config["edge_w_range"]))
                params = cs.ProfileParams(*params.as_array(), edge_amp, edge_w)
            chord_args = None
            if position_balance:
                beta_p_target, bp_info = cs.sample_beta_p_target(proxy, rng, tuple(config["proxy_ratio_range"]), config["beta_p_jitter"],
                                                                 tuple(config["beta_p_bounds"]))
                r_target = float(np.clip(proxy["Rgeo_est"] + config.get("r_init_offset", 0.0), *config["r_init_range"]))   # seed only
                v9_kw = dict(beta_p_target=beta_p_target, beta_p_tol=config.get("beta_p_tol", 0.03), r_init=r_target)
                v9_extra = {"beta_p_proxy_e": proxy["beta_p_e"], "pe_ratio_proxy": bp_info["pe_ratio_proxy"], "beta_p_jitter": bp_info["beta_p_jitter"],
                            "proxy_a_est": proxy["a_est"], "proxy_Rgeo_est": proxy["Rgeo_est"], "proxy_R_axis_est": proxy["R_axis_est"],
                            "proxy_extent_ok": float(proxy["extent_ok"]), "r_seed": r_target}
            elif position_chord:
                # position from the frame's own midplane chord: the estimator target (core centre or LCFS midplane centre) is
                # matched by the equilibrium's own chord observable (cs.solve_frame_chord); the seed radius only starts the search
                target_R = float(proxy["R_axis_est"] if chord_match == "axis" else proxy["Rgeo_est"])
                r_target = float(np.clip(target_R + chord_offset, *config["r_init_range"]))   # seed
                chord_args = (target_R, shot.chord_R, shot.chord_Z)
                v9_extra = {"beta_p_proxy_e": proxy["beta_p_e"], "proxy_a_est": proxy["a_est"], "proxy_Rgeo_est": proxy["Rgeo_est"],
                            "proxy_R_axis_est": proxy["R_axis_est"], "proxy_extent_ok": float(proxy["extent_ok"]), "r_seed": r_target,
                            "beta_p_ref": float(proxy["beta_p_e"]) * float(np.mean(config["proxy_ratio_range"]))}
            if config.get("beta_p_consistency", 0.0) > 0 and proxy is not None and "beta_p_ref" not in v9_extra:
                v9_extra["beta_p_ref"] = float(proxy["beta_p_e"]) * float(np.mean(config["proxy_ratio_range"]))
            # ---- v3 unobserved perturbations (all default-off) ------------------------------------------ #
            extra = {}
            real_currents = shot.currents[local].astype(np.float64)
            solve_currents = real_currents.copy()
            if coil_noise > 0:      # measurement / turn-count uncertainty on the currents used for the SOLVE only
                solve_currents = real_currents * (1.0 + coil_noise * rng.standard_normal(len(real_currents)))
                dvac = solver.vacuum(solve_currents) - solver.vacuum(real_currents)
                extra["coil_noise_std"] = coil_noise
                extra["coil_noise_psi_rms"] = float(np.sqrt(np.mean(dvac ** 2)))
            extra_vacuum = None
            if green is not None and sigma_v > 0:
                passive_kA, pinfo = cs.sample_passive_currents(rng, sigma_v, config["vessel_asym_frac"], config["vessel_share_std"],
                                                                _WORKER["passive_geometry"])
                extra_vacuum = np.tensordot(passive_kA, green, axes=(0, 0))
                extra["passive_currents_kA"] = passive_kA.astype(np.float32)
                extra["passive_total_kA"], extra["passive_asym_kA"] = pinfo["passive_total_kA"], pinfo["passive_asym_kA"]
                extra["psi_passive_boundary_mean"] = float(_boundary(extra_vacuum).mean())
                extra["psi_passive_boundary_absmax"] = float(np.abs(_boundary(extra_vacuum)).max())
                extra["psi_passive_rms"] = float(np.sqrt(np.mean(extra_vacuum ** 2)))
            dn_target = None
            if dn_fraction > 0 and rng.uniform() < dn_fraction:
                dn_target = float(np.clip(rng.normal(0.0, config["dn_asym_std"]), -config["dn_asym_max"], config["dn_asym_max"]))
                z_target = 0.0
            t0 = time.monotonic()
            try:
                solve_kw = dict(z_target=z_target, max_seconds=config["max_seconds"], extra_vacuum=extra_vacuum, dn_target=dn_target,
                                dn_steps=config.get("dn_steps", 4), dn_tolerance=config.get("dn_tolerance", 4e-3), **v9_kw)
                scal = None
                if chord_args is None:
                    result = solver.solve_frame(solve_currents, shot.Ip_kA[local], shot.tf_kA[local], params, r_target=r_target,
                                                initial_current=prev_j, **solve_kw)
                else:
                    result, scal = cs.solve_frame_chord(solver, chord_args[0], chord_args[1], chord_args[2],
                                                        (solve_currents, shot.Ip_kA[local], shot.tf_kA[local], params), match=chord_match,
                                                        steps=config.get("chord_steps", 2), tol=config.get("chord_tol", 0.005),
                                                        r_bounds=tuple(config["r_init_range"]), r_target=r_target, initial_current=prev_j, **solve_kw)
                if "beta_p_ref" in v9_extra:
                    result["beta_p_ref"] = v9_extra["beta_p_ref"]
                if not result["converged"]:
                    ok, reason = False, f"not_converged:{result['status']}"
                else:
                    scal = solver.scalars(result) if scal is None else scal
                    ok, reason = cs.accept_equilibrium(result, scal, rules=rules)
            except Exception as exc:  # noqa: BLE001
                ok, reason = False, f"exception:{type(exc).__name__}:{str(exc)[:60]}"
                result = None
            out["solve_seconds"].append(time.monotonic() - t0)
            if result is not None and result["converged"]:
                prev_j = result["j"]
            if not ok:
                out["reasons"][reason] = out["reasons"].get(reason, 0) + 1
                continue
            # RECORD inputs = REAL currents + the exact P5/P6 control corrections (the solve may have used noisy currents)
            d56 = result["currents_kA"] - solve_currents
            result["currents_kA"] = real_currents + d56
            pe_ratio = float(rng.uniform(*pe_ratio_range)) if pe_ratio_range[1] > pe_ratio_range[0] else float(pe_ratio_range[0])
            pe = solver.thomson(result, scal, shot.chord_R, shot.chord_Z, rng, noise=config["pe_noise"], ratio=pe_ratio)
            diverted = scal["envelope_contact_fraction"] < config["diverted_contact_max"]
            if config.get("v3_metadata", False):
                # the RECORD inputs stay the real coil currents (+ the P5/P6 control corrections); everything else is
                # analysis metadata: the currents whose vacuum field was actually solved, passive currents, X-points
                iz, ir = result["surface"][1]
                extra["currents_solve"] = (solve_currents + d56).astype(np.float32)   # currents whose vacuum field was solved
                extra["pe_ratio"] = pe_ratio
                if extra_vacuum is not None:
                    extra["psi_passive_axis"] = float(extra_vacuum[iz, ir])
                else:
                    extra["passive_total_kA"] = extra["passive_asym_kA"] = 0.0
                    extra["psi_passive_axis"] = extra["psi_passive_boundary_mean"] = extra["psi_passive_boundary_absmax"] = extra["psi_passive_rms"] = 0.0
                    extra["passive_currents_kA"] = np.zeros(1, np.float32)
                if coil_noise > 0:
                    extra["coil_noise_psi_axis"] = float(dvac[iz, ir])
                else:
                    extra["coil_noise_std"] = extra["coil_noise_psi_rms"] = extra["coil_noise_psi_axis"] = 0.0
                metrics = cs.double_null_metrics(result["psi"], solver.R, solver.Z, result["surface"], solver.envelope)
                extra.update(metrics)
                extra["dn_mode"], extra["dn_target"], extra["dn_steps"] = result["dn_mode"], result["dn_target"], result["dn_steps"]
                extra["dn_class"] = float(cs.classify_double_null(metrics["dn_asym"]))
            if v9:
                # v9 metadata (every record of a v9 run carries them so that write_shard stores them)
                extra.update(v9_extra)
                extra.update({"position_mode_balance": float(result.get("position_mode") == "balance"),
                              "position_mode_chord": float(result.get("position_mode") == "chord"),
                              "beta_p_target": float(result.get("beta_p_target", np.nan)), "beta_p_achieved": float(result.get("beta_p_achieved", np.nan)),
                              "beta0_at_bound": float(result.get("beta0_at_bound", False)),
                              "edge_amp": float(result["params"].edge_amp), "edge_w": float(result["params"].edge_w),
                              "centroid_R": solver.centroid_r(result["j"], solver.RR), "centroid_Z": solver.centroid_z(result["j"], solver.ZZ),
                              "chord_R_target": float(result.get("chord_R_target", np.nan)), "chord_R_syn": float(result.get("chord_R_syn", np.nan)),
                              "chord_steps": float(result.get("chord_steps", np.nan))})
                # r_target of the record = the FINAL pinned / balanced centroid target (== centroid_R within tolerance); the
                # seed radius (prior draw, proxy Rgeo, or chord target + offset) is stored separately as r_seed
                extra.setdefault("r_seed", float(r_target))
                for key in ("beta_p_proxy_e", "pe_ratio_proxy", "beta_p_jitter", "proxy_a_est", "proxy_Rgeo_est", "proxy_R_axis_est", "proxy_extent_ok",
                            "beta_p_ref"):
                    extra.setdefault(key, np.nan)
                extra.update(cs.edge_current_ratio(result["j"], result["psi"], result["surface"]))
            record = cs.make_record(result, scal, shot, shot_index, int(local), pe, shot.pe_mask[local], diverted, extra=extra)
            record["params_sampled"] = result["params_sampled"].as_array().astype(np.float32)
            record["beta0_final"] = float(result["params"].beta0)
            out["records"].append(record)
            out["accepted"] += 1
            out["reasons"]["ok"] = out["reasons"].get("ok", 0) + 1
            break
    out["seconds"] = time.monotonic() - started
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", type=int, default=20000, help="number of accepted equilibria to aim for")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--val-shots", type=int, default=60, help="source shots held out as the synthetic validation split")
    p.add_argument("--max-shots", type=int, default=0, help="use only the first N official shots (pilots)")
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--oversample", type=float, default=1.15, help="frames sampled per shot = oversample * target / shots")
    p.add_argument("--r-target-range", type=float, nargs=2, default=(0.78, 0.90), help="current-centroid radius target (m)")
    p.add_argument("--z-std", type=float, default=cs.Z_TARGET_STD)
    p.add_argument("--z-max", type=float, default=cs.Z_TARGET_MAX)
    p.add_argument("--ranges", type=str, default=None, help="JSON dict overriding fusion.c2_synthetic.DEFAULT_RANGES")
    p.add_argument("--pe-noise", type=float, default=0.05)
    p.add_argument("--min-ip-kA", type=float, default=100.0)
    p.add_argument("--lcfs-iter", type=int, default=14)
    p.add_argument("--max-seconds", type=float, default=90.0, help="per-equilibrium solver time limit")
    p.add_argument("--diverted-contact-max", type=float, default=0.01)
    p.add_argument("--time-limit", type=float, default=0.0, help="stop submitting new shots after this many seconds")
    # ---- v3 realism options (all default-off; see fusion/c2_synthetic.py v3 section) -------------------------- #
    p.add_argument("--vessel-current-kA", type=float, default=0.0,
                   help="sigma of the total unobserved passive-structure current I_v ~ N(0, sigma) added to the SOLVE vacuum field (0 = off)")
    p.add_argument("--vessel-asym-frac", type=float, default=0.5, help="sigma of the up/down antisymmetric passive term, as a fraction of --vessel-current-kA")
    p.add_argument("--vessel-share-std", type=float, default=0.4, help="lognormal std of the per-structure share perturbation")
    p.add_argument("--passive-geometry", type=str, default=cs.PASSIVE_GEOMETRY_VERSION, choices=sorted(cs.PASSIVE_GEOMETRIES))
    p.add_argument("--coil-noise", type=float, default=0.0, help="relative Gaussian noise on the coil currents used for the SOLVE only (record keeps the real inputs)")
    p.add_argument("--pe-ratio-range", type=float, nargs=2, default=(2.0, 2.0), help="Thomson total/electron pressure ratio sampled uniformly in this range")
    p.add_argument("--double-null-fraction", type=float, default=0.0, help="fraction of attempts solved in the double-null sampling mode")
    p.add_argument("--dn-asym-std", type=float, default=0.01, help="std of the sampled X-point flux asymmetry target (fraction of axis-edge flux)")
    p.add_argument("--dn-asym-max", type=float, default=0.03)
    p.add_argument("--dn-steps", type=int, default=4)
    p.add_argument("--dn-tolerance", type=float, default=4e-3)
    p.add_argument("--v3-metadata", action="store_true", help="store v3 analysis metadata (currents_solve, passive currents, X-points) in the shards")
    p.add_argument("--compare-report", type=Path, default=None, help="earlier report.json whose distributions are tabulated against this run")
    # ---- v9 options (all default-off; see fusion/c2_synthetic.py v9 section) -------------------------------------- #
    p.add_argument("--position-mode", choices=("prior", "balance", "chord"), default="prior",
                   help="balance: do not sample r_target; beta_p target from the frame's REAL Thomson chord (label-free proxy x ratio x jitter),"
                        " centroid radius iterated until the real vacuum field balances the plasma (dI5 = 0), P6 vertical control kept."
                        " chord (iteration 3): centroid pinned (beta0 balanced as in v1-v3, |dI5| < 1.5 kA-turn) at a radius iterated until the"
                        " midplane-chord estimator (--chord-match) of the equilibrium's own chord pressure equals the value measured on the"
                        " frame's REAL Thomson chord (label-free); no radius prior")
    p.add_argument("--chord-match", choices=("axis", "rgeo"), default="axis", help="chord observable: pressure core centre (axis) or LCFS midplane centre (rgeo)")
    p.add_argument("--chord-steps", type=int, default=2, help="max warm-started position-correction re-solves per attempt")
    p.add_argument("--chord-tol", type=float, default=0.005, help="stop correcting when |target - observable| < tol (m)")
    p.add_argument("--chord-accept-tol", type=float, default=0.02, help="reject a chord-mode equilibrium whose |target - observable| exceeds this (m); 0 = off")
    p.add_argument("--chord-seed-offset", type=float, default=None, help="seed centroid = observable + offset (m); default -0.11 (axis) / 0.0 (rgeo)")
    p.add_argument("--beta-p-consistency", type=float, default=0.0,
                   help="reject equilibria whose balanced beta_p is outside [1/f, f] x (proxy beta_p,e x mean pe ratio); 0 = off")
    p.add_argument("--proxy-ratio-range", type=float, nargs=2, default=(1.5, 2.5), help="total/electron pressure ratio of the beta_p proxy, U(range)")
    p.add_argument("--beta-p-jitter", type=float, default=0.3, help="lognormal sigma on the proxy beta_p target")
    p.add_argument("--beta-p-bounds", type=float, nargs=2, default=(0.03, 2.0))
    p.add_argument("--beta-p-tol", type=float, default=0.03, help="relative tolerance of the beta_p match")
    p.add_argument("--proxy-threshold", type=float, default=0.1, help="hot-extent threshold of the proxy (fraction of the chord maximum)")
    p.add_argument("--proxy-a-range", type=float, nargs=2, default=(0.3, 0.75), help="frames whose proxy minor radius is outside this range are skipped")
    p.add_argument("--r-init-range", type=float, nargs=2, default=(0.70, 1.00), help="clip of the balance seed radius (proxy Rgeo estimate)")
    p.add_argument("--r-init-offset", type=float, default=0.0, help="added to the proxy Rgeo estimate for the seed radius (m)")
    p.add_argument("--edge-current", action="store_true", help="add the v9 edge/pedestal current term with edge_amp ~ U(edge-amp-range), edge_w ~ U(edge-w-range)")
    p.add_argument("--edge-amp-range", type=float, nargs=2, default=(0.0, 0.3))
    p.add_argument("--edge-w-range", type=float, nargs=2, default=(0.03, 0.12))
    p.add_argument("--reject-unbalanced", action="store_true", help="reject equilibria whose beta0 balance hit its bounds or whose beta_p target was not met (|dI5| > 1.5 kA-turn is already rejected)")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranges = dict(cs.DEFAULT_RANGES)
    if args.ranges:
        ranges.update({k: tuple(v) for k, v in json.loads(args.ranges).items()})
    config = {"ranges": ranges, "r_target_range": tuple(args.r_target_range), "z_std": args.z_std, "z_max": args.z_max,
              "max_attempts": args.max_attempts, "pe_noise": args.pe_noise, "min_ip_kA": args.min_ip_kA,
              "lcfs_iter": args.lcfs_iter, "max_seconds": args.max_seconds, "diverted_contact_max": args.diverted_contact_max,
              "vessel_current_kA": args.vessel_current_kA, "vessel_asym_frac": args.vessel_asym_frac, "vessel_share_std": args.vessel_share_std,
              "passive_geometry": args.passive_geometry, "coil_noise": args.coil_noise, "pe_ratio_range": tuple(args.pe_ratio_range),
              "double_null_fraction": args.double_null_fraction, "dn_asym_std": args.dn_asym_std, "dn_asym_max": args.dn_asym_max,
              "dn_steps": args.dn_steps, "dn_tolerance": args.dn_tolerance, "v3_metadata": bool(args.v3_metadata),
              "position_mode": args.position_mode, "proxy_ratio_range": tuple(args.proxy_ratio_range), "beta_p_jitter": args.beta_p_jitter,
              "beta_p_bounds": tuple(args.beta_p_bounds), "beta_p_tol": args.beta_p_tol, "proxy_threshold": args.proxy_threshold,
              "proxy_a_range": tuple(args.proxy_a_range), "r_init_range": tuple(args.r_init_range), "r_init_offset": args.r_init_offset,
              "edge_current": bool(args.edge_current), "edge_amp_range": tuple(args.edge_amp_range), "edge_w_range": tuple(args.edge_w_range),
              "reject_unbalanced": bool(args.reject_unbalanced), "chord_match": args.chord_match, "chord_steps": args.chord_steps,
              "chord_tol": args.chord_tol, "chord_accept_tol": args.chord_accept_tol, "chord_seed_offset": args.chord_seed_offset,
              "beta_p_consistency": args.beta_p_consistency}
    v9_on = args.position_mode in ("balance", "chord") or args.edge_current or args.reject_unbalanced or args.beta_p_consistency > 0
    v3_keys = [k for k in cs.OPTIONAL_RECORD_SCALARS if k != "coil_noise_std"] if args.v3_metadata else []
    if v9_on:
        v3_keys = v3_keys + list(cs.V9_RECORD_SCALARS)
    rng = np.random.default_rng(args.seed)
    paths = cd.mast_parquet_paths()
    indices = list(range(len(paths)))
    if args.max_shots:
        indices = indices[:args.max_shots]
    val = set(rng.choice(indices, size=min(args.val_shots, max(len(indices) // 10, 1)), replace=False).tolist())
    per_shot = max(1, int(np.ceil(args.oversample * args.target / len(indices))))
    order = rng.permutation(indices)
    tasks = [(int(i), str(paths[i]), per_shot, int(rng.integers(0, 2 ** 31 - 1))) for i in order]
    setup = cd.build_machine_setup("MAST")
    table_names = list(setup.table_names)
    print(json.dumps({"shots": len(tasks), "frames_per_shot": per_shot, "val_shots": len(val), "workers": args.workers,
                      "ranges": ranges, "r_target_range": config["r_target_range"]}), flush=True)
    started = time.monotonic()
    buffers = {"train": [], "val": []}
    shards, stats = [], {"attempts": 0, "accepted": 0, "reasons": {}, "solve_seconds": [], "shot_seconds": [], "errors": []}
    summary_rows = []

    def flush(split, force=False):
        while len(buffers[split]) >= args.shard_size or (force and buffers[split]):
            chunk, buffers[split] = buffers[split][:args.shard_size], buffers[split][args.shard_size:]
            name = f"shard_{len(shards):04d}.npz"
            n = cs.write_shard(args.output / name, chunk, table_names)
            shards.append({"path": name, "split": split, "count": n, "shots": sorted(set(int(r["shot_index"]) for r in chunk))})

    total_accepted = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(config,)) as pool:
        for done, out in enumerate(pool.imap_unordered(process_shot, tasks, chunksize=1)):
            split = "val" if out["shot_index"] in val else "train"
            for r in out["records"]:
                summary_rows.append({k: r[k] for k in cs.RECORD_SCALARS} | {"split": split, "beta0_final": r["beta0_final"],
                                                                            "params_sampled": r["params_sampled"].tolist(),
                                                                            "params": r["params"].tolist()}
                                    | {k: float(r[k]) for k in v3_keys if k in r})
            buffers[split].extend(out["records"])
            total_accepted += out["accepted"]
            stats["attempts"] += out["attempts"]
            stats["accepted"] += out["accepted"]
            for k, v in out["reasons"].items():
                stats["reasons"][k] = stats["reasons"].get(k, 0) + v
            stats["solve_seconds"].extend(out["solve_seconds"])
            stats["shot_seconds"].append(out.get("seconds", 0.0))
            stats["errors"].extend(out["errors"])
            flush(split)
            if (done + 1) % 25 == 0:
                el = time.monotonic() - started
                print(f"{done + 1}/{len(tasks)} shots, {total_accepted} accepted, {el:.0f} s, "
                      f"{total_accepted / max(el, 1e-9) * 3600:.0f}/h, mean solve {np.mean(stats['solve_seconds']):.2f} s", flush=True)
            if total_accepted >= args.target or (args.time_limit and time.monotonic() - started > args.time_limit):
                pool.terminate()
                break
    for split in ("train", "val"):
        flush(split, force=True)
    elapsed = time.monotonic() - started
    rows = summary_rows
    def col(key, split=None):
        return [r[key] for r in rows if split is None or r["split"] == split]
    kappa, li, q95, rax = (np.asarray(col(k), float) for k in ("kappa", "li", "q95", "R_axis"))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if v9_on or k not in V9_ARG_KEYS},
        "ranges": ranges, "r_target_range": config["r_target_range"], "shots_processed": len(stats["shot_seconds"]),
        "accepted": int(stats["accepted"]), "attempts": int(stats["attempts"]),
        "acceptance_rate_per_attempt": stats["accepted"] / max(stats["attempts"], 1),
        "reasons": stats["reasons"], "errors": stats["errors"][:50], "elapsed_seconds": elapsed,
        "seconds_per_accepted_equilibrium_wall": elapsed / max(stats["accepted"], 1),
        "solve_seconds_per_attempt": quantiles(stats["solve_seconds"]),
        "cpu_seconds_per_accepted": float(np.sum(stats["solve_seconds"]) / max(stats["accepted"], 1)),
        "throughput_per_hour": stats["accepted"] / max(elapsed, 1e-9) * 3600,
        "splits": {s: int(sum(sh["count"] for sh in shards if sh["split"] == s)) for s in ("train", "val")},
        "distributions": {k: quantiles(col(k)) for k in ("kappa", "li", "volume", "R_axis", "Z_axis", "q95", "betaN", "beta_p", "psi_ptp",
                                                          "psi_axis", "psi_edge", "lcfs_Rmin", "lcfs_Rmax", "lcfs_Zmin", "lcfs_Zmax",
                                                          "tri_top", "tri_bot", "dI5_kAt", "dI6_kAt", "beta0_final", "iterations",
                                                          "residual", "seconds", "ip_kA", "tf_kA", "pressure_axis_Pa",
                                                          "envelope_contact_fraction", "r_target", "z_target")},
        "params_sampled": {k: quantiles([r["params_sampled"][i] for r in rows]) for i, k in enumerate(cs.PARAM_NAMES)},
        "params_final": {k: quantiles([r["params"][i] for r in rows]) for i, k in enumerate(cs.PARAM_NAMES)},
        "target_fractions": {"kappa_in_1.6_2.3": float(np.mean((kappa >= 1.6) & (kappa <= 2.3))) if len(kappa) else None,
                             "kappa_in_1.8_2.2": float(np.mean((kappa >= 1.8) & (kappa <= 2.2))) if len(kappa) else None,
                             "li_in_0.5_1.0": float(np.mean((li >= 0.5) & (li <= 1.0))) if len(li) else None,
                             "q95_in_3_9": float(np.mean((q95 >= 3) & (q95 <= 9))) if len(q95) else None,
                             "R_axis_in_0.8_1.05": float(np.mean((rax >= 0.8) & (rax <= 1.05))) if len(rax) else None},
        "diverted_fraction": float(np.mean(col("diverted"))) if rows else None,
        "radial_free_fraction": float(np.mean(col("radial_free"))) if rows else None,
        "vertical_field_balance": {"note": "beta0 balanced so that the REAL vacuum field holds the plasma at r_target (|dI5| < 0.5 kA-turn);"
                                           " dI6 is the P6 vertical-control correction stored in the record currents",
                                   "dI5_kAt": quantiles(col("dI5_kAt")), "dI6_kAt": quantiles(col("dI6_kAt")),
                                   "lcfs_inside_envelope_fraction": float(np.mean(np.asarray(col("envelope_contact_fraction")) < 0.01)) if rows else None},
        "label_role": "no MAST labels exist locally; public inputs only", "shards": shards,
    }
    if args.v3_metadata and rows:
        dn_class = np.asarray(col("dn_class"), float)
        dn_mode = np.asarray(col("dn_mode"), float)
        names = {0: "single_null_lower", 1: "single_null_upper", 2: "near_double_null_|asym|<0.03", 3: "connected_double_null_|asym|<0.01", -1: "unknown"}

        def class_fractions(sel):
            n = max(int(sel.sum()), 1)
            return {names[k]: float(np.sum(dn_class[sel] == k) / n) for k in names} | {"count": int(sel.sum())}
        dn_asym = np.asarray(col("dn_asym"), float)
        report["double_null"] = {
            "definition": "dn_asym = (psi_x_up - psi_x_low) / (psi_axis - psi_edge) from a grid saddle search on both halves (fusion.c2_synthetic.find_xpoints);"
                          " connected: |dn_asym| < 0.01 (outboard dr_sep ~ 4 mm), near: < 0.03; the scorer's LCFS passes through the limiting (higher-psi) saddle",
            "all": class_fractions(np.ones(len(dn_class), bool)),
            "dn_mode_frames": class_fractions(dn_mode > 0.5), "standard_mode_frames": class_fractions(dn_mode < 0.5),
            "dn_mode_fraction": float(np.mean(dn_mode > 0.5)),
            "dn_asym": quantiles(dn_asym), "abs_dn_asym": quantiles(np.abs(dn_asym)),
            "dn_asym_minus_target_dn_mode": quantiles((dn_asym - np.asarray(col("dn_target"), float))[dn_mode > 0.5]),
            "dr_sep_mm": quantiles(1000.0 * np.asarray(col("dr_sep_m"), float)),
            "dn_steps": quantiles(np.asarray(col("dn_steps"), float)[dn_mode > 0.5]),
            "xpt_up_R": quantiles(col("xpt_up_R")), "xpt_up_Z": quantiles(col("xpt_up_Z")),
            "xpt_low_R": quantiles(col("xpt_low_R")), "xpt_low_Z": quantiles(col("xpt_low_Z")),
            "fraction_two_xpoints_found": float(np.mean(np.asarray(col("n_xpoints"), float) == 2)),
        }
        geometry = cs.PASSIVE_GEOMETRIES[args.passive_geometry]
        report["passive_currents"] = {
            "sigma_total_kA": args.vessel_current_kA, "asym_frac": args.vessel_asym_frac, "share_lognormal_std": args.vessel_share_std,
            "geometry_version": args.passive_geometry, "geometry": geometry, "n_elements": int(sum(g["n"] for g in geometry)),
            "provenance": "adjustable table written from public general knowledge of the MAST machine (vessel ~4 m diameter x 4.4 m tall,"
                          " centre-column casing outside the solenoid, end plates, divertor plates); NOT from FreeGSNKE/MAST-U machine files;"
                          " element positions/sizes/shares are modelling choices",
            "passive_total_kA": quantiles(col("passive_total_kA")), "passive_asym_kA": quantiles(col("passive_asym_kA")),
            "abs_passive_total_over_ip": quantiles(np.abs(np.asarray(col("passive_total_kA"), float)) / np.asarray(col("ip_kA"), float)),
            "psi_passive_axis_Wb_per_rad": quantiles(col("psi_passive_axis")),
            "psi_passive_boundary_mean_Wb_per_rad": quantiles(col("psi_passive_boundary_mean")),
            "psi_passive_boundary_absmax_Wb_per_rad": quantiles(col("psi_passive_boundary_absmax")),
            "psi_passive_rms_Wb_per_rad": quantiles(col("psi_passive_rms")),
            "psi_passive_axis_over_axis_edge_flux": quantiles(np.asarray(col("psi_passive_axis"), float) / (np.asarray(col("psi_axis"), float) - np.asarray(col("psi_edge"), float))),
        }
        report["coil_noise"] = {"relative_std": args.coil_noise, "psi_offset_axis_Wb_per_rad": quantiles(col("coil_noise_psi_axis")),
                                "psi_offset_rms_Wb_per_rad": quantiles(col("coil_noise_psi_rms"))}
        report["pe_ratio"] = {"range": list(args.pe_ratio_range), "achieved": quantiles(col("pe_ratio"))}
    if v9_on and rows:
        r_t = np.asarray(col("r_target"), float)
        r_s = np.asarray(col("r_seed"), float)
        cR = np.asarray(col("centroid_R"), float)
        bal = np.asarray(col("position_mode_balance"), float) > 0.5
        chd = np.asarray(col("position_mode_chord"), float) > 0.5
        chord_stats = None
        if chd.any():
            tgt, syn, stp = (np.asarray(col(k), float) for k in ("chord_R_target", "chord_R_syn", "chord_steps"))
            chord_stats = {"match": config["chord_match"], "target": quantiles(tgt), "observable_achieved": quantiles(syn),
                           "target_minus_observable": quantiles(tgt - syn), "abs_target_minus_observable": quantiles(np.abs(tgt - syn)),
                           "steps": {str(int(k)): int(v) for k, v in zip(*np.unique(stp[np.isfinite(stp)], return_counts=True))},
                           "centroid_R_minus_target": quantiles(cR - tgt), "beta_p_over_ref": quantiles(np.asarray(col("beta_p"), float) / np.asarray(col("beta_p_ref"), float))}
        report["v9"] = {
            "options": {k: config[k] for k in ("position_mode", "proxy_ratio_range", "beta_p_jitter", "beta_p_bounds", "beta_p_tol", "proxy_threshold",
                                               "proxy_a_range", "r_init_range", "r_init_offset", "edge_current", "edge_amp_range", "edge_w_range", "reject_unbalanced",
                                               "chord_match", "chord_steps", "chord_tol", "chord_accept_tol", "chord_seed_offset", "beta_p_consistency")},
            "position": {"mode": args.position_mode, "balance_fraction": float(np.mean(bal)), "chord_fraction": float(np.mean(chd)),
                         "centroid_R": quantiles(cR), "centroid_Z": quantiles(col("centroid_Z")),
                         "r_target_final": quantiles(r_t), "r_seed": quantiles(r_s),
                         "centroid_R_minus_seed": quantiles(cR - r_s), "r_target_final_minus_seed": quantiles(r_t - r_s),
                         "note": "r_target = FINAL pinned/balanced centroid target (== centroid_R within tolerance); r_seed = seed radius"
                                 " (prior draw / proxy Rgeo estimate for balance / chord observable + offset for chord)",
                         "chord": chord_stats,
                         "old_prior_r_target": {"range": list(args.r_target_range), "note": "v1-v3 sampled the centroid radius from this currents-independent prior"},
                         "centroid_R_fraction_inside_old_prior": float(np.mean((cR >= args.r_target_range[0]) & (cR <= args.r_target_range[1]))),
                         "centroid_R_fraction_above_old_prior": float(np.mean(cR > args.r_target_range[1])),
                         "R_axis_minus_centroid_R": quantiles(np.asarray(col("R_axis"), float) - cR)},
            "beta_p_proxy": {"beta_p_e": quantiles(col("beta_p_proxy_e")), "pe_ratio": quantiles(col("pe_ratio_proxy")), "jitter": quantiles(col("beta_p_jitter")),
                             "target": quantiles(col("beta_p_target")), "achieved": quantiles(col("beta_p_achieved")),
                             "achieved_minus_target_rel": quantiles((np.asarray(col("beta_p_achieved"), float) - np.asarray(col("beta_p_target"), float)) / np.asarray(col("beta_p_target"), float)),
                             "a_est": quantiles(col("proxy_a_est")), "Rgeo_est": quantiles(col("proxy_Rgeo_est")), "R_axis_est": quantiles(col("proxy_R_axis_est")),
                             "a_est_minus_lcfs_a": quantiles(np.asarray(col("proxy_a_est"), float) - 0.5 * (np.asarray(col("lcfs_Rmax"), float) - np.asarray(col("lcfs_Rmin"), float))),
                             "Rgeo_est_minus_lcfs_Rgeo": quantiles(np.asarray(col("proxy_Rgeo_est"), float) - 0.5 * (np.asarray(col("lcfs_Rmax"), float) + np.asarray(col("lcfs_Rmin"), float)))},
            "edge_current": {"enabled": bool(args.edge_current), "edge_amp": quantiles(col("edge_amp")), "edge_w": quantiles(col("edge_w")),
                             "j_edge_over_axis": quantiles(col("j_edge_over_axis")), "edge_I_frac_s_gt_0.9": quantiles(col("edge_I_frac_s_gt_0.9")),
                             "reference_d3d_labels": {"j_edge_over_axis": {"median": 0.106, "p10": 0.014, "p90": 0.255}, "edge_I_frac_s_gt_0.9": {"median": 0.059, "p90": 0.113},
                                                      "mast_v1_family_on_d3d": {"j_edge_over_axis_q10_50_90": [0.012, 0.037, 0.126]},
                                                      "source": "results/analysis/synth_fidelity_d3d/report.json label_current_shape"}},
            "beta0_at_bound_fraction_accepted": float(np.mean(np.asarray(col("beta0_at_bound"), float) > 0.5)),
            "rejections": {k: v for k, v in stats["reasons"].items() if k in ("beta0_at_bound", "beta_p_target_not_met", "radial_balance_not_met",
                                                                                 "beta_p_inconsistent", "chord_position_not_met") or k.startswith("proxy_failed")},
        }
    if args.compare_report is not None and args.compare_report.exists():
        other = json.loads(args.compare_report.read_text())
        keys = ("kappa", "li", "q95", "betaN", "R_axis", "Z_axis", "psi_ptp", "psi_axis", "psi_edge", "volume", "beta_p", "dI6_kAt", "dI5_kAt",
                "lcfs_Rmin", "lcfs_Rmax", "lcfs_Zmin", "lcfs_Zmax", "tri_top", "tri_bot", "pressure_axis_Pa", "beta0_final", "r_target")
        report["comparison"] = {"reference": str(args.compare_report),
                                "table": {k: {"reference": {q: other["distributions"][k].get(q) for q in ("p05", "median", "mean", "std", "p95")},
                                              "this": {q: report["distributions"][k].get(q) for q in ("p05", "median", "mean", "std", "p95")}}
                                          for k in keys if k in other.get("distributions", {}) and k in report["distributions"]},
                                "reference_diverted_fraction": other.get("diverted_fraction"), "this_diverted_fraction": report["diverted_fraction"],
                                "reference_throughput_per_hour": other.get("throughput_per_hour"), "this_throughput_per_hour": report["throughput_per_hour"],
                                "reference_acceptance": other.get("acceptance_rate_per_attempt"), "this_acceptance": report["acceptance_rate_per_attempt"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    manifest = {"version": 1, "created_utc": report["created_utc"], "machine": "MAST", "table_names": table_names,
                "param_names": cs.PARAM_NAMES, "shards": shards, "splits": report["splits"], "val_shots": sorted(val),
                "generator": "scripts/challenge2/generate_synthetic_mast.py", "arguments": report["arguments"], "ranges": ranges,
                "target_convention": "psi Wb/rad, axis maximum (MAST); phi = +psi", "profile": "FreeGS-style, see fusion/c2_synthetic.py"}
    if args.v3_metadata:
        manifest["v3"] = {"options": {k: config[k] for k in ("vessel_current_kA", "vessel_asym_frac", "vessel_share_std", "passive_geometry", "coil_noise",
                                                               "pe_ratio_range", "double_null_fraction", "dn_asym_std", "dn_asym_max", "dn_steps", "dn_tolerance")},
                          "passive_geometry": cs.PASSIVE_GEOMETRIES[args.passive_geometry],
                          "record_inputs": "currents = REAL coil currents + P5/P6 control corrections only; passive currents and coil noise are unobserved"
                                           " (metadata arrays currents_solve / passive_currents_kA are for analysis and verification only)",
                          "optional_arrays": cs.OPTIONAL_RECORD_ARRAYS, "optional_scalars": cs.OPTIONAL_RECORD_SCALARS}
    if v9_on:
        manifest["v9"] = {"options": report["v9"]["options"] if "v9" in report else {},
                          "position": {"balance": "centroid radius from radial force balance in the real vacuum field with a label-free Thomson beta_p target",
                                       "chord": "centroid radius iterated until the midplane-chord estimator of the equilibrium's own chord pressure matches the"
                                                " frame's REAL Thomson chord (label-free); beta0 balanced so that the real field holds the plasma",
                                       "prior": "v1-v3 sampling of r_target from --r-target-range"}[args.position_mode],
                          "r_target_semantics": "r_target = FINAL pinned/balanced centroid target (== centroid_R within tolerance); r_seed = seed radius;"
                                                " chord_R_target / chord_R_syn = chord observable measured / achieved; beta_p_ref = proxy reference of the"
                                                " beta_p consistency rule (NaN when unused)",
                          "profile": "params (5) = FreeGS family; edge_amp/edge_w = v9 pedestal term (edge_amp 0 -> v1 family); the stored j is"
                                     " lam [(R/R0)(beta0 f_p + edge_amp exp(-(1-s)/edge_w)) + (R0/R)(1-beta0) f_f] inside the LCFS",
                          "record_scalars": cs.V9_RECORD_SCALARS}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    (args.output / "records_summary.json").write_text(json.dumps(rows, default=float) + "\n")
    print(json.dumps({k: report[k] for k in ("accepted", "attempts", "acceptance_rate_per_attempt", "reasons", "elapsed_seconds",
                                             "seconds_per_accepted_equilibrium_wall", "cpu_seconds_per_accepted", "throughput_per_hour",
                                             "splits", "target_fractions", "diverted_fraction", "radial_free_fraction")}, default=float), flush=True)
    for k in ("kappa", "li", "q95", "betaN", "beta_p", "R_axis", "Z_axis", "volume", "psi_ptp", "dI5_kAt", "dI6_kAt", "beta0_final", "seconds"):
        print(k, json.dumps(report["distributions"][k], default=float), flush=True)


if __name__ == "__main__":
    main()
