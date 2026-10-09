#!/usr/bin/env python3
"""v3-vs-v9 synthetic MAST comparison table from the two report.json files (distributions, X-points, kappa, Zmax/Zmin,
centroid radius, edge current, acceptance). Writes <output>/comparison_v3_v9.json and prints a compact table."""
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--v3", type=Path, default=Path("data/c2_synth_mast_v3/report.json"))
p.add_argument("--v9", type=Path, default=Path("data/c2_synth_mast_v9/report.json"))
p.add_argument("--output", type=Path, default=Path("results/challenge2/c2_synth_mast_v9"))
a = p.parse_args()
r3, r9 = json.loads(a.v3.read_text()), json.loads(a.v9.read_text())
keys = ["kappa", "li", "q95", "betaN", "beta_p", "R_axis", "Z_axis", "lcfs_Rmin", "lcfs_Rmax", "lcfs_Zmin", "lcfs_Zmax", "tri_top", "tri_bot",
        "volume", "psi_ptp", "beta0_final", "dI5_kAt", "dI6_kAt", "r_target", "pressure_axis_Pa", "seconds"]
table = {}
for k in keys:
    d3, d9 = r3["distributions"].get(k, {}), r9["distributions"].get(k, {})
    table[k] = {q: [d3.get(q), d9.get(q)] for q in ("p05", "median", "mean", "p95")}
xp = {}
for k in ("xpt_up_R", "xpt_up_Z", "xpt_low_R", "xpt_low_Z", "dn_asym", "dr_sep_mm"):
    xp[k] = {q: [r3["double_null"][k].get(q), r9["double_null"][k].get(q)] for q in ("p05", "median", "p95")}
xp["classes_all"] = {c: [r3["double_null"]["all"].get(c), r9["double_null"]["all"].get(c)] for c in r3["double_null"]["all"]}
xp["fraction_two_xpoints_found"] = [r3["double_null"]["fraction_two_xpoints_found"], r9["double_null"]["fraction_two_xpoints_found"]]
out = {"columns": ["v3", "v9"], "distributions": table, "xpoints": xp,
       "acceptance": {"v3": {"accepted": r3["accepted"], "attempts": r3["attempts"], "rate": r3["acceptance_rate_per_attempt"], "reasons": r3["reasons"],
                             "throughput_per_hour": r3["throughput_per_hour"], "cpu_seconds_per_accepted": r3["cpu_seconds_per_accepted"]},
                      "v9": {"accepted": r9["accepted"], "attempts": r9["attempts"], "rate": r9["acceptance_rate_per_attempt"], "reasons": r9["reasons"],
                             "throughput_per_hour": r9["throughput_per_hour"], "cpu_seconds_per_accepted": r9["cpu_seconds_per_accepted"]}},
       "diverted_fraction": [r3.get("diverted_fraction"), r9.get("diverted_fraction")],
       "target_fractions": {k: [r3["target_fractions"].get(k), r9["target_fractions"].get(k)] for k in r3["target_fractions"]},
       "v9_only": r9.get("v9")}
a.output.mkdir(parents=True, exist_ok=True)
(a.output / "comparison_v3_v9.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(f"{'key':18s} {'v3 p05':>8s} {'v3 med':>8s} {'v3 p95':>8s} | {'v9 p05':>8s} {'v9 med':>8s} {'v9 p95':>8s}")
for k, v in table.items():
    f = lambda x: f"{x:8.3f}" if isinstance(x, (int, float)) and x is not None else f"{'-':>8s}"
    print(f"{k:18s} {f(v['p05'][0])} {f(v['median'][0])} {f(v['p95'][0])} | {f(v['p05'][1])} {f(v['median'][1])} {f(v['p95'][1])}")
for k, v in xp.items():
    print(k, json.dumps(v, default=float)[:300])
print("acceptance", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "reasons"} for k, v in out["acceptance"].items()}, default=float))
print("reasons v9", out["acceptance"]["v9"]["reasons"])
if out["v9_only"]:
    v9 = out["v9_only"]
    print("centroid_R v9", {q: round(v9["position"]["centroid_R"][q], 3) for q in ("p05", "median", "p95")}, "inside old prior", v9["position"]["centroid_R_fraction_inside_old_prior"],
          "above", v9["position"]["centroid_R_fraction_above_old_prior"])
    print("edge j_edge_over_axis", {q: round(v9["edge_current"]["j_edge_over_axis"][q], 3) for q in ("p05", "median", "p95")})
    bp = v9["beta_p_proxy"]
    if bp.get("target", {}).get("count", 0):        # balance mode: beta_p target/achieved
        print("beta_p target/achieved", {q: round(bp["target"][q], 3) for q in ("p05", "median", "p95")}, {q: round(bp["achieved"][q], 3) for q in ("p05", "median", "p95")})
    ch = v9["position"].get("chord")
    if ch:                                          # chord mode: chord observable target vs achieved, beta_p / proxy reference
        print("chord", ch["match"], "target", {q: round(ch["target"][q], 3) for q in ("p05", "median", "p95")}, "achieved", {q: round(ch["observable_achieved"][q], 3) for q in ("p05", "median", "p95")},
              "|target-achieved| med/p95", round(ch["abs_target_minus_observable"]["median"], 4), round(ch["abs_target_minus_observable"]["p95"], 4),
              "beta_p/ref", {q: round(ch["beta_p_over_ref"][q], 3) for q in ("p05", "median", "p95")}, "steps", ch["steps"])
    print("a_est - lcfs a", {q: round(bp["a_est_minus_lcfs_a"][q], 3) for q in ("p05", "median", "p95")})
