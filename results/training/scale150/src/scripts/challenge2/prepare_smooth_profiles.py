#!/usr/bin/env python3
"""Refit closure targets on identical permitted shots using a continuous edge."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.mast_profiles import FreeBoundaryGS, MU0
from fusion.smooth_profiles import basis, integral
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (a.output / "targets.npz").exists():
        raise ValueError("Use an immutable new target directory")
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.source / "targets.npz", allow_pickle=False) as z:
        d = dict(z)
    metadata = copy.deepcopy(json.loads((a.source / "preparation.json").read_text()))
    solver = FreeBoundaryGS(d["R"], d["Z"])
    valid = d["fit_valid"].copy()
    for i, record in enumerate(metadata["records"]):
        if not valid[i]:
            continue
        retained = {"fold", "shot_index", "frame_index", "source", "time_ms", "label_role", "fit_valid",
                    "Ip_from_target_A", "input_Ip_ratio", "edge_flux", "delta_flux", "R_inner_over_L", "grid_floor_over_L"}
        old_diagnostics = {k: record.pop(k) for k in list(record) if k not in retained}
        record["parent_profile_diagnostics"] = old_diagnostics
        phi = d["phi"][i].astype(float)
        inside, ip, L = d["inside"][i], float(d["Ip_A"][i]), float(d["L"][i])
        edge, delta = record["edge_flux"], record["delta_flux"]
        s = np.clip((phi-edge)/delta, 0, 1)
        r, floor = solver.RR/L, record["R_inner_over_L"]
        source = -(solver.matrix@phi.ravel()).reshape(solver.shape)/(MU0*solver.RR)
        selected = inside & (s > .025) & (s < .975)
        B = basis(s)
        design = np.concatenate(((r-floor**2/r)[..., None]*B, B/r[..., None]), axis=-1)
        bins = np.minimum((s[selected]*12).astype(int), 11)
        weights = 1/np.sqrt(np.bincount(bins, minlength=12)[bins])
        X, y = design[selected], source[selected]*L**2/ip
        Xw, yw = X*weights[:, None], y*weights
        scale = np.maximum(np.linalg.norm(Xw, axis=0), 1e-8)
        try:
            coef = nnls(np.vstack((Xw/scale, np.sqrt(.001)*np.eye(8))), np.r_[yw, np.zeros(8)], maxiter=1000)[0]/scale
            A, C = coef[:4], coef[4:]
            d["A_shape"][i], d["C_shape"][i] = A/max(A.sum(), 1e-12), C/max(C.sum(), 1e-12)
            d["paxis_star"][i] = delta/(MU0*ip*L)*(integral(1.)@A)
            record["paxis_star"] = float(d["paxis_star"][i])
            record["source_relative_RMSE"] = float(np.linalg.norm(X@coef-y)/max(np.linalg.norm(y), 1e-12))
            record["pressure_axis_Pa"] = float(d["paxis_star"][i]*MU0*ip**2/L**2)
        except (RuntimeError, ValueError):
            valid[i] = False
            record["fit_valid"] = False
    d["fit_valid"] = valid
    metadata.update(closure_basis="edge_vanishing", parent_preparation_sha256=hashlib.sha256((a.source / "preparation.json").read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / "fusion/smooth_profiles.py")})
    np.savez_compressed(a.output / "targets.npz", **d)
    write_json(a.output / "preparation.json", metadata)
    print(json.dumps({"frames": len(valid), "valid": int(valid.sum()), "input_features_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
