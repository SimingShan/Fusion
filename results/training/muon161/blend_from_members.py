#!/usr/bin/env python3
"""31-member DIII-D test blend from per-member test exports (equal weights, B-only fallback on gated frames) + 080 closure.
usage: blend_from_members.py <out_closed.npz> <TYPE=dir> x31   (dir holds diii_d_public_test.npz; types B/A; member order m00..m30)
Mirrors finish_ensemble_candidate's export path: gate from the A-family test cache (edge-Thomson availability), fec.blend_frames
semantics (weighted mean; gated frames use the B members only), scalars blended the same way, then the 080 closure."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); sys.path.insert(0, str(ROOT))
import scripts.optimization.finish_ensemble_candidate as fec  # noqa: E402
from scripts.optimization.export_gated_ensemble import input_arrays  # noqa: E402
from scripts.train import prediction_entries  # noqa: E402
from fusion.pod import read_manifest  # noqa: E402

out_path = Path(sys.argv[1]); specs = [s.split("=", 1) for s in sys.argv[2:]]; assert len(specs) >= 31, len(specs)   # 31 shipped r-set members, optionally + added members
types = [t for t, _ in specs]; srcs = [np.load(Path(d) / "diii_d_public_test.npz") for _, d in specs]; weights = [1.0 / len(specs)] * len(specs)
A = ROOT / "data/advanced_test_v1"; entries = prediction_entries(A, "test"); columns = fec.feature_columns(read_manifest(A)["feature_names"])
cl = torch.load(ROOT / "results/candidates/20261004-080/closure/closure64.pt", map_location="cpu", weights_only=False)
net = nn.Sequential(nn.Linear(64, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                    nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 448)); net.load_state_dict(cl["net"]); net.eval()
SD = torch.tensor(cl["sd"]).float(); bz = np.load(ROOT / "results/candidates/20261004-080/closure/basis512.npz")
Bc = torch.tensor(bz["basis"]).double(); Mc = torch.tensor(bz["mean"]).double().reshape(-1)


@torch.no_grad()
def closure(psi):
    x = torch.tensor(psi.reshape(len(psi), -1)).double(); c = (x - Mc) @ Bc.T
    hi = (net((c[:, :64] / SD[:64].double()).float()) * SD[64:]).double()
    return (x + (hi[:, 64:] - c[:, 128:]) @ Bc[128:]).reshape(psi.shape).numpy().astype(np.float32)


out, gated_total, frames = {}, 0, 0
chosen, renorm = fec.fallback_selection(types, weights)
for i, e in enumerate(entries):
    k = f"shot_{i:04d}"; X = input_arrays(A, e)[0]; gated = fec.gate_frames(X, columns, 0.1)
    psis = [s[f"{k}_psirz"].astype(np.float32) for s in srcs]; scal = [np.stack([s[f"{k}_q95"], s[f"{k}_betaN"]], 1).astype(np.float32) for s in srcs]
    full = fec.weighted_sum(psis, weights); gate_psi = fec.weighted_sum([psis[j] for j in chosen], renorm)
    psi = np.where(gated[:, None, None], gate_psi, full).astype(np.float32)
    sc_full = fec.weighted_sum(scal, weights); sc_gate = fec.weighted_sum([scal[j] for j in chosen], renorm); sc = np.where(gated[:, None], sc_gate, sc_full)
    out[f"{k}_psirz"] = closure(psi); out[f"{k}_q95"] = sc[:, 0].astype(np.float32); out[f"{k}_betaN"] = sc[:, 1].astype(np.float32)
    gated_total += int(gated.sum()); frames += len(gated)
np.savez(out_path, **out)
Path(str(out_path) + ".json").write_text(json.dumps({"members": [{"type": t, "dir": d} for t, d in specs], "frames": frames, "gated_frames": gated_total, "shots": len(entries),
                                                       "closure": "080 closure64 (modes 128-511 from leading 64)"}, indent=1))
print(f"wrote {out_path}: shots={len(entries)} frames={frames} gated={gated_total}", flush=True)
