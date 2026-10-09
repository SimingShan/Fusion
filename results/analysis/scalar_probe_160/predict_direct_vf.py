#!/usr/bin/env python3
"""Predict-only companion of train_direct_vf.py: rebuild the DIII-D TEST inputs exactly as train_direct_vf.py does ([708 standardised
member inputs, 126 vessel, 126 future-actuator], vessel/future blocks asinh-scaled and standardised with TRAIN statistics) and average
the saved direct regressors <weights_dir>/w*_d*_s*.pt -> results/analysis/scalar_probe_160/pred_<TAGN>_test.npy (frames, 7).
usage: predict_direct_vf.py <weights_dir> [TAGN=direct_vf60] [device=cuda|cpu]"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); sys.path.insert(0, str(ROOT))
from fusion.pod import standardize_x  # noqa: E402
from scripts.train import prediction_entries  # noqa: E402
from scripts.optimization.export_gated_ensemble import input_arrays  # noqa: E402

W = Path(sys.argv[1]); TAGN = sys.argv[2] if len(sys.argv) > 2 else "direct_vf60"
dev = torch.device(sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu"))
HERE = ROOT / "results/analysis/scalar_probe_160"
norm = np.load(ROOT / "data/pod_history128_robust_v4/normalization.npz")
T = ROOT / "data/advanced_history_test_v2"; ents = prediction_entries(T, "test")
Xt = np.concatenate([standardize_x(input_arrays(T, e)[0], norm) for e in ents])
VD, FD = ROOT / "results/analysis/vessel_196/feats", ROOT / "results/analysis/future_201/feats"
n_tr = len(json.loads((ROOT / "data/advanced_history_v2/manifest.json").read_text())["train"])


def blk(base, sub, n):
    return np.concatenate([np.load(base / sub / f"shot_{i:05d}.npy") for i in range(n)]).astype(np.float64)


Vtr, Vte = blk(VD, "train", n_tr), blk(VD, "test", len(ents)); Ftr, Fte = blk(FD, "train", n_tr), blk(FD, "test", len(ents))
vsc = np.array(json.loads((ROOT / "results/analysis/vessel_196/vessel_scale.json").read_text())["scale"])
fsc = np.median(np.abs(Ftr), 0); fsc = np.where(fsc > 1e-12, fsc, 1.0)


def std(v, ref):
    m, s_ = ref.mean(0), ref.std(0) + 1e-9
    return np.clip((v - m) / s_, -12, 12).astype(np.float32)


Zv = std(np.arcsinh(Vte / vsc), np.arcsinh(Vtr / vsc)); Zf = std(np.arcsinh(Fte / fsc), np.arcsinh(Ftr / fsc))
Xf = np.concatenate([Xt, Zv, Zf], 1).astype(np.float32); print("test inputs", Xf.shape, flush=True)


class Block(nn.Module):
    def __init__(self, w):
        super().__init__(); self.l = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w)); self.n = nn.LayerNorm(w)

    def forward(self, x):
        return x + self.l(self.n(x))


preds = []
for f in sorted(W.glob("w*_d*_s*.pt")):
    c = torch.load(f, map_location="cpu", weights_only=False); width, depth = c["width"], c["depth"]
    net = nn.Sequential(nn.Linear(Xf.shape[1], width), *[Block(width) for _ in range(depth)], nn.LayerNorm(width), nn.Linear(width, 7))
    net.load_state_dict(c["state_dict"]); net.to(dev).eval(); ym, ys = c["ym"].to(dev), c["ys"].to(dev); P = []
    with torch.no_grad():
        for a in range(0, len(Xf), 65536):
            P.append((net(torch.tensor(Xf[a:a + 65536], device=dev)) * ys + ym).cpu().numpy())
    preds.append(np.concatenate(P)); print(f"{f.name}: done", flush=True)
assert preds, f"no regressors in {W}"
out = HERE / f"pred_{TAGN}_test.npy"; np.save(out, np.mean(preds, 0).astype(np.float32))
print(f"wrote {out} from {len(preds)} regressors, frames {len(Xf)}", flush=True)
