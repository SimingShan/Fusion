#!/usr/bin/env python3
"""[X, VESSEL, FUTURE] version of train_direct_final.py (inputs: 708 member inputs + 126 vessel + 126 future-actuator features,
same transforms as feature_probe.py). Outputs pred_direct_vf_{fold}.npy and models in direct_vf/.
Original docstring: Final direct scalar-regressor ensemble (for a shipped shape-consistent adjustment): K seeds x {256x3, 512x4},
100 epochs, trained on TRAIN rows only; saves state dicts and predictions for native, legacy and the DIII-D TEST
inputs (data/advanced_history_test_v2, via the model's standardize_x). Outputs pred_direct_final_{fold}.npy (frames, 7)
in fold frame order (test: prediction_entries order, i.e. the submission shot order)."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT))
from fusion.pod import standardize_x  # noqa: E402
from scripts.train import prediction_entries  # noqa: E402
from scripts.optimization.export_gated_ensemble import input_arrays  # noqa: E402

HERE = ROOT / "results/analysis/scalar_probe_160"; import os
TAGN = os.environ.get("TAGN", "direct_vf"); OUT = HERE / TAGN; OUT.mkdir(exist_ok=True)
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "11,12,13,14,15,16").split(",")]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
dev = torch.device("cuda")
norm = np.load(ROOT / "data/pod_history128_robust_v4/normalization.npz")
tr = np.load(HERE / "train.npz"); X = torch.tensor(tr["X"][tr["valid"]]); Y = torch.tensor(tr["Y"][tr["valid"]])
ym, ys = Y.mean(0), Y.std(0); Yn = (Y - ym) / ys
folds = {f: np.load(HERE / f"{f}.npz")["X"] for f in ("native", "legacy")}
# test inputs in submission shot order
T = ROOT / "data/advanced_history_test_v2"; ents = prediction_entries(T, "test")
folds["test"] = np.concatenate([standardize_x(input_arrays(T, e)[0], norm) for e in ents])
VD, FD = ROOT / "results/analysis/vessel_196/feats", ROOT / "results/analysis/future_201/feats"
n_tr = len(json.loads((ROOT / "data/advanced_history_v2/manifest.json").read_text())["train"])
def blk(base, sub, n):
    return np.concatenate([np.load(base / sub / f"shot_{i:05d}.npy") for i in range(n)]).astype(np.float64)
Vr = {"train": blk(VD, "train", n_tr), "native": blk(VD, "val", 128), "legacy": blk(VD, "legacy_val", 32), "test": blk(VD, "test", len(ents))}
Fr = {"train": blk(FD, "train", n_tr), "native": blk(FD, "val", 128), "legacy": blk(FD, "legacy_val", 32), "test": blk(FD, "test", len(ents))}
vsc = np.array(json.loads((ROOT / "results/analysis/vessel_196/vessel_scale.json").read_text())["scale"]); fsc = np.median(np.abs(Fr["train"]), 0); fsc = np.where(fsc > 1e-12, fsc, 1.0)
def std(d, ref):
    m, s_ = ref.mean(0), ref.std(0) + 1e-9
    return {k: np.clip((v - m) / s_, -12, 12).astype(np.float32) for k, v in d.items()}
Zv = std({k: np.arcsinh(v / vsc) for k, v in Vr.items()}, np.arcsinh(Vr["train"] / vsc)); Zf = std({k: np.arcsinh(v / fsc) for k, v in Fr.items()}, np.arcsinh(Fr["train"] / fsc))
assert len(Zv["train"]) == len(tr["X"]) and all(len(Zv[f]) == len(folds[f]) == len(Zf[f]) for f in folds)
X = torch.tensor(np.concatenate([tr["X"], Zv["train"], Zf["train"]], 1)[tr["valid"]])
folds = {f: np.concatenate([folds[f], Zv[f], Zf[f]], 1).astype(np.float32) for f in folds}
print({f: v.shape for f, v in folds.items()}, flush=True)


class Block(nn.Module):
    def __init__(self, w):
        super().__init__(); self.l = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w)); self.n = nn.LayerNorm(w)
    def forward(self, x):
        return x + self.l(self.n(x))


def make(width, depth, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(X.shape[1], width), *[Block(width) for _ in range(depth)], nn.LayerNorm(width), nn.Linear(width, 7)).to(dev)


preds = {f: [] for f in folds}; BS = 1024; Xg, Yg = X.to(dev), Yn.to(dev); n = len(X); spe = n // BS
SIZES = [tuple(int(v) for v in x.split("x")) for x in os.environ.get("SIZES", "256x3,512x4").split(",")]
for width, depth in SIZES:
    for seed in SEEDS:
        t0 = time.monotonic(); net = make(width, depth, seed)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * spe, pct_start=0.05)
        for ep in range(EPOCHS):
            net.train(); perm = torch.randperm(n, device=dev)
            for j in range(spe):
                ix = perm[j * BS:(j + 1) * BS]; loss = ((net(Xg[ix]) - Yg[ix]) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        net.eval()
        torch.save({"sizes": SIZES, "state_dict": net.state_dict(), "width": width, "depth": depth, "seed": seed, "ym": ym, "ys": ys, "epochs": EPOCHS},
                   OUT / f"w{width}_d{depth}_s{seed}.pt")
        with torch.no_grad():
            for f, Xf in folds.items():
                P = []
                for a in range(0, len(Xf), 65536):
                    P.append((net(torch.tensor(Xf[a:a + 65536], device=dev)) * ys.to(dev) + ym.to(dev)).cpu().numpy())
                preds[f].append(np.concatenate(P))
        print(f"w{width} d{depth} seed {seed} done [{time.monotonic()-t0:.0f}s]", flush=True)
for f in folds:
    np.save(HERE / f"pred_{TAGN}_{f}.npy", np.mean(preds[f], 0).astype(np.float32))
(OUT / "manifest.json").write_text(json.dumps({"seeds": SEEDS, "epochs": EPOCHS, "sizes": [[256, 3], [512, 4]], "models": 2 * len(SEEDS),
                                               "test_shots": len(ents), "test_frames": int(len(folds["test"]))}, indent=2))
print("done", flush=True)
