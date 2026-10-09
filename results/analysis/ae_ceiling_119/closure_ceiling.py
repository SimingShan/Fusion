#!/usr/bin/env python3
"""Nonlinear POD closure: are the high modes a function of the leading d modes?

Latent = exact leading-d POD coefficients (so every arm is >= POD-d by construction). A closure MLP
trained on TRAINING frames predicts modes d..511 from modes 0..d-1 (zero-initialised output, so it
starts at POD-d). Scored on the dev fold with the TRUE leading-d coefficients, the closure's
high modes and the true rank-512 residual. If POD-d + closure approaches the rank-512 level, a model
only needs to predict d numbers well; if it barely moves POD-d, the high modes carry independent
information and a smaller latent cannot help.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT))
from fusion.pod import load_arrays  # noqa: E402

OUT = ROOT / "results/analysis/ae_ceiling_119"
DIMS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "16,32,64").split(",")]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
RANK = 512
dev = torch.device("cuda")
torch.manual_seed(2026)
t0 = time.monotonic()
log = lambda m: print(f"[{time.monotonic()-t0:6.0f}s] {m}", flush=True)

b = np.load(OUT / "basis512.npz")
B = torch.tensor(b["basis"], device=dev); M = torch.tensor(b["mean"], device=dev).reshape(-1)
flat = load_arrays(ROOT / "data/advanced_history_v2", "train", keys=("psi",))["psi"].reshape(-1, 4225)
n = len(flat)
coef = np.empty((n, RANK), dtype=np.float32)
for i in range(0, n, 16384):
    x = torch.tensor(np.asarray(flat[i:i + 16384], dtype=np.float32), device=dev)
    coef[i:i + 16384] = ((x - M) @ B.T).cpu().numpy()
sd = coef.std(0) + 1e-12
log(f"train coefficients {coef.shape}")

dev_dir = ROOT / "data/advanced_val_native_v3"
man = json.loads((dev_dir / "manifest.json").read_text())
truth = [np.load(dev_dir / s["path"]) for s in man["val"]]
psi = [torch.tensor(np.asarray(t["psi"], dtype=np.float32).reshape(-1, 4225), device=dev) for t in truth]
scal = [np.asarray(t["scalars"], dtype=np.float32) for t in truth]
dcoef = [(p - M) @ B.T for p in psi]
dres = [p - M - c @ B for p, c in zip(psi, dcoef)]


def score(name, rc):
    arrays = {}
    for i, (c, r) in enumerate(zip(rc, dres)):
        arrays[f"shot_{i:04d}_psirz"] = (c @ B + M + r).reshape(-1, 65, 65).cpu().numpy().astype(np.float32)
        arrays[f"shot_{i:04d}_q95"] = scal[i][:, 0]; arrays[f"shot_{i:04d}_betaN"] = scal[i][:, 1]
    pred = OUT / f"{name}.npz"; np.savez(pred, **arrays)
    sp = OUT / f"{name}_score.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(dev_dir), "--split",
                    "val", "--pred", str(pred), "--output", str(sp), "--workers", "4"], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    pred.unlink()
    s = json.loads(sp.read_text()); ce = s.get("r2_cons_each") or {}
    mse = float(sum(((c - d) ** 2).sum().item() for c, d in zip(rc, dcoef)) / sum(len(d) for d in dcoef))
    log(f"{name:12s} S={s['S']:.6f} cons={s['consistency']:.6f} kappa={ce.get('kappa',0):.4f} "
        f"li={ce.get('li',0):.4f} Rax={ce.get('R_axis',0):.4f} dlcfs={s['dlcfs']:.5f} mse={mse:.4g}")
    return {"S": s["S"], "consistency": s["consistency"], "dlcfs": s["dlcfs"], "r2_cons_each": ce, "mse": mse}


report = {}
X = torch.tensor(coef)
SD = torch.tensor(sd, device=dev)
for d in DIMS:
    w_out = SD[d:] ** 2 / (SD[d:] ** 2).sum()           # field-MSE weighting in whitened units
    net = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                        nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                        nn.Linear(1024, RANK - d)).to(dev)
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    spe = n // 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * spe, pct_start=0.05)
    for ep in range(EPOCHS):
        perm = torch.randperm(n); tot = 0.0
        for j in range(spe):
            xb = X[perm[j * 1024:(j + 1) * 1024]].to(dev, non_blocking=True) / SD
            err = net(xb[:, :d]) - xb[:, d:]
            loss = (err ** 2 * w_out).sum(1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if ep % 50 == 0 or ep == EPOCHS - 1:
            log(f"closure d={d} epoch {ep} high-mode rel-mse {tot / spe:.4e}")
    net.eval()
    with torch.no_grad():
        rc = [torch.cat([c[:, :d], net(c[:, :d] / SD[:d]) * SD[d:]], 1) for c in dcoef]
    report[d] = score(f"closure{d}", rc)
    torch.save({"net": net.state_dict(), "sd": sd, "d": d}, OUT / f"closure{d}.pt")
    (OUT / "closure_report.json").write_text(json.dumps(report, indent=2))
log("reference POD-d: 16 0.985643, 32 0.997961, 64 0.999393")
