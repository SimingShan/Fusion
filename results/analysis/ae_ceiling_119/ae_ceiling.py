#!/usr/bin/env python3
"""Is the flux manifold more compact under a NONLINEAR latent than under POD?

Gate for the nonlinear-decoder member idea. Fits a rank-512 POD basis on training flux only,
then trains an MLP autoencoder (512 coeffs -> d latent -> 512 coeffs, field-MSE loss, which equals
coefficient MSE because the basis is orthonormal) for several d. Scores the true dev flux
reconstructed through each AE, and through the POD-d linear projection at the same d, with the
official scorer. If AE-d does not clearly beat POD-d per latent dimension, a learned nonlinear
latent offers no compactness advantage and the idea stops here.

Training labels only for basis and AE; the dev fold is used exactly as a scoring fold. Audit untouched.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT))
from fusion.pod import load_arrays  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/analysis/ae_ceiling_119"
DIMS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8,16,32,64").split(",")]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 120
RANK, SAMPLES, WORKERS = 512, 32768, 4
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2026)

train_dir = ROOT / "data/advanced_history_v2"
dev_dir = ROOT / "data/advanced_val_native_v3"
t0 = time.monotonic()
log = lambda m: print(f"[{time.monotonic()-t0:6.0f}s] {m}", flush=True)

flat = load_arrays(train_dir, "train", keys=("psi",))["psi"].reshape(-1, 4225)
n_train = len(flat)
acc = np.zeros(4225);
for i in range(0, n_train, 16384):
    acc += np.asarray(flat[i:i + 16384], dtype=np.float64).sum(0)
mean = acc / n_train
rng = np.random.default_rng(2026)
chosen = np.sort(rng.choice(n_train, SAMPLES, replace=False))
sample = np.asarray(flat[chosen], dtype=np.float64) - mean
cov = sample.T @ sample / len(sample); del sample
from scipy.linalg import eigh  # noqa: E402
vals, vecs = eigh(cov, subset_by_index=(4225 - RANK, 4224), driver="evr", check_finite=False)
basis = np.ascontiguousarray(vecs[:, ::-1].T)  # (RANK, 4225)
log(f"basis rank {RANK} on {SAMPLES} of {n_train} frames")

B = torch.tensor(basis, dtype=torch.float32, device=dev)
M = torch.tensor(mean, dtype=torch.float32, device=dev)
coef = np.empty((n_train, RANK), dtype=np.float32)
for i in range(0, n_train, 16384):
    x = torch.tensor(np.asarray(flat[i:i + 16384], dtype=np.float32), device=dev)
    coef[i:i + 16384] = ((x - M) @ B.T).cpu().numpy()
scale = float(np.sqrt((coef ** 2).sum(1).mean()))   # one global scalar keeps field-MSE geometry
log(f"train coefficients {coef.shape}, rms norm {scale:.4g}")

man = json.loads((dev_dir / "manifest.json").read_text())
truth = [np.load(dev_dir / s["path"]) for s in man["val"]]
psi = [np.asarray(t["psi"], dtype=np.float32) for t in truth]
scal = [np.asarray(t["scalars"], dtype=np.float32) for t in truth]
dev_coef = [((torch.tensor(p.reshape(len(p), -1), device=dev) - M) @ B.T) for p in psi]
dev_resid = [torch.tensor(p.reshape(len(p), -1), device=dev) - M - c @ B for p, c in zip(psi, dev_coef)]
log(f"dev fold {len(psi)} shots, {sum(len(p) for p in psi)} frames")


def score(name, recon_coef):
    """recon_coef: list of (n,RANK) tensors. Adds back the rank-512 residual so ONLY the latent
    bottleneck is measured (the 512->4225 truncation is identical for every arm)."""
    arrays = {}
    for i, (c, r) in enumerate(zip(recon_coef, dev_resid)):
        field = (c @ B + M + r).reshape(-1, 65, 65).cpu().numpy()
        arrays[f"shot_{i:04d}_psirz"] = field.astype(np.float32)
        arrays[f"shot_{i:04d}_q95"] = scal[i][:, 0]
        arrays[f"shot_{i:04d}_betaN"] = scal[i][:, 1]
    pred = OUT / f"{name}.npz"; np.savez(pred, **arrays)
    sp = OUT / f"{name}_score.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(dev_dir),
                    "--split", "val", "--pred", str(pred), "--output", str(sp),
                    "--workers", str(WORKERS)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    pred.unlink()
    s = json.loads(sp.read_text())
    mse = float(sum(((c - d) ** 2).sum().item() for c, d in zip(recon_coef, dev_coef))
                / sum(len(d) for d in dev_coef))
    e = {k: s[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
    e["r2_cons_each"] = s.get("r2_cons_each"); e["coef_mse"] = mse
    ce = e["r2_cons_each"] or {}
    log(f"{name:10s} S={e['S']:.6f} cons={e['consistency']:.6f} kappa={ce.get('kappa', 0):.4f} "
        f"li={ce.get('li', 0):.4f} Rax={ce.get('R_axis', 0):.4f} dlcfs={e['dlcfs']:.5f} mse={mse:.4g}")
    return e


def mlp(i, o, w=1024, depth=3):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, w), nn.LayerNorm(w), nn.GELU()]; d = w
    return nn.Sequential(*layers, nn.Linear(d, o))


report = {"settings": {"rank": RANK, "dims": DIMS, "epochs": EPOCHS, "train_frames": n_train,
                       "dev_dir": str(dev_dir), "scale": scale}, "pod": {}, "ae": {}}
ref = score("pod512", dev_coef); report["pod512"] = ref
for d in DIMS:
    report["pod"][d] = score(f"pod{d}", [torch.cat([c[:, :d], torch.zeros_like(c[:, d:])], 1)
                                         for c in dev_coef])
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

X = torch.tensor(coef / scale)  # CPU pinned, batches to GPU
for d in DIMS:
    enc, decd = mlp(RANK, d).to(dev), mlp(d, RANK).to(dev)
    params = list(enc.parameters()) + list(decd.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)
    steps_per = n_train // 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * steps_per,
                                                pct_start=0.05)
    for ep in range(EPOCHS):
        perm = torch.randperm(n_train)
        tot = 0.0
        for j in range(steps_per):
            xb = X[perm[j * 1024:(j + 1) * 1024]].to(dev, non_blocking=True)
            loss = ((decd(enc(xb)) - xb) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if ep % 20 == 0 or ep == EPOCHS - 1:
            log(f"AE d={d} epoch {ep} train rel-mse {tot / steps_per:.3e}")
    enc.eval(); decd.eval()
    with torch.no_grad():
        rec = [decd(enc(c / scale)) * scale for c in dev_coef]
    report["ae"][d] = score(f"ae{d}", rec)
    torch.save({"enc": enc.state_dict(), "dec": decd.state_dict(), "scale": scale}, OUT / f"ae{d}.pt")
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
np.savez(OUT / "basis512.npz", basis=basis.astype(np.float32), mean=mean.astype(np.float32))
log("done")
