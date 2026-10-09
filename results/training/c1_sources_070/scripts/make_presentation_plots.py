#!/usr/bin/env python3
"""Reproduce the first-result presentation plots from saved data; never retrain."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/presentation/figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, TEAL, ORANGE = "#102238", "#12A89D", "#E98B42"
MODELS = [("deeponet", "DeepONet", TEAL), ("fourier_deeponet", "Fourier-DeepONet", ORANGE)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
    "axes.labelsize": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "text.color": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY,
    "ytick.color": NAVY, "axes.edgecolor": "#CAD2DA", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "savefig.facecolor": "white", "pdf.fonttype": 42,
    "svg.fonttype": "none"})

def read(path):
    return json.loads((ROOT / path).read_text())

manifest = read("data/processed/manifest.json")
scores = {key: read(f"results/{key}/score.json") for key, _, _ in MODELS}
runs = {key: read(f"results/{key}/run.json") for key, _, _ in MODELS}
history = {key: read(f"results/{key}/history.json") for key, _, _ in MODELS}
stats = {"scope": "Local held-out DIII-D validation, 32 shots / 1024 cached frames; not leaderboard scores", "models": {}}
for key, label, _ in MODELS:
    stats["models"][key] = {"label": label, "score": {k: v for k,v in scores[key].items() if k != "evaluation"}, "run": runs[key]}

def save(fig, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=180)
    plt.close(fig)
    print(name, flush=True)

def grid(ax):
    ax.grid(axis="y", color="#E6EBEF", linewidth=.8)
    ax.set_axisbelow(True)

# Fixed validation example: the first shot and its middle cached frame.
entry = manifest["val"][0]
with np.load(ROOT / "data/processed" / entry["path"]) as sample:
    frame = len(sample["psi"]) // 2
    truth, R, Z = sample["psi"][frame], sample["R"], sample["Z"]
    stamp = float(sample["times"][frame])
    original_frame = int(sample["frame_indices"][frame])
predictions = []
for key, _, _ in MODELS:
    with np.load(ROOT / f"results/{key}/validation_predictions.npz") as archive:
        predictions.append(archive["shot_0000_psirz"][frame])
fields = [truth] + predictions
low, high = min(float(x.min()) for x in fields), max(float(x.max()) for x in fields)
limit = max(float(np.abs(x-truth).max()) for x in predictions)
stats["field_selection"] = {"shot_id": entry["shot_id"], "shot_index": entry["shot_index"],
    "cached_frame_index": frame, "original_frame_index": original_frame, "timestamp_ms": stamp,
    "selection": "First validation shot; middle cached frame, fixed before inspecting errors",
    "flux_range": [low, high], "symmetric_error_limit": limit,
    "sample_rmse": {MODELS[i][0]: float(np.sqrt(np.mean((pred-truth)**2))) for i,pred in enumerate(predictions)}}
fig, axes = plt.subplots(2, 3, figsize=(15,8), layout="constrained", gridspec_kw={"width_ratios": [1,1,1]})
for ax, field, title in zip(axes[0], fields, ["EFIT target", "DeepONet prediction", "Fourier-DeepONet prediction"]):
    im = ax.pcolormesh(R, Z, field, cmap="viridis", vmin=low, vmax=high, shading="auto", rasterized=True)
    ax.contour(R, Z, field, levels=np.linspace(low,high,13), colors="white", linewidths=.45, alpha=.55)
    ax.set(title=title, xlabel="R (m)", ylabel="Z (m)", aspect="equal")
fig.colorbar(im, ax=axes[0,:], label="Poloidal flux ψ (V·s/rad)", shrink=.85, pad=.02)
axes[1,0].axis("off")
axes[1,0].text(.05,.87, "Fixed validation example", fontsize=19, fontweight="bold", transform=axes[1,0].transAxes)
axes[1,0].text(.05,.70, f"Shot {entry['shot_id']}\nTime {stamp:,.1f} ms\nCached frame {frame + 1} / {entry['n_frames']}\nOriginal frame index {original_frame}",
    fontsize=16, linespacing=1.6, transform=axes[1,0].transAxes, va="top")
axes[1,0].text(.05,.14, "First shot; middle cached frame.\nShared color scales for comparison.", fontsize=13,
    transform=axes[1,0].transAxes, color="#56677A", linespacing=1.5)
for ax, pred, (_, label, _) in zip(axes[1,1:], predictions, MODELS):
    im = ax.pcolormesh(R, Z, pred-truth, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto", rasterized=True)
    ax.set(title=f"{label} error", xlabel="R (m)", ylabel="Z (m)", aspect="equal")
fig.colorbar(im, ax=axes[1,1:], label="Prediction − target (V·s/rad)", shrink=.85, pad=.02)
save(fig, "field_compare")

# Separate field/error panels keep each physical-aspect map legible on a slide.
with plt.rc_context({"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16}):
    fig, axes = plt.subplots(1, 3, figsize=(11,5.8), layout="constrained")
    for ax, field, title in zip(axes, fields, ["EFIT target", "DeepONet", "Fourier-DeepONet"]):
        im = ax.pcolormesh(R, Z, field, cmap="viridis", vmin=low, vmax=high, shading="auto", rasterized=True)
        ax.contour(R, Z, field, levels=np.linspace(low,high,13), colors="white", linewidths=.5, alpha=.55)
        ax.set(title=title, xlabel="R (m)", aspect="equal")
    axes[0].set_ylabel("Z (m)")
    fig.colorbar(im, ax=axes, label="Poloidal flux ψ (V·s/rad)", shrink=.88, pad=.025, fraction=.035)
    save(fig, "field_flux")

    fig, axes = plt.subplots(1, 2, figsize=(8.5,5.8), layout="constrained")
    for ax, pred, (_, label, _) in zip(axes, predictions, MODELS):
        im = ax.pcolormesh(R, Z, pred-truth, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto", rasterized=True)
        ax.set(title=label, xlabel="R (m)", aspect="equal")
    axes[0].set_ylabel("Z (m)")
    fig.colorbar(im, ax=axes, label="Prediction − target (V·s/rad)", shrink=.88, pad=.03, fraction=.05)
    save(fig, "field_errors")

# Recorded normalized objective, displayed against both epoch and elapsed training time.
fig, axes = plt.subplots(1,2,figsize=(15,5), layout="constrained")
for key, label, color in MODELS:
    h, run = history[key], runs[key]
    best = next(x for x in h if x["epoch"] == run["best_epoch"])
    for ax, field, xlabel in zip(axes, ["epoch", "seconds"], ["Epoch", "Elapsed training time (s)"]):
        ax.plot([x[field] for x in h], [x["train_loss"] for x in h], "--", color=color, lw=1.8, alpha=.8)
        ax.plot([x[field] for x in h], [x["val_loss"] for x in h], color=color, lw=2.4)
        ax.scatter(best[field], best["val_loss"], s=115, marker="*", color=color, edgecolor=NAVY, zorder=5)
        ax.set(xlabel=xlabel, ylabel="Normalized loss (log scale)", yscale="log")
        ax.grid(color="#E6EBEF", linewidth=.8)
    axes[0].annotate(f"{label}: epoch {best['epoch']}", (best['epoch'], best['val_loss']),
        xytext=(10, 15 if key == "deeponet" else -23), textcoords="offset points", color=color, fontsize=12)
axes[0].set_title("Learning curves and chosen checkpoints")
axes[1].set_title("Same objective, different compute budgets")
axes[1].text(.98,.97, "NVIDIA A40\nDeepONet: 31.0 s / 150 epochs\nFourier-DeepONet: 376.1 s / 100 epochs", ha="right",va="top", transform=axes[1].transAxes, fontsize=12)
handles = [Line2D([0],[0],color=c,lw=3,label=l) for _,l,c in MODELS]
handles += [Line2D([0],[0],color=NAVY,lw=2,label="Validation"), Line2D([0],[0],color=NAVY,lw=2,ls="--",label="Training"), Line2D([0],[0],marker="*",color=NAVY,ls="",markersize=12,label="Best checkpoint")]
fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=12)
save(fig, "training_curves")

fig, ax = plt.subplots(figsize=(15,5),layout="constrained")
labels = ["Overall score S", "Flux R²", "Scalar R²", "1 − LCFS distance", "Consistency"]
metrics = ["S", "r2_psi", "r2_qb", "dlcfs", "consistency"]
x, width = np.arange(len(labels)), .34
for i,(key,label,color) in enumerate(MODELS):
    values = [1-scores[key][m] if m == "dlcfs" else scores[key][m] for m in metrics]
    bars = ax.bar(x+(i-.5)*width,values,width,color=color,label=label)
    ax.bar_label(bars,labels=[f"{v:.4f}" for v in values],padding=5,fontsize=12)
ax.set(xticks=x,xticklabels=labels,ylim=(0,1.13),ylabel="Score component (higher is better)")
ax.set_yticks(np.arange(0,1.01,.2));grid(ax)
ax.legend(loc="upper center",ncol=2,bbox_to_anchor=(.5,1.13))
save(fig,"metric_comparison")

# All held-out cached frames are plotted, without filtering by error.
truth_scalars = []
for item in manifest["val"]:
    with np.load(ROOT / "data/processed" / item["path"]) as shot:
        truth_scalars.append(shot["scalars"])
truth_scalars = np.concatenate(truth_scalars)
scalar_pred = {}
for key,_,_ in MODELS:
    with np.load(ROOT / f"results/{key}/validation_predictions.npz") as archive:
        scalar_pred[key] = np.concatenate([np.column_stack([archive[f"shot_{i:04d}_{s}"] for s in ("q95","betaN")]) for i in range(len(manifest["val"]))])
fig, axes = plt.subplots(2,2,figsize=(15,7.5),layout="constrained")
stats["scalar_parity"] = {"frames_per_panel": int(len(truth_scalars)), "r2_convention": "All cached validation frames pooled, direct coefficient of determination", "r2": {}}
for j,(scalar,title) in enumerate([("q95","q₉₅"),("betaN","βN")]):
    all_values = np.concatenate([truth_scalars[:,j]]+[v[:,j] for v in scalar_pred.values()])
    low,high = float(all_values.min()),float(all_values.max()); pad=(high-low)*.05
    for i,(key,label,color) in enumerate(MODELS):
        ax=axes[j,i]; target,pred=truth_scalars[:,j],scalar_pred[key][:,j]
        r2=float(1-np.square(pred-target).sum()/np.square(target-target.mean()).sum())
        stats["scalar_parity"]["r2"].setdefault(key,{})[scalar]=r2
        ax.scatter(target,pred,s=13,alpha=.32,color=color,edgecolors="none",rasterized=True)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],color=NAVY,lw=1.4,ls="--")
        ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),xlabel=f"EFIT target {title}",ylabel=f"Predicted {title}",title=f"{label} · {title}")
        ax.text(.03,.95,f"R² = {r2:.4f}\nn = {len(target):,} frames",va="top",transform=ax.transAxes,fontsize=14,bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
        ax.grid(color="#E6EBEF",linewidth=.8);ax.set_axisbelow(True)
save(fig,"scalar_parity")

fig,ax=plt.subplots(figsize=(15,5),layout="constrained")
keys=["R_axis","Z_axis","kappa","tri_top","tri_bot","volume","li"]
labels=["R axis","Z axis","Elongation κ","Upper δ","Lower δ","Volume","Internal\ninductance ℓᵢ"]
x=np.arange(len(keys))
for i,(key,label,color) in enumerate(MODELS):
    values=[scores[key]["r2_cons_each"][m] for m in keys]
    bars=ax.bar(x+(i-.5)*width,values,width,color=color,label=label)
    ax.bar_label(bars,labels=[f"{v:.3f}" for v in values],padding=5,fontsize=12)
ax.set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel="Derived-scalar R² (higher is better)")
grid(ax);ax.legend(loc="upper center",ncol=2,bbox_to_anchor=(.5,1.13))
save(fig,"derived_consistency")

raw=read("data/raw/download_manifest.json")
counts=Counter(item["path"].split("/")[1] for item in raw["files"] if item["path"].endswith(".parquet") and item["path"].startswith("data/"))
stats["dataset"]={"downloaded_parquet_counts":dict(counts),"experiment_train_shots":len(manifest["train"]),"experiment_validation_shots":len(manifest["val"]),"train_frames":runs["deeponet"]["train_frames"],"validation_frames":runs["deeponet"]["validation_frames"]}
fig,axes=plt.subplots(1,2,figsize=(15,5),layout="constrained",gridspec_kw={"width_ratios":[1.2,1]})
groups=[("diii_d_train","DIII-D training"),("diii_d_public_test","DIII-D public test"),("mast_public_test","MAST public test")]
values=[counts[key] for key,_ in groups]
bars=axes[0].barh([label for _,label in groups],values,color=[NAVY,TEAL,ORANGE],height=.55)
axes[0].bar_label(bars,labels=[f"{v:,}" for v in values],padding=7,fontsize=16)
axes[0].set(xlabel="Downloaded shots",xlim=(0,max(values)*1.18),title="Complete challenge data downloaded")
axes[0].invert_yaxis();axes[0].grid(axis="x",color="#E6EBEF");axes[0].set_axisbelow(True)
bars=axes[1].bar(["Training","Held-out validation"],[256,32],color=[TEAL,ORANGE],width=.55)
axes[1].bar_label(bars,labels=["256 shots\n16,213 frames","32 shots\n1,024 frames"],padding=8,fontsize=15)
axes[1].set(ylabel="Shots used in this first experiment",ylim=(0,330),title="First baseline: 288 DIII-D shots")
grid(axes[1])
save(fig,"dataset_overview")
(OUT/"stats.json").write_text(json.dumps(stats,indent=2)+"\n")
