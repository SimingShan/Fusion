#!/usr/bin/env python3
"""Make exportable validation figures; selects a fixed shot/frame, never by error."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,default=Path("data/processed"))
    p.add_argument("--results",type=Path,default=Path("results"))
    args=p.parse_args()
    manifest=json.loads((args.data_dir/"manifest.json").read_text())
    entry=manifest["val"][0]
    with np.load(args.data_dir/entry["path"]) as shot:
        frame=len(shot["psi"])//2
        truth=shot["psi"][frame];R=shot["R"];Z=shot["Z"]
        stamp=float(shot["times"][frame])
    predictions=[]
    for name in ("deeponet","fourier_deeponet"):
        with np.load(args.results/name/"validation_predictions.npz") as archive:
            predictions.append(archive["shot_0000_psirz"][frame])
    fields=[truth]+predictions
    fig,axes=plt.subplots(2,3,figsize=(11,9),layout="constrained")
    low,high=min(x.min() for x in fields),max(x.max() for x in fields)
    for ax,field,title in zip(axes[0],fields,["EFIT target","DeepONet","Fourier-DeepONet"]):
        im=ax.pcolormesh(R,Z,field,cmap="viridis",vmin=low,vmax=high,shading="auto")
        ax.contour(R,Z,field,levels=12,colors="white",linewidths=.4,alpha=.6)
        ax.set_title(title);ax.set_xlabel("R (m)");ax.set_ylabel("Z (m)");ax.set_aspect("equal")
    fig.colorbar(im,ax=axes[0,:],label="Poloidal flux ψ (V·s/rad)",shrink=.8)
    axes[1,0].axis("off")
    axes[1,0].text(.05,.9,f"Held-out validation shot index {entry['shot_index']}\nTimestamp {stamp:.3f} ms\n\nSelection: first validation shot,\nmiddle cached frame.\n\nErrors use the stored flux sign.\nAll test targets are hidden.",transform=axes[1,0].transAxes,va="top",fontsize=11)
    limit=max(np.abs(x-truth).max() for x in predictions)
    for ax,pred,title in zip(axes[1,1:],predictions,["DeepONet error","Fourier-DeepONet error"]):
        im=ax.pcolormesh(R,Z,pred-truth,cmap="RdBu_r",vmin=-limit,vmax=limit,shading="auto")
        ax.set_title(title);ax.set_xlabel("R (m)");ax.set_ylabel("Z (m)");ax.set_aspect("equal")
    fig.colorbar(im,ax=axes[1,1:],label="Prediction − target (V·s/rad)",shrink=.8)
    for suffix in ("png","pdf"):fig.savefig(args.results/f"field_comparison.{suffix}",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    for name,label in (("deeponet","DeepONet"),("fourier_deeponet","Fourier-DeepONet")):
        history=json.loads((args.results/name/"history.json").read_text())
        line,=ax.plot([x["epoch"] for x in history],[x["val_loss"] for x in history],label=label+" validation")
        ax.plot([x["epoch"] for x in history],[x["train_loss"] for x in history],"--",color=line.get_color(),alpha=.6,label=label+" training")
    ax.set_yscale("log");ax.set_xlabel("Epoch");ax.set_ylabel("Normalized flux + scalar + gradient loss")
    ax.legend();ax.grid(alpha=.25)
    for suffix in ("png","pdf"):fig.savefig(args.results/f"training_curves.{suffix}",dpi=180)
    plt.close(fig)


if __name__=="__main__":main()
