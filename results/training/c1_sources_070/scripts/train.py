#!/usr/bin/env python3
"""Train either operator on a fixed shot split and save reproducible predictions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.models import build_model


def entries(data_dir, split):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    return manifest.get("splits", manifest)[split]


def prediction_entries(data_dir, split):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    selected = manifest.get("splits", manifest)[split]
    if split == "test":
        expected = manifest.get("expected_shots", {}).get("diii_d_public_test")
        if not isinstance(expected, int) or expected < 1:
            raise ValueError("Public-test export requires manifest.expected_shots.diii_d_public_test")
        if len(selected) != expected:
            raise ValueError(f"Incomplete public-test cache: {len(selected)} shots; expected {expected}")
        if [entry.get("shot_index") for entry in selected] != list(range(expected)):
            raise ValueError("Public-test shot_index must follow canonical order 0..expected-1")
        if len({entry["path"] for entry in selected}) != expected:
            raise ValueError("Public-test cache contains duplicate shot files")
        for entry in selected:
            if entry.get("n_frames", 0) != entry.get("raw_frames"):
                raise ValueError(f"Public-test cache must include every frame: {entry['path']}")
            if not (data_dir / entry["path"]).is_file():
                raise ValueError(f"Missing public-test cache file: {entry['path']}")
    return selected


def write_training_manifest(data_dir, output_dir, source="training_startup"):
    """Snapshot exact training selection and hash the source manifest bytes."""
    manifest_path = data_dir / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    splits = manifest.get("splits", manifest)
    selection = {key: manifest[key] for key in
                 ("version", "dataset", "settings", "feature_names", "scalar_names", "selection")
                 if key in manifest}
    selection.update(train=splits["train"], val=splits["val"])
    canonical = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    snapshot = {**selection, "snapshot_metadata": {
        "source": source, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "training_selection_sha256": hashlib.sha256(canonical).hexdigest(),
    }}
    path = output_dir / "training_manifest.json"
    path.write_text(json.dumps(snapshot, indent=2)+"\n")
    return path


def load_arrays(data_dir, split):
    values = {k: [] for k in ("X", "psi", "scalars")}
    selected = entries(data_dir, split)
    if not selected:
        raise ValueError(f"No prepared shots in {split}")
    for entry in selected:
        with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
            for k in values: values[k].append(shot[k])
    return {k: np.concatenate(v).astype(np.float32) for k, v in values.items()}


def validate_shot_split(data_dir):
    """Reject train/validation overlap before fitting any normalization."""
    train_entries, val_entries = entries(data_dir, "train"), entries(data_dir, "val")
    if not train_entries or not val_entries:
        raise ValueError("Training and validation must both contain prepared shots")
    for key in ("path", "shot_id", "shot_index"):
        if not all(key in entry for entry in train_entries + val_entries):
            continue
        train_ids = [entry[key] for entry in train_entries]
        val_ids = [entry[key] for entry in val_entries]
        if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
            raise ValueError(f"Duplicate {key} within a training/validation split")
        overlap = set(train_ids) & set(val_ids)
        if overlap:
            raise ValueError(f"Training/validation {key} overlap: {sorted(overlap)[:5]}")


def fit_normalization(train):
    x = train["X"].astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(train["psi"]).all():
        raise ValueError("Features and flux targets must be finite")
    scalar_values = np.asarray(train["scalars"], dtype=np.float64)
    scalar_values = np.where(np.isfinite(scalar_values), scalar_values, np.nan)
    if not np.isfinite(scalar_values).any(axis=0).all():
        raise ValueError("Each scalar target needs at least one finite training label")
    return {"x_mean": x.mean(0).astype(np.float32),
            "x_std": np.maximum(x.std(0), 1e-6).astype(np.float32),
            "psi_mean": train["psi"].mean(0, dtype=np.float64).astype(np.float32),
            "psi_std": np.float32(max(train["psi"].std(), 1e-6)),
            "scalar_mean": np.nanmean(scalar_values, 0).astype(np.float32),
            "scalar_std": np.maximum(np.nanstd(scalar_values, 0), 1e-6).astype(np.float32)}


def normalized(data, norm, device):
    x = np.clip((data["X"] - norm["x_mean"]) / norm["x_std"], -12, 12)
    y = (data["psi"] - norm["psi_mean"]) / norm["psi_std"]
    scalars = (data["scalars"] - norm["scalar_mean"]) / norm["scalar_std"]
    scalar_mask = np.isfinite(scalars)
    scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
    return [torch.as_tensor(a, device=device) for a in (x, y, scalars, scalar_mask)]


def loss_fn(pred, scalars, y, target_scalars, mask, gradient_weight):
    flux_loss = (pred-y).square().mean()
    scalar_loss = ((scalars-target_scalars).square()*mask).sum()/mask.sum().clamp_min(1)
    gradient_loss = ((pred[:, 1:]-pred[:, :-1])-(y[:, 1:]-y[:, :-1])).square().mean()
    gradient_loss += ((pred[:, :, 1:]-pred[:, :, :-1])-(y[:, :, 1:]-y[:, :, :-1])).square().mean()
    return flux_loss + 0.15*scalar_loss + gradient_weight*gradient_loss


@torch.no_grad()
def validation(model, arrays, batch_size, gradient_weight):
    model.eval()
    x,y,s,m = arrays
    total = scalar_sse = scalar_count = 0.0
    for start in range(0,len(x),batch_size):
        sl = slice(start,start+batch_size)
        p,q = model(x[sl])
        # Flux/gradient terms average over frames; scalar terms average only over
        # observed labels, independently of their distribution between batches.
        total += loss_fn(p,q,y[sl],s[sl],torch.zeros_like(m[sl]),gradient_weight).item()*len(x[sl])
        scalar_sse += ((q-s[sl]).square()*m[sl]).sum().item()
        scalar_count += m[sl].sum().item()
    return total/len(x) + 0.15*scalar_sse/max(scalar_count,1)


@torch.no_grad()
def write_predictions(model, norm, data_dir, split, output, batch_size, device):
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    psi_sse = 0.0
    psi_sum = 0.0
    psi_sq = 0.0
    pixels = frames = 0
    scalar_gt, scalar_pred = [], []
    selected = prediction_entries(data_dir, split)
    tmp = output.with_suffix(".tmp.npz")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i, entry in enumerate(selected):
            with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
                x = np.clip((shot["X"]-norm["x_mean"])/norm["x_std"],-12,12)
                if split == "test" and len(x) != entry["raw_frames"]:
                    raise ValueError(f"Public-test cached frame count mismatch: {entry['path']}")
                p_parts, s_parts = [], []
                for start in range(0,len(x),batch_size):
                    p,s = model(torch.as_tensor(x[start:start+batch_size],device=device))
                    p_parts.append(p.cpu().numpy()*norm["psi_std"]+norm["psi_mean"])
                    s_parts.append(s.cpu().numpy()*norm["scalar_std"]+norm["scalar_mean"])
                p,s = np.concatenate(p_parts),np.concatenate(s_parts)
                if p.shape != (len(x),65,65) or s.shape != (len(x),2) or not np.isfinite(p).all() or not np.isfinite(s).all():
                    raise ValueError(f"Invalid predictions for {entry}")
                for key,array in (("psirz",p),("q95",s[:,0]),("betaN",s[:,1])):
                    with archive.open(f"shot_{i:04d}_{key}.npy","w",force_zip64=True) as member:
                        np.lib.format.write_array(member,array.astype(np.float32),allow_pickle=False)
                frames += len(x)
                if "psi" in shot:
                    gt = shot["psi"].astype(np.float64)
                    psi_sse += float(((p-gt)**2).sum())
                    psi_sum += float(gt.sum()); psi_sq += float((gt**2).sum()); pixels += gt.size
                    scalar_gt.append(shot["scalars"]); scalar_pred.append(s)
            if (i+1)%50==0: print(f"predict {split}: {i+1}/{len(selected)} shots",flush=True)
    tmp.replace(output)
    metrics = {"split":split,"shots":len(selected),"frames":frames,"dtype":"float32","archive":str(output)}
    if pixels:
        psi_sst = psi_sq-psi_sum**2/pixels
        metrics.update(psi_rmse=(psi_sse/pixels)**0.5,r2_psi=1-psi_sse/psi_sst if psi_sst>0 else None)
        g,p = np.concatenate(scalar_gt),np.concatenate(scalar_pred)
        for j,name in enumerate(("q95","betaN")):
            ok = np.isfinite(g[:,j]); a=g[ok,j].astype(np.float64); b=p[ok,j]
            sst = ((a-a.mean())**2).sum() if len(a) else 0.0
            metrics[f"r2_{name}"] = float(1-((a-b)**2).sum()/sst) if sst>0 else None
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics),flush=True)
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["deeponet","fourier_deeponet"],default="deeponet")
    ap.add_argument("--data-dir",type=Path,default=ROOT/"data/processed")
    ap.add_argument("--output",type=Path)
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=0.001)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--seconds",type=int,default=2400)
    ap.add_argument("--width",type=int,default=128)
    ap.add_argument("--latent-dim",type=int,default=64)
    ap.add_argument("--spectral-width",type=int,default=16)
    ap.add_argument("--spectral-layers",type=int,default=2)
    ap.add_argument("--modes",type=int,default=10)
    ap.add_argument("--gradient-weight",type=float,default=1.0)
    ap.add_argument("--predict-only",action="store_true")
    ap.add_argument("--split",choices=["val","test"],default="test")
    args=ap.parse_args()
    if min(args.epochs,args.batch_size,args.seconds) < 1:
        ap.error("epochs, batch-size, and seconds must be positive")
    args.output=args.output or ROOT/"results"/args.model
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    if args.predict_only:
        checkpoint=torch.load(args.output/"best.pt",map_location="cpu",weights_only=False)
        model=build_model(checkpoint["model"],**checkpoint["model_kwargs"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        name="diii_d_public_test.npz" if args.split=="test" else "validation_predictions.npz"
        write_predictions(model,checkpoint["normalization"],args.data_dir,args.split,args.output/name,args.batch_size,device)
        return
    validate_shot_split(args.data_dir)
    write_training_manifest(args.data_dir, args.output)
    train=load_arrays(args.data_dir,"train");val=load_arrays(args.data_dir,"val")
    norm=fit_normalization(train)
    kwargs=dict(input_dim=train["X"].shape[1],width=args.width,latent_dim=args.latent_dim,modes=args.modes,
                spectral_width=args.spectral_width,spectral_layers=args.spectral_layers)
    model=build_model(args.model,**kwargs).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*0.05)
    training=normalized(train,norm,device);valid=normalized(val,norm,device)
    run={"model":args.model,"model_kwargs":kwargs,"seed":args.seed,"train_frames":len(train["X"]),
         "validation_frames":len(val["X"]),"train_shots":len(entries(args.data_dir,"train")),
         "validation_shots":len(entries(args.data_dir,"val")),"parameters":sum(p.numel() for p in model.parameters()),
         "device":str(device),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,
         "torch_version":torch.__version__,"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
         "target_centering":"training mean flux map and global flux standard deviation; training scalar means/std",
         "checkpoint_selection":"minimum held-out normalized flux+scalar+gradient MSE"}
    print(json.dumps(run),flush=True)
    (args.output/"run.json").write_text(json.dumps(run,indent=2)+"\n")
    del train, val
    best=float("inf");started=time.perf_counter();history=[]
    x,y,s,m=training
    for epoch in range(1,args.epochs+1):
        model.train();perm=torch.randperm(len(x),device=device);train_loss=0.0
        for indices in perm.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            p,q=model(x[indices]);loss=loss_fn(p,q,y[indices],s[indices],m[indices],args.gradient_weight)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            train_loss+=loss.item()*len(indices)
        val_loss=validation(model,valid,args.batch_size,args.gradient_weight)
        if not np.isfinite(val_loss): raise RuntimeError("Nonfinite validation loss")
        record={"epoch":epoch,"train_loss":train_loss/len(x),"val_loss":val_loss,"seconds":time.perf_counter()-started,"lr":optimizer.param_groups[0]["lr"]}
        history.append(record);print(json.dumps(record),flush=True)
        if val_loss<best:
            best=val_loss
            torch.save({"model":args.model,"model_kwargs":kwargs,"state_dict":model.state_dict(),
                        "normalization":norm,"epoch":epoch,"val_loss":val_loss,"run":run},args.output/"best.pt")
        scheduler.step()
        (args.output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        if time.perf_counter()-started>=args.seconds: break
    run.update(training_seconds=time.perf_counter()-started,epochs_completed=epoch,best_val_loss=best)
    checkpoint=torch.load(args.output/"best.pt",map_location=device,weights_only=False)
    run["best_epoch"]=checkpoint["epoch"]
    (args.output/"run.json").write_text(json.dumps(run,indent=2)+"\n")
    model.load_state_dict(checkpoint["state_dict"])
    write_predictions(model,norm,args.data_dir,"val",args.output/"validation_predictions.npz",args.batch_size,device)


if __name__=="__main__": main()
