# Fusion Equilibrium Challenge — Challenge 1 (DIII-D) reproduction package

This package rebuilds our DIII-D entry from the public dataset: data preparation (CPU), model training from scratch (GPU, budgeted),
and inference/packaging. The code is the code that trained the submitted members; only machine-specific paths were made relative.

## Quick start

```bash
conda env create -f env/environment.yml && conda activate fusion-c1-final
export FUSION_ROOT=$PWD PYTHON=$(which python)
bash prepare.sh        # CPU: download (pinned revision) -> views -> li stores -> vessel/future features -> scalar targets
bash train.sh          # GPU: 17 LVF + 26 LAV members (4 at a time on one GPU), 6 direct axis regressors, POD-512 closure
bash predict.sh        # blend + closure + gate-aware axis shift -> results/candidates/final/submission.zip
```
Under a batch scheduler, set `FUSION_ROOT` explicitly (job scripts run from a spool copy).

## Model

- **Members** (all trained on every labelled DIII-D shot of the public release: the training split plus the public validation
  splits; the audit split is never used):
  - `LVF` x17: POD-128 DeepONet-style MLP (width 256, depth 3) on 708 history/Thomson/current inputs + 126 vessel-current features
    (exponentially filtered derivatives of the raw 0.05 ms coil/plasma currents, tau 2-50 ms) + 126 future-actuator features
    (the same filters run backwards in time over the next 2-50 ms of the same shot). Cold 600 epochs (AdamW) -> refine 60 epochs
    (amplitude frame weights, fixed-contour li loss) -> 30 epochs Muon (lr ratio 30). Recipe: `configs/LVF.json`.
  - `LAV` x26: the no-history A recipe with the vessel features; cold 600 -> refine 30 -> Muon 30 epochs (ratio 200). `configs/LAV.json`.
- **Blend**: equal weight per member; on Thomson-poor frames (useful-Thomson fraction <= 0.1) only the history (LVF) members are used.
- **Closure**: POD modes 129-512 of the blended flux are filled by a small network trained on training flux (`results/analysis/ae_ceiling_119`).
- **Axis shift**: inside the separatrix only, the magnetic axis is moved half-way (vertical component only) toward the axis predicted by
  6 direct regressors on the same inputs; skipped on Thomson-poor frames. The separatrix is untouched.

## Compute (GPU budget: 24 GPU-hours on one 80 GB A100-class GPU)

| stage | measured |
|---|---|
| one member, solo | 27-37 min (H100/A40/L40S), 4 members per GPU at once: 12-14 min per member (H100) |
| 43 members, 4 at a time | ~9-10 h (H100); ~11-14 h estimated on an uncontended A100 |
| direct regressors | ~6 min |
| closure (POD-512 + autoencoder ceilings + closure net) | ~47 min (A40) |

`train.sh` writes `gpu_timing.json`. CPU preparation (~3-5 h on 8-16 cores, plus ~1.5 h for the li stores on a large-memory node) is not budgeted.

## Data provenance

See `PROVENANCE.md`. Source: Hugging Face `Sophelio/fusion-equilibrium-challenge`, revision pinned in `provenance/download_manifest.json`
(per-file sha256). Starter-kit data fixes applied (`starter/data_fixes.py`, DIII-D plasma-current timestamps). No test labels, no audit
split, no MAST labels or MAST training data. Inputs only: coil currents, plasma current, Thomson profiles, geometry and time metadata.

## Reproducibility

Training is reproducible statistically, not bitwise: GPU floating-point summation order differs between GPU types, and over 600 cold epochs this makes a rebuilt member a fresh draw of the same distribution. On the same GPU type a rebuild stays close to the original (test-flux relative difference 0.002 vs 0.010 between seeds); across GPU types it differs like another seed. The ensemble average (43 members) is insensitive to this.

## Trained weights (the submitted entry)

`weights/` holds the exact trained parameters of the submitted ensemble (checksums in `weights/SHA256SUMS`):
- `weights/members/LVF_s6001 ... LVF_s6017/epoch_0030.pt` and `weights/members/LAV_s7101 ... LAV_s7126/epoch_0030.pt` (final Muon epoch of each member)
- `weights/direct_vf60/w256_d3_s{4..9}.pt` (direct axis regressors)
- `weights/closure/{basis512.npz, closure64.pt}` (POD-512 basis and closure network)

To regenerate the submission from these weights without training: `bash prepare.sh && bash predict.sh` (members missing under `members/`
are exported from `weights/`; the regressor predictions and closure files are restored from `weights/` when `train.sh` has not run).
