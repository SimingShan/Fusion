# Fusion Equilibrium Challenge — Challenge 1 (DIII-D) reproduction package

This package rebuilds our DIII-D entry from the public dataset: data preparation (CPU), model training from scratch (GPU, budgeted),
and inference/packaging. The code is the code that trained the submitted members; only machine-specific paths were made relative.

## Quick start

```bash
conda env create -f env/environment.yml && conda activate fusion-c1-final
export FUSION_ROOT=$PWD PYTHON=$(which python)
bash prepare.sh        # CPU: download (pinned revision) -> views -> li stores -> vessel/future features -> scalar targets
bash train.sh          # GPU: 24 LVF + 40 LAV members (4 at a time on one GPU), 6 direct regressors, POD-512 closure
bash predict.sh        # blend + closure + axis shift + li adjustment -> results/candidates/final/submission.zip
```
Under a batch scheduler, set `FUSION_ROOT` explicitly (job scripts run from a spool copy).

## Model

- **Members** (all trained on every labelled DIII-D shot of the public release: the training split plus the public validation
  splits; the audit split is never used):
  - `LVF` x24 (seeds 6001-6017, 6101-6107): POD-128 DeepONet-style MLP (width 256, depth 3) on 708 history/Thomson/current inputs
    + 126 vessel-current features (exponentially filtered derivatives of the raw 0.05 ms coil/plasma currents, tau 2-50 ms)
    + 126 future-actuator features (the same filters run backwards in time over the next 2-50 ms of the same shot).
    Cold 600 epochs (AdamW) -> refine 60 epochs (amplitude frame weights, fixed-contour li loss) -> 30 epochs Muon (lr ratio 30).
    Recipe: `configs/LVF.json`.
  - `LAV` x40 (seeds 7101-7126, 7201-7214): the no-history A recipe with the vessel features; cold 600 -> refine 30 -> Muon 30
    epochs (ratio 200). `configs/LAV.json`.
- **Blend**: equal weight per member; on Thomson-poor frames (useful-Thomson fraction <= 0.1) only the history (LVF) members are used.
- **Closure**: POD modes 129-512 of the blended flux are filled by a small network trained on training flux (`results/analysis/ae_ceiling_119`).
- **Direct regressors**: 6 small MLPs (256x3) predict the seven flux-derived scalars from the same inputs (training split only).
- **Axis shift**: inside the separatrix only, the magnetic axis is moved half-way (vertical component only) toward the regressors'
  axis; skipped on Thomson-poor frames. The separatrix is untouched.
- **li adjustment** (`results/analysis/lihead_210/li_adjust.py`): inside the separatrix the normalised flux s is remapped,
  h(s) = s + a s (1 - s)^2 (axis value, separatrix level and edge gradient unchanged), with `a` solved per frame so that the internal
  inductance computed from the map (the scorer's li(2) definition) moves 35% of the way toward the regressors' li. Shape, separatrix
  and axis are unchanged; skipped on Thomson-poor frames.

## Compute (GPU budget: 24 GPU-hours on one 80 GB A100-class GPU)

| stage | measured / derived |
|---|---|
| one member, 4 members sharing one GPU | A100 80GB: 15.7 GPU-min (LVF), 15.3 GPU-min (LAV) (measured, one full batch each) |
| 64 members (24 LVF + 40 LAV, 16 full batches of 4) | ~16.5 h on one A100 (16 x 4 x ~15.5 min); ~11.3 h on one H100 (16 batches x ~0.7 h, from the run below) |
| direct regressors | 5.5 min (H100, measured) |
| closure (POD-512 + autoencoder ceilings + closure net) | 30 min (H100, measured) |
| **total** | **~17.3 A100-hours** (24 h cap); ~11.9 H100-hours |

`train.sh` writes `gpu_timing.json`. Measured end to end: a complete from-scratch `train.sh` run of the 43-member subset (17 LVF + 26
LAV, one H100 80GB, 4 members at a time) took 3.30 h (LVF, 5 batches) + 4.98 h (LAV, 7 batches) + 0.09 h (regressors) + 0.51 h
(closure) = 8.9 H100-hours; the 64-member default adds 4 batches. CPU preparation (~3-5 h on 8-16 cores, plus ~1.5 h for the li
stores on a large-memory node) and inference post-processing (closure, axis shift, li adjustment: ~1 h on 10 cores) are not budgeted.

## New test inputs (final phase)

The final phase scores a private test split that is not part of the public dataset. With `prepare.sh` already run:
```bash
bash prepare_test.sh /path/to/new_test_parquet_dir          # CPU: install the shots, rebuild every test-derived view and feature
MEMBERS_DIR=members_final WEIGHTS=weights bash predict.sh final_private
```
Verified with the public test as a stand-in: the rebuilt test views and features were bit-identical to `prepare.sh`'s (6,153 arrays),
and inference from `weights/` with fresh member exports reproduced the full-pipeline prediction bit-identically (same archive sha256).

## Data provenance

See `PROVENANCE.md`. Source: Hugging Face `Sophelio/fusion-equilibrium-challenge`, revision pinned in `provenance/download_manifest.json`
(per-file sha256). Starter-kit data fixes applied (`starter/data_fixes.py`, DIII-D plasma-current timestamps). No test labels, no audit
split, no MAST labels or MAST training data. Inputs only: coil currents, plasma current, Thomson profiles, geometry and time metadata.

## Reproducibility

Training is reproducible statistically, not bitwise: GPU floating-point summation order differs between GPU types, and over 600 cold
epochs this makes a rebuilt member a fresh draw of the same distribution. On the same GPU type a rebuild stays close to the original
(test-flux relative difference 0.002 vs 0.010 between seeds); across GPU types it differs like another seed. Two independent 43-member
draws of this recipe scored within 0.000004 of each other on the public leaderboard. Inference from given weights is deterministic.

## Trained weights (the submitted entry)

`weights/` holds the exact trained parameters of the submitted ensemble (checksums in `weights/SHA256SUMS`), all produced by this
package's own `train_members.py`:
- `weights/members/LVF_s6001 ... LVF_s6017`, `LVF_s6101 ... LVF_s6107`, `LAV_s7101 ... LAV_s7126`, `LAV_s7201 ... LAV_s7214`
  (`epoch_0030.pt` = final Muon epoch of each member; 64 members)
- `weights/direct_vf60/w256_d3_s{4..9}.pt` (direct regressors)
- `weights/closure/{basis512.npz, closure64.pt}` (POD-512 basis and closure network)

To regenerate the submission from these weights without training: `bash prepare.sh && bash predict.sh` (members missing under `members/`
are exported from `weights/`; the regressor predictions and closure files are restored from `weights/` when `train.sh` has not run).
