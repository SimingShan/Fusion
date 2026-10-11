# Fusion Equilibrium Challenge — Challenge 1 (DIII-D) reproduction package

This package rebuilds our DIII-D entry from the public dataset: data preparation (CPU), model training from scratch (GPU, budgeted),
and inference/packaging. The code is the code that trained the submitted members; only machine-specific paths were made relative.

## Quick start

```bash
conda env create -f env/environment.yml && conda activate fusion-c1-final
export FUSION_ROOT=$PWD PYTHON=$(which python)
bash prepare.sh        # CPU: download (pinned revision) -> views -> li stores -> vessel/future features -> scalar targets -> heads view
bash train.sh          # GPU: 24 HLVF + 40 LAV members (4 at a time on one GPU), 6 direct regressors, POD-512 closure
bash predict.sh        # blend + closure + axis shift + elongation and li adjustments -> results/candidates/final/submission.zip
```
Under a batch scheduler, set `FUSION_ROOT` explicitly (job scripts run from a spool copy).

## Model

- **Members** (all trained on every labelled DIII-D shot of the public release: the training split plus the public validation
  splits; the audit split is never used):
  - `HLVF` x24 (seeds 6301-6324): POD-128 DeepONet-style MLP (width 256, depth 3) on 708 history/Thomson/current inputs
    + 126 vessel-current features (exponentially filtered derivatives of the raw 0.05 ms coil/plasma currents, tau 2-50 ms)
    + 126 future-actuator features (the same filters run backwards in time over the next 2-50 ms of the same shot).
    Cold 600 epochs (AdamW) -> refine 60 epochs (amplitude frame weights, fixed-contour li loss) -> 30 epochs Muon (lr ratio 30).
    Besides psi, q95 and betaN, these members have seven extra outputs ("heads") for the seven consistency scalars of the label
    maps (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li; the scorer's own derivation), trained jointly at loss weight 0.05 per
    entry; the q95/betaN loss is exactly the head-less one. Recipe: `configs/HLVF.json` (= `configs/LVF.json` + `--heads-weight 0.05`
    on the 9-scalar view built by `prepare.sh` step 8). `LVF_FAMILY=LVF` trains/uses the head-less recipe (`configs/LVF.json`).
  - `LAV` x40 (seeds 7101-7126, 7201-7214): the no-history A recipe with the vessel features; cold 600 -> refine 30 -> Muon 30
    epochs (ratio 200). `configs/LAV.json`.
- **Blend**: equal weight per member; on Thomson-poor frames (useful-Thomson fraction <= 0.1) only the history (HLVF) members are used.
- **Closure**: POD modes 129-512 of the blended flux are filled by a small network trained on training flux (`results/analysis/ae_ceiling_119`).
- **Direct regressors**: 6 small MLPs (256x3) predict the seven flux-derived scalars from the same inputs (training split only).
- **Axis shift**: inside the separatrix only, the magnetic axis is moved 75% of the way (vertical component only) toward the mean
  of the HLVF heads' axis height and the regressors' axis height (`results/analysis/heads_217/heads_targets.py`); skipped on
  Thomson-poor frames. The separatrix is untouched.
- **Elongation adjustment** (`results/analysis/jkappa_211/`): `member_scalars.py` reads the scorer's kappa off every member's own map;
  the target is the mean of that member-mean kappa, the HLVF heads' kappa and the regressors' kappa, used only where all three agree on
  the direction (`make_targets.py`, `HEADS=`); `kappa_adjust3.py` (KMODE=r) compresses/expands the map radially about the axis inside the plasma, solving per
  frame so the map's kappa moves half-way to the target (`KAPPA_B=0.5`; realised value verified; Thomson-poor frames skipped).
- **li adjustment** (`results/analysis/lihead_210/li_adjust.py`): inside the separatrix the normalised flux s is remapped,
  h(s) = s + a s (1 - s)^2 (axis value, separatrix level and edge gradient unchanged), with `a` solved per frame so that the internal
  inductance computed from the map (the scorer's li(2) definition) moves 35% of the way toward the regressors' li (measured from the li
  before the elongation step). Shape, separatrix and axis are unchanged; skipped on Thomson-poor frames.

## Compute (GPU budget: 24 GPU-hours on one 80 GB A100-class GPU)

| stage | measured / derived |
|---|---|
| one member, 4 members sharing one GPU | A100 80GB: 15.7 GPU-min (LVF), 15.3 GPU-min (LAV) (measured, one full batch each); HLVF = LVF (7 extra outputs; H100 batches of 4: 44-47 min, as LVF) |
| 64 members (24 HLVF + 40 LAV, 16 full batches of 4) | **measured 15.87 h on one A100 80GB PCIe** (24 history members 5.84 h + 40 LAV 10.03 h); ~11.3 h on one H100 |
| direct regressors | 8.2 min (A100 PCIe, measured); 5.5 min (H100) |
| closure (POD-512 + autoencoder ceilings + closure net) | 43 min (A100 PCIe, measured); 30 min (H100) |
| **total** | **16.72 A100-hours measured end to end** (24 h cap); ~11.9 H100-hours |

`train.sh` writes `gpu_timing.json`. Measured end to end on one NVIDIA A100 80GB PCIe (2026-10-10/11, a fresh clone, `prepare.sh`
then `train.sh` with the 64-member defaults): `{"members_LVF_hours": 5.837, "members_LAV_hours": 10.029, "direct_regressors_hours":
0.137, "closure_hours": 0.714, "total_hours": 16.717}`. That run trained the head-less history family (LVF); the HLVF members differ only
by seven extra output units of a 1.2M-parameter network and train in the same time (H100 batches of four: 44-47 min, as LVF). An earlier
43-member run on one H100 80GB took 8.9 H100-hours. CPU preparation (~3-5 h on 8-16 cores, plus ~1.5 h for the li
stores on a large-memory node) and inference post-processing (closure, axis shift, per-member kappa, elongation and li adjustments: ~5 h
on 10 cores) are not budgeted.

## New test inputs (final phase)

The final phase scores a private test split that is not part of the public dataset. With `prepare.sh` already run:
```bash
bash prepare_test.sh /path/to/new_test_parquet_dir          # CPU: install the shots, rebuild every test-derived view and feature
MEMBERS_DIR=members_final WEIGHTS=weights bash predict.sh final_private
```
Verified with the public test as a stand-in: the rebuilt test views and features were bit-identical to `prepare.sh`'s (6,153 arrays).
The submitted entry was re-run from a fresh clone of this repository with only `weights/` (no local training outputs) and CPU-only
member exports: the prediction matched the submitted one to 1.3e-8 of the flux variance (q95/betaN within 6e-6, head targets within
3e-6), with the same per-frame decision in the elongation and li steps on all but ~30 of 193,359 frames (CPU float32 versus GPU
bfloat16 member exports). One member exports in ~2.5 min on 8 CPU cores, so no GPU is needed for inference; the whole `predict.sh`
took ~1.6 h on 64 cores.

## Data provenance

See `PROVENANCE.md`. Source: Hugging Face `Sophelio/fusion-equilibrium-challenge`, revision pinned in `provenance/download_manifest.json`
(per-file sha256). Starter-kit data fixes applied (`starter/data_fixes.py`, DIII-D plasma-current timestamps). No test labels, no audit
split, no MAST labels or MAST training data. Inputs only: coil currents, plasma current, Thomson profiles, geometry and time metadata.

## Reproducibility

Training is reproducible statistically, not bitwise: GPU floating-point summation order differs between GPU types, and over 600 cold
epochs this makes a rebuilt member a fresh draw of the same distribution. On the same GPU type a rebuild stays close to the original
(test-flux relative difference 0.002 vs 0.010 between seeds); across GPU types it differs like another seed. Two independent 43-member
draws of this recipe scored within 0.000004 of each other on the public leaderboard. Inference from given weights is deterministic on
the same CPU type and thread count; with another thread count the closure network's float32 products round differently (flux
differences up to ~1e-4 on ~5% of frames, 2e-13 of the field variance; scalar predictions identical), which does not change the score.

## Trained weights (the submitted entry)

`weights/` holds the exact trained parameters of the submitted ensemble (checksums in `weights/SHA256SUMS`), all produced by this
package's own `train_members.py`:
- `weights/members/HLVF_s6301 ... HLVF_s6324`, `LAV_s7101 ... LAV_s7126`, `LAV_s7201 ... LAV_s7214`
  (`epoch_0030.pt` = final Muon epoch of each member; 64 members)
- `weights/direct_vf60/w256_d3_s{4..9}.pt` (direct regressors)
- `weights/closure/{basis512.npz, closure64.pt}` (POD-512 basis and closure network)

To regenerate the submission from these weights without training: `bash prepare.sh && bash predict.sh` (members missing under `members/`
are exported from `weights/`; the regressor predictions and closure files are restored from `weights/` when `train.sh` has not run).
