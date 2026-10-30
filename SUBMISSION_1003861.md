# Final submission 1003861 (Challenge 1, DIII-D)

Codabench submission **1003861** ("Submission 20261019-203", scored 2026-10-19) was produced by this repository with the trained
parameters in **`weights_rebuild/`**. It contains only `diii_d_private_test.npz` (879 shots, 196,301 frames); the entry is ranked on
DIII-D only and makes no MAST prediction.

| item | where |
|---|---|
| model source | `fusion/`, `scripts/`, `train_members.py`, `results/training/*/src`, `results/analysis/*` (scripts only), `starter/` (scorer functionals and `data_fixes.py`) |
| pinned environment | `env/environment.yml` (Python 3.11.14, torch 2.5.1+cu124), full freeze `env/pip_freeze_full.txt` |
| trained weights of this submission | `weights_rebuild/` (checksums `weights_rebuild/SHA256SUMS`): 24 HLVF + 40 HLAV members, 6 direct regressors, POD-512 basis + closure network |
| training entrypoint (from scratch) | `bash prepare.sh && bash train.sh` |
| inference entrypoint (Final inputs) | below |
| data provenance | `PROVENANCE.md` (per machine), `provenance/download_manifest.json` (training data), `provenance/final_inputs_manifest.json` (Final inputs) |
| compute declaration | below |

## Regenerate the submitted predictions from the shipped weights

```bash
conda env create -f env/environment.yml && conda activate fusion-c1-final
export FUSION_ROOT=$PWD PYTHON=$(which python)
bash prepare.sh                                   # CPU: training data (pinned revision) -> views, features, normalisation
bash scripts/fetch_final_inputs.sh final_inputs   # ganatma/fusion-equilibrium-challenge-final @ 965040c1..., 879 DIII-D shots
bash prepare_test.sh final_inputs/parquet 8       # CPU: install the Final shots as the test split, rebuild every test view/feature
SUBMIT_NAME=diii_d_private_test MEMBERS_DIR=members_rebuild WEIGHTS=weights_rebuild PARALLEL=8 bash predict.sh final_rebuild
# -> results/candidates/final_rebuild/submission.zip  (member diii_d_private_test.npz)
```

Run this in a clone where `train.sh` has not been run (or delete `results/candidates/20261004-080/closure` and
`results/analysis/scalar_probe_160/pred_direct_vf60*_test.npy` first): `predict.sh` reuses closure files and regressor predictions that
already exist, and otherwise restores them from `WEIGHTS`. No GPU is needed for inference.

What we ran for 1003861: a fresh clone at this commit's parent `fd63d2b` (the same code and weights; this commit adds only
documentation and the input-fetch script), the prepared training data of our from-scratch package run, `prepare_test.sh` on the 879
Final shots, then the `predict.sh` line above with `PARALLEL=8 OMP_NUM_THREADS=6 KAPPA_WORKERS=48 LI_WORKERS=48`, 1 h 36 min on
64 CPU cores. Output `submission.zip` SHA-256 `a45366f935c560380a468dd8ea00d3d31578f1eb3ef9699c6a66455ba0a70456`.
Inference from given weights is deterministic on the same CPU type and thread count; another thread count changes float32 rounding
in the closure network (flux differences ~1e-4 on a few % of frames, ~1e-13 of the field variance), far below 1e-4 in S.

## Retrain from scratch

`bash prepare.sh && bash train.sh` retrains all 64 members (default seeds HLVF 6301-6324, HLAV 7401-7440), the six direct regressors
and the closure, writing `members/`, `results/analysis/scalar_probe_160/direct_vf60/` and `results/candidates/20261004-080/closure/`;
then run `prepare_test.sh` and `predict.sh` as above without `WEIGHTS`/`MEMBERS_DIR` (`predict.sh` uses the locally trained members).
`weights_rebuild/` itself is the output of exactly this procedure on one NVIDIA A100 80GB PCIe. GPU training is not bit-reproducible
across runs (summation order), so a retrain is a fresh draw of the same distribution: on the public test, the two complete draws we
have (`weights/` and `weights_rebuild/`) scored 0.996446 and 0.996456, and earlier independent draws of the same structures agreed to
within 0.00001 S.

## Training compute declaration

- **GPU (budgeted):** 17.4 GPU-hours on one NVIDIA A100 80GB PCIe, measured by `train.sh` (`gpu_timing.json`): 24 HLVF members
  5.79 h, 40 HLAV members 10.61 h (members trained four at a time on the one GPU), direct regressors 0.27 h, closure 0.76 h.
  DIII-D only; no GPU work for MAST. Inference uses no GPU.
- **CPU (not budgeted, declared):** `prepare.sh` about 4-6 h on 8-16 cores plus about 1.5 h for the li stores on a large-memory
  node; `prepare_test.sh` 7 min and `predict.sh` 1.5 h on 64 cores for the Final inputs. No synthetic data generation.
- No pre-trained models, no external datasets, no test labels.
