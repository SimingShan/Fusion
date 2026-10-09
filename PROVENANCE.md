# Data provenance and build lineage (DIII-D)

**Data provenance statement (DIII-D).** All training data are the labelled DIII-D shots of the public Fusion Equilibrium Challenge
release (Hugging Face `Sophelio/fusion-equilibrium-challenge`, revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`, per-file sha256 in
`provenance/download_manifest.json`): the training split plus the public development splits (native validation and legacy validation);
the audit split is not used. Splits respect shot boundaries. Test-set labels are never read; test inputs are used only for prediction.
The starter-kit data fixes are applied. Features are computed from inputs only (coil and plasma currents including their raw 0.05 ms
time series within each shot, Thomson profiles, geometry, time metadata); `magnetics_dsep` is not used. No external data or pretrained
models. **MAST:** this package contains no MAST model; no MAST labels or MAST training data were used anywhere.

CPU-only stages (declared, not GPU-budgeted): download, view building, POD bases, li/geometry stores, vessel/future feature extraction,
scalar targets, blending and packaging. GPU stages: member training, direct axis regressors, closure network (see `train.sh`).

---

The lineage below records how the submitted artifacts were originally produced (Slurm job ids, exact commands); `prepare.sh` and
`train.sh` replay the same commands.

Compiled 2026-10-09 from Slurm accounting (`sacct -o SubmitLine`), manifests, run.json files and RESUME.md. Square brackets are Slurm job ids. Paths are relative to the repo root (`/gpfs/radev/scratch/lu_lu/ss5235/Fusion`). Audit splits are not used by this chain.

## A. Raw data and base views (CPU)
1. `data/raw` <- `scripts/download_data.py --workers 12` [2461748]; dataset `Sophelio/fusion-equilibrium-challenge` revision `1e280905...` (pin it: the script only pins when `data/raw/download_manifest.json` exists).
2. `data/processed` <- `scripts/prepare_data.py --train-only` [2461750] (legacy split; `advanced_v1` checks its manifest hash `cb9edb12...`).
3. `data/advanced_v1`, `data/pod_full_v1` <- `scripts/optimization/prepare_full.sh` [2462423]: `prepare_advanced_data.py --output data/advanced_v1 --workers 4 --train-frames 64 --val-frames 32`; `prepare_pod.py --data-dir data/advanced_v1 --output data/pod_full_v1 --rank 128 --basis-samples 16384`.
4. `data/pod_scaled_v2` <- `scripts/optimization/prepare_scaled_features.py` (defaults) [2462452].
5. `data/advanced_history_v2`, `data/pod_history_v2` <- `prepare_history_features.py --base-dir data/advanced_v1 --output data/advanced_history_v2 --source-pod data/pod_scaled_v2 --pod-output data/pod_history_v2 --workers 8` [2462508] (reads raw magnetics).
6. `data/pod_history128_robust_v4` <- `prepare_robust_features.py --data-dir data/advanced_history_v2 --source-pod data/pod_history_v2 --output data/pod_history128_robust_v4` (first line of `train_history128_robust_gpu.sh` [2462775]).
7. Test views: `data/advanced_test_v1` <- `prepare_advanced_data.py --output data/advanced_test_v1 --workers 4 --splits test` [2462440]; `data/advanced_history_test_v2` <- `prepare_history_features.py --base-dir data/advanced_test_v1 --output data/advanced_history_test_v2 --splits test --no-pod --workers 1` [2462535].
8. `data/advanced_val_native_v3`, `data/advanced_history_val_native_v3` <- first two commands of `evaluate_native_validation.sh` [2462722] (dev only).
9. `data/li_full_all_v2` -> `data/geometry_history_all_v5` <- `prepare_li_all_072.sh` [2533590, bigmem, 1.5 h]; generator snapshot `results/training/c1_sources_070/scripts/optimization/prepare_{li,geometry}_targets.py`.

## B. All-labelled views and li store (CPU)
10. `advanced_history_alllab_v1`, `pod_history128_alllab_v1`, `geometry_history_alllab_v1` <- `scripts/optimization/prepare_alllab_view.py` (no args; [2718564] crashed in build_geometry, geometry-only resume [2718579] after a code fix).

## C. Vessel / future features and views (CPU)
11. `results/analysis/vessel_196/feats{,_v2view}` <- `vessel_196/build_features.py <view> <split> <out> [k/n]` on v2 train (8 shards) [2735303-10], v2 val -> feats_v2view [2735369], native-v3 val + v2 legacy_val [2735311], `advanced_history_test_v2` test (4 shards) [2735313-16].
12. `make_vessel_view.py train && make_vessel_view.py native` [2735403] -> `vessel_scale.json`, `advanced_history_vessel_v1`, `pod_history128_vessel_v1`, `advanced_history_val_native_vessel_v1`; `make_vessel_view.py test` [2735404] -> `advanced_history_test_vessel_v1`.
13. `advanced_history_alllab_vessel_v1` + pod <- `make_lv_view.py` [2736303].
14. `make_av_view.py train && native && test` [2739044] -> `advanced_vessel_v1`, `pod_scaled_vessel_v1`, `advanced_val_native_vessel_v1`, `advanced_test_vessel_v1`.
15. `advanced_alllab_vessel_v1`, `pod_scaled_alllab_vessel_v1` <- `make_lav_view.py` [2743581].
16. `results/analysis/future_201/feats{,_v2view}` <- `future_201/build_features.py` [2744116/17/18/22]; test [2744119] built from `advanced_history_test_vessel_v1`.
17. `make_fut_view.py train` [2744187] -> `future_scale.json`, `advanced_history_vf_v1`, `pod_history128_vf_v1`; `make_fut_view.py test` [2744189] -> `advanced_history_test_vf_v1`; `native` -> `advanced_history_val_native_vf_v1`.
18. `advanced_history_alllab_vf_v1`, `pod_history128_alllab_vf_v1` <- `make_lvf_view.py` [2744317].

## D. Members (GPU)
19. LVF x17 (seeds 6001-6017): `member.py LVF_s$i <seed> 256 3 600 60 alllabvf; MEMBER_TRAIN=results/training/scale150/src/scripts/train_pod.py member.py <same> && TEST_DIR=data/advanced_history_test_vf_v1 muon_only_test.py results/training/scale150/LVF_s$i muonLVF_s${i}_r30 30`. Cold/refine on `results/training/muon128/src/scripts/train_pod.py` (cold args from `ens070/B_s3101_cold/run.json`, refine args from `run.arguments` of `20260923-074/checkpoints/m00.pt`, init cold epoch 600); the ratio-9.81 Muon stage (args `muon128/treatment_m00/command.json`) fails on the muon128 trainer and is rerun on `scale150/src`; it is UNUSED except for its command.json, which `muon_only_test.py` replays at ratio 30 from `refine/epoch_0060.pt`. A100-PCIe means: cold 15.5, refine 15.2, Muon-9.81 8.6, Muon-30 8.6 min (H100 ~25% faster).
20. LAV x26 (seeds 7101-7126): `lav_member.py LAV_s$i <seed> 200`, all stages on `scale150/src`; cold args `ens070/A_s4101_cold/run.json`, refine + Muon args from `074/m17.pt` (30 + 30 epochs, ratio 200); test export on `advanced_test_vessel_v1`. A100 means 16.8 / 7.0 / 8.3 min.
    Muon code: `scale150/src/fusion/muon_optimizer.py` + vendored `fusion/vendor/muon_upstream.py` (KellerJordan/Muon commit f98f1cac).

## E. Direct regressors
21. `scalar_probe_160/{train,native,legacy}.npz` <- `prep_targets.py` [2721792, CPU 5.5 min].
22. `pred_direct_vf60_*.npy`, `direct_vf60/w256_d3_s{4..9}.pt` <- `TAGN=direct_vf60 SIZES=256x3 train_direct_vf.py 4,5,6,7,8,9 60` [2744872, GPU A40 5.5 min].
23. `pred_direct_vf60g_*.npy`: NO SCRIPT (inline snippet 2026-10-09: direct_vf60 with NaN rows on gated frames; 1,337 test, 61 legacy) -> must be scripted.

## F. Closure (GPU)
24. `ae_ceiling_119/basis512.npz` <- `ae_ceiling.py results/analysis/ae_ceiling_119 8,16,32,64 120` [2677760, A40 30 min] (rank-512 POD of 32,768 train frames, written at the end of the AE study).
25. `closure64.pt` <- `closure_ceiling.py 16,32,64 200` [2678123, A40 16.7 min] (RNG stream depends on d=16, 32 running first).
26. Both copied by hand to `results/candidates/20261004-080/closure/` (sha256 match).

## G. Inference and package
27. `blend_from_members.py` -> `diii_d_public_test_closed.npz` (gate on `data/advanced_test_v1`, threshold 0.1, 1,337 gated frames; closure).
28. `WARP_SRC=... BAR=0 BAZ=0.5 REG=pred_direct_vf60g_test.npy export_warp_z.py 20261009-106 0.5 1` (copies MAST payload from candidate 085).
29. `scripts/package_dual_submission.py` -> `submission.zip`.

## Gaps to close for a reproduction package
1. Script the gated-NaN regressor file (step 23).
2. Closure files hand-copied; basis512 needs the AE run; closure RNG depends on run order.
3. Unused compute cannot be skipped by the current scripts (LVF 9.81 Muon stage ~2 GPU-h for 17 members; dev-fold exports).
4. Two trainer snapshots (`muon128/src`, `scale150/src`), neither equal to root `scripts/train_pod.py`; run hashes cover only part of the sources; Muon reads env vars (defaults used).
5. Recipe arguments are read from old artifacts (run.json, checkpoints' `run.arguments`, command.json) -> freeze them into explicit config files. `--dynamic-kappa-targets` passed but unused (weight 0).
6. Absolute paths hard-coded in scripts, symlinks, manifests and basis.json (trainer checks manifest hashes) -> a fresh checkout must rebuild views with relative/rooted paths.
7. `.venv` -> conda env `fwi` (Python 3.11.14) outside the repo; dataset revision pinning conditional; MAST payload from 085 not traced.
8. Metadata errors (direct_vf60 manifest/export.json say 12 MLPs; actual 6 x 256x3).
9. `blend_from_members` imports `requests` via `finish_ensemble_candidate` -> `codabench_submit`.
10. Mixed GPUs -> not bit-reproducible (allowed; document).
11. Budget: 17 x 47.9 + 26 x 32.1 min = 27.5 A100-h with the unused stage, ~25 h without, + ~0.9 h regressors/closure -> over 24 h; trim the member count.

## Packages (env `fwi`)
torch 2.5.1+cu124, numpy 1.26.4, scipy 1.17.0, pyarrow 25.0.1, pandas 2.2.2 (lazy), scikit-image 0.26.0 (official scorer image pins 0.25.2), requests 2.32.5, huggingface_hub (download only). Local modules: `fusion`, `scripts`, starter (`local_score`, `derive`, `lcfs`, `contour`, `common`, `metrics`, `o_point`, `data_fixes`).
