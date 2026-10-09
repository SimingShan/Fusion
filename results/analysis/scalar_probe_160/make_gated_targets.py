#!/usr/bin/env python3
"""Gate-aware Z-shift targets: copy a direct-regressor prediction file and set the rows of B-only GATED frames (Thomson-poor, threshold 0.1, the
blend's own gate) to NaN, so export_warp_z.py / shape_corr.py skip the axis shift there (the direct regressors are unreliable on those frames).
Reproduces results/analysis/scalar_probe_160/pred_direct_vf60g_{native,legacy,test}.npy (made inline on 2026-10-09).
usage: make_gated_targets.py <prefix e.g. direct_vf60> [folds: native,legacy,test]"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import scripts.optimization.finish_ensemble_candidate as fec  # noqa: E402
from fusion.pod import read_manifest  # noqa: E402
from scripts.optimization.export_gated_ensemble import input_arrays  # noqa: E402
from scripts.train import prediction_entries  # noqa: E402

P = ROOT / "results/analysis/scalar_probe_160"; pref = sys.argv[1]; folds = (sys.argv[2] if len(sys.argv) > 2 else "native,legacy,test").split(",")
for fold in folds:
    if fold == "test":
        A = ROOT / "data/advanced_test_v1"; cols = fec.feature_columns(read_manifest(A)["feature_names"])
        gated = np.concatenate([fec.gate_frames(input_arrays(A, e)[0], cols, 0.1) for e in prediction_entries(A, "test")])
    else:
        g, split = (ROOT / "data/advanced_val_native_v3", "val") if fold == "native" else (ROOT / "data/advanced_v1", "legacy_val")
        man = read_manifest(g); cols = fec.feature_columns(man["feature_names"]); gated = []
        for e in man[split]:
            with np.load(g / e["path"], allow_pickle=False) as inp:
                gated.append(fec.gate_frames(inp["X"], cols, 0.1))
        gated = np.concatenate(gated)
    r = np.load(P / f"pred_{pref}_{fold}.npy").copy(); assert len(r) == len(gated), (fold, len(r), len(gated))
    r[gated] = np.nan; np.save(P / f"pred_{pref}g_{fold}.npy", r)
    print(f"{fold}: {int(gated.sum())} gated of {len(r)} -> pred_{pref}g_{fold}.npy", flush=True)
