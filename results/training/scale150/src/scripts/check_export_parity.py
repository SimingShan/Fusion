#!/usr/bin/env python3
"""Verify frozen inference against the selected held-out predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimization.export_gated_ensemble import load_expert, input_arrays, predict
from scripts.optimization.quality_gate import feature_columns, gate_from_inputs
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    a = p.parse_args()
    d = json.loads((a.candidate / "decision.json").read_text())
    torch.set_num_threads(2)
    device = torch.device("cpu")
    records = []
    for key, split, directory_a, directory_b in (
            ("native", "val", "advanced_val_native_v3", "advanced_history_val_native_v3"),
            ("legacy", "legacy_val", "advanced_v1", "advanced_history_v2")):
        primary_data, secondary_data = ROOT / "data" / directory_a, ROOT / "data" / directory_b
        primary = load_expert(ROOT / d["checkpoints"][0]["frozen_file"], primary_data, device)
        secondary = load_expert(ROOT / d["checkpoints"][1]["frozen_file"], secondary_data, device)
        manifests = [json.loads((p / "manifest.json").read_text()) for p in (primary_data, secondary_data)]
        selected = sorted(set(np.linspace(0, len(manifests[0][split]) - 1, 8).astype(int)))
        if split == "legacy_val":
            selected = range(len(manifests[0][split]))
        source = (ROOT / d["validation"][key]["path"]).with_name(f"{split}_predictions.npz")
        with np.load(source, allow_pickle=False) as expected:
            for i in selected:
                raw = [input_arrays(p, m[split][i]) for p, m in zip((primary_data, secondary_data), manifests)]
                if not np.array_equal(raw[0][1], raw[1][1]) or not np.array_equal(raw[0][2], raw[1][2]):
                    raise ValueError("Inference query identities differ")
                gate = gate_from_inputs(raw[0][0], feature_columns(manifests[0]["feature_names"]))[0]
                A, B = [predict(e, x[0], 512, device) for e, x in zip((primary, secondary), raw)]
                w = d["primary_weight"]
                flux = np.where(gate[:, None, None], B[0], w*A[0] + (1-w)*B[0])
                scalar = np.where(gate[:, None], B[1], w*A[1] + (1-w)*B[1])
                errors = {}
                for field, value in (("psirz", flux), ("q95", scalar[:, 0]), ("betaN", scalar[:, 1])):
                    ref = expected[f"shot_{i:04d}_{field}"]
                    # Different BLAS kernels accumulate scalar outputs a few
                    # float32 ulps apart. Keep the stricter flux tolerance.
                    atol = 2e-6 if field == "psirz" else 1e-5
                    np.testing.assert_allclose(value, ref, rtol=1e-5, atol=atol)
                    errors[field] = float(np.max(np.abs(value-ref)))
                records.append({"split": split, "shot": int(i), "frames": len(gate), "gated_frames": int(gate.sum()), "max_absolute_error": errors})
    write_json(a.candidate / "export_parity.json", {"passed": True, "shots": len(records),
        "frames": sum(x["frames"] for x in records), "gated_frames": sum(x["gated_frames"] for x in records),
        "records": records, "audit_labels_read": False, "test_labels_read": False})
    print(json.dumps({"passed": True, "shots": len(records), "gated_frames": sum(x["gated_frames"] for x in records)}))


if __name__ == "__main__":
    main()
