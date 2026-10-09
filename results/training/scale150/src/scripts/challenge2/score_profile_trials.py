#!/usr/bin/env python3
"""Compare bounded reconstruction trials on their identical DIII-D queries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/challenge2/development_comparison_001"))
    p.add_argument("--trials", type=int, nargs="+", default=[1, 2])
    a = p.parse_args()
    if not a.trials or min(a.trials) < 1 or len(set(a.trials)) != len(a.trials):
        p.error("Use distinct positive trial identifiers")
    data = a.output / "data"
    data.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/advanced_val_native_v3"
    manifest = json.loads((source / "manifest.json").read_text())
    trials = [ROOT / f"results/challenge2/c2_evaluation_{i:03d}" for i in a.trials]
    records = [[x for x in json.loads((t / "summary.json").read_text())["records"] if x["machine"] == "DIII-D"] for t in trials]
    identities = [[(x["shot"], x["frame"]) for x in group] for group in records]
    if any(selection != identities[0] for selection in identities[1:]):
        raise ValueError("Trial selections differ")
    shot_names = list(dict.fromkeys(x["shot"] for x in records[0]))
    subset = []
    index_map = {}
    for i, name in enumerate(shot_names):
        original_index, entry = next((j, x) for j, x in enumerate(manifest["val"]) if Path(x["shard"]).stem == name)
        frames = np.array([x["frame"] for x in records[0] if x["shot"] == name])
        filename = f"shot_{i:04d}.npz"
        with np.load(source / entry["path"], allow_pickle=False) as z:
            if not np.array_equal(z["frame_indices"][frames], frames):
                raise ValueError("Source cache is not in complete native order")
            arrays = {key: z[key][frames] for key in ("psi", "scalars", "times", "frame_indices")}
            arrays.update(R=z["R"], Z=z["Z"])
        np.savez_compressed(data / filename, **arrays)
        chosen = copy.deepcopy(entry)
        chosen.update(path=filename, n_frames=len(frames))
        subset.append(chosen)
        index_map[name] = (i, original_index, frames)
    local_manifest = {"train": [], "val": subset, "legacy_val": [], "audit": [], "test": [],
        "feature_names": [], "purpose": "Unchanged held-out native targets for a bounded diagnostic only"}
    write_json(data / "manifest.json", local_manifest)

    def write_array(archive, name, values):
        with archive.open(name+".npy", "w") as output:
            np.lib.format.write_array(output, np.asarray(values, np.float32), allow_pickle=False)

    with np.load(ROOT / "results/optimization/native_history128_li_tuned_v9/gated_primary50/val_predictions.npz") as baseline:
        with zipfile.ZipFile(a.output / "baseline.npz", "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, (i, j, frames) in index_map.items():
                for field in ("psirz", "q95", "betaN"):
                    write_array(z, f"shot_{i:04d}_{field}", baseline[f"shot_{j:04d}_{field}"][frames])
    for number, directory, group in zip(a.trials, trials, records):
        with zipfile.ZipFile(a.output / f"trial_{number}.npz", "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, (i, _, frames) in index_map.items():
                flux, q95, beta = [], [], []
                for frame in frames:
                    row = next(x for x in group if x["shot"] == name and x["frame"] == frame)
                    with np.load(directory / f"DIII-D_{name}_{frame:04d}.npz") as values:
                        flux.append(-values["psi"])
                    q95.append(row.get("q95", np.nan))
                    beta.append(row.get("betaN", np.nan))
                for field, values in (("psirz", flux), ("q95", q95), ("betaN", beta)):
                    write_array(z, f"shot_{i:04d}_{field}", values)
    scores = {}
    for name in ["baseline", *[f"trial_{i}" for i in a.trials]]:
        output = a.output / f"{name}_score.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(data),
            "--pred", str(a.output / f"{name}.npz"), "--output", str(output), "--workers", "2"], check=True)
        score = json.loads(output.read_text())
        scores[name] = {k: score[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency", "lcfs_fail_frac")}
    (a.output / "score_profile_trials.py").write_bytes(Path(__file__).read_bytes())
    write_json(a.output / "comparison.json", {"scores": scores, "shots": len(subset), "frames": len(records[0]),
        "trials": a.trials, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "same_queries": True, "audit_labels_read": False, "mast_labels_read": False,
        "role": "Bounded development diagnostic, distinct from full-native scores and official leaderboard",
        "failed_scalars": "Retained as invalid predictions and penalized by the unchanged official scorer; no target imputation in the predictor"})
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
