#!/usr/bin/env python3
"""Train ensemble members FROM SCRATCH (cold -> refine -> Muon) with the frozen recipes in configs/<FAMILY>.json, several members at
once on one GPU, then export each member's DIII-D public-test predictions.

Each stage runs the trainer results/training/scale150/src/scripts/train_pod.py with the frozen argv template (placeholders {seed},
{out}, {stage}); the final Muon epoch is the member's endpoint (all-labelled members have no held-out selection).
  LVF : B recipe + vessel + future-actuator inputs (view data/advanced_history_alllab_vf_v1), Muon lr ratio 30;  test view data/advanced_history_test_vf_v1
  LAV : A recipe + vessel inputs (view data/advanced_alllab_vessel_v1), Muon lr ratio 200;                        test view data/advanced_test_vessel_v1
  HLVF: the LVF recipe with seven extra consistency-scalar heads (configs/HLVF.json), heads exported to test/diii_d_public_test_heads.npz
  HLAV: the LAV recipe with the same seven heads (configs/HLAV.json)
usage: train_members.py <FAMILY> <first_seed> <n_members> [--seeds 6001:17,6101:7] [--parallel 4] [--out members]
       (--seeds lists start:count ranges and replaces first_seed/n_members; all listed members share one pool of --parallel slots)
Writes <out>/<FAMILY>_s<seed>/{cold,refine,muon}/..., test/diii_d_public_test.npz, complete.json (per-stage wall seconds)."""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("FUSION_ROOT") or Path(__file__).resolve().parent)
TRAIN = ROOT / "results/training/scale150/src/scripts/train_pod.py"
TEST_VIEW = {"LVF": "data/advanced_history_test_vf_v1", "HLVF": "data/advanced_history_test_vf_v1", "LAV": "data/advanced_test_vessel_v1", "HLAV": "data/advanced_test_vessel_v1"}


def stage_cmd(template, seed, out, stage):
    return [x.replace("{seed}", str(seed)).replace("{out}", str(out)).replace("{stage}", stage) for x in template]


def member(family, seed, out_root, cfg):
    out = out_root / f"{family}_s{seed}"; out.mkdir(parents=True, exist_ok=True); times = {}
    if (out / "complete.json").exists():
        return json.loads((out / "complete.json").read_text())
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    for stage in ("cold", "refine", "muon"):
        d = out / stage
        if (d / "stage_done").exists():
            times[stage] = float((d / "stage_done").read_text()); continue
        d.mkdir(exist_ok=True); cmd = [sys.executable, "-u", str(TRAIN), *stage_cmd(cfg["stages"][stage], seed, out, stage)]
        (d / "command.json").write_text(json.dumps(cmd, indent=1)); t0 = time.monotonic()
        with open(d / "train.log", "w") as log:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        times[stage] = time.monotonic() - t0; (d / "stage_done").write_text(f"{times[stage]:.0f}\n")
    final = out / "muon/checkpoints/epoch_0030.pt"; td = out / "test"; td.mkdir(exist_ok=True)
    if not (td / "diii_d_public_test.npz").exists():
        (td / "best.pt").write_bytes(final.read_bytes()); t0 = time.monotonic()
        with open(td / "predict.log", "w") as log:
            subprocess.run([sys.executable, str(TRAIN), "--data-dir", TEST_VIEW[family], "--output", str(td), "--predict-only", "--split", "test",
                            "--workers", "2"], check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        times["test_export"] = time.monotonic() - t0
    rec = {"family": family, "seed": seed, "endpoint": str(final), "stage_seconds": times}
    (out / "complete.json").write_text(json.dumps(rec, indent=1)); print(f"{family}_s{seed} done {times}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("family", choices=sorted(TEST_VIEW)); ap.add_argument("first_seed", type=int); ap.add_argument("n", type=int)
    ap.add_argument("--parallel", type=int, default=4); ap.add_argument("--out", default="members")
    ap.add_argument("--seeds", default="", help="start:count[,start:count...] ranges; replaces first_seed/n_members"); a = ap.parse_args()
    seeds = [s0 + i for r in a.seeds.split(",") if r for s0, n in [map(int, r.split(":"))] for i in range(n)] if a.seeds else list(range(a.first_seed, a.first_seed + a.n))
    cfg = json.loads((ROOT / f"configs/{a.family}.json").read_text()); out_root = (ROOT / a.out).resolve(); t0 = time.monotonic()
    with ThreadPoolExecutor(a.parallel) as pool:
        recs = list(pool.map(lambda s: member(a.family, s, out_root, cfg), seeds))
    print(json.dumps({"family": a.family, "members": len(recs), "wall_hours": (time.monotonic() - t0) / 3600}), flush=True)
