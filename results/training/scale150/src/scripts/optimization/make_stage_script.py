#!/usr/bin/env python3
"""Write a staging script for finish_ensemble_candidate.py from a gather report blend tag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gather", type=Path, required=True, help="gather output dir with members.json and report.json")
    parser.add_argument("--blend", required=True, help="blend tag in report.json (e.g. eq_...)")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.gather / "members.json").read_text())
    report = json.loads((args.gather / "report.json").read_text())
    names = report["blends"][args.blend]["members"]
    mem, dev, leg = [], [], []
    for n in names:
        pred = ROOT / meta["members"][n]
        run = pred.parents[1]
        ck, legacy = run / "best.pt", run / "legacy" / "legacy_val_predictions.npz"
        for p in (ck, pred, legacy):
            if not p.exists():
                raise SystemExit(f"missing {p}")
        mem.append(f"{meta['types'][n]}={ck.relative_to(ROOT)}")
        dev.append(str(pred.relative_to(ROOT)))
        leg.append(str(legacy.relative_to(ROOT)))
    script = f"""#!/bin/bash
# Stage candidate {args.candidate}: equal-weight {len(names)}-member gated blend ({args.blend}).
set -euo pipefail
cd /gpfs/radev/scratch/lu_lu/ss5235/Fusion
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
.venv/bin/python -u scripts/optimization/finish_ensemble_candidate.py --candidate {args.candidate} \\
  --members {' '.join(mem)} \\
  --dev-predictions {' '.join(dev)} \\
  --legacy-predictions {' '.join(leg)} \\
  --gate-threshold 0.1 --workers "${{SLURM_CPUS_PER_TASK:-4}}"
echo "=== candidate_evidence ==="
.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.periodic_submission import candidate_evidence
try:
    print(candidate_evidence(Path("results/candidates/{args.candidate}")))
except Exception as exc:
    print("EVIDENCE_ERROR", type(exc).__name__, exc)
PY
"""
    args.output.write_text(script)
    args.output.chmod(0o755)
    print(f"wrote {args.output} with {len(names)} members; expected native S {report['blends'][args.blend]['score']['S']}")


if __name__ == "__main__":
    main()
