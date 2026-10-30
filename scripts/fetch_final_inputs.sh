#!/bin/bash
# Download the Final-phase DIII-D private-test inputs (published 2026-10-19 as ganatma/fusion-equilibrium-challenge-final) into a flat directory of *.parquet shots
# for prepare_test.sh, and record the dataset revision + per-file sha256.   bash scripts/fetch_final_inputs.sh [out dir] [repo id] [config dir] [revision]
set -euo pipefail
OUT=${1:-final_inputs}; PY=${PYTHON:-python}
mkdir -p "$OUT"
$PY - "$OUT" "${2:-ganatma/fusion-equilibrium-challenge-final}" "${3:-data/diii_d_private_test}" "${4:-965040c12b814f33c8e7209f6b1799932247e793}" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
out = Path(sys.argv[1]); rid = sys.argv[2]; cdir = sys.argv[3]; rev = sys.argv[4]
info = HfApi().dataset_info(rid, revision=rev); print("revision", info.sha)
cfg = sorted({Path(s.rfilename).parts[1] for s in info.siblings if s.rfilename.startswith("data/") and len(Path(s.rfilename).parts) > 2})
print("configs", cfg)
snap = Path(snapshot_download(rid, repo_type="dataset", revision=info.sha, allow_patterns=[cdir + "/*", "README.md"],
                              local_dir=str(out / "hf")))
files = sorted((snap / cdir).glob("*.parquet")); print(len(files), "parquet shots")
flat = out / "parquet"; flat.mkdir(exist_ok=True)
man = []
for f in files:
    shutil.copy2(f, flat / f.name); man.append({"path": f"{cdir}/{f.name}", "size": f.stat().st_size,
                                                 "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
(out / "private_manifest.json").write_text(json.dumps({"repo": rid, "revision": info.sha, "files": man}, indent=1))
print("ready:", flat)
PY
