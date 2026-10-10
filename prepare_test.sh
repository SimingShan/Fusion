#!/bin/bash
# Test-only preparation for NEW DIII-D test inputs (e.g. the final phase's private split), in a root where prepare.sh has already run.
# Nothing is retrained: training-derived artifacts (POD bases, normalisations, feature scales, train feature statistics) are reused.
#   FUSION_ROOT=/path/to/root PYTHON=/path/to/python bash prepare_test.sh <directory with the new test *.parquet shots> [N_SHARDS]
# The scripts address the test split by the dataset config name `diii_d_public_test`, so the new shots are installed under that name
# (the public test is kept once as data/raw/data/diii_d_public_test.orig) and their entries in data/raw/download_manifest.json are
# rewritten (path, size, sha256). Every test-derived view, feature and prediction is then rebuilt. Afterwards run inference with a FRESH
# members directory so no export from earlier test inputs is reused:
#   MEMBERS_DIR=members_<tag> WEIGHTS=weights bash predict.sh <tag>
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${PYTHON:-python}; SRC=$(realpath "${1:?directory with the new test parquet shots}"); N=${2:-4}
export CUDA_VISIBLE_DEVICES=""
log() { echo "=== $(date -u +%FT%TZ) $*"; }
shards() { local script=$1 view=$2 split=$3 out=$4 n=$5; for ((k = 0; k < n; k++)); do $PY -u "$script" "$view" "$split" "$out" "$k/$n" & done; wait; }
ls "$SRC"/*.parquet > /dev/null

log "1. install $(ls "$SRC"/*.parquet | wc -l) test shots and rewrite the raw manifest's test entries"
T=data/raw/data/diii_d_public_test
if [ -d "$T" ] && [ ! -e "$T.orig" ]; then mv "$T" "$T.orig"; fi
rm -rf "$T"; mkdir -p "$T"; cp "$SRC"/*.parquet "$T"/
$PY - <<'PY'
import hashlib, json
from pathlib import Path
m = Path("data/raw/download_manifest.json"); d = json.loads(m.read_text())
orig = Path("data/raw/download_manifest.orig.json")
if not orig.exists():
    orig.write_text(m.read_text())
keep = [f for f in d["files"] if not f["path"].startswith("data/diii_d_public_test/")]
new = [{"path": f"data/diii_d_public_test/{p.name}", "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
       for p in sorted(Path("data/raw/data/diii_d_public_test").glob("*.parquet"))]
d["files"] = keep + new; m.write_text(json.dumps(d, indent=1)); print(f"manifest: {len(keep)} train files kept, {len(new)} test files")
PY

log "2. remove every test-derived artifact"
rm -rf data/advanced_test_v1 data/advanced_history_test_v2 data/advanced_history_test_vessel_v1 data/advanced_test_vessel_v1 \
       data/advanced_history_test_vf_v1 results/analysis/vessel_196/feats/test results/analysis/future_201/feats/test \
       results/analysis/scalar_probe_160/pred_direct_vf60_test.npy results/analysis/scalar_probe_160/pred_direct_vf60g_test.npy

log "3. base and history test views (as prepare.sh step 3)"
$PY -u scripts/prepare_advanced_data.py --output data/advanced_test_v1 --workers 4 --splits test
$PY -u scripts/optimization/prepare_history_features.py --base-dir data/advanced_test_v1 --output data/advanced_history_test_v2 --splits test --no-pod --workers 1

log "4. vessel-current test features and views (as prepare.sh step 5)"
V=results/analysis/vessel_196
shards $V/build_features.py data/advanced_history_test_v2 test $V/feats "$N"
$PY -u $V/make_vessel_view.py test && $PY -u $V/make_av_view.py test

log "5. future-actuator test features and view (as prepare.sh step 6)"
F=results/analysis/future_201
shards $F/build_features.py data/advanced_history_test_vessel_v1 test $F/feats "$N"
$PY -u $F/make_fut_view.py test
log "done: run predict.sh with a fresh MEMBERS_DIR"
