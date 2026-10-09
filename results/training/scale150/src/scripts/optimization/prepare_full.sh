#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/prepare_advanced_data.py --output data/advanced_v1 --workers 4 --train-frames 64 --val-frames 32
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/prepare_pod.py --data-dir data/advanced_v1 --output data/pod_full_v1 --rank 128 --basis-samples 16384
