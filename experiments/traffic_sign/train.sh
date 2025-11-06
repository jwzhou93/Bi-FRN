#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/path/to/dataset_root}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../../outputs/traffic_sign}"

python "${SCRIPT_DIR}/train_binary.py" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 60 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --lr-step 20 \
    --lr-gamma 0.2 \
    --train-transform-type 0 \
    --test-transform-type 0 \
    "$@"
