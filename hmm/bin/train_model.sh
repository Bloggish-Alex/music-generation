#!/bin/bash
set -euo pipefail

NAME="${1:-}"
HMM_MODE="${2:-numpy}" # "numpy", "hmmlearn"

if [ -z "$NAME" ]; then
    echo "Usage: $0 <dataset-name> <hmm-mode>"
    echo "  dataset-name: folder under ./datasets/"
    echo "  hmm-mode: backend HMM impl. numpy or hmmlearn, default is numpy"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${ROOT_DIR}/../datasets/${NAME}"
MODEL_DIR="${ROOT_DIR}/models/${NAME}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: dataset '${NAME}' not found at ${DATA_DIR}"
    exit 1
fi

echo "Training model for '${NAME}'..."
echo "  data:   ${DATA_DIR}"
echo "  output: ${MODEL_DIR}"

cd "${ROOT_DIR}/src"
python3 ./train.py \
    --music-dir "$DATA_DIR" \
    --model-dir "$MODEL_DIR" \
    --n-bar-clusters 8 \
    --n-sections 4 \
    --verbose \
    --hmm-backend "${HMM_MODE}"

echo "Done. Model saved to ${MODEL_DIR}"

