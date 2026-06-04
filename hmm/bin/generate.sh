#!/bin/bash
set -euo pipefail

NAME="${1:-test}"
TARGET_MEASURES="${2:-50}"
SEED="${3:-42}"

if [ -z "$NAME" ]; then
    echo "Usage: $0 <model-name> [target-measures] [seed]"
    echo "  model-name:      folder under ./models/ (or ./datasets/ for training)"
    echo "  target-measures: number of measures (default: 50)"
    echo "  seed:            random seed (default: 42)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/${NAME}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model '${NAME}' not found at ${MODEL_DIR}"
    if [ -d "${ROOT_DIR}/datasets/${NAME}" ]; then
        echo "Training dataset found — training model first..."
        "${SCRIPT_DIR}/train_model.sh" "$NAME"
    else
        echo "Error: no dataset at ${ROOT_DIR}/datasets/${NAME} to train from"
        exit 1
    fi
fi

TIMESTAMP="$(date +%Y%m%d%H%M)"

OUTPUT_DIR="${ROOT_DIR}/generated/${NAME}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

OUTPUT_JSON="${OUTPUT_DIR}/${NAME}_${TIMESTAMP}.json"
OUTPUT_FILE="${OUTPUT_DIR}/${NAME}_${TIMESTAMP}.mid"

echo "Generating ${TARGET_MEASURES} measures (seed=${SEED})..."
echo "  model:  ${MODEL_DIR}"
echo "  output: ${OUTPUT_FILE}"

cd "${ROOT_DIR}/src"
python3 ./generate.py \
    --model-dir "$MODEL_DIR" \
    --output-json "$OUTPUT_JSON" \
    --output-midi "$OUTPUT_FILE" \
    --measures "$TARGET_MEASURES" \
    --seed "$SEED"

