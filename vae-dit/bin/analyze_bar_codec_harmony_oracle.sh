#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/init_env.sh"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset-name>"
    echo "  dataset-name: model folder under output/\${MUSICAI_STAGE}/models/"
    exit 1
fi

NAME="$1"
MODEL_DIR="${OUTPUT_DIR}/models/${NAME}"
ENCODED_DIR="${MODEL_DIR}/encoded"

if [ ! -d "${ENCODED_DIR}" ]; then
    echo "Error: encoded artifacts for dataset '${NAME}' not found at ${ENCODED_DIR}"
    echo "Run 'bash bin/encode.sh ${NAME}' first."
    exit 1
fi

echo "Analyze BarCodec harmony for '${NAME}'..."
echo "  model:  ${MODEL_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/bin/analyze_bar_codec_harmony_oracle.py" \
    --model-dir "${MODEL_DIR}"
