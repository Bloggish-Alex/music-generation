#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/init_env.sh"

NAME="${1:-}"

if [ -z "$NAME" ]; then
    echo "Usage: $0 <dataset-name>"
    echo "  dataset-name: folder under ./datasets/"
    exit 1
fi

DATA_DIR="${ROOT_DIR}/../datasets/${NAME}"
ENCODE_DIR="${OUTPUT_DIR}/models/${NAME}/encoded"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: dataset '${NAME}' not found at ${DATA_DIR}"
    exit 1
fi

echo "Encode musics for '${NAME}'..."
echo "  data:   ${DATA_DIR}"
echo "  output: ${ENCODE_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/bin/encode.py" \
    --music-dir "${DATA_DIR}" \
    --output-dir "${ENCODE_DIR}" \
    --config "${CONFIG}"

echo "Done. codec saved to ${ENCODE_DIR}"
