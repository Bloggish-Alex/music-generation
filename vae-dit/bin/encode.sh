#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/init_env.sh"

NAME=""; DATA_DIR=""; RUN_ID=""; CONFIG_PATH="${ROOT_DIR}/config/codec_v2.yaml"; SCHEMA="bar_tensor_schema.v2"
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset-root) DATA_DIR="$2"; shift 2;; --dataset-name) NAME="$2"; shift 2;; --run-id) RUN_ID="$2"; shift 2;; --config) CONFIG_PATH="$2"; shift 2;; --schema-version) SCHEMA="$2"; shift 2;;
    *) [ -z "$NAME" ] && NAME="$1" && shift || { echo "Unknown option: $1" >&2; exit 2; };;
  esac
done
[ -n "$NAME" ] || { echo "--dataset-name is required" >&2; exit 2; }
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/../datasets/${NAME}}"
SCHEMA="${SCHEMA:-bar_tensor_schema.v2}"
[ "$SCHEMA" = "bar_tensor_schema.v2" ] || { echo "Only bar_tensor_schema.v2 is supported." >&2; exit 2; }
[ -n "$RUN_ID" ] || RUN_ID="${NAME}-${SCHEMA##*.}-r001"
ENCODE_DIR="${OUTPUT_DIR}/models/${NAME}/encoded/${RUN_ID}"

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
    --config "${CONFIG_PATH}" --schema-version "${SCHEMA}"

echo "Done. codec saved to ${ENCODE_DIR}"
