#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/init_env.sh"

NAME="${1:-}"
OVERWRITE_ARGS=()

if [ -z "${NAME}" ]; then
    echo "Usage: $0 <dataset-name> [--overwrite]"
    echo "  dataset-name: folder under ../datasets/"
    exit 1
fi

if [ "$#" -gt 2 ]; then
    echo "Error: only the optional --overwrite flag is supported."
    exit 1
fi

if [ "$#" -eq 2 ]; then
    if [ "$2" != "--overwrite" ]; then
        echo "Error: unknown option '$2'. Expected --overwrite."
        exit 1
    fi
    OVERWRITE_ARGS=("--overwrite")
fi

DATA_DIR="${ROOT_DIR}/../datasets/${NAME}"
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: dataset '${NAME}' not found at ${DATA_DIR}"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d%H%M%S)"
DIAGNOSTICS_FILE="${DATA_DIR}/diagnostics_${TIMESTAMP}.json"
TOOL_DIR="${ROOT_DIR}/tools/form_metadata"

"${PYTHON_BIN}" "${TOOL_DIR}/form_json_generator.py" \
    --music-dir "${DATA_DIR}" \
    --diagnostics-output "${DIAGNOSTICS_FILE}" \
    --config "${TOOL_DIR}/config/style_defaults.yaml" \
    "${OVERWRITE_ARGS[@]}"

echo "Done. Metadata: ${DATA_DIR}/form.json"
echo "Done. Diagnostics: ${DIAGNOSTICS_FILE}"
