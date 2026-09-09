#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/init_env.sh"
"${PYTHON_BIN}" "${ROOT_DIR}/bin/raw_pairing_census.py" "$@"
