#!/bin/bash

export MUSICAI_STAGE="stage3"

export SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export OUTPUT_DIR="${ROOT_DIR}/output/${MUSICAI_STAGE}"
export CONFIG="${ROOT_DIR}/config/style_defaults.yaml"
export DEVICE="cuda"

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python >/dev/null 2>&1; then export PYTHON_BIN="python"; else export PYTHON_BIN="python3"; fi
fi

mkdir -p "$OUTPUT_DIR"
