#!/usr/bin/env bash
set -euo pipefail

module="${1:?module name is required}"
shift

case "$module" in
    attribution|renderer_consistency)
        exec "$(dirname "$0")/evaluate_module.sh" "$module" "$@"
        ;;
    *)
        if [[ "${1:-}" == --* || $# -eq 0 ]]; then
            exec "$(dirname "$0")/evaluate_module.sh" "$module" "$@"
        fi
        model="$1"
        shift
        exec "$(dirname "$0")/evaluate_module.sh" "$module" --model "$model" "$@"
        ;;
esac
