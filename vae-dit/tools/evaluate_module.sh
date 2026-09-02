#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../bin/init_env.sh"

usage() {
    cat <<'EOF'
Usage: evaluate_module.sh <module> (--model <model-name> | --source-dir <directory>) [--run-dir <directory>]

Exports one diagnostics module and writes its report into a flat run directory.
`--model` resolves to $OUTPUT_DIR/models/<model-name>; use `--source-dir` for
generation-run diagnostics such as renderer consistency and attribution.
EOF
}

MODULE=""
MODEL=""
SOURCE_DIR=""
RUN_DIR=""

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
MODULE="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:?--model requires a name}"; shift 2 ;;
        --source-dir) SOURCE_DIR="${2:?--source-dir requires a directory}"; shift 2 ;;
        --run-dir) RUN_DIR="${2:?--run-dir requires a directory}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$MODULE" in
    anchor_transport|attribution|codec_fidelity|dataset_tonality|dvae_fidelity|dvae_pitch_diagnostics|latent_probe|physical_trajectory_objective|renderer_consistency|trajectory_anchor_context|parser_integrity|quantization_audit|performance_controls|form_action_alignment) ;;
    *) echo "Unsupported module: $MODULE" >&2; exit 2 ;;
esac
[[ -z "$MODEL" || -z "$SOURCE_DIR" ]] || { echo "Use either --model or --source-dir." >&2; exit 2; }
case "$MODULE" in
    renderer_consistency|attribution)
        [[ -z "$MODEL" ]] || { echo "$MODULE requires --source-dir <generation-run-dir>; it cannot be evaluated from a model directory." >&2; exit 2; }
        ;;
esac
if [[ -n "$MODEL" ]]; then
    SOURCE_DIR="${OUTPUT_DIR}/models/${MODEL}"
    [[ "$MODULE" != "physical_trajectory_objective" ]] || SOURCE_DIR="${SOURCE_DIR}/physical_trajectory"
fi
[[ -n "$SOURCE_DIR" && -d "$SOURCE_DIR" ]] || { echo "An existing source directory is required." >&2; exit 2; }

if [[ -z "$RUN_DIR" ]]; then
    if [[ -n "$MODEL" ]]; then
        RUN_DIR="${OUTPUT_DIR}/${MODEL}__${MODULE}__$(date +%Y%m%d_%H%M%S)"
    else
        RUN_DIR="${SOURCE_DIR}/${MODULE}__evaluation"
    fi
fi
mkdir -p "$RUN_DIR"

revision="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
"${PYTHON_BIN}" "$ROOT_DIR/tools/export_module.py" --module "$MODULE" --source-dir "$SOURCE_DIR" --output-dir "$RUN_DIR"

case "$MODULE" in
    dvae_pitch_diagnostics) evaluation_modules="dvae_pitch_supervision_audit,dvae_pitch_gradient_probe" ;;
    *) evaluation_modules="$MODULE" ;;
esac
"${PYTHON_BIN}" "$ROOT_DIR/tools/evaluate.py" --run-dir "$RUN_DIR" --modules "$evaluation_modules" --mode all --code-revision "$revision"
echo "${MODULE} report written to: ${RUN_DIR}"
