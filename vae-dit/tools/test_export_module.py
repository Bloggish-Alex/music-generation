from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_module import COMMANDS


def test_each_dispatcher_module_has_a_concrete_cli_command() -> None:
    assert set(COMMANDS) == {
        "anchor_transport", "attribution", "codec_fidelity", "dataset_tonality",
        "dvae_fidelity", "dvae_pitch_diagnostics", "latent_probe",
        "physical_trajectory_objective", "renderer_consistency", "trajectory_anchor_context",
        "parser_integrity", "quantization_audit", "performance_controls", "form_action_alignment",
    }
    assert COMMANDS["trajectory_anchor_context"].source_option == "--model-dir"


def test_encode_time_modules_resolve_an_explicit_canonical_encoded_run() -> None:
    script = (ROOT / "tools" / "evaluate_module.sh").read_text(encoding="utf-8")

    assert "--encoded-run-id" in script
    assert 'ENCODED_RUN_ID="${ENCODED_RUN_ID:-${MODEL}-v2-r001}"' in script
    assert 'SOURCE_DIR="${SOURCE_DIR}/encoded/${ENCODED_RUN_ID}"' in script
    for module in (
        "codec_fidelity", "dataset_tonality", "parser_integrity",
        "quantization_audit", "performance_controls", "form_action_alignment",
    ):
        assert module in script
