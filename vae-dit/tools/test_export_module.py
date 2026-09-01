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
    }
    assert COMMANDS["trajectory_anchor_context"].source_option == "--model-dir"


def test_oracle_evidence_shell_uses_existing_module_and_ladder_entry_points() -> None:
    script = (ROOT / "tools" / "evaluate_oracle_evidence.sh").read_text(encoding="utf-8")

    assert "evaluate_module.sh" in script
    assert "export_evaluation_artifacts.py" in script
    assert "evaluate_oracle_ladder.sh" in script
