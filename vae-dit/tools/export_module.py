#!/usr/bin/env python3
"""Dispatch one public-artifact export without exposing module-specific wrappers."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class ExportCommand:
    cli_module: str
    source_option: str = "--source-dir"
    module_name: str | None = None


COMMANDS = {
    "anchor_transport": ExportCommand("cli.export_anchor_transport_evaluation_artifacts"),
    "attribution": ExportCommand("cli.export_attribution_evaluation_artifacts"),
    "codec_fidelity": ExportCommand("cli.export_codec_fidelity_evaluation_artifacts"),
    "dataset_tonality": ExportCommand("cli.export_dataset_tonality_evaluation_artifacts"),
    "dvae_fidelity": ExportCommand("cli.export_dvae_fidelity_evaluation_artifacts"),
    "dvae_pitch_diagnostics": ExportCommand("cli.export_dvae_pitch_diagnostics_evaluation_artifacts"),
    "latent_probe": ExportCommand("cli.export_latent_probe_evaluation_artifacts"),
    "physical_trajectory_objective": ExportCommand("cli.export_physical_trajectory_objective_evaluation_artifacts"),
    "renderer_consistency": ExportCommand("cli.export_renderer_consistency_evaluation_artifacts"),
    "trajectory_anchor_context": ExportCommand(
        "cli.export_trajectory_anchor_context_evaluation_artifacts", "--model-dir"
    ),
    "parser_integrity": ExportCommand("cli.export_final_v2_diagnostic_evaluation_artifacts", module_name="parser_integrity"),
    "quantization_audit": ExportCommand("cli.export_final_v2_diagnostic_evaluation_artifacts", module_name="quantization_audit"),
    "performance_controls": ExportCommand("cli.export_final_v2_diagnostic_evaluation_artifacts", module_name="performance_controls"),
    "form_action_alignment": ExportCommand("cli.export_final_v2_diagnostic_evaluation_artifacts", module_name="form_action_alignment"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export one Evaluation Framework module.")
    parser.add_argument("--module", required=True, choices=sorted(COMMANDS))
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    command = COMMANDS[args.module]
    cli = importlib.import_module(command.cli_module)
    original_argv = sys.argv
    try:
        sys.argv = [command.cli_module, command.source_option, str(args.source_dir), "--output-dir", str(args.output_dir)]
        if command.module_name:
            sys.argv.extend(["--module", command.module_name])
        cli.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
