from __future__ import annotations

import argparse
from pathlib import Path

from export.final_v2_diagnostics_artifact_export import export_final_v2_diagnostic_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module", required=True, choices=("parser_integrity", "quantization_audit", "performance_controls", "form_action_alignment"))
    args = parser.parse_args()
    export_final_v2_diagnostic_artifacts(args.source_dir, args.output_dir, args.module)


if __name__ == "__main__":
    main()
