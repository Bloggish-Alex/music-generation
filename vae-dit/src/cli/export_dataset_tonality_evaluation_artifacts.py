"""CLI for public dataset-tonality evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from export.dataset_tonality_artifact_export import DatasetTonalityArtifactExportConfig, export_dataset_tonality_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset-tonality diagnostics as public evaluation artifacts.")
    parser.add_argument("--source-dir", required=True, type=Path, help="Model directory containing raw dataset-tonality diagnostics.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Single evaluation or analysis run directory.")
    args = parser.parse_args()
    print(export_dataset_tonality_artifacts(DatasetTonalityArtifactExportConfig(args.source_dir, args.output_dir)))


if __name__ == "__main__":
    main()
