"""Publish one final Codec V2 diagnostic raw observation unchanged."""
from __future__ import annotations

import shutil
from pathlib import Path


def export_final_v2_diagnostic_artifacts(source_dir: Path, output_dir: Path, module: str) -> None:
    source = source_dir / f"{module}__raw_observation.v2.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {module} raw observation.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_dir / source.name)
