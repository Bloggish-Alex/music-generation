"""Tests for the engineering-side public copy of trajectory anchor-context facts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from export.trajectory_anchor_context_artifact_export import (
    TrajectoryAnchorContextArtifactExportConfig,
    export_trajectory_anchor_context_artifacts,
)


def test_export_copies_only_recognized_flat_public_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    trajectory_dir = model_dir / "trajectory_diffusion"
    trajectory_dir.mkdir(parents=True)
    (model_dir / "encoded_input_manifest.v1.json").write_text(
        json.dumps({"schema_version": "encoded_input_manifest.v1"}),
        encoding="utf-8",
    )
    (trajectory_dir / "trajectory_training_input_lineage__raw.v2.json").write_text(
        json.dumps({"schema_version": "trajectory_training_input_lineage_raw.v2"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    paths = export_trajectory_anchor_context_artifacts(
        TrajectoryAnchorContextArtifactExportConfig(model_dir, output_dir)
    )

    assert [path.name for path in paths] == [
        "trajectory_anchor_context__encoded_input_manifest.v1.json",
        "trajectory_anchor_context__training_lineage_raw.v2.json",
    ]
    assert all(not path.is_dir() for path in output_dir.iterdir())


def test_export_rejects_wrong_schema_before_copying(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "encoded_input_manifest.v1.json").write_text(
        json.dumps({"schema_version": "wrong.v1"}),
        encoding="utf-8",
    )

    try:
        export_trajectory_anchor_context_artifacts(
            TrajectoryAnchorContextArtifactExportConfig(model_dir, tmp_path / "run")
        )
    except ValueError as error:
        assert "Unsupported trajectory anchor-context schema" in str(error)
    else:
        raise AssertionError("Expected public export to reject an unsupported schema.")
