"""Tests for physical trajectory objective public copying."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from export.physical_trajectory_objective_artifact_export import PhysicalTrajectoryObjectiveArtifactExportConfig, export_physical_trajectory_objective_artifacts


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_copies_only_status_led_physical_trajectory_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    arrays = source / "physical_trajectory_objective__raw_arrays.v2.npz"; arrays.write_bytes(b"arrays")
    observation = source / "physical_trajectory_objective__raw_observation.v2.json"; observation.write_text(json.dumps({"schema_version": "physical_trajectory_objective_raw_observation.v2", "arrays": {"path": arrays.name, "sha256": _sha(arrays)}}), encoding="utf-8")
    status = {"schema_version": "physical_trajectory_objective_raw_status.v2", "status": "AVAILABLE", "artifacts": {"observation": {"path": observation.name, "sha256": _sha(observation)}}}
    (source / "physical_trajectory_objective__raw_status.v2.json").write_text(json.dumps(status), encoding="utf-8")
    paths = export_physical_trajectory_objective_artifacts(PhysicalTrajectoryObjectiveArtifactExportConfig(source, tmp_path / "out"))
    assert {path.name for path in paths} == {"physical_trajectory_objective__raw_status.v2.json", observation.name, arrays.name}
