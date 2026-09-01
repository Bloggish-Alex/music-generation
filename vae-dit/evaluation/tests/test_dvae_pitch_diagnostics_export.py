"""Tests for public copying of DVAE pitch diagnostic facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from export.dvae_pitch_diagnostics_artifact_export import DVAEPitchDiagnosticsArtifactExportConfig, export_dvae_pitch_diagnostics_artifacts


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_copies_only_status_led_pitch_observations(tmp_path: Path) -> None:
    source = tmp_path / "model"; source.mkdir()
    raw = source / "dvae_pitch_supervision_audit__raw.v1.json"
    raw.write_text(json.dumps({"schema_version": "dvae_pitch_supervision_audit_raw.v1"}), encoding="utf-8")
    status = {"schema_version": "dvae_pitch_supervision_audit_status.v1", "run": {"identity": "fixture"}, "status": "AVAILABLE", "availability": {"observation": True}, "unavailable_reasons": [], "artifacts": {"observation": {"path": raw.name, "sha256": _hash(raw)}}}
    (source / "dvae_pitch_supervision_audit__raw_status.v1.json").write_text(json.dumps(status), encoding="utf-8")
    output = tmp_path / "run"
    paths = export_dvae_pitch_diagnostics_artifacts(DVAEPitchDiagnosticsArtifactExportConfig(source, output))
    assert {path.name for path in paths} == {raw.name, "dvae_pitch_supervision_audit__raw_status.v1.json"}
    assert all(path.is_file() for path in output.iterdir())
