"""Tests for the engineering-side DVAE fidelity public copy."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from export.dvae_fidelity_artifact_export import DVAEFidelityArtifactExportConfig, export_dvae_fidelity_artifacts


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_export_copies_status_led_dvae_raw_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    observation = source / "dvae_fidelity__raw_observation__validation.v1.json"
    arrays = source / "dvae_fidelity__raw_arrays__validation.v1.npz"
    arrays.write_bytes(b"fixture")
    observation.write_text(json.dumps({"schema_version": "dvae_fidelity_raw_observation.v1"}), encoding="utf-8")
    status = {
        "schema_version": "dvae_fidelity_raw_status.v1",
        "dataset": {"split": "validation"}, "status": "AVAILABLE",
        "artifacts": {
            "observation": {"path": observation.name, "sha256": _sha256(observation)},
            "arrays": {"path": arrays.name, "sha256": _sha256(arrays)},
        },
    }
    (source / "dvae_fidelity__raw_status__validation.v1.json").write_text(json.dumps(status), encoding="utf-8")

    output = tmp_path / "run"
    paths = export_dvae_fidelity_artifacts(DVAEFidelityArtifactExportConfig(source, output))

    assert {path.name for path in paths} == {"dvae_fidelity__raw_status__validation.v1.json", observation.name, arrays.name}
    assert all(not path.is_dir() for path in output.iterdir())
