"""Artifact-only tests for held-out DVAE latent readability probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_split(root: Path, split: str, song: str, *, anchorless: bool = False) -> None:
    rows, features = 8, 18
    tensor = np.zeros((rows, 1, 2, features), dtype=np.float32)
    tensor[..., 1] = 1.0
    tensor[:, 0, 0, 1] = 0.0
    tensor[:, 0, 0, 2] = 1.0
    tensor[:, 0, 1, 1] = 0.0
    tensor[:, 0, 1, 3] = 1.0
    tensor[:, 0, :, 0] = np.arange(rows, dtype=np.float32)[:, None] / 24.0
    latent = np.column_stack([np.arange(rows), np.arange(rows) ** 2]).astype(np.float32)
    decoded = np.zeros_like(tensor)
    arrays_path = root / f"dvae_fidelity__raw_arrays__{split}.v1.npz"
    np.savez_compressed(arrays_path, source_tensor=tensor, latent_mu=latent, decoded_tensor=decoded)
    names = ["relative_pitch", "is_rest", "is_note_on", "is_hold", "normalized_velocity", "velocity_ratio", "density_gradient", *[f"relative_chroma_embed_{value:02d}" for value in range(11)]]
    observation_path = root / f"dvae_fidelity__raw_observation__{split}.v1.json"
    observation = {
        "schema_version": "dvae_fidelity_raw_observation.v1",
        "dataset": {"identity": "fixture", "identity_kind": "test", "content_sha256": None, "split": split, "split_unit": "base_song_id"},
        "arrays": {"path": arrays_path.name, "sha256": _sha256(arrays_path)},
        "tensor_schema": {"schema_version": "bar_tensor_schema.v1", "feature_names": names, "pitch_scale_semitones": 24.0},
        "alignment": [{"row": row, "song_id": song, "base_song_id": song, "source_bar_index": row, "applied_transpose_semitones": 0, "base_pitch_semitones": None if anchorless else 60} for row in range(rows)],
    }
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    status = {"schema_version": "dvae_fidelity_raw_status.v1", "dataset": observation["dataset"], "status": "AVAILABLE", "artifacts": {"observation": {"path": observation_path.name, "sha256": _sha256(observation_path)}, "arrays": {"path": arrays_path.name, "sha256": _sha256(arrays_path)}}}
    (root / f"dvae_fidelity__raw_status__{split}.v1.json").write_text(json.dumps(status), encoding="utf-8")


def _report(tmp_path: Path, *, same_song: bool = False, anchorless: bool = False) -> tuple[dict[str, object], Path]:
    inputs = tmp_path / "input"; inputs.mkdir()
    _write_split(inputs, "train", "train_song", anchorless=anchorless)
    _write_split(inputs, "validation", "train_song" if same_song else "validation_song", anchorless=anchorless)
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(input_root=inputs, output_root=tmp_path / "runs", run_id="probe", modules=("latent_probe",), mode=EvaluationMode.ALL))
    report_path = store.run_dir / "probe__latent_probe__report.v1.json"
    return (json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}), store.run_dir


def test_latent_probe_uses_train_fit_and_validation_report(tmp_path: Path) -> None:
    report, run_dir = _report(tmp_path)
    relative = report["metrics"]["validation"]["relative_register_center"]
    assert report["status"] == "MONITOR"
    assert relative["train_bar_count"] == 8
    assert relative["validation_bar_count"] == 8
    assert "linear" in relative and "mlp" in relative
    assert (run_dir / "probe__latent_probe__readability_summary.v1.png").read_bytes().startswith(b"\x89PNG")


def test_latent_probe_marks_only_absolute_targets_unavailable_without_anchor(tmp_path: Path) -> None:
    report, _ = _report(tmp_path, anchorless=True)
    targets = report["metrics"]["validation"]
    assert targets["absolute_chroma"]["status"] == "UNAVAILABLE"
    assert targets["absolute_register_center"]["status"] == "UNAVAILABLE"
    assert targets["relative_chroma"]["status"] == "MONITOR"


def test_latent_probe_rejects_base_song_overlap(tmp_path: Path) -> None:
    _, run_dir = _report(tmp_path, same_song=True)
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    assert index["modules"]["latent_probe"]["evaluate"]["status"] == "FAIL"


def test_default_registry_exposes_latent_probe() -> None:
    assert "latent_probe" in DEFAULT_MODULE_REGISTRY.names()
