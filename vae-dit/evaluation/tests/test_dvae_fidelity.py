"""Artifact-only tests for deterministic DVAE fidelity reporting."""

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
from evaluation_framework.evaluation_dvae_fidelity import _pitch_metrics, _summary_png
from evaluation_framework.evaluation_dvae_fidelity_presentation import _markdown


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _schema() -> dict[str, object]:
    names = [
        "relative_pitch", "is_rest", "is_note_on", "is_hold", "normalized_velocity",
        "velocity_ratio", "density_gradient", *[f"relative_chroma_embed_{index:02d}" for index in range(11)],
    ]
    return {
        "schema_version": "bar_tensor_schema.v1",
        "axis_order": ["bar", "track", "step", "feature"],
        "feature_names": names,
        "feature_units": {},
        "pitch_scale_semitones": 24.0,
        "track_names": ["melody"],
    }


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.zeros((2, 1, 2, 18), dtype=np.float32)
    decoded = np.zeros_like(source)
    for tensor in (source, decoded):
        tensor[..., 1] = 1.0
        tensor[:, 0, 0, 1] = 0.0
        tensor[:, 0, 0, 2] = 1.0
        tensor[:, 0, 1, 1] = 0.0
        tensor[:, 0, 1, 3] = 1.0
        tensor[:, 0, :, 4] = 0.5
    source[0, 0, :, 0] = [0.0, 0.0]
    source[1, 0, :, 0] = [2.0 / 24.0, 2.0 / 24.0]
    decoded[:] = source
    decoded[1, 0, :, 0] = 3.0 / 24.0
    latent = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return source, latent, decoded


def _write_available_split(root: Path, split: str) -> None:
    source, latent, decoded = _arrays()
    arrays_path = root / f"dvae_fidelity__raw_arrays__{split}.v1.npz"
    np.savez_compressed(arrays_path, source_tensor=source, latent_mu=latent, decoded_tensor=decoded)
    observation_path = root / f"dvae_fidelity__raw_observation__{split}.v1.json"
    observation = {
        "schema_version": "dvae_fidelity_raw_observation.v1",
        "dataset": {"identity": "fixture", "identity_kind": "stage_label_unverified", "content_sha256": None, "split": split, "split_unit": "base_song_id"},
        "parent_encoded_input_manifest": {"sha256": "sha256:" + "1" * 64, "run_identity": "fixture", "bar_tensor_index_sha256": "sha256:" + "2" * 64, "bar_tensors_sha256": "sha256:" + "3" * 64, "tensor_schema_version": "bar_tensor_schema.v1"},
        "checkpoint": {"role": "dvae", "sha256": "sha256:" + "4" * 64},
        "arrays": {
            "path": arrays_path.name, "sha256": _sha256(arrays_path),
            "source_tensor": {"array_name": "source_tensor", "dtype": "float32", "shape": [2, 1, 2, 18]},
            "latent_mu": {"array_name": "latent_mu", "dtype": "float32", "shape": [2, 2]},
            "decoded_tensor": {"array_name": "decoded_tensor", "dtype": "float32", "shape": [2, 1, 2, 18]},
        },
        "tensor_schema": _schema(),
        "decoder_projection": {"latent_path": "deterministic_mu", "state_projection": "argmax_state_logits_to_one_hot", "feature_projection": "pitch_state_velocity_chord_into_bar_tensor_schema.v1"},
        "alignment": [
            {"row": 0, "song_id": "song", "base_song_id": "song", "source_bar_index": 0, "applied_transpose_semitones": 0, "base_pitch_semitones": 60},
            {"row": 1, "song_id": "song", "base_song_id": "song", "source_bar_index": 1, "applied_transpose_semitones": 0, "base_pitch_semitones": 60},
        ],
        "availability": {"source_tensor": True, "latent_mu": True, "decoded_tensor": True, "row_alignment": True, "tensor_schema": True, "deterministic_mu": True},
    }
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    status = {
        "schema_version": "dvae_fidelity_raw_status.v1",
        "dataset": observation["dataset"], "status": "AVAILABLE",
        "availability": observation["availability"], "unavailable_reasons": [],
        "artifacts": {"observation": {"path": observation_path.name, "sha256": _sha256(observation_path)}, "arrays": {"path": arrays_path.name, "sha256": _sha256(arrays_path)}},
    }
    (root / f"dvae_fidelity__raw_status__{split}.v1.json").write_text(json.dumps(status), encoding="utf-8")


def _run(tmp_path: Path, splits: tuple[str, ...] = ("train", "validation")) -> dict[str, object]:
    input_root = tmp_path / "input"
    input_root.mkdir()
    for split in splits:
        _write_available_split(input_root, split)
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(
        EvaluationRunRequest(input_root=input_root, output_root=tmp_path / "runs", run_id="dvae_1", modules=("dvae_fidelity",), mode=EvaluationMode.ALL)
    )
    return json.loads((store.run_dir / "dvae_1__dvae_fidelity__report.v1.json").read_text(encoding="utf-8"))


def test_dvae_fidelity_reports_independent_reconstruction_groups(tmp_path: Path) -> None:
    report = _run(tmp_path)
    validation = report["metrics"]["splits"]["validation"]
    metrics = validation["metrics"]

    assert report["status"] == "MONITOR"
    assert metrics["state"]["accuracy"] == 1.0
    assert metrics["relative_pitch"]["mae_semitones"] == 0.5
    assert metrics["register"]["delta_mae_semitones"] == 1.0
    assert metrics["feature_group_mse"]["relative_pitch"] > 0.0
    assert metrics["texture_diversity"]["pair_count"] == 2


def test_dvae_fidelity_is_unavailable_without_any_available_status(tmp_path: Path) -> None:
    report = _run(tmp_path, splits=())

    assert report["status"] == "UNAVAILABLE"
    assert report["missing_inputs"]


def test_dvae_fidelity_summary_is_a_real_png(tmp_path: Path) -> None:
    report = _run(tmp_path)
    png = _summary_png(report["metrics"]["splits"])

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1_000


def test_default_registry_exposes_dvae_fidelity() -> None:
    assert "dvae_fidelity" in DEFAULT_MODULE_REGISTRY.names()


def test_dvae_fidelity_markdown_is_utf8_chinese_text() -> None:
    markdown = _markdown({"status": "UNAVAILABLE"})

    assert markdown.startswith("# DVAE 保真度")
    assert "原始观察数据不可用" in markdown


def test_relative_pitch_metrics_distinguish_octave_errors_from_pitch_class_errors() -> None:
    source = np.zeros((1, 3, 1), dtype=np.float32)
    decoded = np.asarray([[[12.0], [-12.0], [24.0]]], dtype=np.float32)
    mask = np.ones_like(source, dtype=bool)
    metrics = _pitch_metrics(source, decoded, mask, {"track_names": ["melody", "harmony", "bass"]})

    assert metrics["mae_semitones"] == 16.0
    assert metrics["pitch_class_mae_semitones"] == 0.0
    assert metrics["octave_component_mae_semitones"] == 16.0
    assert metrics["octave_equivalent_error_rate"] == 1.0
    assert metrics["error_bands"]["plus_12_rate"] == 1.0 / 3.0
    assert metrics["error_bands"]["minus_12_rate"] == 1.0 / 3.0
    assert metrics["error_bands"]["plus_24_rate"] == 1.0 / 3.0
    assert metrics["voice_roles"]["bass"]["mae_semitones"] == 24.0


def test_relative_pitch_metrics_report_voice_slope_bias_and_anchor_distance() -> None:
    source = np.asarray([[[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]], dtype=np.float32)
    decoded = source.copy()
    decoded[0, 0] = [0.0, 5.0, 10.0]
    mask = np.ones_like(source, dtype=bool)

    metrics = _pitch_metrics(source, decoded, mask, {"track_names": ["melody", "harmony", "bass"]})

    melody = metrics["voice_roles"]["melody"]
    assert melody["slope"] == 0.5
    assert np.isclose(melody["intercept_semitones"], 0.0)
    assert melody["signed_bias_semitones"] == -5.0
    bands = metrics["distance_to_anchor"]["voice_roles"]["melody"]["bands"]
    assert bands[0]["range_semitones"] == "0-5"
    assert bands[1]["mae_semitones"] == 5.0
    assert bands[3]["mae_semitones"] == 10.0
