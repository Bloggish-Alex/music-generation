"""Artifact-only tests for encoded-to-trajectory anchor lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner
from evaluation_framework.evaluation_trajectory_anchor_context import _summary_png


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _parent() -> dict[str, object]:
    return {
        "schema_version": "encoded_input_manifest.v1",
        "run": {"identity": "fixture_preparation", "identity_kind": "model_preparation_run"},
        "dataset": {"identity": "fixture", "identity_kind": "stage_label_unverified", "content_sha256": None},
        "encoded_artifacts": {
            "bar_tensor_index": {"filename": "encoded/bar_tensor_index.json", "sha256": "sha256:" + "1" * 64, "schema_version": "encoded_bar_tensor_index.v1"},
            "bar_tensors": {"filename": "encoded/bar_tensors.npz", "sha256": "sha256:" + "2" * 64, "schema_version": "bar_tensor_archive.v1"},
            "songs": {"filename": "encoded/songs.json", "sha256": "sha256:" + "3" * 64, "schema_version": "encoded_song_records.v1"},
        },
        "tensor_identity": {
            "key_field": "tensor_key", "song_id_field": "song_id", "source_bar_index_field": "bar_index",
            "transpose_field": "applied_transpose_semitones", "base_pitch_field": "diagnostics.base_pitch",
            "tensor_schema_version": "bar_tensor_schema.v1",
        },
        "availability": {"bar_tensor_index": True, "bar_tensors": True, "songs": True, "stable_row_identity": True},
        "unavailable_reasons": [],
    }


def _raw(parent_hash: str, *, offset: int = 0, parent_hash_override: str | None = None) -> dict[str, object]:
    parent_hash = parent_hash if parent_hash_override is None else parent_hash_override
    rows = [
        {
            "tensor_key": "song__bar_0000", "song_id": "song", "base_song_id": "song",
            "source_bar_index": 0, "applied_transpose_semitones": 0,
            "encoded_base_pitch": 60, "trajectory_input_base_pitch": 60, "song_anchor": 61,
            "register_offset": -1 + offset, "base_pitch_origin": "serialized_anchor",
        },
        {
            "tensor_key": "song__bar_0001", "song_id": "song", "base_song_id": "song",
            "source_bar_index": 1, "applied_transpose_semitones": 0,
            "encoded_base_pitch": 62, "trajectory_input_base_pitch": 62, "song_anchor": 61,
            "register_offset": 1, "base_pitch_origin": "serialized_anchor",
        },
    ]
    return {
        "schema_version": "trajectory_training_input_lineage_raw.v2",
        "parent_encoded_input_manifest": {
            "filename": "encoded_input_manifest.v1.json", "sha256": parent_hash,
            "run_identity": "fixture_preparation", "bar_tensor_index_sha256": "sha256:" + "1" * 64,
            "bar_tensors_sha256": "sha256:" + "2" * 64, "tensor_schema_version": "bar_tensor_schema.v1",
        },
        "availability": {
            "parent_manifest": True, "encoded_index_hash_match": True, "encoded_tensor_hash_match": True,
            "row_identity_alignment": True, "base_pitch_alignment": True,
        },
        "unavailable_reasons": [],
        "row_summary": {"loaded_row_count": 2, "unique_tensor_key_count": 2, "song_count": 1},
        "register_parameterization": {
            "base_pitch_definition": "serialized encoded_index.diagnostics.base_pitch; otherwise configured runtime fallback",
            "song_anchor_definition": "round(median(base_pitch) within song_id)",
            "register_offset_definition": "base_pitch - song_anchor", "unit": "midi_semitone", "fallback_base_pitch": 60,
        },
        "observations": rows,
    }


def _run(
    tmp_path: Path,
    *,
    offset: int = 0,
    parent_hash_override: str | None = None,
) -> dict[str, object]:
    input_root = tmp_path / "input"
    input_root.mkdir()
    parent_path = input_root / "trajectory_anchor_context__encoded_input_manifest.v1.json"
    parent_path.write_text(json.dumps(_parent(), indent=2), encoding="utf-8")
    raw = _raw(
        _sha256(parent_path),
        offset=offset,
        parent_hash_override=parent_hash_override,
    )
    (input_root / "trajectory_anchor_context__training_lineage_raw.v2.json").write_text(
        json.dumps(raw, indent=2),
        encoding="utf-8",
    )
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(
        EvaluationRunRequest(
            input_root=input_root,
            output_root=tmp_path / "runs",
            run_id="anchor_context_1",
            modules=("trajectory_anchor_context",),
            mode=EvaluationMode.ALL,
        )
    )
    return json.loads(
        (store.run_dir / "anchor_context_1__trajectory_anchor_context__report.v1.json").read_text(encoding="utf-8")
    )


def test_anchor_context_passes_observed_training_boundaries_and_marks_runtime_unavailable(tmp_path: Path) -> None:
    report = _run(tmp_path)
    stages = report["metrics"]["stages"]

    assert report["status"] == "UNAVAILABLE"
    assert stages["model_preparation_to_training_input"]["status"] == "PASS"
    assert stages["empty_bar_fallback_to_training_input"]["status"] == "UNAVAILABLE"
    assert stages["training_anchor_derivation"]["status"] == "PASS"
    assert stages["training_anchor_derivation"]["comparable_unit"] == "song"
    assert stages["training_offset_reconstruction"]["status"] == "PASS"
    assert stages["generation_runtime_to_renderer"]["status"] == "UNAVAILABLE"


def test_anchor_context_fails_the_offset_boundary_without_hiding_other_evidence(tmp_path: Path) -> None:
    report = _run(tmp_path, offset=2)
    stages = report["metrics"]["stages"]

    assert report["status"] == "FAIL"
    assert stages["model_preparation_to_training_input"]["status"] == "PASS"
    assert stages["training_offset_reconstruction"]["status"] == "FAIL"


def test_anchor_context_fails_when_raw_parent_provenance_does_not_match(tmp_path: Path) -> None:
    report = _run(tmp_path, parent_hash_override="sha256:" + "f" * 64)
    stages = report["metrics"]["stages"]

    assert report["status"] == "FAIL"
    assert stages["model_preparation_to_training_input"]["status"] == "FAIL"
    assert "Parent provenance mismatch: sha256." in stages["model_preparation_to_training_input"]["reasons"]


def test_anchor_context_summary_is_a_real_png_with_missing_runtime_observation(tmp_path: Path) -> None:
    report = _run(tmp_path)
    png = _summary_png(report["metrics"]["stages"])

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1_000


def test_default_registry_exposes_trajectory_anchor_context() -> None:
    assert "trajectory_anchor_context" in DEFAULT_MODULE_REGISTRY.names()
