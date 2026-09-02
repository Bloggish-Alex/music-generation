from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError

from evaluation_framework.evaluation_artifact_store import EvaluationArtifactStore
from evaluation_framework.evaluation_context import EvaluationContext, ExportContext
from evaluation_framework.evaluation_final_v2_diagnostics import FinalV2DiagnosticEvaluator, FinalV2DiagnosticExporter
from diagnostics.final_v2_evaluation_raw_capture import FinalV2EvaluationRawCapture
from data.core import SongRecord


def _digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(event_count: int = 2, *, onset_max: float = .1) -> dict:
    return {"by_meter": {"4/4": {"event_count": event_count, "nonzero_residual_count": 2, "onset_residual_ql": {"max": onset_max, "p95": onset_max}, "end_residual_ql": {"max": .2, "p95": .2}}}}


def test_final_v2_diagnostic_raw_schema_rejects_available_without_capture(tmp_path) -> None:
    schema_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2" / "quantization_audit__raw_observation.v2.schema.json"
    schema = json.loads(schema_path.read_text())
    payload = {"schema_version": "quantization_audit_raw_observation.v2", "status": "AVAILABLE", "run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "0" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "x", "content_sha256": None}, "availability": {"raw_capture": False, "source_boundaries": True}, "grid_policy": {"quantum_ql": .25, "epsilon_ql": 1e-6, "capacity": 48}, "by_file_meter": []}
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(payload)


def test_final_v2_diagnostic_export_and_evaluate(tmp_path) -> None:
    public = tmp_path / "public"; public.mkdir(); run = EvaluationArtifactStore.create(tmp_path, "run")
    raw = {"schema_version": "parser_integrity_raw_observation.v2", "status": "AVAILABLE", "run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "x", "content_sha256": None}, "availability": {"raw_capture": True, "measure_map": True}, "measure_map": {"song_count": 1, "measure_count": 1, "meter_distribution": {"4/4": 1}, "opus_tune_count": 0, "over_capacity_count": 0}, "track_retention": {"hard_safety_limit": 48, "policy": "retain_all", "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0}, "parser_failures": [], "unavailable_reasons": []}
    path = public / "parser_integrity__raw_observation.v2.json"; path.write_text(json.dumps(raw))
    exporter = FinalV2DiagnosticExporter("parser_integrity"); bundle = exporter.export(ExportContext("run", public, run))
    result = FinalV2DiagnosticEvaluator("parser_integrity").evaluate(EvaluationContext("run", public, run), bundle)
    assert result.report["status"] == "MONITOR"


def test_unavailable_final_v2_raw_observations_remain_schema_valid(tmp_path) -> None:
    paths = FinalV2EvaluationRawCapture().capture(tmp_path, [], {"identity": "fixture", "content_sha256": None}, [])
    root = __import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2"
    for module, path in paths.items():
        Draft202012Validator(json.loads((root / f"{module}__raw_observation.v2.schema.json").read_text())).validate(json.loads(path.read_text()))


def test_quantization_audit_merges_same_opus_source_and_meter(tmp_path) -> None:
    audit = _audit()
    samples = {"4/4": {"onset_residual_samples_ql": [.0, .1], "end_residual_samples_ql": [.0, .2]}}
    songs = [SongRecord("opus__tune_000", "opus.abc", metadata={"source_file_identity": "same", "quantization_audit": audit}, runtime_diagnostics={"quantization_residual_samples":samples}), SongRecord("opus__tune_001", "opus.abc", metadata={"source_file_identity": "same", "quantization_audit": audit}, runtime_diagnostics={"quantization_residual_samples":samples})]
    common = {"run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "fixture", "content_sha256": None}}
    payload = FinalV2EvaluationRawCapture._quantization(common, songs, tmp_path)
    assert payload["by_file_meter"] == [{"source_file_identity": "same", "meter": "4/4", "event_count": 4, "nonzero_residual_count": 4, "onset_residual_ql": {"max": .1, "p95": .1}, "end_residual_ql": {"max": .2, "p95": .2}}]
    archive_path = tmp_path / payload["residual_samples"]["path"]
    assert payload["residual_samples"]["sha256"] == _digest(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"source_file_identities", "meters", "group_offsets", "onset_residuals_ql", "end_residuals_ql"}
        assert archive["group_offsets"].dtype == np.dtype("int64")
        assert archive["group_offsets"].tolist() == [0, 4]
        assert archive["onset_residuals_ql"].dtype == np.dtype("float32")
    schema_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2" / "quantization_audit__raw_observation.v2.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text())).validate(payload)


@pytest.mark.parametrize(
    "samples",
    [
        None,
        {"4/4": {"onset_residual_samples_ql": [.0], "end_residual_samples_ql": [.0]}},
        {"4/4": {"onset_residual_samples_ql": [.0, .1], "end_residual_samples_ql": [.0]}},
    ],
)
def test_quantization_audit_rejects_missing_or_inconsistent_runtime_samples(tmp_path, samples) -> None:
    audit = _audit()
    runtime = {} if samples is None else {"quantization_residual_samples": samples}
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "source", "quantization_audit": audit}, runtime_diagnostics=runtime)
    with pytest.raises(ValueError, match="quantization residual samples"):
        FinalV2EvaluationRawCapture._quantization({}, [song], tmp_path)


def test_quantization_audit_rejects_summary_statistic_mismatch(tmp_path) -> None:
    samples = {"4/4": {"onset_residual_samples_ql": [.0, .1], "end_residual_samples_ql": [.0, .2]}}
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "source", "quantization_audit": _audit(onset_max=.09)}, runtime_diagnostics={"quantization_residual_samples": samples})
    with pytest.raises(ValueError, match="quantization residual samples disagree with summary"):
        FinalV2EvaluationRawCapture._quantization({}, [song], tmp_path)


def test_song_json_excludes_quantization_runtime_samples() -> None:
    song = SongRecord("song", "song.mid", metadata={"quantization_audit": {"by_meter": {"4/4": {"event_count": 1}}}}, runtime_diagnostics={"quantization_residual_samples": {"4/4": {"onset_residual_samples_ql": [.0], "end_residual_samples_ql": [.0]}}})
    serialized = json.dumps(song.to_dict())
    assert "residual_samples" not in serialized
