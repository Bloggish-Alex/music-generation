from __future__ import annotations

import hashlib
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from evaluation_framework.evaluation_artifact_store import EvaluationArtifactStore
from evaluation_framework.evaluation_context import EvaluationContext, ExportContext
from evaluation_framework.evaluation_final_v2_diagnostics import FinalV2DiagnosticEvaluator, FinalV2DiagnosticExporter
from diagnostics.final_v2_evaluation_raw_capture import FinalV2EvaluationRawCapture
from data.core import SongRecord


def _digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_quantization_audit_merges_same_opus_source_and_meter() -> None:
    audit = {"by_meter": {"4/4": {"onset_residual_samples_ql": [.0, .1], "end_residual_samples_ql": [.0, .2]}}}
    songs = [SongRecord("opus__tune_000", "opus.abc", metadata={"source_file_identity": "same", "quantization_audit": audit}), SongRecord("opus__tune_001", "opus.abc", metadata={"source_file_identity": "same", "quantization_audit": audit})]
    payload = FinalV2EvaluationRawCapture._quantization({}, songs)
    assert payload["by_file_meter"] == [{"source_file_identity": "same", "meter": "4/4", "event_count": 4, "nonzero_residual_count": 4, "onset_residual_ql": {"max": .1, "p95": .1}, "end_residual_ql": {"max": .2, "p95": .2}}]
