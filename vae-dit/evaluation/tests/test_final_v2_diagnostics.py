from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError

from evaluation_framework.evaluation_artifact_store import EvaluationArtifactStore
from evaluation_framework.evaluation_context import EvaluationContext, ExportContext
from evaluation_framework.evaluation_final_v2_diagnostics import FinalV2DiagnosticEvaluator, FinalV2DiagnosticExporter
from codec.action_labeler import ACTION_RETURN, ActionLabeler, ActionLabelerConfig
from diagnostics.final_v2_evaluation_raw_capture import FinalV2EvaluationRawCapture
from data.core import BarRecord, NoteEvent, SongRecord, TrackRecord


def _digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _samples(prefix: str = "note") -> list[dict]:
    return [
        {"source_note_id": f"{prefix}:0", "canonical_bar_index": 0, "meter": "4/4", "raw_local_start_ql": .0, "raw_local_end_ql": 1.0, "quantized_local_start_ql": .0, "quantized_local_end_ql": 1.0, "onset_residual_ql": .0, "end_residual_ql": .0},
        {"source_note_id": f"{prefix}:1", "canonical_bar_index": 1, "meter": "4/4", "raw_local_start_ql": .1, "raw_local_end_ql": 1.2, "quantized_local_start_ql": .0, "quantized_local_end_ql": 1.0, "onset_residual_ql": .1, "end_residual_ql": .2},
    ]


def _audit(samples: list[dict]) -> dict:
    return {"audit_unit": "source_note_fragment", "fragment_count": len(samples), "by_meter": {"4/4": {"fragment_count": len(samples), "nonzero_residual_count": sum(value > 1e-9 for sample in samples for value in (sample["onset_residual_ql"], sample["end_residual_ql"])), "onset_residual_ql": {"max": .1, "p95": .1}, "end_residual_ql": {"max": .2, "p95": .2}}}}


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
    first, second = _samples("first"), _samples("second")
    songs = [SongRecord("opus__tune_000", "opus.mid", metadata={"source_file_identity": "same", "tune_index": 0, "quantization_audit": _audit(first)}, runtime_diagnostics={"quantization_fragment_samples": first}), SongRecord("opus__tune_001", "opus.mid", metadata={"source_file_identity": "same", "tune_index": 1, "quantization_audit": _audit(second)}, runtime_diagnostics={"quantization_fragment_samples": second})]
    common = {"run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "fixture", "content_sha256": None}}
    payload = FinalV2EvaluationRawCapture._quantization(common, songs, tmp_path)
    assert payload["audit_unit"] == "source_note_fragment"
    assert payload["fragment_count"] == 4
    assert [row["tune_index"] for row in payload["by_file_meter"]] == [0, 1]
    archive_path = tmp_path / payload["residual_samples"]["path"]
    assert payload["residual_samples"]["sha256"] == _digest(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        assert {"source_note_ids", "canonical_bar_indexes", "raw_local_start_ql", "quantized_local_end_ql"} <= set(archive.files)
        assert archive["group_offsets"].dtype == np.dtype("int64")
        assert archive["group_offsets"].tolist() == [0, 2, 4]
        assert archive["onset_residuals_ql"].dtype == np.dtype("float32")
    schema_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2" / "quantization_audit__raw_observation.v2.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text())).validate(payload)


@pytest.mark.parametrize("runtime", [{}, {"quantization_residual_samples": {"4/4": {"onset_residual_samples_ql": [.0], "end_residual_samples_ql": [.0]}}}])
def test_quantization_audit_rejects_missing_or_legacy_runtime_samples(tmp_path, runtime) -> None:
    samples = _samples()
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "source", "quantization_audit": _audit(samples)}, runtime_diagnostics=runtime)
    with pytest.raises(ValueError, match="quantization fragment samples"):
        FinalV2EvaluationRawCapture._quantization({}, [song], tmp_path)


def test_quantization_audit_rejects_summary_statistic_mismatch(tmp_path) -> None:
    samples = _samples(); audit = _audit(samples); audit["by_meter"]["4/4"]["onset_residual_ql"]["max"] = .09
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "source", "quantization_audit": audit}, runtime_diagnostics={"quantization_fragment_samples": samples})
    with pytest.raises(ValueError, match="quantization fragment samples disagree with summary"):
        FinalV2EvaluationRawCapture._quantization({}, [song], tmp_path)


def test_song_json_excludes_quantization_runtime_samples() -> None:
    song = SongRecord("song", "song.mid", metadata={"quantization_audit": {"by_meter": {"4/4": {"event_count": 1}}}}, runtime_diagnostics={"quantization_residual_samples": {"4/4": {"onset_residual_samples_ql": [.0], "end_residual_samples_ql": [.0]}}})
    serialized = json.dumps(song.to_dict())
    assert "residual_samples" not in serialized


def test_form_action_alignment_exports_mid_song_return_through_framework(tmp_path) -> None:
    """An ABACA-style middle return must survive label, raw capture, export, and evaluation."""
    bars = []
    for index in range(40):
        is_a = index < 4 or index in {16, 17}
        bars.append(BarRecord(
            "abaca", "fixture.mid", index, 4.0,
            form="A" if is_a else "B",
            tracks=[TrackRecord(0, "track", [NoteEvent(60 if is_a else 67, 0.0, 1.0)])],
        ))
    song = SongRecord("abaca", "fixture.mid", bars=bars)
    ActionLabeler(ActionLabelerConfig(
        theme_anchor_bars=4,
        return_min_consecutive=2,
        return_similarity_threshold=.99,
        repeat_similarity_threshold=1.1,
    )).label_song(song)
    assert [bar.action for bar in bars[16:18]] == [ACTION_RETURN, ACTION_RETURN]

    raw = FinalV2EvaluationRawCapture._form_action({}, [song])
    assert {"form": "A", "action": "RETURN", "count": 2} in raw["confusion_table"]

    public = tmp_path / "public"; public.mkdir()
    raw_path = public / "form_action_alignment__raw_observation.v2.json"
    raw_path.write_text(json.dumps({
        "schema_version": "form_action_alignment_raw_observation.v2",
        "status": "AVAILABLE",
        "run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"},
        "dataset": {"identity": "fixture", "content_sha256": None},
        **raw,
    }), encoding="utf-8")
    run = EvaluationArtifactStore.create(tmp_path, "run")
    bundle = FinalV2DiagnosticExporter("form_action_alignment").export(ExportContext("run", public, run))
    result = FinalV2DiagnosticEvaluator("form_action_alignment").evaluate(EvaluationContext("run", public, run), bundle)
    assert {"form": "A", "action": "RETURN", "count": 2} in result.report["metrics"]["observation"]["confusion_table"]
