from __future__ import annotations

import hashlib
import json
import math

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
        {"source_note_id": f"{prefix}:0", "canonical_bar_index": 0, "meter": "4/4", "raw_local_start_ql": .0, "raw_local_end_ql": 1.0, "ordinary_quantized_local_start_ql": .0, "ordinary_quantized_local_end_ql": 1.0, "final_quantized_local_start_ql": .0, "final_quantized_local_end_ql": 1.0, "quantization_repair_kind": None, "repair_slot_index": None, "projection_overlap_ql": None, "projection_endpoint_error_ql": None, "onset_residual_ql": .0, "end_residual_ql": .0},
        {"source_note_id": f"{prefix}:1", "canonical_bar_index": 1, "meter": "4/4", "raw_local_start_ql": .1, "raw_local_end_ql": 1.2, "ordinary_quantized_local_start_ql": .0, "ordinary_quantized_local_end_ql": 1.0, "final_quantized_local_start_ql": .0, "final_quantized_local_end_ql": 1.0, "quantization_repair_kind": None, "repair_slot_index": None, "projection_overlap_ql": None, "projection_endpoint_error_ql": None, "onset_residual_ql": .1, "end_residual_ql": .2},
    ]


def _audit(samples: list[dict]) -> dict:
    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {"max": max(ordered, default=0.0), "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)] if ordered else 0.0}

    by_meter = {}
    for meter in sorted({sample["meter"] for sample in samples}):
        rows = [sample for sample in samples if sample["meter"] == meter]
        onset = [sample["onset_residual_ql"] for sample in rows]
        end = [sample["end_residual_ql"] for sample in rows]
        by_meter[meter] = {
            "fragment_count": len(rows),
            "projected_fragment_count": sum(sample["quantization_repair_kind"] is not None for sample in rows),
            "nonzero_residual_count": sum(value > 1e-9 for value in onset + end),
            "onset_residual_ql": summary(onset),
            "end_residual_ql": summary(end),
        }
    projected = sum(sample["quantization_repair_kind"] is not None for sample in samples)
    return {"audit_unit": "source_note_fragment", "fragment_count": len(samples), "projected_fragment_count": projected, "projected_fragment_rate": projected / len(samples) if samples else 0.0, "by_meter": by_meter}


def _song(samples: list[dict], bars: list[BarRecord] | None = None) -> SongRecord:
    return SongRecord(
        "song", "song.mid",
        metadata={"source_file_identity": "source", "quantization_audit": _audit(samples)},
        runtime_diagnostics={"quantization_fragment_samples": samples},
        bars=bars or [BarRecord("song", "song.mid", index, 4.0, canonical_bar_index=index, time_signature="4/4") for index in range(2)],
    )


def _repaired_sample() -> dict:
    return {
        "source_note_id": "projected:0", "canonical_bar_index": 0, "meter": "4/4",
        "raw_local_start_ql": .11, "raw_local_end_ql": .12,
        "ordinary_quantized_local_start_ql": .0, "ordinary_quantized_local_end_ql": .0,
        "final_quantized_local_start_ql": .0, "final_quantized_local_end_ql": .25,
        "quantization_repair_kind": "minimum_representable_slot_projection", "repair_slot_index": 0,
        "projection_overlap_ql": .01, "projection_endpoint_error_ql": .24,
        "onset_residual_ql": .11, "end_residual_ql": .13,
    }


def test_final_v2_diagnostic_raw_schema_rejects_available_without_capture(tmp_path) -> None:
    schema_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2" / "quantization_audit__raw_observation.v2.schema.json"
    schema = json.loads(schema_path.read_text())
    payload = {"schema_version": "quantization_audit_raw_observation.v2", "status": "AVAILABLE", "run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "0" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "x", "content_sha256": None}, "availability": {"raw_capture": False, "source_boundaries": True}, "grid_policy": {"quantum_ql": .25, "epsilon_ql": 1e-6, "capacity": 48}, "by_file_meter": []}
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(payload)


def test_controls_capture_is_schema_valid_unavailable_when_raw_controls_are_missing() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    schema = json.loads((root / "contracts" / "evaluation" / "v2" / "performance_controls__raw_observation.v2.schema.json").read_text())
    song = SongRecord("song", "song.mid", metadata={"performance_controls": {"tempo_available": False, "key_available": False, "cc64_available": False, "cc64_unavailable_reason": "canonical_raw_controls_pending"}}, bars=[BarRecord("song", "song.mid", 0, 4.0, tracks=[TrackRecord(0, "track", [NoteEvent(60, 0.0, 1.0, 80)])])])
    payload = FinalV2EvaluationRawCapture._controls({"run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "fixture", "content_sha256": None}}, [song])
    assert payload["status"] == "UNAVAILABLE"
    assert payload["availability"] == {"raw_capture": True, "tempo": False, "key": False, "velocity": True, "cc64": False}
    assert payload["velocity"]["note_count"] == 1
    Draft202012Validator(schema).validate(payload)
    payload["availability"]["raw_capture"] = False
    Draft202012Validator(schema).validate(payload)


def test_controls_capture_requires_velocity_for_available_status() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    schema = json.loads((root / "contracts" / "evaluation" / "v2" / "performance_controls__raw_observation.v2.schema.json").read_text())
    song = SongRecord("song", "song.mid", metadata={"performance_controls": {"tempo_available": True, "key_available": True, "cc64_available": True}})
    payload = FinalV2EvaluationRawCapture._controls({"run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "fixture", "content_sha256": None}}, [song])
    assert payload["status"] == "UNAVAILABLE"
    assert payload["availability"]["velocity"] is False
    assert {"field": "velocity", "reason": "no_note_velocity_facts"} in payload["unavailable_reasons"]
    Draft202012Validator(schema).validate(payload)


def test_controls_capture_adds_cc64_reason_when_source_omits_it() -> None:
    song = SongRecord("song", "song.mid", metadata={"performance_controls": {"tempo_available": True, "key_available": True, "cc64_available": False}}, bars=[BarRecord("song", "song.mid", 0, 4.0, tracks=[TrackRecord(0, "track", [NoteEvent(60, 0.0, 1.0, 80)])])])
    payload = FinalV2EvaluationRawCapture._controls({}, [song])
    assert {"field": "cc64", "reason": "canonical_raw_controls_pending"} in payload["unavailable_reasons"]


def test_final_v2_diagnostic_export_and_evaluate(tmp_path) -> None:
    public = tmp_path / "public"; public.mkdir(); run = EvaluationArtifactStore.create(tmp_path, "run")
    raw = {"schema_version": "parser_integrity_raw_observation.v2", "status": "AVAILABLE", "run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "x", "content_sha256": None}, "availability": {"raw_capture": True, "measure_map": True}, "measure_map": {"song_count": 1, "measure_count": 1, "meter_distribution": {"4/4": 1}, "opus_tune_count": 0, "over_capacity_count": 0}, "track_retention": {"hard_safety_limit": 48, "policy": "retain_all", "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0}, "normalization_policy_version": "raw_pairing_normalization.v1", "repair_artifact": {"path": "raw_pairing_repairs.v1.json", "sha256": "sha256:" + "2" * 64}, "repair_count": 0, "repair_counts_by_kind": {"same_tick_zero_duration_pair": 0, "redundant_orphan_note_off": 0}, "repair_affected_file_count": 0, "partial_span_count": 0, "partial_reason_counts": {"time_signature_change": 0}, "partial_spans": [], "parser_failures": [], "unavailable_reasons": []}
    schema = json.loads((__import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2" / "parser_integrity__raw_observation.v2.schema.json").read_text())
    Draft202012Validator(schema).validate(raw)
    path = public / "parser_integrity__raw_observation.v2.json"; path.write_text(json.dumps(raw))
    exporter = FinalV2DiagnosticExporter("parser_integrity"); bundle = exporter.export(ExportContext("run", public, run))
    result = FinalV2DiagnosticEvaluator("parser_integrity").evaluate(EvaluationContext("run", public, run), bundle)
    assert result.report["status"] == "MONITOR"
    assert result.report["metrics"]["observation"]["partial_span_count"] == 0


@pytest.mark.parametrize("mutation", ["bad_hash", "capture_false", "incomplete_measure_map"])
def test_parser_integrity_schema_preserves_existing_gates(mutation) -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    schema = json.loads((root / "contracts" / "evaluation" / "v2" / "parser_integrity__raw_observation.v2.schema.json").read_text())
    raw = {"schema_version": "parser_integrity_raw_observation.v2", "status": "AVAILABLE", "run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "x", "content_sha256": None}, "availability": {"raw_capture": True, "measure_map": True}, "measure_map": {"song_count": 1, "measure_count": 1, "meter_distribution": {}, "opus_tune_count": 0, "over_capacity_count": 0}, "track_retention": {"hard_safety_limit": 48, "policy": "retain_all", "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0}, "normalization_policy_version": "raw_pairing_normalization.v1", "repair_artifact": {"path": "raw_pairing_repairs.v1.json", "sha256": "sha256:" + "2" * 64}, "repair_count": 0, "repair_counts_by_kind": {}, "repair_affected_file_count": 0, "partial_span_count": 0, "partial_reason_counts": {"time_signature_change": 0}, "partial_spans": [], "parser_failures": []}
    if mutation == "bad_hash": raw["run"]["encoding_manifest_sha256"] = "bad"
    elif mutation == "capture_false": raw["availability"]["raw_capture"] = False
    else: del raw["measure_map"]["measure_count"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(raw)


def test_unavailable_final_v2_raw_observations_remain_schema_valid(tmp_path) -> None:
    paths = FinalV2EvaluationRawCapture().capture(tmp_path, [], {"identity": "fixture", "content_sha256": None}, [])
    root = __import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2"
    for module, path in paths.items():
        Draft202012Validator(json.loads((root / f"{module}__raw_observation.v2.schema.json").read_text())).validate(json.loads(path.read_text()))


def test_parser_integrity_rejects_tampered_raw_pairing_repair_artifact(tmp_path) -> None:
    repair = {"repair_kind": "same_tick_zero_duration_pair", "source_file_identity": "a" * 64, "dataset_relative_posix_path": "a.mid", "tune_index": 0, "physical_track_index": 0, "channel": 0, "pitch": 60, "queue_depth_before": 1, "same_tick_events": [], "smf_format": 1, "ppqn": 480, "on_tick": 10, "on_event_ordinal": 1, "on_velocity": 80, "off_tick": 10, "off_event_ordinal": 2}
    artifact = tmp_path / "raw_pairing_repairs.v1.json"
    artifact.write_text(json.dumps({"schema_version": "raw_pairing_repairs.v1", "normalization_policy_version": "raw_pairing_normalization.v1", "repairs": [repair]}), encoding="utf-8")
    manifest = {"normalization_policy_version": "raw_pairing_normalization.v1", "repair_artifact": {"path": artifact.name, "sha256": _digest(artifact)}, "repair_count": 1, "repair_counts_by_kind": {"same_tick_zero_duration_pair": 1, "redundant_orphan_note_off": 0}, "repair_affected_file_count": 1}
    common = {"dataset": {"identity": "fixture", "content_sha256": None}}
    song = SongRecord("song", "a.mid", metadata={"source_file_identity": "a" * 64, "raw_pairing_repairs": [repair]})
    payload = FinalV2EvaluationRawCapture._parser_integrity(common, [song], [], manifest, tmp_path)
    assert payload["repair_count"] == 1
    repair["off_tick"] = 11
    artifact.write_text(json.dumps({"schema_version": "raw_pairing_repairs.v1", "normalization_policy_version": "raw_pairing_normalization.v1", "repairs": [repair]}), encoding="utf-8")
    with pytest.raises(ValueError, match="repair provenance|repair artifact"):
        FinalV2EvaluationRawCapture._parser_integrity(common, [song], [], manifest, tmp_path)


@pytest.mark.parametrize("tamper", ["index_length", "index_reason", "slot_duration", "arrays_hash", "raw_end", "trigger_tick"])
def test_parser_integrity_rejects_partial_index_raw_or_slot_tampering(tmp_path, tamper) -> None:
    repair_path = tmp_path / "raw_pairing_repairs.v1.json"
    repair_path.write_text(json.dumps({"schema_version": "raw_pairing_repairs.v1", "normalization_policy_version": "raw_pairing_normalization.v1", "repairs": []}), encoding="utf-8")
    bar = BarRecord("song", "song.mid", 0, 2.0, time_signature="3/4", canonical_bar_index=0, is_partial=True, partial_reason="time_signature_change", nominal_meter="3/4", triggering_ts_tick=960, triggering_ts_provenance={"physical_track_index": 0, "event_ordinal": 2, "numerator": 4, "denominator": 4}, canonical_start_tick=0, canonical_end_tick=960, ppqn=480)
    following = BarRecord("song", "song.mid", 1, 4.0, time_signature="4/4", canonical_bar_index=1, nominal_meter="4/4", canonical_start_tick=960, canonical_end_tick=2880, ppqn=480)
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "a" * 64, "ppqn": 480, "raw_pairing_repairs": []}, bars=[bar, following])
    rows = [{"row": 0, "song_id": "song", "canonical_bar_index": 0, "canonical_start_tick": 0, "canonical_end_tick": 960, "ppqn": 480, "is_partial": True, "partial_reason": "time_signature_change", "actual_bar_length_ql": 2.0, "nominal_meter": "3/4", "triggering_ts_tick": 960}, {"row": 1, "song_id": "song", "canonical_bar_index": 1, "canonical_start_tick": 960, "canonical_end_tick": 2880, "ppqn": 480, "is_partial": False, "partial_reason": None, "actual_bar_length_ql": 4.0, "nominal_meter": "4/4", "triggering_ts_tick": None}]
    masks = np.zeros((2, 48), dtype=bool); masks[0, :8] = True; masks[1, :16] = True
    durations = np.zeros((2, 48), dtype=np.float32); durations[0, :8] = .25; durations[1, :16] = .25
    if tamper == "index_length": rows[0]["actual_bar_length_ql"] = 2.25
    if tamper == "index_reason": rows[0]["partial_reason"] = None
    if tamper == "slot_duration": durations[0, 7] = .5
    if tamper == "raw_end": bar.canonical_end_tick = 1080
    if tamper == "trigger_tick": bar.triggering_ts_tick = 961
    index_path = tmp_path / "bar_tensor_index.json"; index_path.write_text(json.dumps(rows), encoding="utf-8")
    arrays_path = tmp_path / "voice_tensors.npz"; np.savez_compressed(arrays_path, slot_valid_mask=masks, slot_durations_ql=durations)
    manifest = {"normalization_policy_version": "raw_pairing_normalization.v1", "repair_artifact": {"path": repair_path.name, "sha256": _digest(repair_path)}, "repair_count": 0, "repair_counts_by_kind": {"same_tick_zero_duration_pair": 0, "redundant_orphan_note_off": 0}, "repair_affected_file_count": 0, "index": {"path": index_path.name}, "arrays": {"path": arrays_path.name, "sha256": _digest(arrays_path)}}
    if tamper == "arrays_hash": manifest["arrays"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="canonical partial"):
        FinalV2EvaluationRawCapture._parser_integrity({"dataset": {"identity": "fixture"}}, [song], [], manifest, tmp_path)


def test_quantization_audit_merges_same_opus_source_and_meter(tmp_path) -> None:
    first, second = _samples("first"), _samples("second")
    bars = [BarRecord("opus", "opus.mid", index, 4.0, canonical_bar_index=index, time_signature="4/4") for index in range(2)]
    songs = [SongRecord("opus__tune_000", "opus.mid", metadata={"source_file_identity": "same", "tune_index": 0, "quantization_audit": _audit(first)}, runtime_diagnostics={"quantization_fragment_samples": first}, bars=bars), SongRecord("opus__tune_001", "opus.mid", metadata={"source_file_identity": "same", "tune_index": 1, "quantization_audit": _audit(second)}, runtime_diagnostics={"quantization_fragment_samples": second}, bars=bars)]
    common = {"run": {"encoding_manifest_sha256": "sha256:" + "0" * 64, "bar_tensor_index_sha256": "sha256:" + "1" * 64, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": "fixture", "content_sha256": None}}
    payload = FinalV2EvaluationRawCapture._quantization(common, songs, tmp_path)
    assert payload["audit_unit"] == "source_note_fragment"
    assert payload["fragment_count"] == 4
    assert len(payload["by_file_meter"]) == 1
    archive_path = tmp_path / payload["residual_samples"]["path"]
    assert payload["residual_samples"]["sha256"] == _digest(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        assert {"source_note_ids", "canonical_bar_indexes", "raw_local_start_ql", "ordinary_quantized_local_end_ql", "final_quantized_local_end_ql", "quantization_repair_kinds"} <= set(archive.files)
        assert archive["group_offsets"].dtype == np.dtype("int64")
        assert archive["group_offsets"].tolist() == [0, 4]
        assert archive["sample_tune_indexes"].tolist() == [0, 0, 1, 1]
        timing_arrays = {
            "raw_local_start_ql", "raw_local_end_ql",
            "ordinary_quantized_local_start_ql", "ordinary_quantized_local_end_ql",
            "final_quantized_local_start_ql", "final_quantized_local_end_ql",
            "projection_overlap_ql", "projection_endpoint_error_ql",
            "onset_residuals_ql", "end_residuals_ql",
        }
        assert all(archive[name].dtype == np.dtype("float64") for name in timing_arrays)
        assert all(payload["residual_samples"]["arrays"][name]["dtype"] == "float64" for name in timing_arrays)
    schema_path = __import__("pathlib").Path(__file__).resolve().parents[2] / "contracts" / "evaluation" / "v2" / "quantization_audit__raw_observation.v2.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text())).validate(payload)


def test_quantization_archive_preserves_residual_recomputation_precision(tmp_path) -> None:
    sample = _samples()[0]
    sample.update({
        "raw_local_end_ql": 1.4979166666666666,
        "ordinary_quantized_local_end_ql": 1.5,
        "final_quantized_local_end_ql": 1.5,
        "end_residual_ql": 0.002083333333333437,
    })
    bar = BarRecord("song", "song.mid", 0, 4.0, canonical_bar_index=0, time_signature="4/4")
    payload = FinalV2EvaluationRawCapture._quantization({}, [_song([sample], [bar])], tmp_path)
    archive_path = tmp_path / payload["residual_samples"]["path"]
    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["raw_local_end_ql"].dtype == np.dtype("float64")
        recomputed = abs(archive["final_quantized_local_end_ql"][0] - archive["raw_local_end_ql"][0])
        assert recomputed == pytest.approx(archive["end_residuals_ql"][0], abs=1e-15)


def test_quantization_archive_rejects_float32_timing_arrays(tmp_path) -> None:
    payload = FinalV2EvaluationRawCapture._quantization({}, [_song(_samples())], tmp_path)
    archive_path = tmp_path / payload["residual_samples"]["path"]
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["raw_local_start_ql"] = arrays["raw_local_start_ql"].astype(np.float32)
    invalid_path = tmp_path / "float32_quantization_residual_samples.v2.npz"
    np.savez_compressed(invalid_path, **arrays)
    descriptors = dict(payload["residual_samples"]["arrays"])
    descriptors["raw_local_start_ql"] = {"dtype": "float32", "shape": list(arrays["raw_local_start_ql"].shape)}
    with pytest.raises(ValueError, match="residual arrays"):
        FinalV2EvaluationRawCapture._validate_residual_archive(invalid_path, descriptors)


@pytest.mark.parametrize("runtime", [{}, {"quantization_residual_samples": {"4/4": {"onset_residual_samples_ql": [.0], "end_residual_samples_ql": [.0]}}}])
def test_quantization_audit_rejects_missing_or_legacy_runtime_samples(tmp_path, runtime) -> None:
    samples = _samples()
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "source", "quantization_audit": _audit(samples)}, runtime_diagnostics=runtime)
    with pytest.raises(ValueError, match="quantization fragment samples"):
        FinalV2EvaluationRawCapture._quantization({}, [song], tmp_path)


def test_quantization_audit_rejects_summary_statistic_mismatch(tmp_path) -> None:
    samples = _samples(); audit = _audit(samples); audit["by_meter"]["4/4"]["onset_residual_ql"]["max"] = .09
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "source", "quantization_audit": audit}, runtime_diagnostics={"quantization_fragment_samples": samples}, bars=[BarRecord("song", "song.mid", index, 4.0, canonical_bar_index=index, time_signature="4/4") for index in range(2)])
    with pytest.raises(ValueError, match="quantization fragment samples disagree with summary"):
        FinalV2EvaluationRawCapture._quantization({}, [song], tmp_path)


def test_quantization_audit_accepts_partial_final_slot_for_5_32_bar(tmp_path) -> None:
    sample = {
        "source_note_id": "partial:0", "canonical_bar_index": 0, "meter": "5/32",
        "raw_local_start_ql": .51, "raw_local_end_ql": .625,
        "ordinary_quantized_local_start_ql": .5, "ordinary_quantized_local_end_ql": .625,
        "final_quantized_local_start_ql": .5, "final_quantized_local_end_ql": .625,
        "quantization_repair_kind": None, "repair_slot_index": None,
        "projection_overlap_ql": None, "projection_endpoint_error_ql": None,
        "onset_residual_ql": .01, "end_residual_ql": .0,
    }
    bar = BarRecord("song", "song.mid", 0, .625, canonical_bar_index=0, time_signature="5/32")
    payload = FinalV2EvaluationRawCapture._quantization({}, [_song([sample], [bar])], tmp_path)
    assert payload["fragment_count"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("repair_slot_index", 1),
        ("projection_overlap_ql", .02),
        ("projection_endpoint_error_ql", .23),
        ("final_quantized_local_end_ql", .5),
    ],
)
def test_quantization_audit_rejects_forged_projection_facts(tmp_path, field, value) -> None:
    sample = _repaired_sample(); sample[field] = value
    with pytest.raises(ValueError, match="sample values|repair facts"):
        FinalV2EvaluationRawCapture._quantization({}, [_song([sample])], tmp_path)


def test_quantization_audit_rejects_collapsed_fragment_without_repair(tmp_path) -> None:
    sample = _repaired_sample()
    sample.update({
        "quantization_repair_kind": None,
        "repair_slot_index": None,
        "projection_overlap_ql": None,
        "projection_endpoint_error_ql": None,
        "final_quantized_local_end_ql": .0,
        "end_residual_ql": .12,
    })
    with pytest.raises(ValueError, match="sample values|repair facts"):
        FinalV2EvaluationRawCapture._quantization({}, [_song([sample])], tmp_path)


def test_quantization_audit_rejects_normal_fragment_with_forged_repair(tmp_path) -> None:
    sample = _samples()[0]
    sample.update({
        "quantization_repair_kind": "minimum_representable_slot_projection",
        "repair_slot_index": 0,
        "projection_overlap_ql": 1.0,
        "projection_endpoint_error_ql": 0.0,
    })
    with pytest.raises(ValueError, match="repair facts"):
        FinalV2EvaluationRawCapture._quantization({}, [_song([sample])], tmp_path)


@pytest.mark.parametrize("path", [("projected_fragment_count",), ("by_meter", "4/4", "projected_fragment_count")])
def test_quantization_audit_rejects_projection_summary_mismatch(tmp_path, path) -> None:
    sample = _repaired_sample(); audit = _audit([sample])
    target = audit
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = 0
    song = _song([sample]); song.metadata["quantization_audit"] = audit
    with pytest.raises(ValueError, match="projection summary"):
        FinalV2EvaluationRawCapture._quantization({}, [song], tmp_path)


@pytest.mark.parametrize(
    "array_name,value",
    [("repair_slot_indexes", 0), ("projection_overlap_ql", 0.0)],
)
def test_quantization_audit_rejects_incoherent_none_archive_repair_fields(tmp_path, array_name, value) -> None:
    payload = FinalV2EvaluationRawCapture._quantization({}, [_song(_samples())], tmp_path)
    archive_path = tmp_path / payload["residual_samples"]["path"]
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[array_name][0] = value
    invalid_path = tmp_path / "invalid_quantization_residual_samples.v2.npz"
    np.savez_compressed(invalid_path, **arrays)
    with pytest.raises(ValueError, match="repair arrays"):
        FinalV2EvaluationRawCapture._validate_residual_archive(invalid_path, payload["residual_samples"]["arrays"])


@pytest.mark.parametrize("field,value", [("meter", "3/4"), ("canonical_bar_index", 9), ("ordinary_quantized_local_start_ql", .1), ("ordinary_quantized_local_end_ql", .4), ("final_quantized_local_start_ql", .1), ("final_quantized_local_end_ql", .4), ("raw_local_end_ql", .0)])
def test_quantization_audit_rejects_fragments_outside_canonical_slot_grid(tmp_path, field, value) -> None:
    samples = _samples(); samples[0][field] = value
    song = SongRecord("song", "song.mid", metadata={"source_file_identity": "source", "quantization_audit": _audit(samples)}, runtime_diagnostics={"quantization_fragment_samples": samples}, bars=[BarRecord("song", "song.mid", 0, .625, canonical_bar_index=0, time_signature="5/32"), BarRecord("song", "song.mid", 1, 4.0, canonical_bar_index=1, time_signature="4/4")])
    with pytest.raises(ValueError, match="canonical bar alignment|sample values"):
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
