"""Run-level raw observations for the final Codec V2 framework modules."""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np
from jsonschema import Draft202012Validator, ValidationError
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.core import SongRecord
from codec.slot_grid import SlotGrid


MODULES = ("parser_integrity", "quantization_audit", "performance_controls", "form_action_alignment")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class FinalV2EvaluationRawCapture:
    """Publish only observations bound to the canonical manifest and index."""

    def capture(self, output_dir: Path, songs: Sequence[SongRecord], dataset: Mapping[str, Any], parser_failures: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
        manifest_path, index_path = output_dir / "encoding_manifest.json", output_dir / "bar_tensor_index.json"
        if not manifest_path.is_file() or not index_path.is_file():
            return self._unavailable_all(output_dir, dataset, "canonical manifest or index is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "bar_tensor_schema.v2" or manifest.get("index", {}).get("sha256") != _sha256(index_path):
            return self._unavailable_all(output_dir, dataset, "canonical manifest/index alignment failed")
        run = {"encoding_manifest_sha256": _sha256(manifest_path), "bar_tensor_index_sha256": _sha256(index_path), "tensor_schema_version": "bar_tensor_schema.v2"}
        common = {"run": run, "dataset": {"identity": dataset["identity"], "content_sha256": dataset.get("content_sha256")}}
        payloads = {
            "parser_integrity": self._parser_integrity(common, songs, parser_failures, manifest, output_dir),
            "quantization_audit": self._quantization(common, songs, output_dir),
            "performance_controls": self._controls(common, songs),
            "form_action_alignment": self._form_action(common, songs),
        }
        paths = {}
        for module, payload in payloads.items():
            path = output_dir / f"{module}__raw_observation.v2.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            paths[module] = path
        return paths

    def _unavailable_all(self, output_dir: Path, dataset: Mapping[str, Any], reason: str) -> dict[str, Path]:
        paths = {}
        for module in MODULES:
            availability = {"raw_capture": False}
            if module == "parser_integrity": availability["measure_map"] = False
            elif module == "quantization_audit": availability.update({"source_boundaries": False, "residual_samples": False})
            elif module == "performance_controls": availability.update({"tempo": False, "key": False, "velocity": False, "cc64": False})
            else: availability.update({"form_metadata": False, "action_labels": False})
            path = output_dir / f"{module}__raw_observation.v2.json"
            unavailable_hash = "sha256:" + "0" * 64
            path.write_text(json.dumps({"schema_version": f"{module}_raw_observation.v2", "status": "UNAVAILABLE", "run": {"encoding_manifest_sha256": unavailable_hash, "bar_tensor_index_sha256": unavailable_hash, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": dataset["identity"], "content_sha256": dataset.get("content_sha256")}, "availability": availability, "unavailable_reasons": [{"field": "canonical_artifacts", "reason": reason}]}, indent=2) + "\n", encoding="utf-8")
            paths[module] = path
        return paths

    @staticmethod
    def _parser_integrity(common: Mapping[str, Any], songs: Sequence[SongRecord], failures: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
        bars = [bar for song in songs for bar in song.bars]
        retained = [song.metadata.get("track_retention", {}) for song in songs]
        opus_sources = {song.metadata.get("source_file_identity") for song in songs if int(song.metadata.get("opus_tune_count", 1)) > 1}
        artifact = manifest.get("repair_artifact", {})
        path = output_dir / str(artifact.get("path", ""))
        if manifest.get("normalization_policy_version") != "raw_pairing_normalization.v1" or not path.is_file() or artifact.get("sha256") != _sha256(path):
            raise ValueError("raw pairing repair provenance is invalid")
        payload = json.loads(path.read_text(encoding="utf-8"))
        repair_schema = json.loads((Path(__file__).resolve().parents[2] / "contracts" / "diagnostics" / "raw_pairing_repairs.v1.schema.json").read_text(encoding="utf-8"))
        try:
            Draft202012Validator(repair_schema).validate(payload)
        except ValidationError as error:
            raise ValueError("raw pairing repair artifact is invalid") from error
        repairs = payload.get("repairs")
        expected_repairs = [repair for song in songs for repair in song.metadata.get("raw_pairing_repairs", [])]
        counts = Counter(item.get("repair_kind") for item in repairs) if isinstance(repairs, list) else Counter()
        expected = {"same_tick_zero_duration_pair": counts["same_tick_zero_duration_pair"], "redundant_orphan_note_off": counts["redundant_orphan_note_off"]}
        dataset_identity = str(common.get("dataset", {}).get("identity", ""))
        canonical = lambda items: sorted(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in items)
        source_ids = {str(song.metadata.get("source_file_identity")) for song in songs}
        if payload.get("schema_version") != "raw_pairing_repairs.v1" or payload.get("normalization_policy_version") != manifest.get("normalization_policy_version") or not isinstance(repairs, list) or canonical(repairs) != canonical(expected_repairs) or not {item["source_file_identity"] for item in repairs} <= source_ids or manifest.get("repair_count") != len(repairs) or manifest.get("repair_counts_by_kind") != expected or manifest.get("repair_affected_file_count") != len({item["source_file_identity"] for item in repairs}) or not dataset_identity or any(item.get("repair_kind") not in expected or (item.get("repair_kind") == "same_tick_zero_duration_pair" and item.get("on_tick") != item.get("off_tick")) or (item.get("repair_kind") == "redundant_orphan_note_off" and item.get("on_tick") is not None) for item in repairs):
            raise ValueError("raw pairing repair artifact is invalid")
        partials = [{"source_file_identity": song.metadata.get("source_file_identity"), "tune_index": song.metadata.get("tune_index", 0), "canonical_bar_index": bar.canonical_bar_index, "exact_start_ql": None, "exact_end_ql": None, "actual_bar_length_ql": bar.bar_length_ql, "nominal_meter": bar.nominal_meter, "partial_reason": bar.partial_reason, "triggering_ts_tick": bar.triggering_ts_tick, "triggering_ts_provenance": bar.triggering_ts_provenance} for song in songs for bar in song.bars if bar.is_partial]
        return {"schema_version": "parser_integrity_raw_observation.v2", "status": "AVAILABLE", **common, "availability": {"raw_capture": True, "measure_map": True}, "measure_map": {"song_count": len(songs), "measure_count": len(bars), "meter_distribution": dict(Counter(bar.time_signature for bar in bars)), "opus_tune_count": len(opus_sources), "over_capacity_count": 0}, "partial_span_count": len(partials), "partial_reason_counts": {"time_signature_change": len(partials)}, "partial_spans": partials, "track_retention": {"hard_safety_limit": 48, "policy": "truncate" if any(item.get("policy") == "truncate" for item in retained) else "retain_all", "dropped_part_count": sum(int(item.get("dropped_part_count", 0)) for item in retained), "dropped_note_count": sum(int(item.get("dropped_note_count", 0)) for item in retained), "dropped_note_ratio": float(sum(float(item.get("dropped_note_ratio", 0.0)) for item in retained) / max(1, len(retained)))}, "normalization_policy_version": manifest["normalization_policy_version"], "repair_artifact": artifact, "repair_count": len(repairs), "repair_counts_by_kind": expected, "repair_affected_file_count": len({item["source_file_identity"] for item in repairs}), "parser_failures": list(failures), "unavailable_reasons": []}

    @staticmethod
    def _quantization(common: Mapping[str, Any], songs: Sequence[SongRecord], output_dir: Path) -> dict[str, Any]:
        """Capture only complete local-fragment timing facts; no legacy fallback."""
        def residual(values: list[float]) -> dict[str, float]:
            ordered = sorted(values)
            return {"max": max(ordered, default=0.0), "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)] if ordered else 0.0}

        required_sample = {
            "source_note_id", "canonical_bar_index", "meter",
            "raw_local_start_ql", "raw_local_end_ql",
            "ordinary_quantized_local_start_ql", "ordinary_quantized_local_end_ql",
            "final_quantized_local_start_ql", "final_quantized_local_end_ql",
            "quantization_repair_kind", "repair_slot_index", "projection_overlap_ql", "projection_endpoint_error_ql",
            "onset_residual_ql", "end_residual_ql",
        }
        grouped: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = {}
        for song in songs:
            audit = song.metadata.get("quantization_audit", {})
            samples = song.runtime_diagnostics.get("quantization_fragment_samples")
            if audit.get("audit_unit") != "source_note_fragment" or not isinstance(samples, list):
                raise ValueError(f"quantization fragment samples are unavailable for {song.song_id}")
            if int(audit.get("fragment_count", -1)) != len(samples):
                raise ValueError(f"quantization fragment samples disagree with summary for {song.song_id}")
            per_meter: dict[str, list[Mapping[str, Any]]] = {}
            identities: set[tuple[str, int]] = set()
            bars: dict[int, Any] = {}
            for bar in song.bars:
                index = int(bar.canonical_bar_index if bar.canonical_bar_index is not None else bar.bar_index)
                if index in bars:
                    raise ValueError(f"canonical bar index is duplicated for {song.song_id}")
                bars[index] = bar
            for sample in samples:
                if not isinstance(sample, Mapping) or set(sample) != required_sample:
                    raise ValueError(f"quantization fragment sample has an invalid shape for {song.song_id}")
                note_id, bar_index, meter = sample["source_note_id"], sample["canonical_bar_index"], sample["meter"]
                if not isinstance(note_id, str) or not note_id or not isinstance(bar_index, int) or bar_index < 0 or not isinstance(meter, str) or "/" not in meter:
                    raise ValueError(f"quantization fragment sample identity is invalid for {song.song_id}")
                identity = (note_id, bar_index)
                if identity in identities:
                    raise ValueError(f"quantization fragment identity is duplicated for {song.song_id}")
                identities.add(identity)
                bar = bars.get(bar_index)
                if bar is None or meter != str(bar.time_signature):
                    raise ValueError(f"quantization fragment canonical bar alignment is invalid for {song.song_id}")
                try:
                    raw_start, raw_end, ordinary_start, ordinary_end, quantized_start, quantized_end, onset_error, end_error = (float(sample[name]) for name in ("raw_local_start_ql", "raw_local_end_ql", "ordinary_quantized_local_start_ql", "ordinary_quantized_local_end_ql", "final_quantized_local_start_ql", "final_quantized_local_end_ql", "onset_residual_ql", "end_residual_ql"))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"quantization fragment sample is non-numeric for {song.song_id}") from error
                grid = SlotGrid.for_bar(float(bar.bar_length_ql))
                starts = [grid.interval(slot)[0] for slot in range(grid.valid_slot_count)]
                ends = [*starts, float(bar.bar_length_ql)]
                epsilon = 1e-6
                if (not all(math.isfinite(value) and value >= 0.0 for value in (raw_start, raw_end, quantized_start, quantized_end, onset_error, end_error)) or raw_end <= raw_start or raw_end > float(bar.bar_length_ql) + epsilon or quantized_end <= quantized_start or not any(math.isclose(quantized_start, value, abs_tol=epsilon) for value in starts) or not any(math.isclose(quantized_end, value, abs_tol=epsilon) for value in ends) or not math.isclose(onset_error, abs(quantized_start - raw_start), abs_tol=1e-9) or not math.isclose(end_error, abs(quantized_end - raw_end), abs_tol=1e-9)):
                    raise ValueError(f"quantization fragment sample values are invalid for {song.song_id}")
                repaired = sample["quantization_repair_kind"] == "minimum_representable_slot_projection"
                ordinary_start_valid = any(math.isclose(ordinary_start, value, abs_tol=epsilon) for value in starts)
                ordinary_end_valid = any(math.isclose(ordinary_end, value, abs_tol=epsilon) for value in ends)
                if not ordinary_start_valid or not ordinary_end_valid:
                    raise ValueError(f"quantization fragment ordinary boundaries are invalid for {song.song_id}")
                if sample["quantization_repair_kind"] not in {None, "minimum_representable_slot_projection"}:
                    raise ValueError(f"quantization fragment repair facts are invalid for {song.song_id}")
                if not repaired:
                    if ordinary_end <= ordinary_start or not math.isclose(quantized_start, ordinary_start, abs_tol=epsilon) or not math.isclose(quantized_end, ordinary_end, abs_tol=epsilon) or any(sample[name] is not None for name in ("repair_slot_index", "projection_overlap_ql", "projection_endpoint_error_ql")):
                        raise ValueError(f"quantization fragment repair facts are invalid for {song.song_id}")
                else:
                    slot = sample["repair_slot_index"]
                    if ordinary_end > ordinary_start or not isinstance(slot, int) or not 0 <= slot < grid.valid_slot_count:
                        raise ValueError(f"quantization fragment repair facts are invalid for {song.song_id}")
                    candidates = []
                    for index, start in enumerate(starts):
                        end = ends[index + 1]
                        overlap = max(0.0, min(raw_end, end) - max(raw_start, start))
                        error = abs(start - raw_start) + abs(end - raw_end)
                        candidates.append((overlap, error, index, start, end))
                    overlap, error, expected_slot, expected_start, expected_end = max(candidates, key=lambda item: (item[0], -item[1], item[2]))
                    if overlap <= 0 or slot != expected_slot or not math.isclose(quantized_start, expected_start, abs_tol=epsilon) or not math.isclose(quantized_end, expected_end, abs_tol=epsilon) or not math.isclose(float(sample["projection_overlap_ql"]), overlap, abs_tol=epsilon) or not math.isclose(float(sample["projection_endpoint_error_ql"]), error, abs_tol=epsilon):
                        raise ValueError(f"quantization fragment repair facts are invalid for {song.song_id}")
                per_meter.setdefault(meter, []).append(sample)
            for meter, summary in audit.get("by_meter", {}).items():
                meter_samples = per_meter.get(meter, [])
                onset_values = [float(sample["onset_residual_ql"]) for sample in meter_samples]
                end_values = [float(sample["end_residual_ql"]) for sample in meter_samples]
                if int(summary.get("fragment_count", summary.get("event_count", -1))) != len(meter_samples) or int(summary.get("nonzero_residual_count", -1)) != sum(value > 1e-9 for value in onset_values + end_values):
                    raise ValueError(f"quantization fragment samples disagree with summary for {song.song_id}/{meter}")
                projected_count = sum(sample["quantization_repair_kind"] is not None for sample in meter_samples)
                if int(summary.get("projected_fragment_count", -1)) != projected_count:
                    raise ValueError(f"quantization fragment projection summary disagrees for {song.song_id}/{meter}")
                for field, values in (("onset_residual_ql", onset_values), ("end_residual_ql", end_values)):
                    declared = summary.get(field)
                    if not isinstance(declared, Mapping) or any(not math.isclose(float(declared.get(statistic, float("nan"))), residual(values)[statistic], abs_tol=1e-9) for statistic in ("max", "p95")):
                        raise ValueError(f"quantization fragment samples disagree with summary for {song.song_id}/{meter}")
            if set(per_meter) != set(audit.get("by_meter", {})):
                raise ValueError(f"quantization fragment samples disagree with summary for {song.song_id}")
            projected_count = sum(sample["quantization_repair_kind"] is not None for sample in samples)
            if int(audit.get("projected_fragment_count", -1)) != projected_count or not math.isclose(float(audit.get("projected_fragment_rate", float("nan"))), projected_count / len(samples) if samples else 0.0, abs_tol=1e-9):
                raise ValueError(f"quantization fragment projection summary disagrees for {song.song_id}")
            source = str(song.metadata.get("source_file_identity", song.song_id))
            tune_index = int(song.metadata.get("tune_index", 0))
            for meter, meter_samples in per_meter.items():
                grouped.setdefault((source, meter), []).extend((tune_index, sample) for sample in meter_samples)
        keys = sorted(grouped); offsets = [0]; flat: list[Mapping[str, Any]] = []
        for key in keys:
            rows = sorted(grouped[key], key=lambda item: (item[0], int(item[1]["canonical_bar_index"]), str(item[1]["source_note_id"])))
            flat.extend(rows)
            offsets.append(len(flat))
        path = output_dir / "quantization_residual_samples.v2.npz"
        archive = {
            "source_file_identities": np.asarray([key[0] for key in keys], dtype=np.str_),
            "meters": np.asarray([key[1] for key in keys], dtype=np.str_),
            "group_offsets": np.asarray(offsets, dtype=np.int64),
            "sample_tune_indexes": np.asarray([item[0] for item in flat], dtype=np.int64),
            "source_note_ids": np.asarray([item[1]["source_note_id"] for item in flat], dtype=np.str_),
            "canonical_bar_indexes": np.asarray([item[1]["canonical_bar_index"] for item in flat], dtype=np.int64),
            "raw_local_start_ql": np.asarray([item[1]["raw_local_start_ql"] for item in flat], dtype=np.float32),
            "raw_local_end_ql": np.asarray([item[1]["raw_local_end_ql"] for item in flat], dtype=np.float32),
            "ordinary_quantized_local_start_ql": np.asarray([item[1]["ordinary_quantized_local_start_ql"] for item in flat], dtype=np.float32),
            "ordinary_quantized_local_end_ql": np.asarray([item[1]["ordinary_quantized_local_end_ql"] for item in flat], dtype=np.float32),
            "final_quantized_local_start_ql": np.asarray([item[1]["final_quantized_local_start_ql"] for item in flat], dtype=np.float32),
            "final_quantized_local_end_ql": np.asarray([item[1]["final_quantized_local_end_ql"] for item in flat], dtype=np.float32),
            "quantization_repair_kinds": np.asarray([item[1]["quantization_repair_kind"] or "none" for item in flat], dtype=np.str_),
            "repair_slot_indexes": np.asarray([item[1]["repair_slot_index"] if item[1]["repair_slot_index"] is not None else -1 for item in flat], dtype=np.int64),
            "projection_overlap_ql": np.asarray([item[1]["projection_overlap_ql"] if item[1]["projection_overlap_ql"] is not None else np.nan for item in flat], dtype=np.float32),
            "projection_endpoint_error_ql": np.asarray([item[1]["projection_endpoint_error_ql"] if item[1]["projection_endpoint_error_ql"] is not None else np.nan for item in flat], dtype=np.float32),
            "onset_residuals_ql": np.asarray([item[1]["onset_residual_ql"] for item in flat], dtype=np.float32),
            "end_residuals_ql": np.asarray([item[1]["end_residual_ql"] for item in flat], dtype=np.float32),
        }
        np.savez_compressed(path, **archive)
        arrays = {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in archive.items()}
        FinalV2EvaluationRawCapture._validate_residual_archive(path, arrays)
        rows = []
        for source, meter in keys:
            values = [item[1] for item in grouped[(source, meter)]]
            onset_values = [float(item["onset_residual_ql"]) for item in values]
            end_values = [float(item["end_residual_ql"]) for item in values]
            rows.append({"source_file_identity": source, "meter": meter, "fragment_count": len(values), "projected_fragment_count": sum(item["quantization_repair_kind"] is not None for item in values), "nonzero_residual_count": sum(value > 1e-9 for value in onset_values + end_values), "onset_residual_ql": residual(onset_values), "end_residual_ql": residual(end_values)})
        projected = sum(item[1]["quantization_repair_kind"] is not None for item in flat)
        return {"schema_version": "quantization_audit_raw_observation.v2", "status": "AVAILABLE", **common, "availability": {"raw_capture": True, "source_boundaries": True, "residual_samples": True}, "audit_unit": "source_note_fragment", "fragment_count": len(flat), "projected_fragment_count": projected, "projected_fragment_rate": projected / len(flat) if flat else 0.0, "grid_policy": {"quantum_ql": .25, "epsilon_ql": 1e-6, "capacity": 48}, "by_file_meter": rows, "residual_samples":{"path":path.name,"sha256":_sha256(path),"arrays":arrays}, "unavailable_reasons": []}

    @staticmethod
    def _validate_residual_archive(path: Path, declared_arrays: Mapping[str, Mapping[str, Any]]) -> None:
        """Reject a non-canonical residual archive before publishing AVAILABLE."""
        required = {"source_file_identities", "meters", "group_offsets", "sample_tune_indexes", "source_note_ids", "canonical_bar_indexes", "raw_local_start_ql", "raw_local_end_ql", "ordinary_quantized_local_start_ql", "ordinary_quantized_local_end_ql", "final_quantized_local_start_ql", "final_quantized_local_end_ql", "quantization_repair_kinds", "repair_slot_indexes", "projection_overlap_ql", "projection_endpoint_error_ql", "onset_residuals_ql", "end_residuals_ql"}
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError("quantization residual archive has an invalid array set")
            arrays = {name: archive[name] for name in required}
        for name, value in arrays.items():
            descriptor = declared_arrays.get(name, {})
            if descriptor.get("dtype") != str(value.dtype) or descriptor.get("shape") != list(value.shape):
                raise ValueError(f"quantization residual archive descriptor mismatch: {name}")
        source_ids, meters = arrays["source_file_identities"], arrays["meters"]
        offsets = arrays["group_offsets"]
        onset, end = arrays["onset_residuals_ql"], arrays["end_residuals_ql"]
        if source_ids.ndim != 1 or meters.ndim != 1 or source_ids.shape != meters.shape:
            raise ValueError("quantization residual archive group dimensions are invalid")
        if source_ids.dtype.kind not in {"U", "S"} or meters.dtype.kind not in {"U", "S"}:
            raise ValueError("quantization residual archive string arrays are invalid")
        if offsets.dtype != np.dtype("int64") or offsets.ndim != 1 or len(offsets) != len(source_ids) + 1:
            raise ValueError("quantization residual archive offsets are invalid")
        sample_arrays = (arrays["sample_tune_indexes"], arrays["source_note_ids"], arrays["canonical_bar_indexes"], arrays["raw_local_start_ql"], arrays["raw_local_end_ql"], arrays["ordinary_quantized_local_start_ql"], arrays["ordinary_quantized_local_end_ql"], arrays["final_quantized_local_start_ql"], arrays["final_quantized_local_end_ql"], onset, end)
        if any(value.ndim != 1 or len(value) != len(onset) for value in sample_arrays) or arrays["sample_tune_indexes"].dtype != np.dtype("int64") or np.any(arrays["sample_tune_indexes"] < 0) or arrays["source_note_ids"].dtype.kind not in {"U", "S"} or arrays["canonical_bar_indexes"].dtype != np.dtype("int64") or any(value.dtype != np.dtype("float32") for value in sample_arrays[3:]):
            raise ValueError("quantization residual archive residual arrays are invalid")
        if int(offsets[0]) != 0 or int(offsets[-1]) != len(onset) or np.any(np.diff(offsets) < 0):
            raise ValueError("quantization residual archive offsets do not align")
        repair_kinds = arrays["quantization_repair_kinds"]
        repair_slots = arrays["repair_slot_indexes"]
        overlaps = arrays["projection_overlap_ql"]
        errors = arrays["projection_endpoint_error_ql"]
        if any(value.ndim != 1 or len(value) != len(onset) for value in (repair_kinds, repair_slots, overlaps, errors)) or repair_kinds.dtype.kind not in {"U", "S"} or repair_slots.dtype != np.dtype("int64") or overlaps.dtype != np.dtype("float32") or errors.dtype != np.dtype("float32"):
            raise ValueError("quantization residual archive repair arrays are invalid")
        for kind, slot, overlap, error in zip(repair_kinds.tolist(), repair_slots.tolist(), overlaps.tolist(), errors.tolist()):
            if kind == "none":
                if slot != -1 or not math.isnan(overlap) or not math.isnan(error):
                    raise ValueError("quantization residual archive repair arrays are invalid")
            elif kind != "minimum_representable_slot_projection" or slot < 0 or not math.isfinite(overlap) or not math.isfinite(error) or overlap < 0 or error < 0:
                raise ValueError("quantization residual archive repair arrays are invalid")
        if not all(np.isfinite(value).all() for value in sample_arrays[3:]) or any(np.any(value < 0.0) for value in sample_arrays[3:]) or np.any(arrays["raw_local_end_ql"] <= arrays["raw_local_start_ql"]) or np.any(arrays["final_quantized_local_end_ql"] <= arrays["final_quantized_local_start_ql"]) or not np.allclose(onset, np.abs(arrays["final_quantized_local_start_ql"] - arrays["raw_local_start_ql"])) or not np.allclose(end, np.abs(arrays["final_quantized_local_end_ql"] - arrays["raw_local_end_ql"])):
            raise ValueError("quantization residual archive values are invalid")

    @staticmethod
    def _controls(common: Mapping[str, Any], songs: Sequence[SongRecord]) -> dict[str, Any]:
        facts = [song.metadata.get("performance_controls", {}) for song in songs]
        velocities = [note.velocity for song in songs for bar in song.bars for track in bar.tracks for note in track.notes]
        tempo_available = all(bool(item.get("tempo_available")) for item in facts)
        key_available = all(bool(item.get("key_available")) for item in facts)
        cc64_available = all(bool(item.get("cc64_available")) for item in facts)
        velocity_available = bool(velocities)
        reasons = [str(item.get("cc64_unavailable_reason")) for item in facts if item.get("cc64_unavailable_reason")]
        if not cc64_available and not reasons:
            reasons = ["canonical_raw_controls_pending"]
        unavailable_reasons = ([{"field": "tempo", "reason": "canonical_raw_controls_pending"}] if not tempo_available else []) + ([{"field": "key", "reason": "canonical_raw_controls_pending"}] if not key_available else []) + ([{"field": "velocity", "reason": "no_note_velocity_facts"}] if not velocity_available else []) + ([{"field": "cc64", "reason": reason} for reason in reasons] if not cc64_available else [])
        payload = {"schema_version": "performance_controls_raw_observation.v2", "status": "AVAILABLE" if tempo_available and key_available and velocity_available and cc64_available else "UNAVAILABLE", **common, "availability": {"raw_capture": True, "tempo": tempo_available, "key": key_available, "velocity": velocity_available, "cc64": cc64_available}, "unavailable_reasons": unavailable_reasons}
        if velocities:
            payload["velocity"] = {"note_count": len(velocities), "mean": float(sum(velocities) / len(velocities))}
        if tempo_available:
            payload["tempo"] = {"song_count": len(songs)}
        if key_available:
            payload["key"] = {"song_count": len(songs)}
        if cc64_available:
            payload["cc64"] = {"available": True, "format_coverage": {"song_count": len(songs), "available_song_count": len(songs)}, "unavailable_reasons": []}
        return payload

    @staticmethod
    def _form_action(common: Mapping[str, Any], songs: Sequence[SongRecord]) -> dict[str, Any]:
        bars = [bar for song in songs for bar in song.bars]
        table = Counter((bar.form or "UNLABELED", bar.action or "UNLABELED") for bar in bars)
        return {"schema_version": "form_action_alignment_raw_observation.v2", "status": "AVAILABLE", **common, "availability": {"raw_capture": True, "form_metadata": any(bar.form for bar in bars), "action_labels": any(bar.action for bar in bars)}, "coverage": {"bar_count": len(bars), "form_labeled_bar_count": sum(bool(bar.form) for bar in bars), "action_labeled_bar_count": sum(bool(bar.action) for bar in bars), "joint_labeled_bar_count": sum(bool(bar.form and bar.action) for bar in bars)}, "confusion_table": [{"form": form, "action": action, "count": count} for (form, action), count in sorted(table.items())], "unavailable_reasons": []}
