"""Run-level raw observations for the final Codec V2 framework modules."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.core import SongRecord


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
            "parser_integrity": self._parser_integrity(common, songs, parser_failures),
            "quantization_audit": self._quantization(common, songs),
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
            elif module == "quantization_audit": availability["source_boundaries"] = False
            elif module == "performance_controls": availability.update({"tempo": False, "key": False, "velocity": False, "cc64": False})
            else: availability.update({"form_metadata": False, "action_labels": False})
            path = output_dir / f"{module}__raw_observation.v2.json"
            unavailable_hash = "sha256:" + "0" * 64
            path.write_text(json.dumps({"schema_version": f"{module}_raw_observation.v2", "status": "UNAVAILABLE", "run": {"encoding_manifest_sha256": unavailable_hash, "bar_tensor_index_sha256": unavailable_hash, "tensor_schema_version": "bar_tensor_schema.v2"}, "dataset": {"identity": dataset["identity"], "content_sha256": dataset.get("content_sha256")}, "availability": availability, "unavailable_reasons": [{"field": "canonical_artifacts", "reason": reason}]}, indent=2) + "\n", encoding="utf-8")
            paths[module] = path
        return paths

    @staticmethod
    def _parser_integrity(common: Mapping[str, Any], songs: Sequence[SongRecord], failures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        bars = [bar for song in songs for bar in song.bars]
        retained = [song.metadata.get("track_retention", {}) for song in songs]
        opus_sources = {song.metadata.get("source_file_identity") for song in songs if int(song.metadata.get("opus_tune_count", 1)) > 1}
        return {"schema_version": "parser_integrity_raw_observation.v2", "status": "AVAILABLE", **common, "availability": {"raw_capture": True, "measure_map": True}, "measure_map": {"song_count": len(songs), "measure_count": len(bars), "meter_distribution": dict(Counter(bar.time_signature for bar in bars)), "opus_tune_count": len(opus_sources), "over_capacity_count": 0}, "track_retention": {"hard_safety_limit": 48, "policy": "truncate" if any(item.get("policy") == "truncate" for item in retained) else "retain_all", "dropped_part_count": sum(int(item.get("dropped_part_count", 0)) for item in retained), "dropped_note_count": sum(int(item.get("dropped_note_count", 0)) for item in retained), "dropped_note_ratio": float(sum(float(item.get("dropped_note_ratio", 0.0)) for item in retained) / max(1, len(retained)))}, "parser_failures": list(failures), "unavailable_reasons": []}

    @staticmethod
    def _quantization(common: Mapping[str, Any], songs: Sequence[SongRecord]) -> dict[str, Any]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for song in songs:
            audit = song.metadata.get("quantization_audit", {})
            for meter, values in audit.get("by_meter", {}).items():
                key = (str(song.metadata.get("source_file_identity", song.song_id)), str(meter))
                bucket = grouped.setdefault(key, {"onset": [], "end": []})
                bucket["onset"].extend(float(value) for value in values.get("onset_residual_samples_ql", []))
                bucket["end"].extend(float(value) for value in values.get("end_residual_samples_ql", []))
        def residual(values: list[float]) -> dict[str, float]:
            ordered = sorted(values)
            return {"max": max(ordered, default=0.0), "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)] if ordered else 0.0}
        rows = [{"source_file_identity": source, "meter": meter, "event_count": len(values["onset"]), "nonzero_residual_count": sum(value > 1e-9 for value in values["onset"] + values["end"]), "onset_residual_ql": residual(values["onset"]), "end_residual_ql": residual(values["end"])} for (source, meter), values in sorted(grouped.items())]
        return {"schema_version": "quantization_audit_raw_observation.v2", "status": "AVAILABLE", **common, "availability": {"raw_capture": True, "source_boundaries": True}, "grid_policy": {"quantum_ql": .25, "epsilon_ql": 1e-6, "capacity": 48}, "by_file_meter": rows, "unavailable_reasons": []}

    @staticmethod
    def _controls(common: Mapping[str, Any], songs: Sequence[SongRecord]) -> dict[str, Any]:
        facts = [song.metadata.get("performance_controls", {}) for song in songs]
        cc64_available = sum(bool(item.get("cc64_available")) for item in facts)
        velocities = [note.velocity for song in songs for bar in song.bars for track in bar.tracks for note in track.notes]
        return {"schema_version": "performance_controls_raw_observation.v2", "status": "AVAILABLE", **common, "availability": {"raw_capture": True, "tempo": True, "key": True, "velocity": bool(velocities), "cc64": True}, "tempo": {"song_count": len(songs)}, "key": {"song_count": len(songs)}, "velocity": {"note_count": len(velocities), "mean": float(sum(velocities) / len(velocities)) if velocities else 0.0}, "cc64": {"available": bool(cc64_available), "format_coverage": {"song_count": len(songs), "available_song_count": cc64_available}, "unavailable_reasons": [item.get("cc64_unavailable_reason") for item in facts if item.get("cc64_unavailable_reason")]}, "unavailable_reasons": []}

    @staticmethod
    def _form_action(common: Mapping[str, Any], songs: Sequence[SongRecord]) -> dict[str, Any]:
        bars = [bar for song in songs for bar in song.bars]
        table = Counter((bar.form or "UNLABELED", bar.action or "UNLABELED") for bar in bars)
        return {"schema_version": "form_action_alignment_raw_observation.v2", "status": "AVAILABLE", **common, "availability": {"raw_capture": True, "form_metadata": any(bar.form for bar in bars), "action_labels": any(bar.action for bar in bars)}, "coverage": {"bar_count": len(bars), "form_labeled_bar_count": sum(bool(bar.form) for bar in bars), "action_labeled_bar_count": sum(bool(bar.action) for bar in bars), "joint_labeled_bar_count": sum(bool(bar.form and bar.action) for bar in bars)}, "confusion_table": [{"form": form, "action": action, "count": count} for (form, action), count in sorted(table.items())], "unavailable_reasons": []}
