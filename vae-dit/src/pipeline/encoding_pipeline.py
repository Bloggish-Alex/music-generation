#!/usr/bin/env python3
"""Pipeline that parses music, encodes bar tensors, and labels actions."""

from __future__ import annotations

import json
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from codec.action_labeler import ActionLabeler
from codec.bar_feature_extractor import BarFeatureExtractor, EncodedBarFeatureStore, V2_BAR_FEATURE_NAMES
from codec.bar_tensor_codec_factory import BarTensorCodecFactory
from codec.semantic_harmony_set_codec import SemanticHarmonySetCodec
from data.core import BarTensorRecord, SongRecord
from data.music_parser import MusicDirectoryParser
from diagnostics.diagnostics import DiagnosticsBase
from diagnostics.codec_fidelity_raw_capture import CodecFidelityV2RawCaptureRequest, JsonNpzCodecFidelityV2RawCapture
from diagnostics.dataset_tonality_raw_source import DatasetTonalityRawSourceRequest, JsonDatasetTonalityRawSourceWriter
from diagnostics.final_v2_evaluation_raw_capture import FinalV2EvaluationRawCapture


@dataclass
class EncodingPipelineResult:
    """All outputs produced by the encoding pipeline."""

    songs: List[SongRecord]
    tensors: List[BarTensorRecord]
    diagnostics: Dict[str, Any]


class EncodingPipeline:
    """Parse files, encode bars into tensors, label actions, and save outputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store runtime configuration and initialize encoding diagnostics."""
        self.config = config
        self.diagnostics = DiagnosticsBase("encoding")

    def run(self, music_dir: str | Path, output_dir: str | Path) -> EncodingPipelineResult:
        """Run the encoding pipeline and write artifacts to output_dir."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        parser = MusicDirectoryParser.from_config(self.config)
        songs = parser.parse_directory(music_dir)
        provenance = self._dataset_provenance(Path(music_dir))
        self.diagnostics.record_stage("input", {
            "music_dir": str(music_dir),
            "parsed_file_count": int(len(songs)),
            "failed_file_count": int(len(parser.failed_files)),
            "failed_files": parser.failed_files,
            "bar_count": int(sum(len(song.bars) for song in songs)),
        })
        labeler = ActionLabeler.from_config(self.config)
        action_diagnostics = [labeler.label_song(song) for song in songs]
        self.diagnostics.record_stage("action_labeling", {
            "song_count": int(len(action_diagnostics)),
            "songs": action_diagnostics,
            "global_action_counts": self._global_action_counts(action_diagnostics),
        })
        codec = BarTensorCodecFactory.create(self.config)
        tensors = self._encode_tensors(codec, songs)
        self.diagnostics.record_stage("bar_tensor_encoding", self._tensor_diagnostics(tensors))
        feature_summary = self._write_outputs(output_path, songs, tensors, provenance)
        if self._is_v2():
            self._capture_v2_evaluation_raw(output_path, songs, provenance)
            FinalV2EvaluationRawCapture().capture(output_path, songs, provenance or {}, parser.failed_files)
        self.diagnostics.record_stage("bar_feature_encoding", feature_summary)
        diagnostics = self.diagnostics.to_dict()
        self.diagnostics.write(output_path / "encoding_diagnostics.json")
        return EncodingPipelineResult(songs=songs, tensors=tensors, diagnostics=diagnostics)

    def _capture_v2_evaluation_raw(
        self,
        output_dir: Path,
        songs: List[SongRecord],
        provenance: Dict[str, str] | None = None,
    ) -> None:
        """Materialize the source and codec facts consumed by the shared evaluator."""
        train_ids, validation_ids = self._resolve_evaluation_splits(songs)
        groups = {
            "train": [song for song in songs if self._base_song_id(song) in train_ids],
            "validation": [song for song in songs if self._base_song_id(song) in validation_ids],
        }
        groups = {name: values for name, values in groups.items() if values}
        details = provenance or {
            "identity": "unverified",
            "identity_kind": "unverified",
            "content_sha256": "sha256:" + "0" * 64,
        }
        manifest_path = output_dir / "encoding_manifest.json"
        source_paths = JsonDatasetTonalityRawSourceWriter().write(
            DatasetTonalityRawSourceRequest(
                output_dir,
                details["identity"],
                details["identity_kind"],
                groups,
                details["content_sha256"],
                self._sha256(manifest_path) if manifest_path.is_file() else None,
            )
        )
        JsonNpzCodecFidelityV2RawCapture().capture(
            CodecFidelityV2RawCaptureRequest(
                output_dir,
                details["identity"],
                details["content_sha256"],
                source_paths,
                train_ids,
                validation_ids,
            )
        )

    def _encode_tensors(self, codec: SemanticHarmonySetCodec, songs: List[SongRecord]) -> List[BarTensorRecord]:
        """Encode every parsed bar with diagnostics."""
        records: List[BarTensorRecord] = []
        for song in songs:
            records.extend(codec.encode_song(song))
        return records

    def _write_outputs(
        self,
        output_dir: Path,
        songs: List[SongRecord],
        tensors: List[BarTensorRecord],
        provenance: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Write JSON metadata and compressed tensor artifacts."""
        (output_dir / "songs.json").write_text(
            json.dumps([song.to_dict() for song in songs], indent=2),
            encoding="utf-8",
        )
        if not self._is_v2():
            raise ValueError("Only final bar_tensor_schema.v2 output is supported; re-encode legacy data.")
        return self._write_v2_outputs(output_dir, songs, tensors, provenance)

    def _is_v2(self) -> bool:
        section = self.config.get("bar_tensor", {})
        return isinstance(section, dict) and section.get("schema_version") == "bar_tensor_schema.v2"

    def _write_v2_outputs(
        self,
        output_dir: Path,
        songs: List[SongRecord],
        tensors: List[BarTensorRecord],
        provenance: Dict[str, str] | None,
    ) -> Dict[str, Any]:
        """Write the V2 row-aligned arrays and manifest declared by the public contract."""
        song_by_id = {song.song_id: song for song in songs}
        ordered = sorted(tensors, key=lambda record: (
            str(song_by_id[record.song_id].metadata.get("base_song_id", record.song_id)),
            int(song_by_id[record.song_id].metadata.get("transpose_semitones", 0)), str(record.song_id), int(record.bar_index),
        ))
        voice_tensors = np.stack([np.asarray(record.tensor, dtype=np.float32) for record in ordered]) if ordered else np.zeros((0, 18, 48, 6), dtype=np.float32)
        slot_valid_mask = np.asarray([record.diagnostics["slot_valid_mask"] for record in ordered], dtype=bool) if ordered else np.zeros((0, 48), dtype=bool)
        slot_durations_ql = np.asarray([record.diagnostics["slot_durations_ql"] for record in ordered], dtype=np.float32) if ordered else np.zeros((0, 48), dtype=np.float32)
        bar_contexts = np.asarray([record.diagnostics["bar_context"] for record in ordered], dtype=np.float32) if ordered else np.zeros((0, 12), dtype=np.float32)
        base_pitches = np.asarray([record.diagnostics.get("base_pitch") or 0 for record in ordered], dtype=np.int16)
        base_pitch_valid = np.asarray([bool(record.diagnostics.get("base_pitch_valid")) for record in ordered], dtype=bool)
        arrays_path = output_dir / "voice_tensors.npz"
        np.savez_compressed(arrays_path, voice_tensors=voice_tensors, slot_valid_mask=slot_valid_mask, slot_durations_ql=slot_durations_ql, bar_contexts=bar_contexts, base_pitches=base_pitches, base_pitch_valid=base_pitch_valid)
        index_rows = []
        for row, record in enumerate(ordered):
            song = song_by_id[record.song_id]
            index_rows.append({"row": row, "tensor_key": f"{record.song_id}__bar_{record.bar_index:06d}", "song_id": record.song_id, "base_song_id": song.metadata.get("base_song_id", record.song_id), "source_bar_index": int(record.bar_index), "source_measure_index": song.bars[record.bar_index].source_measure_index, "applied_transpose_semitones": int(song.metadata.get("transpose_semitones", 0)), "schema_version": "bar_tensor_schema.v2", "base_pitch_valid": bool(base_pitch_valid[row]), "voice_tensor_shape": [18, 48, 6], "slot_valid_count": int(slot_valid_mask[row].sum()), "bar_context_shape": [12]})
        index_path = output_dir / "bar_tensor_index.json"
        index_path.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
        details = provenance or {"identity": "unverified", "identity_kind": "unverified", "content_sha256": None}
        config_json = json.dumps(self.config.get("bar_tensor", {}), sort_keys=True, separators=(",", ":")).encode("utf-8")
        manifest = {"schema_version": "bar_tensor_schema.v2", "row_count": len(index_rows), "arrays": {"path": arrays_path.name, "sha256": self._sha256(arrays_path), "names": {"voice_tensors": {"dtype": "float32", "shape": list(voice_tensors.shape)}, "slot_valid_mask": {"dtype": "bool", "shape": list(slot_valid_mask.shape)}, "slot_durations_ql": {"dtype": "float32", "shape": list(slot_durations_ql.shape)}, "bar_contexts": {"dtype": "float32", "shape": list(bar_contexts.shape)}, "base_pitches": {"dtype": "int16", "shape": list(base_pitches.shape)}, "base_pitch_valid": {"dtype": "bool", "shape": list(base_pitch_valid.shape)}}}, "index": {"path": index_path.name, "sha256": self._sha256(index_path)}, "voice_names": ["melody", *[f"harmony_{index:02d}" for index in range(16)], "bass"], "feature_names": ["relative_pitch", "is_rest", "is_note_on", "is_hold", "normalized_velocity", "velocity_ratio"], "configuration": self.config.get("bar_tensor", {}), "configuration_sha256": "sha256:" + hashlib.sha256(config_json).hexdigest(), "dataset_identity": details["identity"], "dataset_identity_kind": details["identity_kind"], "dataset_content_sha256": details["content_sha256"], "source_revision": str(self.config.get("source_revision", "unknown")), "lane_capacity": 16, "overflow_policy": self.config.get("bar_tensor", {}).get("overflow_policy"), "slot_grid_policy": {"quantum_ql": 0.25, "capacity": 48, "epsilon_ql": 1e-6}}
        (output_dir / "encoding_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        extractor = BarFeatureExtractor()
        v2_features = np.stack([extractor.v2_features(voice_tensors[index], bar_contexts[index], slot_valid_mask[index]) for index in range(len(index_rows))]) if index_rows else np.zeros((0, 31), dtype=np.float32)
        np.savez_compressed(output_dir / "bar_features.npz", features=v2_features)
        summary = {"source": "codec_v2", "feature_count": 31, "feature_names": V2_BAR_FEATURE_NAMES, "row_count": len(index_rows), "shape": list(v2_features.shape)}
        (output_dir / "bar_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @staticmethod
    def _sha256(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _base_song_id(song: SongRecord) -> str:
        value = song.metadata.get("base_song_id") if isinstance(song.metadata, dict) else None
        return str(value) if value else str(song.song_id).split("_T", 1)[0]

    def _resolve_evaluation_splits(self, songs: List[SongRecord]) -> tuple[frozenset[str], frozenset[str]]:
        configured = self.config.get("evaluation_splits")
        if not isinstance(configured, dict):
            raise ValueError("Codec V2 requires an explicit evaluation_splits configuration")
        observed = {self._base_song_id(song) for song in songs}
        if configured.get("strategy") == "sha256_modulo":
            modulus = int(configured.get("validation_modulus", 0))
            remainder = int(configured.get("validation_remainder", -1))
            if modulus < 2 or not 0 <= remainder < modulus:
                raise ValueError("Codec V2 sha256_modulo split requires a valid modulus and remainder")
            validation = frozenset(
                base_song_id for base_song_id in observed
                if int(hashlib.sha256(base_song_id.encode("utf-8")).hexdigest(), 16) % modulus == remainder
            )
            train = frozenset(observed - validation)
        else:
            train = frozenset(str(value) for value in configured.get("train_base_song_ids", []))
            validation = frozenset(str(value) for value in configured.get("validation_base_song_ids", []))
        if not train or not validation:
            raise ValueError("Codec V2 evaluation_splits must resolve non-empty train and validation base_song_id sets")
        if train & validation:
            raise ValueError("Codec V2 evaluation_splits overlap on base_song_id")
        if observed != train | validation:
            missing = sorted(observed - (train | validation))
            unknown = sorted((train | validation) - observed)
            raise ValueError(f"Codec V2 evaluation_splits must partition parsed base_song_id values; missing={missing}, unknown={unknown}")
        return train, validation

    @staticmethod
    def _dataset_provenance(music_dir: Path) -> Dict[str, str]:
        root = music_dir.resolve()
        if not root.is_dir():
            raise ValueError(f"music directory does not exist: {root}")
        lines: list[str] = []
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: unicodedata.normalize("NFC", item.relative_to(root).as_posix())):
            relative = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{relative}\t{digest}\n")
        content = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
        return {"identity": f"content-manifest:{content}", "identity_kind": "content_manifest_sha256", "content_sha256": f"sha256:{content}"}

    def _tensor_diagnostics(self, tensors: List[BarTensorRecord]) -> Dict[str, Any]:
        """Summarize tensor output shapes and density."""
        shapes = [tuple(record.tensor_shape) for record in tensors]
        nonzero_ratios = [
            float(np.count_nonzero(record.tensor) / max(1, int(np.prod(record.tensor.shape))))
            for record in tensors
        ]
        return {
            "bar_count": int(len(tensors)),
            "unique_shapes": [list(shape) for shape in sorted(set(shapes))],
            "mean_nonzero_ratio": float(np.mean(nonzero_ratios)) if nonzero_ratios else 0.0,
            "min_nonzero_ratio": float(np.min(nonzero_ratios)) if nonzero_ratios else 0.0,
            "max_nonzero_ratio": float(np.max(nonzero_ratios)) if nonzero_ratios else 0.0,
        }

    def _global_action_counts(self, action_diagnostics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate action counts across songs."""
        counts: Dict[str, int] = {}
        for song_diag in action_diagnostics:
            for action, count in song_diag.get("action_counts", {}).items():
                counts[str(action)] = counts.get(str(action), 0) + int(count)
        return counts
