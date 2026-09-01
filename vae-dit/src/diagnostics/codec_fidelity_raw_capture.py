"""Diagnostics-side raw tensor observations for codec fidelity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from codec.bar_tensor_codec import FEATURE_NAMES
from codec.semantic_bar_tensor_codec import SEMANTIC_FEATURE_NAMES, SEMANTIC_TRACK_NAMES
from data.core import BarTensorRecord, SongRecord


RAW_OBSERVATION_SCHEMA_VERSION = "codec_fidelity_raw_observation.v1"
RAW_STATUS_SCHEMA_VERSION = "codec_fidelity_raw_status.v1"
RAW_OBSERVATION_V2_SCHEMA_VERSION = "codec_fidelity_raw_observation.v2"
RAW_STATUS_V2_SCHEMA_VERSION = "codec_fidelity_raw_status.v2"
_TRANSPOSE_SUFFIX = re.compile(r"_T[+-]?\d+$")


@dataclass(frozen=True)
class CodecFidelityRawCaptureRequest:
    """Existing encoding facts needed to describe split-specific codec tensors."""

    model_dir: Path
    dataset_identity: str
    dataset_identity_kind: str
    dataset_content_sha256: str | None
    source_raw_paths: Mapping[str, Path]
    songs: Sequence[SongRecord]
    tensors: Sequence[BarTensorRecord]
    train_base_song_ids: frozenset[str]
    validation_base_song_ids: frozenset[str]
    codec_config: Mapping[str, Any]


@dataclass(frozen=True)
class CodecFidelityRawCaptureResult:
    """Paths emitted for available splits and explicit reasons for unavailable ones."""

    artifacts: Mapping[str, Path]
    unavailable: Mapping[str, str]
    status_artifacts: Mapping[str, Path]


@dataclass(frozen=True)
class CodecFidelityV2RawCaptureRequest:
    """Materialized V2 encoding facts for one public fidelity capture."""

    encoded_dir: Path
    dataset_identity: str
    dataset_content_sha256: str | None
    source_raw_paths: Mapping[str, Path]
    train_base_song_ids: frozenset[str]
    validation_base_song_ids: frozenset[str]


class JsonNpzCodecFidelityV2RawCapture:
    """Capture split-specific V2 arrays without computing fidelity metrics."""

    def capture(self, request: CodecFidelityV2RawCaptureRequest) -> CodecFidelityRawCaptureResult:
        encoded_dir = Path(request.encoded_dir)
        arrays_path, index_path, manifest_path = (encoded_dir / "codec_v2_arrays.npz", encoded_dir / "bar_tensor_index.json", encoded_dir / "encoding_manifest.json")
        if not all(path.is_file() for path in (arrays_path, index_path, manifest_path)):
            return self._unavailable_all(request, "V2 canonical encoding artifacts are unavailable")
        try:
            rows = json.loads(index_path.read_text(encoding="utf-8")); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "bar_tensor_schema.v2" or not isinstance(rows, list):
                raise ValueError("V2 manifest or index schema is invalid")
            with np.load(arrays_path, allow_pickle=False) as archive:
                values = {name: np.asarray(archive[name]) for name in ("voice_tensors", "bar_contexts", "base_pitches", "base_pitch_valid")}
            count = len(rows)
            if any(array.shape[0] != count for array in values.values()) or values["voice_tensors"].shape[1:] != (18, 16, 6) or values["bar_contexts"].shape[1:] != (12,) or not np.isfinite(values["voice_tensors"]).all() or not np.isfinite(values["bar_contexts"]).all():
                raise ValueError("V2 arrays are non-finite or misaligned")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            return self._unavailable_all(request, str(error))
        groups: dict[str, list[int]] = {"train": [], "validation": [], "excluded_unpaired": []}
        for index, row in enumerate(rows):
            base = str(row.get("base_song_id", "")); split = "train" if base in request.train_base_song_ids else "validation" if base in request.validation_base_song_ids else "excluded_unpaired"
            groups[split].append(index)
        artifacts: dict[str, Path] = {}; statuses: dict[str, Path] = {}; unavailable: dict[str, str] = {}
        for split, positions in groups.items():
            status_path = encoded_dir / f"codec_fidelity__raw_status__{split}.v2.json"; observation_path = encoded_dir / f"codec_fidelity__raw_observation__{split}.v2.json"; split_arrays_path = encoded_dir / f"codec_fidelity__raw_arrays__{split}.v2.npz"
            source_ref = request.source_raw_paths.get(split)
            if not positions or source_ref is None or not source_ref.is_file():
                reason = "matching dataset_tonality raw source artifact is unavailable"; unavailable[split] = reason
                observation_path.unlink(missing_ok=True); split_arrays_path.unlink(missing_ok=True)
                self._write_json(status_path, self._status(request, split, "UNAVAILABLE", {}, reason)); statuses[split] = status_path; continue
            source_bars, reason = _source_bar_index(source_ref, split)
            if source_bars is None:
                unavailable[split] = reason
                observation_path.unlink(missing_ok=True); split_arrays_path.unlink(missing_ok=True)
                self._write_json(status_path, self._status(request, split, "UNAVAILABLE", {}, reason)); statuses[split] = status_path; continue
            alignment_error = next((
                "source raw artifact does not align with canonical V2 index"
                for position in positions
                if (str(rows[position].get("song_id")), int(rows[position].get("source_bar_index", -1))) not in source_bars
                or source_bars[(str(rows[position].get("song_id")), int(rows[position].get("source_bar_index", -1)))]["base_song_id"] != str(rows[position].get("base_song_id"))
                or source_bars[(str(rows[position].get("song_id")), int(rows[position].get("source_bar_index", -1)))]["applied_transpose_semitones"] != int(rows[position].get("applied_transpose_semitones", 0))
            ), None)
            if alignment_error is not None:
                unavailable[split] = alignment_error
                observation_path.unlink(missing_ok=True); split_arrays_path.unlink(missing_ok=True)
                self._write_json(status_path, self._status(request, split, "UNAVAILABLE", {}, alignment_error)); statuses[split] = status_path; continue
            selected = np.asarray(positions, dtype=np.int64)
            np.savez_compressed(split_arrays_path, **{name: value[selected] for name, value in values.items()})
            alignment = [{**rows[position], "tensor_row": local_row} for local_row, position in enumerate(positions)]
            observation = {"schema_version": RAW_OBSERVATION_V2_SCHEMA_VERSION, "dataset": {"identity": request.dataset_identity, "content_sha256": request.dataset_content_sha256, "split": split, "split_unit": "base_song_id"}, "arrays": {"path": split_arrays_path.name, "sha256": _sha256(split_arrays_path), "names": {name: {"dtype": str(value[selected].dtype), "shape": list(value[selected].shape)} for name, value in values.items()}}, "encoding_manifest": {"path": manifest_path.name, "sha256": _sha256(manifest_path)}, "bar_tensor_index": {"path": index_path.name, "sha256": _sha256(index_path)}, "source_raw": {"path": source_ref.name, "sha256": _sha256(source_ref)}, "tensor_schema": {"schema_version": "bar_tensor_schema.v2", "voice_names": manifest.get("voice_names"), "feature_names": manifest.get("feature_names")}, "alignment": alignment, "availability": {"voice_tensors": True, "bar_contexts": True, "base_pitches": True, "base_pitch_valid": True, "row_alignment": True, "source_raw_reference": True}}
            self._write_json(observation_path, observation); artifacts[split] = observation_path
            self._write_json(status_path, self._status(request, split, "AVAILABLE", {"observation": observation_path, "arrays": split_arrays_path}, None)); statuses[split] = status_path
        return CodecFidelityRawCaptureResult(artifacts, unavailable, statuses)

    def _unavailable_all(self, request: CodecFidelityV2RawCaptureRequest, reason: str) -> CodecFidelityRawCaptureResult:
        statuses = {}; unavailable = {}
        for split in ("train", "validation", "excluded_unpaired"):
            path = Path(request.encoded_dir) / f"codec_fidelity__raw_status__{split}.v2.json"; self._write_json(path, self._status(request, split, "UNAVAILABLE", {}, reason)); statuses[split] = path; unavailable[split] = reason
        return CodecFidelityRawCaptureResult({}, unavailable, statuses)

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _status(request: CodecFidelityV2RawCaptureRequest, split: str, status: str, artifacts: Mapping[str, Path], reason: str | None) -> Mapping[str, Any]:
        return {"schema_version": RAW_STATUS_V2_SCHEMA_VERSION, "dataset": {"identity": request.dataset_identity, "content_sha256": request.dataset_content_sha256, "split": split, "split_unit": "base_song_id"}, "status": status, "artifacts": {name: {"path": path.name, "sha256": _sha256(path)} for name, path in artifacts.items()}, "unavailable_reasons": [] if reason is None else [{"field": "v2_encoding", "reason": reason}]}


class CodecFidelityRawCapture(Protocol):
    """Persist codec raw observations without evaluation calculations."""

    def capture(self, request: CodecFidelityRawCaptureRequest) -> CodecFidelityRawCaptureResult:
        """Write raw observations from already materialized encoding facts."""


class JsonNpzCodecFidelityRawCapture:
    """Write stable split tensors, schema and source alignment atomically."""

    def capture(self, request: CodecFidelityRawCaptureRequest) -> CodecFidelityRawCaptureResult:
        song_by_id = self._song_by_id(request.songs)
        records_by_split = self._records_by_split(request, song_by_id)
        artifacts: dict[str, Path] = {}
        unavailable: dict[str, str] = {}
        status_artifacts: dict[str, Path] = {}
        for split in sorted(records_by_split):
            records = records_by_split[split]
            if not records:
                continue
            prepared, reason = self._prepare_split(request, split, records, song_by_id)
            status_path = request.model_dir / f"codec_fidelity__raw_status__{split}.v1.json"
            tensor_path = request.model_dir / f"codec_fidelity__raw_tensors__{split}.v1.npz"
            observation_path = request.model_dir / f"codec_fidelity__raw_observation__{split}.v1.json"
            if prepared is None:
                unavailable[split] = reason
                self._remove_stale_artifacts(observation_path, tensor_path)
                self._atomic_write_json(status_path, self._unavailable_status_payload(request, split, reason))
                status_artifacts[split] = status_path
                continue
            self._atomic_write_npz(tensor_path, prepared["bar_tensors"])
            payload = {
                "schema_version": RAW_OBSERVATION_SCHEMA_VERSION,
                "dataset": {
                    "identity": request.dataset_identity,
                    "identity_kind": request.dataset_identity_kind,
                    "content_sha256": request.dataset_content_sha256,
                    "split": split,
                    "split_unit": "base_song_id",
                },
                "source_raw": self._source_raw_reference(request.source_raw_paths.get(split)),
                "tensor": {
                    "path": tensor_path.name,
                    "array_name": "bar_tensors",
                    "sha256": _sha256(tensor_path),
                    "dtype": "float32",
                    "shape": [int(value) for value in prepared["bar_tensors"].shape],
                },
                "tensor_schema": prepared["tensor_schema"],
                "alignment": prepared["alignment"],
                "availability": {
                    "source_bar_notes": True,
                    "source_raw_reference": prepared["source_raw_available"],
                    "bar_tensors": True,
                    "tensor_schema": True,
                    "row_alignment": True,
                    "base_pitch": True,
                },
            }
            self._atomic_write_json(observation_path, payload)
            artifacts[split] = observation_path
            self._atomic_write_json(status_path, self._available_status_payload(
                request,
                split,
                observation_path,
                tensor_path,
            ))
            status_artifacts[split] = status_path
        return CodecFidelityRawCaptureResult(
            artifacts=artifacts,
            unavailable=unavailable,
            status_artifacts=status_artifacts,
        )

    @staticmethod
    def _available_status_payload(
        request: CodecFidelityRawCaptureRequest,
        split: str,
        observation_path: Path,
        tensor_path: Path,
    ) -> Mapping[str, Any]:
        return {
            "schema_version": RAW_STATUS_SCHEMA_VERSION,
            "dataset": _dataset_payload(request, split),
            "status": "AVAILABLE",
            "availability": _availability(True, True, True, True, True, True),
            "unavailable_reasons": [],
            "artifacts": {
                "observation": {"path": observation_path.name, "sha256": _sha256(observation_path)},
                "tensors": {"path": tensor_path.name, "sha256": _sha256(tensor_path)},
            },
        }

    @staticmethod
    def _unavailable_status_payload(
        request: CodecFidelityRawCaptureRequest,
        split: str,
        reason: str,
    ) -> Mapping[str, Any]:
        field, availability = _unavailable_status_details(reason)
        return {
            "schema_version": RAW_STATUS_SCHEMA_VERSION,
            "dataset": _dataset_payload(request, split),
            "status": "UNAVAILABLE",
            "availability": availability,
            "unavailable_reasons": [{"field": field, "reason": reason}],
            "artifacts": {},
        }

    @staticmethod
    def _remove_stale_artifacts(observation_path: Path, tensor_path: Path) -> None:
        observation_path.unlink(missing_ok=True)
        tensor_path.unlink(missing_ok=True)

    @staticmethod
    def _song_by_id(songs: Sequence[SongRecord]) -> dict[str, SongRecord]:
        result = {str(song.song_id): song for song in songs}
        if len(result) != len(songs):
            raise ValueError("codec raw capture requires unique song_id values")
        return result

    @staticmethod
    def _records_by_split(
        request: CodecFidelityRawCaptureRequest,
        song_by_id: Mapping[str, SongRecord],
    ) -> Mapping[str, list[BarTensorRecord]]:
        train = set(request.train_base_song_ids)
        validation = set(request.validation_base_song_ids)
        if train & validation:
            raise ValueError("a base_song_id cannot be both train and validation")
        result: dict[str, list[BarTensorRecord]] = {"train": [], "validation": [], "excluded_unpaired": []}
        for record in request.tensors:
            song = song_by_id.get(str(record.song_id))
            if song is None:
                raise ValueError(f"tensor row references unknown song_id: {record.song_id!r}")
            base_song_id = _base_song_id(song)
            split = "train" if base_song_id in train else "validation" if base_song_id in validation else "excluded_unpaired"
            result[split].append(record)
        return result

    def _prepare_split(
        self,
        request: CodecFidelityRawCaptureRequest,
        split: str,
        records: Sequence[BarTensorRecord],
        song_by_id: Mapping[str, SongRecord],
    ) -> tuple[dict[str, Any] | None, str]:
        source_path = request.source_raw_paths.get(split)
        source_bars, source_reason = _source_bar_index(source_path, split)
        if source_bars is None:
            return None, source_reason
        ordered = sorted(records, key=lambda item: (str(item.song_id), int(item.bar_index)))
        try:
            bar_tensors = np.stack([np.asarray(record.tensor, dtype=np.float32) for record in ordered])
        except ValueError:
            return None, "bar tensors do not share one [track, step, feature] shape"
        if (
            bar_tensors.ndim != 4
            or bar_tensors.dtype != np.float32
            or bar_tensors.dtype.hasobject
            or not np.isfinite(bar_tensors).all()
        ):
            return None, "bar tensors must be finite float32 [bar, track, step, feature] values"
        try:
            tensor_schema = _tensor_schema(request.codec_config, bar_tensors.shape[1:])
        except ValueError as error:
            return None, str(error)
        alignment: list[dict[str, Any]] = []
        for tensor_row, record in enumerate(ordered):
            song = song_by_id[str(record.song_id)]
            base_pitch = _base_pitch(record)
            if base_pitch is None:
                return None, "base_pitch is unavailable for one or more encoded bars"
            source_bar = source_bars.get((str(record.song_id), int(record.bar_index)))
            if source_bar is None:
                return None, "source raw artifact has no matching song_id/source_bar_index"
            if (
                source_bar["base_song_id"] != _base_song_id(song)
                or source_bar["applied_transpose_semitones"] != _transpose_semitones(song)
            ):
                return None, "source raw alignment disagrees on base_song_id or applied transpose"
            alignment.append({
                "tensor_row": tensor_row,
                "song_id": str(record.song_id),
                "base_song_id": _base_song_id(song),
                "bar_index": int(record.bar_index),
                "source_bar_index": int(record.bar_index),
                "applied_transpose_semitones": _transpose_semitones(song),
                "base_pitch_semitones": base_pitch,
            })
        return {
            "bar_tensors": bar_tensors,
            "tensor_schema": tensor_schema,
            "alignment": alignment,
            "source_raw_available": True,
        }, ""

    @staticmethod
    def _source_raw_reference(path: Path | None) -> Mapping[str, str | None]:
        if path is None or not path.is_file():
            return {"path": None, "sha256": None}
        return {"path": path.name, "sha256": _sha256(path)}

    @staticmethod
    def _atomic_write_npz(path: Path, values: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, bar_tensors=values)
        try:
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
        try:
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _tensor_schema(config: Mapping[str, Any], shape: Sequence[int]) -> Mapping[str, Any]:
    if len(shape) != 3:
        raise ValueError("codec tensor shape must be [track, step, feature]")
    track_count, step_count, feature_count = (int(value) for value in shape)
    section = config.get("bar_tensor") if isinstance(config.get("bar_tensor"), Mapping) else {}
    backend = str(section.get("backend", "legacy_physical")).strip().lower()
    pitch_scale = float(section.get("pitch_scale", 24.0))
    if not math.isfinite(pitch_scale) or pitch_scale <= 0:
        raise ValueError("codec pitch_scale must be finite and positive")
    if backend in {"semantic_3voice", "semantic", "melody_harmony_bass"}:
        if track_count != len(SEMANTIC_TRACK_NAMES) or feature_count != len(SEMANTIC_FEATURE_NAMES):
            raise ValueError("semantic codec configuration does not match runtime tensor shape")
        feature_names = list(SEMANTIC_FEATURE_NAMES)
        track_names = list(SEMANTIC_TRACK_NAMES)
        feature_units = {
            "relative_pitch": "normalized by pitch_scale_semitones",
            "is_rest": "binary 1=rest",
            "is_note_on": "binary 1=note onset",
            "is_hold": "binary 1=held note",
            "normalized_velocity": "unit interval MIDI velocity",
            "velocity_ratio": "slot-local velocity share",
            "density_gradient": "normalized per-track slot density change",
            "relative_chroma_embedding": "relative chroma projection coordinates",
        }
    elif backend in {"legacy", "legacy_physical", "physical"}:
        if feature_count != len(FEATURE_NAMES):
            raise ValueError("legacy codec configuration does not match runtime tensor shape")
        feature_names = list(FEATURE_NAMES)
        track_names = [f"track_{index}" for index in range(track_count)]
        feature_units = {
            "relative_pitch": "normalized by pitch_scale_semitones",
            "is_rest": "binary 1=rest",
            "is_note_on": "binary 1=note onset",
            "is_hold": "binary 1=held note",
            "normalized_velocity": "unit interval MIDI velocity",
            "chord_embedding": "relative chord projection coordinates",
        }
    else:
        raise ValueError("runtime codec has no supported public tensor schema")
    if int(section.get("steps_per_bar", step_count)) != step_count:
        raise ValueError("runtime tensor step count does not match codec configuration")
    return {
        "axis_order": ["bar", "track", "step", "feature"],
        "feature_names": feature_names,
        "track_names": track_names,
        "pitch_scale_semitones": pitch_scale,
        "feature_units": feature_units,
    }


def _base_song_id(song: SongRecord) -> str:
    metadata = song.metadata if isinstance(song.metadata, Mapping) else {}
    return str(metadata.get("base_song_id")) if metadata.get("base_song_id") else _TRANSPOSE_SUFFIX.sub("", str(song.song_id))


def _transpose_semitones(song: SongRecord) -> int:
    metadata = song.metadata if isinstance(song.metadata, Mapping) else {}
    return int(metadata.get("transpose_semitones", 0))


def _base_pitch(record: BarTensorRecord) -> float | None:
    value = record.diagnostics.get("base_pitch") if isinstance(record.diagnostics, Mapping) else None
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _dataset_payload(request: CodecFidelityRawCaptureRequest, split: str) -> Mapping[str, Any]:
    return {
        "identity": request.dataset_identity,
        "identity_kind": request.dataset_identity_kind,
        "content_sha256": request.dataset_content_sha256,
        "split": split,
        "split_unit": "base_song_id",
    }


def _availability(
    source_bar_notes: bool,
    source_raw_reference: bool,
    bar_tensors: bool,
    tensor_schema: bool,
    row_alignment: bool,
    base_pitch: bool,
) -> Mapping[str, bool]:
    return {
        "source_bar_notes": source_bar_notes,
        "source_raw_reference": source_raw_reference,
        "bar_tensors": bar_tensors,
        "tensor_schema": tensor_schema,
        "row_alignment": row_alignment,
        "base_pitch": base_pitch,
    }


def _unavailable_status_details(reason: str) -> tuple[str, Mapping[str, bool]]:
    if "base_pitch" in reason:
        return "base_pitch", _availability(True, True, True, True, True, False)
    if "bar tensors" in reason:
        return "bar_tensors", _availability(True, True, False, False, False, False)
    if "tensor shape" in reason or "codec" in reason or "pitch_scale" in reason:
        return "tensor_schema", _availability(True, True, True, False, False, False)
    if "alignment" in reason or "song_id/source_bar_index" in reason:
        return "row_alignment", _availability(True, True, True, True, False, True)
    return "source_raw_reference", _availability(False, False, False, False, False, False)


def _source_bar_index(
    path: Path | None,
    expected_split: str,
) -> tuple[dict[tuple[str, int], Mapping[str, Any]] | None, str]:
    if path is None or not path.is_file():
        return None, "matching dataset_tonality raw source artifact is unavailable"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"dataset_tonality raw source artifact cannot be read: {error}"
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "dataset_tonality_raw_source.v1":
        return None, "dataset_tonality raw source artifact has an unsupported schema_version"
    dataset = payload.get("dataset")
    availability = payload.get("availability")
    songs = payload.get("songs")
    if not isinstance(dataset, Mapping) or dataset.get("split") != expected_split:
        return None, "dataset_tonality raw source split does not match codec split"
    if not isinstance(availability, Mapping) or not availability.get("bar_note_events") or not availability.get("split_membership"):
        return None, "dataset_tonality raw source does not make bar notes and split membership available"
    if not isinstance(songs, list):
        return None, "dataset_tonality raw source has no song observations"
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for song in songs:
        if not isinstance(song, Mapping):
            return None, "dataset_tonality raw source contains an invalid song observation"
        song_id = song.get("song_id")
        base_song_id = song.get("base_song_id")
        transpose = song.get("applied_transpose_semitones")
        bars = song.get("bars")
        if not isinstance(song_id, str) or not isinstance(base_song_id, str) or not isinstance(transpose, int) or not isinstance(bars, list):
            return None, "dataset_tonality raw source song observations are incomplete"
        for bar in bars:
            if not isinstance(bar, Mapping) or not isinstance(bar.get("bar_index"), int):
                return None, "dataset_tonality raw source bar observations are incomplete"
            key = (song_id, int(bar["bar_index"]))
            if key in result:
                return None, "dataset_tonality raw source repeats a song_id/bar_index observation"
            result[key] = {
                "base_song_id": base_song_id,
                "applied_transpose_semitones": transpose,
            }
    return result, ""


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
