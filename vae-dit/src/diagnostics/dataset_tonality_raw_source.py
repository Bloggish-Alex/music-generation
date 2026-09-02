"""Versioned source-note observations for dataset tonality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Protocol, Sequence

from data.core import SongRecord


RAW_SOURCE_SCHEMA_VERSION = "dataset_tonality_raw_source.v1"
_SPLIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TIME_SIGNATURE = re.compile(r"^[0-9]+/[0-9]+$")
_TRANSPOSE_SUFFIX = re.compile(r"_T[+-]?\d+$")


@dataclass(frozen=True)
class DatasetTonalityRawSourceRequest:
    """Parsed source facts and resolved base-song split membership."""

    model_dir: Path
    dataset_identity: str
    dataset_identity_kind: str
    songs_by_split: Mapping[str, Sequence[SongRecord]]
    dataset_content_sha256: str | None = None
    encoding_artifact_sha256: str | None = None


class DatasetTonalityRawSourceWriter(Protocol):
    """Persist source-level observations without loading model artifacts."""

    def write(self, request: DatasetTonalityRawSourceRequest) -> Mapping[str, Path]:
        """Write one raw-source artifact for every resolved split."""


class JsonDatasetTonalityRawSourceWriter:
    """Write deterministic, path-free source note observations atomically."""

    def write(self, request: DatasetTonalityRawSourceRequest) -> Mapping[str, Path]:
        self._validate_request(request)
        base_split = self._base_song_split(request.songs_by_split)
        availability = {
            "bar_note_events": True,
            "split_membership": True,
            "source_content_hashes": all(
                _source_content_hash(song) is not None
                for songs in request.songs_by_split.values()
                for song in songs
            ),
        }
        paths: dict[str, Path] = {}
        for split in sorted(request.songs_by_split):
            songs = request.songs_by_split[split]
            payload = {
                "schema_version": RAW_SOURCE_SCHEMA_VERSION,
                "dataset": {
                    "identity": request.dataset_identity,
                    "identity_kind": request.dataset_identity_kind,
                    "content_sha256": request.dataset_content_sha256,
                    "split": split,
                    "split_unit": "base_song_id",
                },
                "source": {
                    "encoding_artifact_sha256": request.encoding_artifact_sha256,
                    "note_representation": "bar_note_events",
                    "bar_index_semantics": "zero_based_within_song",
                },
                "songs": [self._song_payload(song) for song in sorted(songs, key=lambda item: item.song_id)],
                "availability": availability,
            }
            self._validate_split_payload(payload, base_split, split)
            path = request.model_dir / f"dataset_tonality__raw_source__{split}.v1.json"
            self._atomic_write_json(path, payload)
            paths[split] = path
        return paths

    @staticmethod
    def _validate_request(request: DatasetTonalityRawSourceRequest) -> None:
        if not request.dataset_identity or not request.dataset_identity_kind:
            raise ValueError("dataset identity and identity_kind are required")
        if not request.songs_by_split:
            raise ValueError("at least one resolved dataset split is required")
        for split, songs in request.songs_by_split.items():
            if not _SPLIT_NAME.fullmatch(str(split)):
                raise ValueError(f"invalid split name: {split!r}")
            if not songs:
                raise ValueError(f"split {split!r} has no source songs")

    @staticmethod
    def _base_song_split(songs_by_split: Mapping[str, Sequence[SongRecord]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for split, songs in songs_by_split.items():
            for song in songs:
                base_song_id = _base_song_id(song)
                previous = result.setdefault(base_song_id, str(split))
                if previous != str(split):
                    raise ValueError(
                        f"base_song_id {base_song_id!r} appears in both {previous!r} and {split!r}"
                    )
        return result

    @staticmethod
    def _song_payload(song: SongRecord) -> dict[str, Any]:
        bars = sorted(song.bars, key=lambda item: int(item.bar_index))
        expected_indices = list(range(len(bars)))
        observed_indices = [int(bar.bar_index) for bar in bars]
        if observed_indices != expected_indices:
            raise ValueError(f"song {song.song_id!r} bars must be zero-based and contiguous")
        return {
            "song_id": str(song.song_id),
            "base_song_id": _base_song_id(song),
            "source_content_sha256": _source_content_hash(song),
            "applied_transpose_semitones": _transpose_semitones(song),
            "bars": [JsonDatasetTonalityRawSourceWriter._bar_payload(bar) for bar in bars],
        }

    @staticmethod
    def _bar_payload(bar: Any) -> dict[str, Any]:
        bar_length = float(bar.bar_length_ql)
        if not math.isfinite(bar_length) or bar_length <= 0:
            raise ValueError("bar_length_ql must be finite and positive")
        time_signature = str(bar.time_signature)
        if not _TIME_SIGNATURE.fullmatch(time_signature):
            raise ValueError(f"invalid time signature: {time_signature!r}")
        notes = [
            {
                "track_index": int(track.track_index),
                "physical_track_index": int(note.physical_track_index if note.physical_track_index is not None else track.track_index),
                "source_note_ordinal": int(note.source_note_ordinal if note.source_note_ordinal is not None else 0),
                "source_note_id": str(note.source_note_id or f"{note.source_file_identity or ''}:{note.physical_track_index if note.physical_track_index is not None else track.track_index}:{note.source_note_ordinal if note.source_note_ordinal is not None else 0}"),
                "source_onset_ql": float(note.source_onset_ql if note.source_onset_ql is not None else note.onset_ql),
                "pitch": int(note.pitch),
                "onset_ql": float(note.onset_ql),
                "duration_ql": float(note.duration_ql),
                "velocity": int(note.velocity),
                "continues_from_previous_bar": bool(note.continues_from_previous_bar),
                "continues_into_next_bar": bool(note.continues_into_next_bar),
            }
            for track in bar.tracks
            for note in track.notes
        ]
        for note in notes:
            if (
                note["track_index"] < 0
                or not 0 <= note["pitch"] <= 127
                or not 0 <= note["velocity"] <= 127
                or not math.isfinite(note["onset_ql"])
                or not math.isfinite(note["duration_ql"])
                or note["onset_ql"] < 0
                or note["duration_ql"] <= 0
            ):
                raise ValueError("source note facts are outside the raw-source contract")
        return {
            "bar_index": int(bar.bar_index),
            "bar_length_ql": bar_length,
            "time_signature": time_signature,
            "source_measure_index": int(bar.source_measure_index) if bar.source_measure_index is not None else None,
            "meter_numerator": int(bar.meter_numerator) if bar.meter_numerator is not None else None,
            "meter_denominator": int(bar.meter_denominator) if bar.meter_denominator is not None else None,
            "is_pickup": bool(bar.is_pickup),
            "tempo_bpm": None,
            "notes": sorted(
                notes,
                key=lambda item: (
                    item["track_index"],
                    item["onset_ql"],
                    item["pitch"],
                    item["duration_ql"],
                    item["velocity"],
                    item["source_note_id"],
                ),
            ),
        }

    @staticmethod
    def _validate_split_payload(payload: Mapping[str, Any], base_split: Mapping[str, str], split: str) -> None:
        for song in payload["songs"]:
            if base_split[song["base_song_id"]] != split:
                raise ValueError("song split membership does not match its base_song_id")

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=False)
            handle.write("\n")
        try:
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _base_song_id(song: SongRecord) -> str:
    metadata_value = song.metadata.get("base_song_id") if isinstance(song.metadata, Mapping) else None
    return str(metadata_value) if metadata_value else _TRANSPOSE_SUFFIX.sub("", str(song.song_id))


def _transpose_semitones(song: SongRecord) -> int:
    if not isinstance(song.metadata, Mapping):
        return 0
    return int(song.metadata.get("transpose_semitones", 0))


def _source_content_hash(song: SongRecord) -> str | None:
    if not isinstance(song.metadata, Mapping):
        return None
    value = song.metadata.get("source_content_sha256")
    if isinstance(value, str) and value.startswith("sha256:"):
        return value
    return None
