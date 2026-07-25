#!/usr/bin/env python3
"""Encode parsed bars into [tracks, slots, features] tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from common.config_loader import ConfigView
from data.core import BarRecord, BarTensorRecord, NoteEvent, TrackRecord


FEATURE_NAMES = [
    "relative_pitch",
    "is_rest",
    "is_note_on",
    "is_hold",
    "normalized_velocity",
    "chord_embed_00",
    "chord_embed_01",
    "chord_embed_02",
    "chord_embed_03",
    "chord_embed_04",
    "chord_embed_05",
    "chord_embed_06",
    "chord_embed_07",
    "chord_embed_08",
    "chord_embed_09",
    "chord_embed_10",
]


@dataclass(frozen=True)
class BarTensorConfig:
    """Configuration for bar tensor encoding."""

    max_tracks: int = 3
    steps_per_bar: int = 16
    feature_dim: int = 16
    pitch_scale: float = 24.0
    velocity_scale: float = 127.0
    rest_feature_value: float = 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BarTensorConfig":
        """Build tensor codec configuration from style config."""
        section = ConfigView(config).section("bar_tensor")
        return cls(
            max_tracks=int(section.get("max_tracks", 3)),
            steps_per_bar=int(section.get("steps_per_bar", 16)),
            feature_dim=int(section.get("feature_dim", 16)),
            pitch_scale=float(section.get("pitch_scale", 24.0)),
            velocity_scale=float(section.get("velocity_scale", 127.0)),
            rest_feature_value=float(section.get("rest_feature_value", 1.0)),
        )


class FixedChromagramProjector:
    """Deterministically compress a 12-bin chromagram into 11 dimensions."""

    def project(self, chromagram: Sequence[float]) -> np.ndarray:
        """Return an 11-dimensional normalized chord texture vector."""
        values = np.asarray(chromagram, dtype=np.float32)
        if values.shape != (12,):
            raise ValueError("chromagram must have shape [12].")
        compressed = np.zeros(11, dtype=np.float32)
        compressed[:10] = values[:10]
        compressed[10] = float(values[10] + values[11]) * 0.5
        return compressed


class BarTensorCodec:
    """Convert BarRecord objects into physical multi-task tensors."""

    def __init__(self, config: BarTensorConfig, projector: FixedChromagramProjector | None = None) -> None:
        self.config = config
        self.projector = projector or FixedChromagramProjector()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BarTensorCodec":
        """Create a tensor codec from the full style config."""
        return cls(BarTensorConfig.from_config(config))

    def encode(self, bar: BarRecord) -> BarTensorRecord:
        """Encode one bar into a [3, 16, 16] tensor record."""
        if self.config.feature_dim != len(FEATURE_NAMES):
            raise ValueError(f"feature_dim must be {len(FEATURE_NAMES)}.")
        tensor = np.zeros(
            (self.config.max_tracks, self.config.steps_per_bar, self.config.feature_dim),
            dtype=np.float32,
        )
        chromagram = self._bar_chromagram(bar)
        chord_embedding = self.projector.project(chromagram)
        base_pitch = self._bar_base_pitch(bar)
        track_diagnostics = []
        for track_index in range(self.config.max_tracks):
            if track_index >= len(bar.tracks):
                track_diagnostics.append({"track_index": track_index, "padded": True, "note_count": 0})
                continue
            track = bar.tracks[track_index]
            self._encode_track(tensor[track_index], track, bar, base_pitch, chord_embedding)
            track_diagnostics.append({
                "track_index": int(track_index),
                "padded": False,
                "note_count": int(len(track.notes)),
            })
        diagnostics = {
            "song_id": bar.song_id,
            "bar_index": int(bar.bar_index),
            "shape": [int(value) for value in tensor.shape],
            "feature_names": list(FEATURE_NAMES),
            "base_pitch": int(base_pitch) if base_pitch is not None else None,
            "chromagram": [float(value) for value in chromagram.tolist()],
            "chord_embedding": [float(value) for value in chord_embedding.tolist()],
            "track_count": int(len(bar.tracks)),
            "tracks": track_diagnostics,
        }
        return BarTensorRecord(
            song_id=bar.song_id,
            bar_index=int(bar.bar_index),
            tensor_shape=[int(value) for value in tensor.shape],
            tensor=tensor,
            diagnostics=diagnostics,
        )

    def _encode_track(
        self,
        track_tensor: np.ndarray,
        track: TrackRecord,
        bar: BarRecord,
        base_pitch: int | None,
        chord_embedding: np.ndarray,
    ) -> None:
        """Fill one track slice of the tensor."""
        slot_len = float(bar.bar_length_ql) / float(self.config.steps_per_bar)
        for slot_index in range(self.config.steps_per_bar):
            slot_start = float(slot_index) * slot_len
            slot_end = slot_start + slot_len
            onset_note = self._onset_note(track.notes, slot_start, slot_end)
            active_note = self._active_note(track.notes, slot_start, slot_end)
            track_tensor[slot_index, 5:16] = chord_embedding
            if onset_note is not None:
                self._write_note_on(track_tensor[slot_index], onset_note, base_pitch)
            elif active_note is not None:
                self._write_hold(track_tensor[slot_index], active_note, base_pitch)
            else:
                self._write_rest(track_tensor[slot_index])

    def _write_note_on(self, features: np.ndarray, note: NoteEvent, base_pitch: int | None) -> None:
        """Write NOTE_ON state features."""
        features[0] = self._relative_pitch(note.pitch, base_pitch)
        features[1] = 0.0
        features[2] = 1.0
        features[3] = 0.0
        features[4] = self._normalized_velocity(note.velocity)

    def _write_hold(self, features: np.ndarray, note: NoteEvent, base_pitch: int | None) -> None:
        """Write NOTE_HOLD state features."""
        features[0] = self._relative_pitch(note.pitch, base_pitch)
        features[1] = 0.0
        features[2] = 0.0
        features[3] = 1.0
        features[4] = self._normalized_velocity(note.velocity)

    def _write_rest(self, features: np.ndarray) -> None:
        """Write REST state features."""
        features[0] = 0.0
        features[1] = self.config.rest_feature_value
        features[2] = 0.0
        features[3] = 0.0
        features[4] = 0.0

    def _relative_pitch(self, pitch: int, base_pitch: int | None) -> float:
        """Normalize a pitch offset relative to the bar base pitch."""
        if base_pitch is None:
            return 0.0
        return float(int(pitch) - int(base_pitch)) / float(self.config.pitch_scale)

    def _normalized_velocity(self, velocity: int) -> float:
        """Normalize MIDI velocity into [0, 1]."""
        return float(max(0, min(int(velocity), int(self.config.velocity_scale)))) / float(self.config.velocity_scale)

    def _onset_note(self, notes: Sequence[NoteEvent], slot_start: float, slot_end: float) -> NoteEvent | None:
        """Find the strongest note onset inside a slot."""
        candidates = [note for note in notes if slot_start <= float(note.onset_ql) < slot_end]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (int(item.velocity), int(item.pitch)))

    def _active_note(self, notes: Sequence[NoteEvent], slot_start: float, slot_end: float) -> NoteEvent | None:
        """Find the strongest note sustained through a slot."""
        candidates = [
            note for note in notes
            if float(note.onset_ql) < slot_end and float(note.onset_ql) + float(note.duration_ql) > slot_start
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (int(item.velocity), int(item.pitch)))

    def _bar_base_pitch(self, bar: BarRecord) -> int | None:
        """Choose a stable base pitch for relative-pitch encoding."""
        notes = bar.all_notes()
        if not notes:
            return None
        pitches = np.asarray([int(note.pitch) for note in notes], dtype=np.int32)
        return int(np.median(pitches))

    def _bar_chromagram(self, bar: BarRecord) -> np.ndarray:
        """Compute a normalized 12-bin pitch-class profile for the whole bar."""
        values = np.zeros(12, dtype=np.float32)
        for note in bar.all_notes():
            values[int(note.pitch) % 12] += max(0.0, float(note.duration_ql))
        total = float(values.sum())
        if total <= 0.0:
            return values
        return values / total
