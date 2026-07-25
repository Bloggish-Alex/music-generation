#!/usr/bin/env python3
"""MidiTok-style event extraction from encoded bar tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


MIDITOK_STYLE_FEATURE_NAMES = [
    "event_count",
    "event_density",
    "active_density",
    "rest_density",
    "mean_position",
    "std_position",
    "position_entropy",
    "strong_beat_ratio",
    "offbeat_ratio",
    "mean_duration",
    "std_duration",
    "max_duration",
    "short_duration_ratio",
    "long_duration_ratio",
    "mean_pitch",
    "std_pitch",
    "min_pitch",
    "max_pitch",
    "pitch_range",
    "mean_abs_interval",
    "max_abs_interval",
    "direction_change_ratio",
    "repeat_pitch_ratio",
    "ascending_interval_ratio",
    "descending_interval_ratio",
    "mean_velocity",
    "std_velocity",
    "track0_event_density",
    "track1_event_density",
    "track2_event_density",
    "track_switch_ratio",
    "polyphonic_onset_ratio",
    "chord_mean_00",
    "chord_mean_01",
    "chord_mean_02",
    "chord_mean_03",
    "chord_mean_04",
    "chord_mean_05",
    "chord_mean_06",
    "chord_mean_07",
    "chord_mean_08",
    "chord_mean_09",
    "chord_mean_10",
]


@dataclass(frozen=True)
class MidiTokStyleEvent:
    """One event-level note representation."""

    position: int
    track: int
    pitch: float
    velocity: float
    duration: int

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly event."""
        return {
            "position": int(self.position),
            "track": int(self.track),
            "pitch": float(self.pitch),
            "velocity": float(self.velocity),
            "duration": int(self.duration),
        }


class MidiTokStyleBarEventEncoder:
    """Extract MidiTok-style note events and fixed event features from one bar tensor."""

    feature_names = MIDITOK_STYLE_FEATURE_NAMES

    def events(self, tensor: np.ndarray) -> List[MidiTokStyleEvent]:
        """Return note-on events with duration from a [tracks, steps, features] tensor."""
        values = np.asarray(tensor, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] < 5:
            raise ValueError("bar tensor must have shape [tracks, steps, features>=5].")
        note_on = values[..., 2] > 0.5
        hold = values[..., 3] > 0.5
        events: List[MidiTokStyleEvent] = []
        tracks, steps = int(values.shape[0]), int(values.shape[1])
        for slot in range(steps):
            for track in range(tracks):
                if not bool(note_on[track, slot]):
                    continue
                duration = 1
                cursor = slot + 1
                while cursor < steps and bool(hold[track, cursor]):
                    duration += 1
                    cursor += 1
                events.append(MidiTokStyleEvent(
                    position=int(slot),
                    track=int(track),
                    pitch=float(values[track, slot, 0]),
                    velocity=float(values[track, slot, 4]),
                    duration=int(duration),
                ))
        return events

    def features(self, tensor: np.ndarray) -> np.ndarray:
        """Return fixed MidiTok-style event feature vector."""
        values = np.asarray(tensor, dtype=np.float32)
        events = self.events(values)
        tracks, steps = int(values.shape[0]), int(values.shape[1])
        total_slots = max(1, tracks * steps)
        note_on = values[..., 2] > 0.5
        hold = values[..., 3] > 0.5
        rest = values[..., 1] > 0.5
        active = note_on | hold
        if values.shape[2] >= 16:
            chord_mean = np.mean(values[..., -11:].reshape(-1, 11), axis=0)
        else:
            chord_mean = np.zeros((11,), dtype=np.float32)
        if not events:
            base = [
                0.0, 0.0,
                float(active.sum() / total_slots),
                float(rest.sum() / total_slots),
                *([0.0] * 28),
            ]
            return np.asarray([*base, *[float(item) for item in chord_mean.tolist()]], dtype=np.float32)

        positions = np.asarray([event.position for event in events], dtype=np.float32)
        tracks_array = np.asarray([event.track for event in events], dtype=np.int64)
        pitches = np.asarray([event.pitch for event in events], dtype=np.float32)
        velocities = np.asarray([event.velocity for event in events], dtype=np.float32)
        durations = np.asarray([event.duration for event in events], dtype=np.float32)
        intervals = np.diff(pitches) if pitches.size > 1 else np.zeros((0,), dtype=np.float32)
        abs_intervals = np.abs(intervals)
        position_counts = np.bincount(positions.astype(np.int64), minlength=steps).astype(np.float32)
        position_prob = position_counts / max(1.0, float(position_counts.sum()))
        position_entropy = float(-(position_prob * np.log(np.clip(position_prob, 1.0e-8, 1.0))).sum() / np.log(max(2, steps)))
        onset_slots = np.count_nonzero(position_counts)
        polyphonic_onsets = int(np.count_nonzero(position_counts > 1.0))
        signs = np.sign(intervals)
        nonzero_signs = signs[signs != 0]
        direction_change_ratio = 0.0
        if nonzero_signs.size > 1:
            direction_change_ratio = float(np.mean(nonzero_signs[1:] != nonzero_signs[:-1]))
        track_switch_ratio = 0.0
        if tracks_array.size > 1:
            track_switch_ratio = float(np.mean(tracks_array[1:] != tracks_array[:-1]))
        track_event_density = [
            float(np.count_nonzero(tracks_array == track) / max(1, steps))
            for track in range(3)
        ]
        pitch_min = float(np.min(pitches))
        pitch_max = float(np.max(pitches))
        features = [
            float(len(events)),
            float(len(events) / max(1, steps)),
            float(active.sum() / total_slots),
            float(rest.sum() / total_slots),
            float(np.mean(positions) / max(1, steps - 1)),
            float(np.std(positions) / max(1, steps - 1)),
            position_entropy,
            float(np.mean((positions.astype(np.int64) % 4) == 0)),
            float(np.mean((positions.astype(np.int64) % 4) != 0)),
            float(np.mean(durations) / max(1, steps)),
            float(np.std(durations) / max(1, steps)),
            float(np.max(durations) / max(1, steps)),
            float(np.mean(durations <= 1.0)),
            float(np.mean(durations >= 4.0)),
            float(np.mean(pitches)),
            float(np.std(pitches)),
            pitch_min,
            pitch_max,
            float(pitch_max - pitch_min),
            float(np.mean(abs_intervals)) if abs_intervals.size else 0.0,
            float(np.max(abs_intervals)) if abs_intervals.size else 0.0,
            direction_change_ratio,
            float(np.mean(abs_intervals <= 1.0)) if abs_intervals.size else 0.0,
            float(np.mean(intervals > 0.0)) if intervals.size else 0.0,
            float(np.mean(intervals < 0.0)) if intervals.size else 0.0,
            float(np.mean(velocities)),
            float(np.std(velocities)),
            *track_event_density,
            track_switch_ratio,
            float(polyphonic_onsets / max(1, onset_slots)),
            *[float(item) for item in chord_mean.tolist()],
        ]
        return np.asarray(features, dtype=np.float32)


class MidiTokStyleBarEventStore:
    """Read and write MidiTok-style event feature artifacts."""

    FEATURE_FILE = "miditok_style_bar_features.npz"
    EVENT_FILE = "miditok_style_bar_events.jsonl"
    SUMMARY_FILE = "miditok_style_bar_summary.json"

    def __init__(self, encoded_dir: str | Path) -> None:
        self.encoded_dir = Path(encoded_dir)

    def matrix_for_rows(self, rows: Sequence[Dict[str, Any]], fallback_tensor_file: str = "bar_tensors.npz") -> tuple[np.ndarray, Dict[str, Any]]:
        """Return [rows, event_feature_dim] features, using cache or tensor fallback."""
        feature_path = self.encoded_dir / self.FEATURE_FILE
        if feature_path.exists():
            archive = np.load(feature_path)
            try:
                missing = self._missing_archive_keys(archive, rows)
                if not missing:
                    return self._matrix_from_archive(archive, rows), {
                        "event_feature_source": "cached_miditok_style_bar_features",
                        "event_feature_path": str(feature_path),
                    }
            finally:
                archive.close()
        tensor_path = self.encoded_dir / fallback_tensor_file
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing {self.FEATURE_FILE} and fallback tensor archive: {tensor_path}")
        matrix = self._matrix_from_tensor_archive(rows, tensor_path)
        info = {
            "event_feature_source": "computed_miditok_style_from_tensors",
            "event_feature_path": str(feature_path),
            "fallback_tensor_path": str(tensor_path),
        }
        if feature_path.exists():
            info["cache_key_miss"] = True
        return matrix, info

    def write_from_tensor_archive(self, tensor_path: str | Path, rows: Sequence[Dict[str, Any]], write_events: bool = True) -> Dict[str, Any]:
        """Compute and cache event features from a tensor archive."""
        tensor_archive = Path(tensor_path)
        matrix_by_key: Dict[str, np.ndarray] = {}
        encoder = MidiTokStyleBarEventEncoder()
        archive = np.load(tensor_archive)
        event_path = self.encoded_dir / self.EVENT_FILE
        self.encoded_dir.mkdir(parents=True, exist_ok=True)
        event_handle = event_path.open("w", encoding="utf-8") if write_events else None
        try:
            for row in rows:
                key = str(row.get("tensor_key", ""))
                if key not in archive.files:
                    raise KeyError(f"Missing tensor_key in archive: {key}")
                tensor = np.asarray(archive[key], dtype=np.float32)
                matrix_by_key[key] = encoder.features(tensor)
                if event_handle is not None:
                    payload = {
                        "tensor_key": key,
                        "song_id": str(row.get("song_id", "UNKNOWN")),
                        "bar_index": int(row.get("bar_index", 0)),
                        "events": [event.to_dict() for event in encoder.events(tensor)],
                    }
                    event_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        finally:
            archive.close()
            if event_handle is not None:
                event_handle.close()
        np.savez_compressed(self.encoded_dir / self.FEATURE_FILE, **matrix_by_key)
        matrix = np.stack(list(matrix_by_key.values()), axis=0).astype(np.float32) if matrix_by_key else np.zeros((0, len(MIDITOK_STYLE_FEATURE_NAMES)), dtype=np.float32)
        summary = self.summary(matrix, source="computed")
        (self.encoded_dir / self.SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def summary(self, matrix: np.ndarray, source: str) -> Dict[str, Any]:
        """Return compact event feature summary."""
        values = np.asarray(matrix, dtype=np.float32)
        return {
            "source": str(source),
            "feature_count": int(len(MIDITOK_STYLE_FEATURE_NAMES)),
            "feature_names": list(MIDITOK_STYLE_FEATURE_NAMES),
            "row_count": int(values.shape[0]),
            "shape": [int(item) for item in values.shape],
            "mean": [float(item) for item in values.mean(axis=0).tolist()] if values.size else [],
            "std": [float(item) for item in values.std(axis=0).tolist()] if values.size else [],
        }

    def _matrix_from_archive(self, archive: Any, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Load event features by tensor_key order."""
        values = []
        for row in rows:
            key = str(row.get("tensor_key", ""))
            if key not in archive.files:
                raise KeyError(f"Missing tensor_key in {self.FEATURE_FILE}: {key}")
            values.append(np.asarray(archive[key], dtype=np.float32))
        return np.stack(values, axis=0).astype(np.float32)

    def _matrix_from_tensor_archive(self, rows: Sequence[Dict[str, Any]], tensor_path: Path) -> np.ndarray:
        """Compute event feature matrix from encoded tensors."""
        encoder = MidiTokStyleBarEventEncoder()
        archive = np.load(tensor_path)
        try:
            values = []
            for row in rows:
                key = str(row.get("tensor_key", ""))
                if key not in archive.files:
                    raise KeyError(f"Missing tensor_key in archive: {key}")
                values.append(encoder.features(np.asarray(archive[key], dtype=np.float32)))
            return np.stack(values, axis=0).astype(np.float32)
        finally:
            archive.close()

    def _missing_archive_keys(self, archive: Any, rows: Sequence[Dict[str, Any]]) -> List[str]:
        """Return tensor keys absent from archive."""
        available = set(str(key) for key in archive.files)
        missing: List[str] = []
        for row in rows:
            key = str(row.get("tensor_key", ""))
            if key not in available:
                missing.append(key)
        return missing
