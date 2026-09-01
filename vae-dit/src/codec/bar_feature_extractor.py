#!/usr/bin/env python3
"""Explicit bar-level feature extraction from encoded bar tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


BAR_FEATURE_NAMES = [
    "note_density",
    "active_density",
    "rest_density",
    "hold_density",
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "first_pitch",
    "last_pitch",
    "last_minus_first_pitch",
    "velocity_mean",
    "velocity_std",
    "rhythm_centroid",
    "rhythm_std",
    "rhythm_entropy",
    "onset_slot_coverage",
    "track0_note_density",
    "track1_note_density",
    "track2_note_density",
    "track0_active_density",
    "track1_active_density",
    "track2_active_density",
    "interval_mean",
    "interval_max",
    "interval_std",
]

V2_BAR_FEATURE_NAMES = [
    "global_note_on_density", "global_active_density", "global_rest_density", "global_hold_density",
    "active_pitch_mean", "active_pitch_std", "active_pitch_min", "active_pitch_max", "active_pitch_range", "active_pitch_first", "active_pitch_last", "active_pitch_delta",
    "active_velocity_mean", "active_velocity_std", "onset_centroid", "onset_spread", "onset_entropy", "occupied_slot_ratio",
    "melody_note_on_density", "harmony_set_note_on_density", "bass_note_on_density", "melody_active_density", "harmony_set_active_density", "bass_active_density",
    "onset_interval_mean", "onset_interval_max", "onset_interval_std", "mean_active_harmony_cardinality", "maximum_harmony_cardinality", "non_empty_harmony_slot_ratio", "relative_chroma_entropy",
]


class BarFeatureExtractor:
    """Compute explicit 27D music features for a [tracks, steps, features] bar tensor."""

    feature_names = BAR_FEATURE_NAMES

    def features(self, tensor: np.ndarray) -> np.ndarray:
        """Return explicit normalized music features for one [tracks, steps, features] bar."""
        values = np.asarray(tensor, dtype=np.float32)
        note = values[..., 2] > 0.5
        hold = values[..., 3] > 0.5
        rest = values[..., 1] > 0.5
        active = note | hold
        total_slots = max(1, int(values.shape[0] * values.shape[1]))
        note_pitches = values[..., 0][note]
        note_velocity = values[..., 4][note]
        onset_by_slot = note.sum(axis=0).astype(np.float32)
        track_note_density = note.sum(axis=1).astype(np.float32) / max(1, int(values.shape[1]))
        track_active_density = active.sum(axis=1).astype(np.float32) / max(1, int(values.shape[1]))
        first_pitch, last_pitch = self._first_last_pitch(values, note)
        if len(note_pitches):
            pitch_mean = float(np.mean(note_pitches))
            pitch_std = float(np.std(note_pitches))
            pitch_min = float(np.min(note_pitches))
            pitch_max = float(np.max(note_pitches))
            pitch_range = float(pitch_max - pitch_min)
            velocity_mean = float(np.mean(note_velocity)) if len(note_velocity) else 0.0
            velocity_std = float(np.std(note_velocity)) if len(note_velocity) else 0.0
            ordered = self._ordered_note_pitches(values, note)
            intervals = np.abs(np.diff(ordered)) if len(ordered) > 1 else np.zeros((0,), dtype=np.float32)
            interval_mean = float(np.mean(intervals)) if len(intervals) else 0.0
            interval_max = float(np.max(intervals)) if len(intervals) else 0.0
            interval_std = float(np.std(intervals)) if len(intervals) else 0.0
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = pitch_range = 0.0
            velocity_mean = velocity_std = 0.0
            interval_mean = interval_max = interval_std = 0.0
        if float(onset_by_slot.sum()) > 0.0:
            slots = np.arange(len(onset_by_slot), dtype=np.float32)
            denom = max(1, len(onset_by_slot) - 1)
            rhythm_centroid = float(np.sum(slots * onset_by_slot) / np.sum(onset_by_slot) / denom)
            rhythm_std = float(np.sqrt(np.sum(((slots / denom) - rhythm_centroid) ** 2 * onset_by_slot) / np.sum(onset_by_slot)))
            probabilities = onset_by_slot / float(onset_by_slot.sum())
            rhythm_entropy = float(-(probabilities * np.log(np.clip(probabilities, 1.0e-8, 1.0))).sum() / np.log(len(probabilities)))
        else:
            rhythm_centroid = rhythm_std = rhythm_entropy = 0.0
        # This fixed ordering is the persisted 27D feature contract used by
        # downstream dataset readers; BAR_FEATURE_NAMES names every column.
        return np.asarray([
            float(note.sum() / total_slots),
            float(active.sum() / total_slots),
            float(rest.sum() / total_slots),
            float(hold.sum() / total_slots),
            pitch_mean,
            pitch_std,
            pitch_min,
            pitch_max,
            pitch_range,
            first_pitch,
            last_pitch,
            float(last_pitch - first_pitch),
            velocity_mean,
            velocity_std,
            rhythm_centroid,
            rhythm_std,
            rhythm_entropy,
            float(np.count_nonzero(onset_by_slot) / max(1, len(onset_by_slot))),
            *[float(item) for item in track_note_density[:3]],
            *[float(item) for item in track_active_density[:3]],
            interval_mean,
            interval_max,
            interval_std,
        ], dtype=np.float32)

    def v2_features(self, tensor: np.ndarray, bar_context: np.ndarray) -> np.ndarray:
        """Return the contract-defined 31D V2 diagnostic vector."""
        values = np.asarray(tensor, dtype=np.float32); context = np.asarray(bar_context, dtype=np.float32)
        note, hold, rest = values[..., 2] > .5, values[..., 3] > .5, values[..., 1] > .5
        active = note | hold; pitches = values[..., 0][active]; velocities = values[..., 4][active]
        by_slot = note.sum(axis=0); occupied = active.any(axis=0); onset_slots = np.flatnonzero(by_slot)
        ordered = self._ordered_note_pitches(values, note)
        intervals = np.diff(onset_slots).astype(np.float32) if len(onset_slots) > 1 else np.zeros(0, dtype=np.float32)
        harmony = active[1:17].sum(axis=0).astype(np.float32)
        probs = by_slot / max(1.0, float(by_slot.sum()))
        chroma = context / max(1.0e-8, float(context.sum()))
        pitch_values = [float(np.mean(pitches)) if pitches.size else 0., float(np.std(pitches)) if pitches.size else 0., float(np.min(pitches)) if pitches.size else 0., float(np.max(pitches)) if pitches.size else 0., float(np.ptp(pitches)) if pitches.size else 0., float(ordered[0]) if ordered.size else 0., float(ordered[-1]) if ordered.size else 0., float(ordered[-1] - ordered[0]) if ordered.size else 0.]
        steps = max(1, values.shape[1] - 1)
        onset_centroid = float(np.dot(np.arange(values.shape[1]), by_slot) / max(1.0, float(by_slot.sum())) / steps)
        onset_spread = float(np.std(np.repeat(np.arange(values.shape[1]), by_slot.astype(int))) / steps) if by_slot.sum() else 0.0
        onset_entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1))).sum() / np.log(max(2, values.shape[1])))
        chroma_entropy = float(-(chroma * np.log(np.clip(chroma, 1e-8, 1))).sum())
        result = [float(note.mean()), float(active.mean()), float(rest.mean()), float(hold.mean()), *pitch_values, float(np.mean(velocities)) if velocities.size else 0.0, float(np.std(velocities)) if velocities.size else 0.0, onset_centroid, onset_spread, onset_entropy, float(occupied.mean()), float(note[0].mean()), float(note[1:17].sum() / (16 * values.shape[1])), float(note[17].mean()), float(active[0].mean()), float(active[1:17].sum() / (16 * values.shape[1])), float(active[17].mean()), float(intervals.mean()) if intervals.size else 0.0, float(intervals.max()) if intervals.size else 0.0, float(intervals.std()) if intervals.size else 0.0, float(harmony.mean()), float(harmony.max()), float((harmony > 0).mean()), chroma_entropy]
        return np.asarray(result, dtype=np.float32)

    def _first_last_pitch(self, tensor: np.ndarray, note: np.ndarray) -> tuple[float, float]:
        """Return first and last normalized note-on pitch."""
        pitches = self._ordered_note_pitches(tensor, note)
        if len(pitches) == 0:
            return 0.0, 0.0
        return float(pitches[0]), float(pitches[-1])

    def _ordered_note_pitches(self, tensor: np.ndarray, note: np.ndarray) -> np.ndarray:
        """Return note-on pitches ordered by slot then track."""
        items: List[float] = []
        for slot in range(tensor.shape[1]):
            for track in range(tensor.shape[0]):
                if bool(note[track, slot]):
                    items.append(float(tensor[track, slot, 0]))
        return np.asarray(items, dtype=np.float32)


class EncodedBarFeatureStore:
    """Read and write cached explicit bar feature artifacts."""

    FEATURE_FILE = "bar_features.npz"
    SUMMARY_FILE = "bar_feature_summary.json"

    def __init__(self, encoded_dir: str | Path) -> None:
        """Bind feature artifacts to one encoded dataset directory."""
        self.encoded_dir = Path(encoded_dir)

    def write(self, tensors_by_key: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute and write bar_features.npz and bar_feature_summary.json."""
        extractor = BarFeatureExtractor()
        features = {
            key: extractor.features(np.asarray(tensor, dtype=np.float32))
            for key, tensor in tensors_by_key.items()
        }
        self.encoded_dir.mkdir(parents=True, exist_ok=True)
        if features:
            np.savez_compressed(self.encoded_dir / self.FEATURE_FILE, **features)
            matrix = np.stack(list(features.values()), axis=0).astype(np.float32)
        else:
            np.savez_compressed(self.encoded_dir / self.FEATURE_FILE)
            matrix = np.zeros((0, len(BAR_FEATURE_NAMES)), dtype=np.float32)
        summary = self.summary(matrix, source="encoded")
        (self.encoded_dir / self.SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def matrix_for_rows(self, rows: Sequence[Dict[str, Any]], fallback_tensor_file: str = "bar_tensors.npz") -> tuple[np.ndarray, Dict[str, Any]]:
        """Return [rows, 27] features, using cached artifacts or legacy fallback."""
        feature_path = self.encoded_dir / self.FEATURE_FILE
        if feature_path.exists():
            archive = np.load(feature_path)
            missing_keys = []
            try:
                missing_keys = self._missing_archive_keys(archive, rows)
                if not missing_keys:
                    matrix = self._matrix_from_archive(archive, rows)
                    return matrix, {"feature_source": "cached_bar_features", "feature_path": str(feature_path)}
            finally:
                archive.close()
            tensor_path = self.encoded_dir / fallback_tensor_file
            if not tensor_path.exists():
                raise KeyError(
                    f"{self.FEATURE_FILE} is missing {len(missing_keys)} tensor keys and fallback tensor archive does not exist: {tensor_path}. "
                    f"First missing key: {missing_keys[0]}"
                )
            matrix = self._matrix_from_tensor_archive(rows, tensor_path)
            return matrix, {
                "feature_source": "computed_fallback_cache_key_miss",
                "feature_path": str(feature_path),
                "fallback_tensor_path": str(tensor_path),
                "missing_feature_key_count": int(len(missing_keys)),
                "missing_feature_key_examples": missing_keys[:10],
            }
        tensor_path = self.encoded_dir / fallback_tensor_file
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing {self.FEATURE_FILE} and fallback tensor archive: {tensor_path}")
        matrix = self._matrix_from_tensor_archive(rows, tensor_path)
        return matrix, {
            "feature_source": "computed_legacy_fallback",
            "feature_path": str(feature_path),
            "fallback_tensor_path": str(tensor_path),
        }

    def summary(self, matrix: np.ndarray, source: str) -> Dict[str, Any]:
        """Return compact feature artifact summary."""
        values = np.asarray(matrix, dtype=np.float32)
        return {
            "source": str(source),
            "feature_count": int(len(BAR_FEATURE_NAMES)),
            "feature_names": list(BAR_FEATURE_NAMES),
            "row_count": int(values.shape[0]),
            "shape": [int(item) for item in values.shape],
            "mean": [float(item) for item in values.mean(axis=0).tolist()] if values.size else [],
            "std": [float(item) for item in values.std(axis=0).tolist()] if values.size else [],
        }

    def _matrix_from_archive(self, archive: Any, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Load features by tensor_key order."""
        values = []
        for row in rows:
            key = str(row.get("tensor_key", ""))
            if key not in archive.files:
                raise KeyError(f"Missing tensor_key in bar_features.npz: {key}")
            values.append(np.asarray(archive[key], dtype=np.float32))
        return np.stack(values, axis=0).astype(np.float32)

    def _missing_archive_keys(self, archive: Any, rows: Sequence[Dict[str, Any]]) -> List[str]:
        """Return tensor keys from rows that are absent in an npz archive."""
        available = set(str(key) for key in archive.files)
        missing = []
        for row in rows:
            key = str(row.get("tensor_key", ""))
            if key not in available:
                missing.append(key)
        return missing

    def _matrix_from_tensor_archive(self, rows: Sequence[Dict[str, Any]], tensor_path: Path) -> np.ndarray:
        """Compute feature matrix from cached encoded bar tensors."""
        extractor = BarFeatureExtractor()
        archive = np.load(tensor_path)
        try:
            values = []
            for row in rows:
                key = str(row.get("tensor_key", ""))
                if key not in archive.files:
                    raise KeyError(f"Missing tensor_key in archive: {key}")
                values.append(extractor.features(np.asarray(archive[key], dtype=np.float32)))
            matrix = np.stack(values, axis=0).astype(np.float32)
        finally:
            archive.close()
        return matrix
