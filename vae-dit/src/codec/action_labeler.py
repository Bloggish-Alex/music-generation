#!/usr/bin/env python3
"""Automatic action labeling for parsed bar sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from common.config_loader import ConfigView
from data.core import BarRecord, SongRecord


ACTION_INTRODUCE = "INTRODUCE"
ACTION_REPEAT = "REPEAT"
ACTION_RETURN = "RETURN"
ACTION_DEVELOP = "DEVELOP"
ACTION_CADENCE = "CADENCE"
ACTION_VARY = "VARY"


@dataclass(frozen=True)
class ActionLabelerConfig:
    """Configuration for heuristic action labeling."""

    repeat_similarity_threshold: float = 0.95
    return_similarity_threshold: float = 0.85
    return_min_consecutive: int = 2
    theme_anchor_bars: int = 4
    introduce_bars: int = 8
    develop_multiplier: float = 1.3
    phrase_length_bars: int = 8
    cadence_tail_bars: int = 2
    cadence_density_multiplier: float = 0.75
    cadence_variance_multiplier: float = 0.75
    tonic_pitch_classes: tuple[int, ...] = (0, 7)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ActionLabelerConfig":
        """Build labeler configuration from style config."""
        section = ConfigView(config).section("action_labeler")
        return cls(
            repeat_similarity_threshold=float(section.get("repeat_similarity_threshold", 0.95)),
            return_similarity_threshold=float(section.get("return_similarity_threshold", 0.85)),
            return_min_consecutive=int(section.get("return_min_consecutive", 2)),
            theme_anchor_bars=int(section.get("theme_anchor_bars", 4)),
            introduce_bars=int(section.get("introduce_bars", 8)),
            develop_multiplier=float(section.get("develop_multiplier", 1.3)),
            phrase_length_bars=int(section.get("phrase_length_bars", 8)),
            cadence_tail_bars=int(section.get("cadence_tail_bars", 2)),
            cadence_density_multiplier=float(section.get("cadence_density_multiplier", 0.75)),
            cadence_variance_multiplier=float(section.get("cadence_variance_multiplier", 0.75)),
            tonic_pitch_classes=tuple(int(x) for x in section.get("tonic_pitch_classes", [0, 7])),
        )


class BarFeatureVectorizer:
    """Compute compact vectors and scalar features for action labeling."""

    def vector(self, bar: BarRecord) -> np.ndarray:
        """Return a fixed vector combining rhythm and pitch-class content."""
        chroma = np.zeros(12, dtype=np.float32)
        rhythm = np.zeros(16, dtype=np.float32)
        if not bar.all_notes():
            return np.concatenate([chroma, rhythm], axis=0)
        slot_len = float(bar.bar_length_ql) / 16.0
        for note in bar.all_notes():
            chroma[int(note.pitch) % 12] += max(0.0, float(note.duration_ql))
            slot = min(15, max(0, int(round(float(note.onset_ql) / slot_len))))
            rhythm[slot] += 1.0
        return self._normalize(np.concatenate([chroma, rhythm], axis=0))

    def scalars(self, bar: BarRecord) -> Dict[str, Any]:
        """Return scalar statistics used by label rules."""
        notes = bar.all_notes()
        pitches = np.asarray([int(note.pitch) for note in notes], dtype=np.float32)
        intervals = np.abs(np.diff(pitches)) if pitches.size > 1 else np.asarray([], dtype=np.float32)
        density = float(len(notes)) / max(1.0, float(bar.bar_length_ql))
        last_pitch = int(pitches[-1]) if pitches.size else None
        return {
            "note_count": int(len(notes)),
            "density": float(density),
            "pitch_variance": float(np.var(pitches)) if pitches.size else 0.0,
            "interval_variance": float(np.var(intervals)) if intervals.size else 0.0,
            "last_pitch": last_pitch,
            "last_pitch_class": int(last_pitch % 12) if last_pitch is not None else None,
        }

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return cosine similarity in [0, 1] for nonnegative vectors."""
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 0.0:
            return 1.0 if float(np.linalg.norm(left) + np.linalg.norm(right)) <= 0.0 else 0.0
        return float(np.clip(float(np.dot(left, right)) / denom, 0.0, 1.0))

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a nonnegative vector by its L1 mass."""
        total = float(vector.sum())
        if total <= 0.0:
            return vector
        return vector / total


class ActionLabeler:
    """Assign one high-level temporal action label to each bar."""

    def __init__(self, config: ActionLabelerConfig, vectorizer: BarFeatureVectorizer | None = None) -> None:
        """Store action thresholds and the bar-level comparison vectorizer."""
        self.config = config
        self.vectorizer = vectorizer or BarFeatureVectorizer()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ActionLabeler":
        """Create an action labeler from the full style configuration."""
        return cls(ActionLabelerConfig.from_config(config))

    def label_song(self, song: SongRecord) -> Dict[str, Any]:
        """Label all bars in one song and return diagnostics."""
        bars = list(song.bars)
        vectors = [self.vectorizer.vector(bar) for bar in bars]
        scalars = [self.vectorizer.scalars(bar) for bar in bars]
        medians = self._feature_medians(scalars)
        labels = [None for _ in bars]
        reasons = [None for _ in bars]
        self._mark_repeat(bars, vectors, labels, reasons)
        self._mark_return(bars, vectors, labels, reasons)
        self._mark_develop(scalars, medians, labels, reasons)
        self._mark_cadence(scalars, medians, labels, reasons)
        self._fill_defaults(labels, reasons)
        for bar, label, reason in zip(bars, labels, reasons):
            bar.action = str(label)
            bar.action_reason = str(reason)
        return {
            "song_id": song.song_id,
            "bar_count": int(len(bars)),
            "feature_medians": medians,
            "action_counts": self._action_counts(labels),
            "bars": [
                {
                    "bar_index": int(bar.bar_index),
                    "action": str(label),
                    "reason": str(reason),
                    "features": scalars[index],
                }
                for index, (bar, label, reason) in enumerate(zip(bars, labels, reasons))
            ],
        }

    def _mark_repeat(
        self,
        bars: Sequence[BarRecord],
        vectors: Sequence[np.ndarray],
        labels: List[str | None],
        reasons: List[str | None],
    ) -> None:
        """Mark immediate repetitions against the previous bar."""
        for index in range(1, len(bars)):
            similarity = self.vectorizer.similarity(vectors[index], vectors[index - 1])
            if similarity >= self.config.repeat_similarity_threshold:
                labels[index] = ACTION_REPEAT
                reasons[index] = f"prev_similarity={similarity:.4f}"

    def _mark_return(
        self,
        bars: Sequence[BarRecord],
        vectors: Sequence[np.ndarray],
        labels: List[str | None],
        reasons: List[str | None],
    ) -> None:
        """Mark later runs that return to the opening theme anchor."""
        anchor_size = min(max(1, self.config.theme_anchor_bars), len(vectors))
        if anchor_size <= 0:
            return
        anchor = np.mean(np.stack(vectors[:anchor_size], axis=0), axis=0)
        consecutive: List[int] = []
        start_index = max(anchor_size, len(bars) // 2)
        for index in range(start_index, len(bars)):
            similarity = self.vectorizer.similarity(vectors[index], anchor)
            if similarity >= self.config.return_similarity_threshold:
                consecutive.append(index)
                continue
            self._flush_return_run(consecutive, labels, reasons)
            consecutive = []
        self._flush_return_run(consecutive, labels, reasons)

    def _flush_return_run(self, indices: Sequence[int], labels: List[str | None], reasons: List[str | None]) -> None:
        """Apply RETURN when a matching run is long enough."""
        if len(indices) < self.config.return_min_consecutive:
            return
        for index in indices:
            if labels[index] is None:
                labels[index] = ACTION_RETURN
                reasons[index] = f"theme_anchor_run_len={len(indices)}"

    def _mark_develop(
        self,
        scalars: Sequence[Dict[str, Any]],
        medians: Dict[str, float],
        labels: List[str | None],
        reasons: List[str | None],
    ) -> None:
        """Mark bars with high density and pitch variance as DEVELOP."""
        density_cutoff = medians["density"] * self.config.develop_multiplier
        variance_cutoff = medians["pitch_variance"] * self.config.develop_multiplier
        for index, feature in enumerate(scalars):
            if labels[index] is not None:
                continue
            if float(feature["density"]) > density_cutoff and float(feature["pitch_variance"]) > variance_cutoff:
                labels[index] = ACTION_DEVELOP
                reasons[index] = f"density>{density_cutoff:.4f};pitch_variance>{variance_cutoff:.4f}"

    def _mark_cadence(
        self,
        scalars: Sequence[Dict[str, Any]],
        medians: Dict[str, float],
        labels: List[str | None],
        reasons: List[str | None],
    ) -> None:
        """Mark low-density phrase endings landing on tonic or dominant."""
        phrase = max(1, int(self.config.phrase_length_bars))
        tail = max(1, int(self.config.cadence_tail_bars))
        density_cutoff = medians["density"] * self.config.cadence_density_multiplier
        variance_cutoff = medians["pitch_variance"] * self.config.cadence_variance_multiplier
        for index, feature in enumerate(scalars):
            phrase_position = index % phrase
            if phrase_position < phrase - tail:
                continue
            last_pc = feature.get("last_pitch_class")
            if last_pc is None or int(last_pc) not in self.config.tonic_pitch_classes:
                continue
            if float(feature["density"]) <= density_cutoff and float(feature["pitch_variance"]) <= variance_cutoff:
                labels[index] = ACTION_CADENCE
                reasons[index] = f"phrase_tail;last_pc={last_pc};low_density_low_variance"

    def _fill_defaults(self, labels: List[str | None], reasons: List[str | None]) -> None:
        """Fill remaining bars as INTRODUCE or VARY."""
        for index, label in enumerate(labels):
            if label is not None:
                continue
            if index < self.config.introduce_bars:
                labels[index] = ACTION_INTRODUCE
                reasons[index] = "default_opening_region"
            else:
                labels[index] = ACTION_VARY
                reasons[index] = "default_unmatched"

    def _feature_medians(self, scalars: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        """Compute robust per-song medians for label thresholds."""
        if not scalars:
            return {"density": 0.0, "pitch_variance": 0.0, "interval_variance": 0.0}
        return {
            "density": float(np.median([float(row["density"]) for row in scalars])),
            "pitch_variance": float(np.median([float(row["pitch_variance"]) for row in scalars])),
            "interval_variance": float(np.median([float(row["interval_variance"]) for row in scalars])),
        }

    def _action_counts(self, labels: Sequence[str | None]) -> Dict[str, int]:
        """Count final action labels."""
        counts: Dict[str, int] = {}
        for label in labels:
            key = str(label)
            counts[key] = counts.get(key, 0) + 1
        return counts
