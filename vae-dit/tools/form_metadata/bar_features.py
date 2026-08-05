#!/usr/bin/env python3
"""Method-independent bar feature extraction.

These features are musical facts derived from parsed notes and grid tokens.
They are intentionally computed before any encoder strategy such as
autoencoder, edit distance, or clustering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

from core_data import BarRecord


@dataclass(frozen=True)
class DensityFeature:
    """Summarize note count, duration, and silence for one bar."""

    note_count: int
    note_density: float
    mean_duration_ql: float
    duration_variance: float
    short_note_ratio: float
    silence_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize density values without changing units."""
        return asdict(self)


@dataclass(frozen=True)
class PitchFeature:
    """Summarize pitch location, span, intervals, and token variance."""

    pitch_mean: float
    pitch_min: int | None
    pitch_max: int | None
    pitch_range: int
    pitch_intervals: List[int]
    mean_abs_interval: float
    token_variance: float
    sharing_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pitch statistics and interval values."""
        return asdict(self)


@dataclass(frozen=True)
class RhythmFeature:
    """Count rest, sustain, and note-on token states for one bar."""

    rest_count: int
    sustain_count: int
    note_on_count: int
    rest_ratio: float
    sustain_ratio: float
    note_on_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rhythm state counts and ratios."""
        return asdict(self)


@dataclass(frozen=True)
class PositionFeature:
    """Represent zero-based and normalized position within a source song."""

    bar_index: int
    source_bar_count: int | None
    position_ratio: float
    mod8: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bar position and source length metadata."""
        return asdict(self)


@dataclass(frozen=True)
class BarFeatures:
    """Combine the method-independent feature groups for one bar."""

    density: DensityFeature
    pitch: PitchFeature
    rhythm: RhythmFeature
    position: PositionFeature
    feature_vector: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize nested feature groups and the fixed numeric vector."""
        return {
            "density": self.density.to_dict(),
            "pitch": self.pitch.to_dict(),
            "rhythm": self.rhythm.to_dict(),
            "position": self.position.to_dict(),
            "feature_vector": list(self.feature_vector),
        }


class BarFeatureExtractor:
    """Compute reusable bar features from notes and tokens."""

    def __init__(self, rest_token: int = -1, sustain_token: int = -2) -> None:
        """Set the special grid-token values used by rhythm extraction."""
        self.rest_token = int(rest_token)
        self.sustain_token = int(sustain_token)

    def extract(self, bar: BarRecord) -> BarFeatures:
        """Derive density, pitch, rhythm, position, and an 8D feature vector."""
        density = self._density_feature(bar)
        pitch = self._pitch_feature(bar)
        rhythm = self._rhythm_feature(bar.relative_tokens)
        position = self._position_feature(bar)
        feature_vector = [
            float(density.note_density),
            float(density.mean_duration_ql),
            float(density.duration_variance),
            float(density.short_note_ratio),
            float(density.silence_ratio),
            float(pitch.pitch_mean),
            float(pitch.pitch_range),
            float(pitch.mean_abs_interval),
        ]
        return BarFeatures(
            density=density,
            pitch=pitch,
            rhythm=rhythm,
            position=position,
            feature_vector=feature_vector,
        )

    def apply(self, bar: BarRecord) -> BarFeatures:
        """Populate legacy BarRecord feature fields from the canonical feature object."""
        features = self.extract(bar)
        bar.token_variance = float(features.pitch.token_variance)
        bar.sharing_score = float(features.pitch.sharing_score)
        bar.pitch_intervals = list(features.pitch.pitch_intervals)
        bar.feature_vector = list(features.feature_vector)
        return features

    def _density_feature(self, bar: BarRecord) -> DensityFeature:
        """Measure event density and silence in quarter-length units."""
        notes = bar.notes
        if not notes:
            return DensityFeature(
                note_count=0,
                note_density=0.0,
                mean_duration_ql=0.0,
                duration_variance=0.0,
                short_note_ratio=0.0,
                silence_ratio=1.0,
            )
        durations = [float(note.duration_ql) for note in notes]
        mean_duration = sum(durations) / len(durations)
        duration_variance = sum((value - mean_duration) ** 2 for value in durations) / len(durations)
        short_ratio = sum(1 for value in durations if value < 0.5) / len(durations)
        total_duration = sum(durations)
        silence_ratio = max(0.0, 1.0 - total_duration / max(1e-9, float(bar.bar_length_ql)))
        return DensityFeature(
            note_count=len(notes),
            note_density=float(len(notes) / max(1e-9, float(bar.bar_length_ql))),
            mean_duration_ql=float(mean_duration),
            duration_variance=float(duration_variance),
            short_note_ratio=float(short_ratio),
            silence_ratio=float(silence_ratio),
        )

    def _pitch_feature(self, bar: BarRecord) -> PitchFeature:
        """Measure ordered note pitches and adjacent melodic intervals."""
        pitches = [int(note.pitch) for note in sorted(bar.notes, key=lambda note: note.onset_ql)]
        intervals = [right - left for left, right in zip(pitches, pitches[1:])]
        token_variance = self._token_variance(bar.relative_tokens)
        if not pitches:
            return PitchFeature(
                pitch_mean=0.0,
                pitch_min=None,
                pitch_max=None,
                pitch_range=0,
                pitch_intervals=[],
                mean_abs_interval=0.0,
                token_variance=token_variance,
                sharing_score=self._sharing_score(token_variance),
            )
        return PitchFeature(
            pitch_mean=float(sum(pitches) / len(pitches)),
            pitch_min=min(pitches),
            pitch_max=max(pitches),
            pitch_range=max(pitches) - min(pitches),
            pitch_intervals=intervals,
            mean_abs_interval=(
                float(sum(abs(value) for value in intervals) / len(intervals))
                if intervals else 0.0
            ),
            token_variance=token_variance,
            sharing_score=self._sharing_score(token_variance),
        )

    def _rhythm_feature(self, tokens: Sequence[int]) -> RhythmFeature:
        """Measure the occupancy ratio of each special token state."""
        values = [int(token) for token in tokens]
        total = len(values)
        rest_count = sum(1 for token in values if token == self.rest_token)
        sustain_count = sum(1 for token in values if token == self.sustain_token)
        note_on_count = sum(1 for token in values if token >= 0)
        return RhythmFeature(
            rest_count=rest_count,
            sustain_count=sustain_count,
            note_on_count=note_on_count,
            rest_ratio=self._ratio(rest_count, total),
            sustain_ratio=self._ratio(sustain_count, total),
            note_on_ratio=self._ratio(note_on_count, total),
        )

    def _position_feature(self, bar: BarRecord) -> PositionFeature:
        """Normalize a bar index over its source-song bar count."""
        total = int(bar.source_bar_count or 0)
        position_ratio = (
            max(0.0, min(1.0, float(bar.bar_index) / float(total - 1)))
            if total > 1 else 0.0
        )
        return PositionFeature(
            bar_index=int(bar.bar_index),
            source_bar_count=total if total > 0 else None,
            position_ratio=float(position_ratio),
            mod8=int(bar.bar_index) % 8,
        )

    def _token_variance(self, tokens: Sequence[int]) -> float:
        """Return variance over non-negative relative pitch tokens."""
        values = [float(token) for token in tokens if int(token) >= 0]
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return float(sum((value - mean) ** 2 for value in values) / len(values))

    def _sharing_score(self, variance: float) -> float:
        """Convert token variance to the historical inverse sharing score."""
        return float(1.0 / (1.0 + max(0.0, variance)))

    def _ratio(self, count: int, total: int) -> float:
        """Return a rounded ratio and handle empty token sequences."""
        if total <= 0:
            return 0.0
        return round(float(count) / float(total), 6)
