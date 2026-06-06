#!/usr/bin/env python3
"""Lightweight harmonic context annotation for edit-distance tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from config_loader import ConfigView


@dataclass(frozen=True)
class HarmonicAnalysisConfig:
    key: str = "C"
    mode: str = "major"
    degree_stride: int = 1000
    relative_pitch_stride: int = 20
    interval_center: int = 10
    interval_clip: int = 9
    rest_token: int = -1
    sustain_token: int = -2
    unknown_degree_id: int = 0
    bind_rest_to_degree: bool = False


@dataclass(frozen=True)
class HarmonicDegree:
    name: str
    degree_id: int


class PitchClassKeyResolver:
    """Resolve user-facing key names to tonic pitch classes."""

    SEMITONES = {
        "C": 0,
        "B#": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "FB": 4,
        "E#": 5,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
        "CB": 11,
    }

    def resolve(self, key: str) -> int:
        normalized = str(key).strip().upper()
        if normalized not in self.SEMITONES:
            raise ValueError(f"Unsupported key '{key}'. Use names like C, F#, Bb.")
        return int(self.SEMITONES[normalized])


class DegreePitchClassMap:
    """Map pitch classes to nearest diatonic degree ids."""

    DEGREE_NAMES = {
        0: "UNKNOWN",
        1: "I",
        2: "II",
        3: "III",
        4: "IV",
        5: "V",
        6: "VI",
        7: "VII",
    }
    MODE_OFFSETS = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
    }

    def __init__(self, key: str, mode: str) -> None:
        self.tonic_pc = PitchClassKeyResolver().resolve(key)
        self.mode = str(mode).lower()
        if self.mode not in self.MODE_OFFSETS:
            raise ValueError(f"Unsupported mode '{mode}'. Expected major or minor.")
        self.degree_pc_to_id = {
            (self.tonic_pc + offset) % 12: index + 1
            for index, offset in enumerate(self.MODE_OFFSETS[self.mode])
        }

    def degree_for_pitch_class(self, pitch_class: int) -> HarmonicDegree:
        degree_id = int(self.degree_pc_to_id.get(int(pitch_class) % 12, 0))
        return HarmonicDegree(self.DEGREE_NAMES[degree_id], degree_id)


class BarHarmonicDegreeAnnotator:
    """Choose one coarse Roman-degree context for a bar."""

    def __init__(self, config: HarmonicAnalysisConfig) -> None:
        self.config = config
        self.degree_map = DegreePitchClassMap(config.key, config.mode)

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "BarHarmonicDegreeAnnotator":
        engine = ConfigView(config).section("harmonic_engine")
        distance = ConfigView(config).section("distance_matrix")
        grid = ConfigView(config).section("grid_tokenizer")
        return cls(HarmonicAnalysisConfig(
            key=str(engine.get("key", "C")),
            mode=str(engine.get("mode", "major")),
            degree_stride=int(distance.get("harmonic_degree_stride", 100)),
            relative_pitch_stride=int(distance.get("relative_pitch_stride", 20)),
            interval_center=int(distance.get("interval_center", 10)),
            interval_clip=int(distance.get("interval_clip", 9)),
            rest_token=int(grid.get("rest_token", -1)),
            sustain_token=int(grid.get("sustain_token", -2)),
            unknown_degree_id=int(distance.get("unknown_degree_id", 0)),
            bind_rest_to_degree=bool(distance.get("bind_rest_to_degree", False)),
        ))

    def annotate(self, pitches: Sequence[int]) -> HarmonicDegree:
        counts: Dict[int, int] = {}
        for pitch in pitches:
            degree = self.degree_map.degree_for_pitch_class(int(pitch) % 12)
            if degree.degree_id == 0:
                continue
            counts[degree.degree_id] = counts.get(degree.degree_id, 0) + 1
        if not counts:
            return HarmonicDegree("UNKNOWN", self.config.unknown_degree_id)
        degree_id = max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
        return HarmonicDegree(DegreePitchClassMap.DEGREE_NAMES[degree_id], degree_id)


class HarmonicTokenEncoder:
    """Bind relative edit-distance tokens to a bar-level harmonic degree."""

    def __init__(self, config: HarmonicAnalysisConfig) -> None:
        self.config = config

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "HarmonicTokenEncoder":
        return cls(BarHarmonicDegreeAnnotator.from_style_config(config).config)

    def encode(self, relative_tokens: Sequence[int], degree_id: Optional[int]) -> List[int]:
        if degree_id is None or int(degree_id) == self.config.unknown_degree_id:
            return [int(token) for token in relative_tokens]
        prefix = int(degree_id) * self.config.degree_stride
        encoded = []
        for token in relative_tokens:
            value = int(token)
            if value == self.config.rest_token and not self.config.bind_rest_to_degree:
                encoded.append(value)
            else:
                encoded.append(prefix + value)
        return encoded

    def encode_with_intervals(self, relative_tokens: Sequence[int], degree_id: Optional[int]) -> List[int]:
        if degree_id is None or int(degree_id) == self.config.unknown_degree_id:
            return [int(token) for token in relative_tokens]
        prefix = int(degree_id) * self.config.degree_stride
        encoded = []
        previous_note: Optional[int] = None
        for token in relative_tokens:
            value = int(token)
            if value == self.config.rest_token and not self.config.bind_rest_to_degree:
                encoded.append(value)
                continue
            if value == self.config.sustain_token:
                encoded.append(prefix + value)
                continue
            if value < 0:
                encoded.append(prefix + value)
                continue
            interval = 0 if previous_note is None else value - previous_note
            interval_bucket = self._interval_bucket(interval)
            encoded.append(prefix + value * self.config.relative_pitch_stride + interval_bucket)
            previous_note = value
        return encoded

    def _interval_bucket(self, interval: int) -> int:
        clipped = max(-self.config.interval_clip, min(self.config.interval_clip, int(interval)))
        return int(self.config.interval_center + clipped)
