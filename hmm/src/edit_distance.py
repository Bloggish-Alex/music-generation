#!/usr/bin/env python3
"""Edit-distance utilities over BarRecord token APIs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from Levenshtein import distance as levenshtein_distance

from config_loader import ConfigLoader, ConfigView
from core_data import BarRecord, SongRecord


@dataclass(frozen=True)
class RootPitchClassFeatureConfig:
    enabled: bool = False
    weight: float = 0.05


@dataclass(frozen=True)
class PositionBucketFeatureConfig:
    enabled: bool = False
    weight: float = 0.03
    bucket_count: int = 8


@dataclass(frozen=True)
class EditDistanceConfig:
    normalize_distance: bool = True
    token_offset: int = 2
    token_strategy: str = "relative"
    root_pitch_class_token_multiplier: int = 100
    root_pitch_class: RootPitchClassFeatureConfig = field(default_factory=RootPitchClassFeatureConfig)
    position_bucket: PositionBucketFeatureConfig = field(default_factory=PositionBucketFeatureConfig)


class EditDistanceDiagnosticsAnalyzer:
    """Summarize pairwise edit-distance matrices for training diagnostics."""

    def __init__(self, config: EditDistanceConfig) -> None:
        self.config = config

    def summarize(
        self,
        matrix: np.ndarray,
        thresholds: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        values = matrix[np.triu_indices_from(matrix, k=1)] if matrix.size else np.asarray([])
        values = values.astype(np.float64, copy=False)
        positive_values = values[values > 0.0] if values.size else np.asarray([])
        payload: Dict[str, Any] = {
            "config": asdict(self.config),
            "shape": [int(item) for item in matrix.shape],
            "pair_count": int(values.size),
            "zero_pair_count": int(np.sum(values == 0.0)) if values.size else 0,
            "nonzero_pair_count": int(positive_values.size),
            "min": self._value(np.min(values)) if values.size else 0.0,
            "max": self._value(np.max(values)) if values.size else 0.0,
            "mean": self._value(np.mean(values)) if values.size else 0.0,
            "std": self._value(np.std(values)) if values.size else 0.0,
            "median": self._value(np.median(values)) if values.size else 0.0,
            "nonzero_min": self._value(np.min(positive_values)) if positive_values.size else 0.0,
            "nonzero_median": self._value(np.median(positive_values)) if positive_values.size else 0.0,
            "quantiles": self._quantiles(values),
        }
        if thresholds:
            payload["threshold_counts"] = self._threshold_counts(values, thresholds)
        return payload

    def _quantiles(self, values: np.ndarray) -> Dict[str, float]:
        if not values.size:
            return {}
        return {
            str(percentile): self._value(np.percentile(values, percentile))
            for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        }

    def _threshold_counts(
        self,
        values: np.ndarray,
        thresholds: Sequence[float],
    ) -> List[Dict[str, Any]]:
        total = int(values.size)
        result: List[Dict[str, Any]] = []
        for threshold in thresholds:
            count = int(np.sum(values <= float(threshold))) if total else 0
            result.append({
                "threshold": self._value(threshold),
                "count": count,
                "ratio": round(float(count) / float(total), 6) if total else 0.0,
            })
        return result

    def _value(self, value: Any) -> float:
        return round(float(value), 6)


class EditDistanceCalculator:
    """Build pairwise Levenshtein matrices from BarRecord token strategies."""

    def __init__(self, config: EditDistanceConfig) -> None:
        self.config = config

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "EditDistanceCalculator":
        section = ConfigView(config).section("distance_matrix")
        context = section.get("context_features", {})
        context = context if isinstance(context, dict) else {}
        root_pitch_class = context.get("root_pitch_class", {})
        root_pitch_class = root_pitch_class if isinstance(root_pitch_class, dict) else {}
        position_bucket = context.get("position_bucket", {})
        position_bucket = position_bucket if isinstance(position_bucket, dict) else {}
        return cls(EditDistanceConfig(
            normalize_distance=bool(section.get("normalize_distance", True)),
            token_offset=int(section.get("token_offset", 2)),
            token_strategy=str(section.get("token_strategy", "relative")),
            root_pitch_class_token_multiplier=int(section.get("root_pitch_class_token_multiplier", 100)),
            root_pitch_class=RootPitchClassFeatureConfig(
                enabled=bool(root_pitch_class.get("enabled", False)),
                weight=float(root_pitch_class.get("weight", 0.05)),
            ),
            position_bucket=PositionBucketFeatureConfig(
                enabled=bool(position_bucket.get("enabled", False)),
                weight=float(position_bucket.get("weight", 0.03)),
                bucket_count=int(position_bucket.get("bucket_count", 8)),
            ),
        ))

    def build_matrix(self, bars: Sequence[BarRecord]) -> np.ndarray:
        encoded = [self._encode(self.tokens_for_bar(bar)) for bar in bars]
        matrix = np.zeros((len(encoded), len(encoded)), dtype=np.float64)
        for i in range(len(encoded)):
            for j in range(i + 1, len(encoded)):
                value = self._distance_pair(bars[i], bars[j], encoded[i], encoded[j])
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix

    def distance(self, left: BarRecord, right: BarRecord) -> float:
        matrix = self.build_matrix([left, right])
        return float(matrix[0, 1])

    def diagnostics(self, matrix: np.ndarray) -> Dict[str, Any]:
        return EditDistanceDiagnosticsAnalyzer(self.config).summarize(matrix)

    def tokens_for_bar(self, bar: BarRecord) -> List[int]:
        """Return the exact token sequence used by this calculator."""
        if self.config.token_strategy == "root_pitch_class_relative":
            root_pitch_class = self._bar_root_pitch_class(bar)
            if root_pitch_class is None:
                return bar.tokens_for_edit_distance("relative")
            base = int(root_pitch_class) * int(self.config.root_pitch_class_token_multiplier)
            return [
                base + int(token) if int(token) >= 0 else int(token)
                for token in bar.tokens_for_edit_distance("relative")
            ]
        return bar.tokens_for_edit_distance(self.config.token_strategy)

    def _encode(self, tokens: Sequence[int]) -> str:
        values = [int(token) + self.config.token_offset for token in tokens]
        if min(values, default=0) < 0:
            raise ValueError("token_offset is too small for special negative tokens.")
        return "".join(chr(value + 1) for value in values)

    def _distance_pair(
        self,
        left: BarRecord,
        right: BarRecord,
        encoded_left: str,
        encoded_right: str,
    ) -> float:
        value = float(levenshtein_distance(encoded_left, encoded_right))
        if self.config.normalize_distance:
            value = value / max(1, max(len(encoded_left), len(encoded_right)))
        value += self._root_pitch_class_distance(left, right) * self.config.root_pitch_class.weight
        value += self._position_bucket_distance(left, right) * self.config.position_bucket.weight
        return float(value)

    def _root_pitch_class_distance(self, left: BarRecord, right: BarRecord) -> float:
        if not self.config.root_pitch_class.enabled:
            return 0.0
        left_root = self._bar_root_pitch_class(left)
        right_root = self._bar_root_pitch_class(right)
        if left_root is None and right_root is None:
            return 0.0
        if left_root is None or right_root is None:
            return 1.0
        diff = abs(left_root - right_root) % 12
        return float(min(diff, 12 - diff) / 6.0)

    def _bar_root_pitch_class(self, bar: BarRecord) -> Optional[int]:
        pitches = [int(token) for token in bar.absolute_tokens if int(token) >= 0]
        if not pitches:
            return None
        return min(pitches) % 12

    def _position_bucket_distance(self, left: BarRecord, right: BarRecord) -> float:
        if not self.config.position_bucket.enabled:
            return 0.0
        left_bucket = self._position_bucket(left)
        right_bucket = self._position_bucket(right)
        if left_bucket is None or right_bucket is None:
            return 0.0
        bucket_count = max(1, self.config.position_bucket.bucket_count)
        if bucket_count <= 1:
            return 0.0
        return float(abs(left_bucket - right_bucket) / float(bucket_count - 1))

    def _position_bucket(self, bar: BarRecord) -> Optional[int]:
        total = int(bar.source_bar_count or 0)
        if total <= 1:
            return None
        ratio = max(0.0, min(1.0, float(bar.bar_index) / float(total - 1)))
        bucket_count = max(1, self.config.position_bucket.bucket_count)
        return min(bucket_count - 1, int(math.floor(ratio * bucket_count)))


class EditDistanceCLI:
    """Standalone CLI for computing a distance matrix from parsed songs."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Build edit-distance matrix from parsed SongRecord JSON.")
        parser.add_argument("--songs", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        payload = json.loads(args.songs.read_text(encoding="utf-8"))
        songs = [SongRecord.from_dict(item) for item in payload.get("songs", [])]
        bars = [bar for song in songs for bar in song.bars]
        calculator = EditDistanceCalculator.from_style_config(config)
        matrix = calculator.build_matrix(bars)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, distance=matrix)
        if args.diagnostics_output:
            args.diagnostics_output.write_text(
                json.dumps(calculator.diagnostics(matrix), indent=2),
                encoding="utf-8",
            )
        print(f"Wrote distance matrix {matrix.shape} -> {args.output}")


def main() -> None:
    EditDistanceCLI().run()


if __name__ == "__main__":
    main()
