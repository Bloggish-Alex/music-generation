"""Schema-aware decoding of semantic bar tensors for evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SemanticTensorDecoder:
    """Maps named semantic features to observations without positional constants."""

    feature_names: tuple[str, ...]
    track_names: tuple[str, ...]
    pitch_scale_semitones: float

    @classmethod
    def from_schema(cls, schema: Mapping[str, Any]) -> "SemanticTensorDecoder":
        schema_version = schema.get("schema_version")
        # Raw v1 observation contracts embed this mapping under ``tensor_schema``
        # and establish its version from that enclosing contract.  Public v1
        # artifacts therefore may omit this redundant nested marker.
        if schema_version not in (None, "bar_tensor_schema.v1"):
            raise ValueError("Unsupported bar tensor schema.")
        features = tuple(schema.get("feature_names") or ())
        tracks = tuple(schema.get("track_names") or ())
        scale = schema.get("pitch_scale_semitones")
        required = {"relative_pitch", "is_note_on", "is_hold"}
        if not required.issubset(features) or not tracks or not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError("Bar tensor schema cannot decode semantic pitch activity.")
        return cls(features, tracks, float(scale))

    def feature_index(self, name: str) -> int:
        try:
            return self.feature_names.index(name)
        except ValueError as error:
            raise ValueError(f"Tensor schema has no '{name}' feature.") from error

    def active_mask(self, bars: np.ndarray) -> np.ndarray:
        onset = self.feature_index("is_note_on")
        hold = self.feature_index("is_hold")
        return (bars[..., onset] > 0.5) | (bars[..., hold] > 0.5)

    def absolute_pitch(self, bars: np.ndarray, base_pitches: np.ndarray) -> np.ndarray:
        pitch = self.feature_index("relative_pitch")
        anchors = np.asarray(base_pitches, dtype=float).reshape(-1)
        if bars.ndim != 4 or bars.shape[0] != anchors.shape[0]:
            raise ValueError("Bar tensor and base-pitch paths must align by bar.")
        return np.asarray(bars[..., pitch], dtype=float) * self.pitch_scale_semitones + anchors[:, None, None]
