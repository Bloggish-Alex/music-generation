"""Calibration for CandidateReranker scores.

The reranker estimates whether a candidate looks like corpus material.  That
raw probability is useful, but it should not be allowed to overrule strong
musical evidence in every context.  This module maps raw reranker logits into
generation logits with role-aware, diagnostic-aware calibration.

This is not note repair and does not introduce one-bar exceptions.  It only
controls how much trust to place in the learned prior for a given phrase role.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CalibrationResult:
    raw_logit: float
    calibrated_logit: float
    calibrated_probability: float
    adjustment: float
    good_cadence_confidence: float


class CandidateRerankerCalibration:
    """Role-conditioned mapping from raw reranker logit to usable logit."""

    @classmethod
    def calibrate(
        cls,
        raw_logit: float,
        features: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> CalibrationResult:
        cfg = config.get("calibration", {})
        if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", True)):
            probability = cls._sigmoid(raw_logit)
            return CalibrationResult(
                raw_logit=raw_logit,
                calibrated_logit=raw_logit,
                calibrated_probability=probability,
                adjustment=0.0,
                good_cadence_confidence=0.0,
            )

        role = str(features.get("cadence_role", "none"))
        calibrated = float(raw_logit)
        good_cadence_confidence = 0.0

        if role == "CADENCE":
            good_cadence_confidence = cls._good_cadence_confidence(features, cfg)
            min_logit = float(cfg.get("cadence_min_logit_when_good", -0.20))
            # Smooth lower-bound interpolation: weak cadence keeps the raw
            # logit; diagnostically strong cadence cannot receive a severe
            # reranker penalty.
            lower_bound = raw_logit + good_cadence_confidence * (min_logit - raw_logit)
            calibrated = max(calibrated, lower_bound)

        clip = float(config.get("logit_clip", 3.0))
        calibrated = max(-clip, min(clip, calibrated))
        probability = cls._sigmoid(calibrated)
        return CalibrationResult(
            raw_logit=float(raw_logit),
            calibrated_logit=float(calibrated),
            calibrated_probability=float(probability),
            adjustment=float(calibrated - raw_logit),
            good_cadence_confidence=float(good_cadence_confidence),
        )

    @classmethod
    def _good_cadence_confidence(
        cls,
        features: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> float:
        harmony_score = cls._as_float(features.get("harmony_score"))
        chord_ratio = cls._as_float(features.get("chord_tone_ratio"))
        resolution_cost = cls._as_float(features.get("non_chord_resolution_cost"))
        unresolved_cost = cls._as_float(features.get("unresolved_dissonance_cost"))
        final_chord = cls._as_float(features.get("final_is_chord_tone"))

        harmony_center = float(cfg.get("good_cadence_harmony_score", 2.0))
        chord_center = float(cfg.get("good_cadence_chord_tone_ratio", 0.80))
        resolution_center = float(cfg.get("good_cadence_resolution_cost", 0.12))
        unresolved_center = float(cfg.get("good_cadence_unresolved_cost", 0.12))

        harmony_term = cls._sigmoid((harmony_score - harmony_center) * 1.4)
        chord_term = cls._sigmoid((chord_ratio - chord_center) * 6.0)
        resolution_term = cls._sigmoid((resolution_center - resolution_cost) * 6.0)
        unresolved_term = cls._sigmoid((unresolved_center - unresolved_cost) * 6.0)
        final_term = 1.0 if final_chord >= 0.5 else 0.35

        confidence = (
            harmony_term
            + chord_term
            + resolution_term
            + unresolved_term
        ) / 4.0
        return max(0.0, min(1.0, confidence * final_term))

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
