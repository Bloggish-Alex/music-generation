#!/usr/bin/env python3
"""Soft candidate scorer for dual-theme development.

The scorer never rewrites notes.  It compares a candidate with the recalled
source bar and the partner-theme bar, then rewards candidates that occupy a
continuous position between them according to the dual-theme target.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from hierarchical_types import BarGenerationTarget, NoteEvent


class DualThemeCandidateScorer:
    """Score candidate bars against a learned source/partner relationship."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("dual_theme_development", {})
        if not isinstance(cfg, dict):
            cfg = {}
        scoring = cfg.get("candidate_scoring", {})
        if not isinstance(scoring, dict):
            scoring = {}
        self.enabled = bool(scoring.get("enabled", True))
        self.weight = float(scoring.get("weight", 1.65))
        self.partner_echo_weight = float(scoring.get("partner_echo_weight", 0.42))
        self.source_retention_weight = float(scoring.get("source_retention_weight", 0.34))
        self.blended_anchor_weight = float(scoring.get("blended_anchor_weight", 0.24))
        self.exact_copy_weight = float(scoring.get("exact_copy_weight", 0.22))
        self.partner_target_base = float(scoring.get("partner_target_base", 0.26))
        self.partner_target_blend = float(scoring.get("partner_target_blend", 0.72))
        self.partner_target_transform = float(scoring.get("partner_target_transform", 0.42))
        self.partner_target_contrast = float(scoring.get("partner_target_contrast", 0.20))
        self.source_target_base = float(scoring.get("source_target_base", 0.72))
        self.source_target_transform_relief = float(
            scoring.get("source_target_transform_relief", 0.34)
        )
        self.partner_target_width = float(scoring.get("partner_target_width", 0.26))
        self.source_target_width = float(scoring.get("source_target_width", 0.26))

    def score(
        self,
        candidate_notes: List[NoteEvent],
        source_notes: Optional[List[NoteEvent]],
        partner_notes: Optional[List[NoteEvent]],
        target: BarGenerationTarget,
    ) -> float:
        """Return a weighted soft score for the candidate's theme relation."""
        return float(self.diagnostics(
            candidate_notes,
            source_notes=source_notes,
            partner_notes=partner_notes,
            target=target,
        ).get("score", 0.0))

    def diagnostics(
        self,
        candidate_notes: List[NoteEvent],
        source_notes: Optional[List[NoteEvent]],
        partner_notes: Optional[List[NoteEvent]],
        target: BarGenerationTarget,
    ) -> Dict[str, float]:
        """Return score components for diagnostics and candidate selection."""
        if not self.enabled or not target.dual_theme:
            return {"active": 0.0, "score": 0.0}
        candidate = self._melody(candidate_notes)
        source = self._melody(source_notes or [])
        partner = self._melody(partner_notes or [])
        if len(candidate) < 2 or len(source) < 2 or len(partner) < 2:
            return {"active": 0.0, "score": 0.0}

        relation = target.dual_theme
        blend = self._clip01(float(relation.get("blend", 0.0)))
        transform = self._clip01(float(relation.get("transform", 0.0)))
        affinity = self._clip01(float(relation.get("affinity", 0.5)))
        contrast = self._clip01(float(relation.get("contrast", 0.5)))

        source_similarity = self._gesture_similarity(candidate, source)
        partner_similarity = self._gesture_similarity(candidate, partner)
        exact_source = self._exact_copy_ratio(candidate, source)
        exact_partner = self._exact_copy_ratio(candidate, partner)

        # Desired position is a continuum.  V1.6 deliberately gives the
        # partner target enough range to be audible, while preserving a
        # source target so the current theme remains identifiable.
        desired_partner = self._clip01(
            self.partner_target_base
            + self.partner_target_blend * blend
            + self.partner_target_transform * transform
            + self.partner_target_contrast * contrast * max(blend, transform)
        )
        desired_source = self._clip01(
            self.source_target_base
            + 0.18 * affinity
            - self.source_target_transform_relief * max(blend, transform)
        )
        partner_echo = self._target_fit(
            partner_similarity,
            desired_partner,
            width=self.partner_target_width,
        )
        source_retention = self._target_fit(
            source_similarity,
            desired_source,
            width=self.source_target_width,
        )

        candidate_mean = self._mean_pitch(candidate)
        source_mean = self._mean_pitch(source)
        partner_mean = self._mean_pitch(partner)
        anchor = (1.0 - desired_partner) * source_mean + desired_partner * partner_mean
        blended_anchor = math.exp(-abs(candidate_mean - anchor) / 7.5)

        exact_copy_cost = (
            exact_source * (0.65 + 0.35 * transform)
            + exact_partner * (0.40 + 0.60 * contrast)
        )

        raw = (
            self.partner_echo_weight * partner_echo
            + self.source_retention_weight * source_retention
            + self.blended_anchor_weight * blended_anchor
            - self.exact_copy_weight * exact_copy_cost
        )
        strength = self._clip01(0.35 + 0.65 * max(blend, transform))
        score = float(self.weight * strength * raw)
        return {
            "active": 1.0,
            "score": score,
            "source_similarity": float(source_similarity),
            "partner_similarity": float(partner_similarity),
            "desired_source": float(desired_source),
            "desired_partner": float(desired_partner),
            "partner_echo": float(partner_echo),
            "source_retention": float(source_retention),
            "blended_anchor": float(blended_anchor),
            "exact_source": float(exact_source),
            "exact_partner": float(exact_partner),
            "exact_copy_cost": float(exact_copy_cost),
            "strength": float(strength),
        }

    @staticmethod
    def _melody(notes: List[NoteEvent]) -> List[NoteEvent]:
        return sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )

    @staticmethod
    def _intervals(melody: List[NoteEvent]) -> List[int]:
        return [
            int(melody[i + 1].pitch - melody[i].pitch)
            for i in range(len(melody) - 1)
        ]

    def _gesture_similarity(self, a: List[NoteEvent], b: List[NoteEvent]) -> float:
        contour = self._contour_similarity(self._intervals(a), self._intervals(b))
        rhythm = self._rhythm_similarity(a, b)
        register = math.exp(-abs(self._mean_pitch(a) - self._mean_pitch(b)) / 10.0)
        return float(np.clip(0.50 * contour + 0.32 * rhythm + 0.18 * register, 0.0, 1.0))

    @staticmethod
    def _contour_similarity(a: List[int], b: List[int]) -> float:
        length = min(len(a), len(b))
        if length <= 0:
            return 0.0
        same = 0
        for x, y in zip(a[:length], b[:length]):
            if (x == 0 and y == 0) or (x * y > 0):
                same += 1
        return same / length

    @staticmethod
    def _rhythm_similarity(a: List[NoteEvent], b: List[NoteEvent]) -> float:
        length = min(len(a), len(b))
        if length <= 0:
            return 0.0
        distance = 0.0
        for x, y in zip(a[:length], b[:length]):
            distance += abs(float(x.duration_ql) - float(y.duration_ql))
            distance += 0.35 * abs(float(x.beat_offset) - float(y.beat_offset))
        return float(1.0 / (1.0 + distance / length))

    @staticmethod
    def _exact_copy_ratio(a: List[NoteEvent], b: List[NoteEvent]) -> float:
        length = min(len(a), len(b))
        if length <= 0:
            return 0.0
        same = 0
        for x, y in zip(a[:length], b[:length]):
            if (
                int(x.pitch) == int(y.pitch)
                and round(float(x.duration_ql), 3) == round(float(y.duration_ql), 3)
                and round(float(x.beat_offset), 3) == round(float(y.beat_offset), 3)
            ):
                same += 1
        return float(same / length)

    @staticmethod
    def _mean_pitch(melody: List[NoteEvent]) -> float:
        if not melody:
            return 64.0
        return float(np.mean([n.pitch for n in melody]))

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _target_fit(value: float, target: float, width: float) -> float:
        width = max(0.05, float(width))
        return float(np.clip(1.0 - abs(float(value) - float(target)) / width, 0.0, 1.0))
