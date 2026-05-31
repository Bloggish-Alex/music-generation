#!/usr/bin/env python3
"""Continuous dual-theme relationship layer.

This module does not assign fixed dramatic roles such as "conflict" or
"resolution".  It reads the learned theme identities/skeletons and produces a
small continuous influence vector for the current bar.  Downstream generators
can use that vector as a soft candidate-selection target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from hierarchical_types import ThemeIdentity, ThemeSkeleton


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mean(values: Iterable[float], fallback: float = 0.0) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else fallback


def _std(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.std(vals)) if vals else 0.0


def _flatten_ints(groups: Tuple[Tuple[int, ...], ...]) -> List[int]:
    return [int(x) for group in groups for x in group]


def _flatten_floats(groups: Tuple[Tuple[float, ...], ...]) -> List[float]:
    return [float(x) for group in groups for x in group]


def _circular_pc_features(pc: int) -> Tuple[float, float]:
    radians = 2.0 * math.pi * (int(pc) % 12) / 12.0
    return math.sin(radians), math.cos(radians)


@dataclass(frozen=True)
class ThemeEmbedding:
    """Compact continuous description of one learned theme family."""

    label: str
    vector: Tuple[float, ...]
    confidence: float
    register_mean: float
    tension_mean: float
    note_density: float


@dataclass(frozen=True)
class ThemeRelation:
    """Measured relationship between two theme embeddings."""

    source_label: str
    partner_label: str
    confidence: float
    affinity: float
    contrast: float
    register_delta: float
    tension_delta: float
    density_delta: float


class DualThemeDevelopment:
    """Produce soft A/B relationship targets from learned theme material.

    The layer is intentionally data-informed rather than rule-driven:
    all values are derived from distances between current theme embeddings and
    the current narrative energy.  It never edits notes directly.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("dual_theme_development", {})
        if not isinstance(cfg, dict):
            cfg = {}
        self.enabled = bool(cfg.get("enabled", True))
        self.min_partner_confidence = float(cfg.get("min_partner_confidence", 0.32))
        self.influence = float(cfg.get("influence", 0.42))
        self.register_coupling = float(cfg.get("register_coupling", 0.30))
        self.tension_coupling = float(cfg.get("tension_coupling", 0.34))
        self.similarity_freedom = float(cfg.get("similarity_freedom", 0.22))
        self.target_attraction_scale = float(cfg.get("target_attraction_scale", 0.10))

    def target_for_bar(
        self,
        label: str,
        local_bar: int,
        section_len: int,
        narrative_tension: float,
        narrative_intensity: float,
        theme_identities: Dict[str, ThemeIdentity],
        theme_skeletons: Dict[str, ThemeSkeleton],
    ) -> Optional[Dict[str, Any]]:
        """Return a soft influence vector for the current bar, if available."""
        if not self.enabled or label in ("FREE", "FLAT"):
            return None
        relation = self._best_relation(label, theme_identities, theme_skeletons)
        if relation is None or relation.confidence < self.min_partner_confidence:
            return None

        tension = _clip01(float(narrative_tension))
        intensity = _clip01(float(narrative_intensity))
        if section_len <= 0:
            phrase_pos = 0.5
        else:
            phrase_pos = _clip01((float(local_bar) + 0.5) / float(section_len))

        # The phrase curve keeps the influence audible inside the phrase while
        # avoiding a mechanical bar-by-bar switch.  It is continuous and style
        # configurable through the global influence/coupling parameters.
        phrase_curve = 0.35 + 0.65 * math.sin(math.pi * phrase_pos)
        relation_energy = (
            0.35
            + 0.35 * relation.contrast
            + 0.20 * relation.affinity
            + 0.10 * abs(relation.density_delta)
        )
        blend = _clip01(
            self.influence
            * relation.confidence
            * phrase_curve
            * relation_energy
            * (0.45 + 0.55 * intensity)
        )
        transform = _clip01(
            self.influence
            * relation.confidence
            * (0.35 + 0.65 * tension)
            * (0.40 + 0.60 * relation.contrast)
        )

        register_shift = float(
            np.clip(
                relation.register_delta * blend * self.register_coupling,
                -7.0,
                7.0,
            )
        )
        tension_bias = float(
            np.clip(
                relation.tension_delta * blend * self.tension_coupling
                + (relation.contrast - 0.5) * (tension - 0.5) * 0.18,
                -0.22,
                0.22,
            )
        )

        return {
            "source_label": relation.source_label,
            "partner_label": relation.partner_label,
            "confidence": round(relation.confidence, 4),
            "affinity": round(relation.affinity, 4),
            "contrast": round(relation.contrast, 4),
            "blend": round(blend, 4),
            "transform": round(transform, 4),
            "register_shift": round(register_shift, 4),
            "tension_bias": round(tension_bias, 4),
            "similarity_freedom": round(self.similarity_freedom * blend * relation.contrast, 4),
            "target_attraction_delta": round(
                self.target_attraction_scale * blend * (relation.affinity - relation.contrast),
                4,
            ),
            "exact_copy_penalty_delta": round(0.28 * transform * relation.contrast, 4),
        }

    def _best_relation(
        self,
        label: str,
        identities: Dict[str, ThemeIdentity],
        skeletons: Dict[str, ThemeSkeleton],
    ) -> Optional[ThemeRelation]:
        source = self._embedding(label, identities, skeletons)
        if source is None:
            return None
        best: Optional[ThemeRelation] = None
        best_weight = -1.0
        for partner_label in sorted(identities.keys()):
            if partner_label == label:
                continue
            partner = self._embedding(partner_label, identities, skeletons)
            if partner is None:
                continue
            relation = self._relation(source, partner)
            weight = relation.confidence * (0.55 * relation.affinity + 0.45 * relation.contrast)
            if weight > best_weight:
                best_weight = weight
                best = relation
        return best

    def _embedding(
        self,
        label: str,
        identities: Dict[str, ThemeIdentity],
        skeletons: Dict[str, ThemeSkeleton],
    ) -> Optional[ThemeEmbedding]:
        identity = identities.get(label)
        if identity is None:
            return None

        intervals = _flatten_ints(identity.bar_intervals)
        durations = _flatten_floats(identity.bar_durations)
        offsets = [float(x) for x in identity.bar_mean_offsets]
        sizes = [float(x) for x in identity.bar_sizes]
        skeleton = skeletons.get(label)
        bars = list(skeleton.bars) if skeleton is not None else []
        registers = [float(b.register_zone) for b in bars]
        tensions = [float(b.tension) for b in bars]
        note_counts = [float(b.note_count) for b in bars]
        contours = [int(x) for b in bars for x in b.contour]

        interval_abs = [abs(x) for x in intervals]
        ascending = sum(1 for x in intervals if x > 0)
        descending = sum(1 for x in intervals if x < 0)
        direction_balance = (
            (ascending - descending) / max(1.0, float(len(intervals)))
        )
        cadence_sin, cadence_cos = _circular_pc_features(identity.cadence_pc)

        vector = (
            _mean(interval_abs) / 12.0,
            _std(interval_abs) / 10.0,
            direction_balance,
            _mean([abs(x) >= 7 for x in intervals]),
            _mean(durations, 0.75) / 2.0,
            _std(durations) / 1.5,
            _mean([d <= 0.5 for d in durations]),
            _mean(sizes, 3.0) / 8.0,
            _std(sizes) / 4.0,
            _mean(offsets) / 12.0,
            _std(offsets) / 10.0,
            (max(offsets) - min(offsets)) / 16.0 if offsets else 0.0,
            _mean(registers, 64.0) / 84.0,
            _std(registers) / 12.0,
            _mean(tensions, 0.35),
            _std(tensions),
            _mean(note_counts, _mean(sizes, 3.0)) / 16.0,
            _mean([abs(x) for x in contours]) / 12.0 if contours else 0.0,
            cadence_sin,
            cadence_cos,
        )
        bar_count = max(len(identity.bar_intervals), len(bars))
        confidence = _clip01(
            0.20
            + 0.30 * min(1.0, bar_count / 4.0)
            + 0.30 * min(1.0, len(intervals) / 10.0)
            + 0.20 * min(1.0, len(bars) / 4.0)
        )
        return ThemeEmbedding(
            label=label,
            vector=tuple(float(x) for x in vector),
            confidence=confidence,
            register_mean=_mean(registers, 64.0),
            tension_mean=_mean(tensions, 0.35),
            note_density=_mean(note_counts, _mean(sizes, 3.0)),
        )

    @staticmethod
    def _relation(source: ThemeEmbedding, partner: ThemeEmbedding) -> ThemeRelation:
        a = np.array(source.vector, dtype=float)
        b = np.array(partner.vector, dtype=float)
        distance = float(np.linalg.norm(a - b) / max(1.0, math.sqrt(len(a))))
        affinity = _clip01(math.exp(-2.35 * distance))
        register_delta = partner.register_mean - source.register_mean
        tension_delta = partner.tension_mean - source.tension_mean
        density_delta = (partner.note_density - source.note_density) / 12.0
        contrast = _clip01(
            0.55 * (1.0 - affinity)
            + 0.22 * min(1.0, abs(register_delta) / 12.0)
            + 0.15 * min(1.0, abs(tension_delta))
            + 0.08 * min(1.0, abs(density_delta))
        )
        return ThemeRelation(
            source_label=source.label,
            partner_label=partner.label,
            confidence=min(source.confidence, partner.confidence),
            affinity=affinity,
            contrast=contrast,
            register_delta=register_delta,
            tension_delta=tension_delta,
            density_delta=density_delta,
        )
