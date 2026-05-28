#!/usr/bin/env python3
"""Data-informed scoring for theme-development candidates.

This module deliberately avoids editing notes. It evaluates generated
candidates with soft costs learned from the trained clusterer whenever
available, then lets the generator select the best candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from hierarchical_types import BarGenerationTarget, NoteEvent


@dataclass(frozen=True)
class DevelopmentScore:
    """Score plus compact diagnostics for a developed bar candidate."""

    total: float
    diagnostics: Dict[str, float]


class DevelopmentCandidateScorer:
    """Score theme-development candidates using corpus priors and soft goals."""

    def __init__(
        self,
        step_histograms: Optional[np.ndarray],
        pitch_histograms: Optional[np.ndarray],
        config: Dict,
    ) -> None:
        self._step_histograms = step_histograms
        self._pitch_histograms = pitch_histograms
        cfg = config.get("development_scorer", {})
        self._cfg = cfg if isinstance(cfg, dict) else {}

    def score(
        self,
        notes: List[NoteEvent],
        target: BarGenerationTarget,
        cluster_label: int,
        source_notes: Optional[List[NoteEvent]] = None,
        previous_notes: Optional[List[NoteEvent]] = None,
    ) -> DevelopmentScore:
        """Return a weighted soft score for one developed candidate."""
        melody = self._melody(notes)
        if not melody:
            return DevelopmentScore(-1e6, {"empty": 1.0})

        source = self._melody(source_notes or [])
        previous = self._melody(previous_notes or [])
        weights = self._weights()

        style = self._style_likelihood(melody, cluster_label)
        pitch_pc = self._pitch_pc_likelihood(melody, cluster_label)
        leap = self._leap_cost(melody, cluster_label, target.development_role)
        register = self._register_continuity_cost(melody, previous, target)
        direction = self._development_direction_fit(melody, target)
        cadence = self._cadence_fit(melody, target)
        similarity = self._theme_similarity_fit(melody, source, target)
        exact_copy = self._exact_copy_cost(melody, source)

        total = (
            weights["style"] * style
            + weights["pitch_pc"] * pitch_pc
            + weights["direction"] * direction
            + weights["cadence"] * cadence
            + weights["similarity"] * similarity
            - weights["leap"] * leap
            - weights["register"] * register
            - weights["exact_copy"] * exact_copy
        )
        diagnostics = {
            "style": style,
            "pitch_pc": pitch_pc,
            "leap_cost": leap,
            "register_cost": register,
            "direction": direction,
            "cadence": cadence,
            "similarity": similarity,
            "exact_copy": exact_copy,
        }
        return DevelopmentScore(float(total), diagnostics)

    def _weights(self) -> Dict[str, float]:
        default = {
            "style": 0.85,
            "pitch_pc": 0.35,
            "leap": 0.85,
            "register": 0.55,
            "direction": 0.45,
            "cadence": 0.70,
            "similarity": 0.65,
            "exact_copy": 0.55,
        }
        weights = self._cfg.get("weights", {})
        if isinstance(weights, dict):
            for key, value in weights.items():
                if key in default:
                    default[key] = float(value)
        return default

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

    @staticmethod
    def _step_bin(interval: int) -> int:
        value = abs(int(interval))
        if value == 0:
            return 0
        if value == 1:
            return 1
        if value == 2:
            return 2
        if value == 3:
            return 3
        if value <= 5:
            return 4
        if value <= 8:
            return 5
        return 6

    def _style_likelihood(self, melody: List[NoteEvent], cluster_label: int) -> float:
        intervals = self._intervals(melody)
        if not intervals:
            return 0.0
        prior = self._cluster_step_prior(cluster_label)
        return float(np.mean([math.log(prior[self._step_bin(iv)]) for iv in intervals]))

    def _pitch_pc_likelihood(self, melody: List[NoteEvent], cluster_label: int) -> float:
        if self._pitch_histograms is None or len(self._pitch_histograms) == 0:
            return 0.0
        idx = cluster_label % len(self._pitch_histograms)
        prior = np.asarray(self._pitch_histograms[idx], dtype=np.float64) + 1e-4
        prior /= prior.sum()
        return float(np.mean([math.log(prior[n.pitch % 12]) for n in melody]))

    def _cluster_step_prior(self, cluster_label: int) -> np.ndarray:
        if self._step_histograms is None or len(self._step_histograms) == 0:
            return np.array([0.12, 0.22, 0.24, 0.14, 0.15, 0.09, 0.04], dtype=np.float64)
        idx = cluster_label % len(self._step_histograms)
        prior = np.asarray(self._step_histograms[idx], dtype=np.float64) + 1e-4
        total = prior.sum()
        if total <= 0:
            return np.array([0.12, 0.22, 0.24, 0.14, 0.15, 0.09, 0.04], dtype=np.float64)
        return prior / total

    def _leap_cost(
        self,
        melody: List[NoteEvent],
        cluster_label: int,
        development_role: str,
    ) -> float:
        intervals = self._intervals(melody)
        if not intervals:
            return 0.0
        prior = self._cluster_step_prior(cluster_label)
        role_cfg = self._role_cfg(development_role)
        role_allowance = float(role_cfg.get("leap_allowance", 1.0))
        large_leap_cost = 0.0
        for iv in intervals:
            bin_idx = self._step_bin(iv)
            learned_cost = -math.log(max(1e-4, prior[bin_idx]))
            excess = max(0, abs(iv) - 7)
            large_leap_cost += learned_cost + (excess * excess * 0.10 / max(0.25, role_allowance))
        return float(large_leap_cost / len(intervals))

    def _register_continuity_cost(
        self,
        melody: List[NoteEvent],
        previous: List[NoteEvent],
        target: BarGenerationTarget,
    ) -> float:
        mean_pitch = float(np.mean([n.pitch for n in melody]))
        target_cost = abs(mean_pitch - target.register_target) / 12.0
        if not previous:
            return target_cost
        prev_mean = float(np.mean([n.pitch for n in previous]))
        jump = abs(mean_pitch - prev_mean)
        role_cfg = self._role_cfg(target.development_role)
        allowed = float(role_cfg.get("register_jump_allowance", 6.0))
        jump_cost = max(0.0, jump - allowed) / 12.0
        return float(target_cost + jump_cost)

    def _development_direction_fit(
        self,
        melody: List[NoteEvent],
        target: BarGenerationTarget,
    ) -> float:
        if len(melody) < 2:
            return 0.0
        first = melody[0].pitch
        last = melody[-1].pitch
        slope = last - first
        role = target.development_role
        if role in ("SEQUENCE_UP", "INTENSIFY"):
            return float(np.clip(slope / 7.0, -1.0, 1.0))
        if role in ("SEQUENCE_DOWN", "RELAX"):
            return float(np.clip(-slope / 7.0, -1.0, 1.0))
        if role == "CADENTIAL":
            return -abs(last - target.target_pitch) / 12.0
        return -abs(float(np.mean([n.pitch for n in melody])) - target.register_target) / 18.0

    def _cadence_fit(self, melody: List[NoteEvent], target: BarGenerationTarget) -> float:
        if target.cadence_strength <= 0.4:
            return 0.0
        last = melody[-1]
        pitch_fit = -abs(last.pitch - target.target_pitch) / 12.0
        duration_fit = min(1.0, last.duration_ql / 1.0)
        return float(pitch_fit + 0.35 * duration_fit)

    def _theme_similarity_fit(
        self,
        melody: List[NoteEvent],
        source: List[NoteEvent],
        target: BarGenerationTarget,
    ) -> float:
        if not source or len(melody) < 2 or len(source) < 2:
            return 0.0
        contour_sim = self._contour_similarity(self._intervals(melody), self._intervals(source))
        rhythm_sim = self._rhythm_similarity(melody, source)
        similarity = 0.65 * contour_sim + 0.35 * rhythm_sim
        if similarity < target.similarity_min:
            return -float(target.similarity_min - similarity)
        if similarity > target.similarity_max:
            return -float((similarity - target.similarity_max) * 1.25)
        center = 0.5 * (target.similarity_min + target.similarity_max)
        width = max(0.05, target.similarity_max - target.similarity_min)
        return float(1.0 - abs(similarity - center) / width)

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
            distance += abs(x.duration_ql - y.duration_ql)
            distance += 0.35 * abs(x.beat_offset - y.beat_offset)
        return float(1.0 / (1.0 + distance / length))

    @staticmethod
    def _exact_copy_cost(melody: List[NoteEvent], source: List[NoteEvent]) -> float:
        length = min(len(melody), len(source))
        if length <= 0:
            return 0.0
        same = 0
        for x, y in zip(melody[:length], source[:length]):
            if (
                x.pitch == y.pitch
                and round(x.duration_ql, 3) == round(y.duration_ql, 3)
                and round(x.beat_offset, 3) == round(y.beat_offset, 3)
            ):
                same += 1
        return same / length

    def _role_cfg(self, role: str) -> Dict:
        theme_cfg = self._cfg.get("roles", {})
        if not isinstance(theme_cfg, dict):
            return {}
        role_cfg = theme_cfg.get(role, {})
        return role_cfg if isinstance(role_cfg, dict) else {}
