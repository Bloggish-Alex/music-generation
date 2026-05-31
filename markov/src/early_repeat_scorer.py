#!/usr/bin/env python3
"""Soft scorer for early repeated motifs before dual-theme context exists."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from harmonic_planner import HarmonicPlanner
from hierarchical_types import BarGenerationTarget, NoteEvent


class EarlyRepeatCandidateScorer:
    """Penalize harmony-obscuring early repeat candidates.

    This scorer does not edit notes.  It only affects candidate selection for
    repeated motifs that occur before a partner theme is available.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("early_repeat_adaptation", {})
        if not isinstance(cfg, dict):
            cfg = {}
        scoring = cfg.get("scoring", {})
        if not isinstance(scoring, dict):
            scoring = {}
        self.enabled = bool(scoring.get("enabled", True))
        self.weight = float(scoring.get("weight", 1.15))
        self.chord_tone_target = float(scoring.get("chord_tone_target", 0.42))
        self.non_chord_resolution_target = float(
            scoring.get("non_chord_resolution_target", 0.72)
        )
        self.unresolved_target = float(scoring.get("unresolved_target", 0.55))
        self.chord_tone_weight = float(scoring.get("chord_tone_weight", 1.25))
        self.resolution_weight = float(scoring.get("resolution_weight", 1.35))
        self.unresolved_weight = float(scoring.get("unresolved_weight", 0.85))

    def diagnostics(
        self,
        notes: List[NoteEvent],
        role: str,
        target: BarGenerationTarget,
        partner_notes: Optional[List[NoteEvent]],
        config: Dict[str, Any],
    ) -> Dict[str, float]:
        if not self.enabled or role != "REPEAT" or not target.harmony:
            return {"active": 0.0, "score": 0.0}
        if target.dual_theme or partner_notes:
            return {"active": 0.0, "score": 0.0}

        diag = HarmonicPlanner.diagnostics(notes, target.harmony, config)
        chord_ratio = diag.get("chord_tone_ratio")
        resolution_cost = diag.get("non_chord_resolution_cost")
        unresolved_cost = diag.get("unresolved_dissonance_cost")
        if chord_ratio is None:
            chord_ratio = 0.0
        if resolution_cost is None:
            resolution_cost = 0.0
        if unresolved_cost is None:
            unresolved_cost = 0.0

        chord_deficit = max(0.0, self.chord_tone_target - float(chord_ratio))
        resolution_excess = max(0.0, float(resolution_cost) - self.non_chord_resolution_target)
        unresolved_excess = max(0.0, float(unresolved_cost) - self.unresolved_target)
        penalty = (
            self.chord_tone_weight * chord_deficit
            + self.resolution_weight * resolution_excess
            + self.unresolved_weight * unresolved_excess
        )
        score = -self.weight * penalty
        return {
            "active": 1.0,
            "score": float(score),
            "chord_tone_ratio": float(chord_ratio),
            "non_chord_resolution_cost": float(resolution_cost),
            "unresolved_dissonance_cost": float(unresolved_cost),
            "chord_deficit": float(chord_deficit),
            "resolution_excess": float(resolution_excess),
            "unresolved_excess": float(unresolved_excess),
        }
