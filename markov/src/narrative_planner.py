#!/usr/bin/env python3
"""Global narrative planning for hierarchical generation.

NarrativePlanner does not generate notes.  It assigns each bar a dramatic
function so lower layers can interpret repeated material differently depending
on where it appears in the whole piece.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class NarrativeBar:
    """Narrative targets for one bar."""

    role: str
    tension: float
    intensity: float
    register_shift: float
    development_role: str
    cadence_bias: float

    def to_affect(self) -> Dict[str, float | str]:
        return {
            "narrative_role": self.role,
            "narrative_tension": self.tension,
            "narrative_intensity": self.intensity,
            "narrative_register_shift": self.register_shift,
            "narrative_cadence_bias": self.cadence_bias,
        }


class NarrativePlanner:
    """Create a piece-level dramatic arc."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("narrative", {})
        self.config = cfg if isinstance(cfg, dict) else {}
        self.enabled = bool(self.config.get("enabled", True))

    def build(
        self,
        measure_context: List[Tuple[str, int, str, int, int]],
        labels: List[int],
        seed: int = 0,
    ) -> Dict[int, NarrativeBar]:
        if not self.enabled or not measure_context:
            return {}

        n = len(measure_context)
        climax_pos = float(self.config.get("climax_position", 0.72))
        recap_pos = float(self.config.get("recap_position", 0.84))
        coda_pos = float(self.config.get("coda_position", 0.94))
        contrast_pos = float(self.config.get("contrast_position", 0.24))
        development_pos = float(self.config.get("development_position", 0.42))
        rng = np.random.RandomState(seed)

        plan: Dict[int, NarrativeBar] = {}
        for i, (section_label, local_bar, structural_role, _, section_len) in enumerate(measure_context):
            pos = i / max(1, n - 1)
            role = self._macro_role(pos, contrast_pos, development_pos, climax_pos, recap_pos, coda_pos)
            section_pos = local_bar / max(1, section_len - 1)
            is_cadence = section_len > 1 and local_bar >= section_len - 1 and structural_role not in ("FREE", "FLAT")
            role_tension = self._tension_curve(pos, climax_pos, coda_pos)
            role_intensity = self._intensity_curve(pos, climax_pos, coda_pos)
            register_shift = self._register_shift(role, pos, climax_pos)
            development_role = self._development_role(
                role,
                section_label,
                structural_role,
                section_pos,
                is_cadence,
                rng,
            )
            cadence_bias = 0.0
            if role in ("RECAP", "CODA"):
                cadence_bias += 0.18
            if role == "CLIMAX":
                cadence_bias += 0.08
            if is_cadence:
                cadence_bias += 0.20
            plan[i] = NarrativeBar(
                role=role,
                tension=role_tension,
                intensity=role_intensity,
                register_shift=register_shift,
                development_role=development_role,
                cadence_bias=float(np.clip(cadence_bias, 0.0, 0.45)),
            )
        return plan

    @staticmethod
    def _macro_role(
        pos: float,
        contrast_pos: float,
        development_pos: float,
        climax_pos: float,
        recap_pos: float,
        coda_pos: float,
    ) -> str:
        if pos >= coda_pos:
            return "CODA"
        if pos >= recap_pos:
            return "RECAP"
        if pos >= climax_pos:
            return "CLIMAX"
        if pos >= development_pos:
            return "DEVELOPMENT"
        if pos >= contrast_pos:
            return "CONTRAST"
        return "EXPOSITION"

    @staticmethod
    def _tension_curve(pos: float, climax_pos: float, coda_pos: float) -> float:
        if pos <= climax_pos:
            x = pos / max(0.01, climax_pos)
            return float(np.clip(0.18 + 0.62 * (x ** 1.35), 0.0, 1.0))
        x = (pos - climax_pos) / max(0.01, coda_pos - climax_pos)
        return float(np.clip(0.80 - 0.52 * x, 0.12, 0.85))

    @staticmethod
    def _intensity_curve(pos: float, climax_pos: float, coda_pos: float) -> float:
        if pos <= climax_pos:
            x = pos / max(0.01, climax_pos)
            return float(np.clip(0.30 + 0.58 * (x ** 1.20), 0.0, 1.0))
        x = (pos - climax_pos) / max(0.01, coda_pos - climax_pos)
        return float(np.clip(0.88 - 0.36 * x, 0.25, 0.95))

    @staticmethod
    def _register_shift(role: str, pos: float, climax_pos: float) -> float:
        if role == "CLIMAX":
            return 5.0
        if role == "DEVELOPMENT":
            return 2.0 + 2.0 * min(1.0, pos / max(0.01, climax_pos))
        if role == "RECAP":
            return -1.0
        if role == "CODA":
            return -3.0
        if role == "CONTRAST":
            return 1.0
        return 0.0

    @staticmethod
    def _development_role(
        narrative_role: str,
        section_label: str,
        structural_role: str,
        section_pos: float,
        is_cadence: bool,
        rng: np.random.RandomState,
    ) -> str:
        if is_cadence:
            return "CADENTIAL"
        if narrative_role == "EXPOSITION":
            return "STATEMENT" if structural_role == "NEW" else "REPEAT"
        if narrative_role == "CONTRAST":
            return "EXTENSION" if section_label == "FREE" else "SEQUENCE_DOWN"
        if narrative_role == "DEVELOPMENT":
            palette = ["FRAGMENT", "SEQUENCE_UP", "EXTENSION", "INTENSIFY"]
            return palette[int(rng.randint(0, len(palette)))]
        if narrative_role == "CLIMAX":
            return "INTENSIFY" if section_pos < 0.75 else "CADENTIAL"
        if narrative_role == "RECAP":
            return "REPEAT" if structural_role in ("REPEAT", "RETURN") else "RELAX"
        if narrative_role == "CODA":
            return "CADENTIAL" if section_pos > 0.45 else "RELAX"
        return "CONTRAST"
