#!/usr/bin/env python3
"""Optional harmonic planning and scoring layer.

The planner is intentionally self-contained. The generator can turn it on or
off through config without changing the section/theme/development algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hierarchical_types import NoteEvent


_CHORD_QUALITIES: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "dom7": (0, 4, 7, 10),
    "min7": (0, 3, 7, 10),
}

_MAJOR_ROMANS: Dict[str, Tuple[int, str, str]] = {
    "I": (0, "maj", "T"),
    "ii": (2, "min", "PD"),
    "IV": (5, "maj", "PD"),
    "V": (7, "dom7", "D"),
    "vi": (9, "min", "T"),
    "V/V": (2, "dom7", "D"),
}


@dataclass(frozen=True)
class HarmonicContext:
    """Harmony target for one generated bar."""

    tonic_pc: int
    roman: str
    function: str
    root_pc: int
    quality: str
    chord_tones: Tuple[int, ...]
    cadence_role: str
    bass_pc: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tonic_pc": self.tonic_pc,
            "roman": self.roman,
            "function": self.function,
            "root_pc": self.root_pc,
            "quality": self.quality,
            "chord_tones": list(self.chord_tones),
            "cadence_role": self.cadence_role,
            "bass_pc": self.bass_pc,
        }


class HarmonicPlanner:
    """Build harmonic contexts and score melody/bass fit."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("harmony", {})
        self.config = cfg if isinstance(cfg, dict) else {}
        self.enabled = bool(self.config.get("enabled", False))

    def build_plan(
        self,
        measure_context: List[Tuple[str, int, str, int, int]],
        measure_affects: Dict[int, Dict[str, float]],
        global_tonic_pc: int,
    ) -> Dict[int, HarmonicContext]:
        """Assign a simple functional harmony to each bar."""
        if not self.enabled:
            return {}

        plan: Dict[int, HarmonicContext] = {}
        for bar_index, (_, local_bar, role, _, section_len) in enumerate(measure_context):
            affect = measure_affects.get(bar_index, {})
            tonic_pc = int(affect.get("tonal_pc", global_tonic_pc)) % 12
            phrase_pos = local_bar / max(1, section_len - 1)
            structural_bar = role not in ("FREE", "FLAT")
            is_last = structural_bar and local_bar >= section_len - 1
            is_penultimate = structural_bar and local_bar == section_len - 2
            roman = self._choose_roman(phrase_pos, is_penultimate, is_last, role)
            plan[bar_index] = self._context_from_roman(tonic_pc, roman, is_last, is_penultimate)
        return plan

    def _choose_roman(
        self,
        phrase_pos: float,
        is_penultimate: bool,
        is_last: bool,
        structural_role: str,
    ) -> str:
        if is_last:
            cadence = self.config.get("cadence", {})
            if isinstance(cadence, dict) and structural_role in ("RETURN", "VARIANT", "REPEAT"):
                return str(cadence.get("final", "I"))
            return "I"
        if is_penultimate:
            return "V"

        progression = self.config.get("progression", ["I", "vi", "IV", "V"])
        if not isinstance(progression, list) or not progression:
            progression = ["I", "vi", "IV", "V"]
        idx = min(len(progression) - 1, int(phrase_pos * len(progression)))
        roman = str(progression[idx])
        return roman if roman in _MAJOR_ROMANS else "I"

    @staticmethod
    def _context_from_roman(
        tonic_pc: int,
        roman: str,
        is_last: bool,
        is_penultimate: bool,
    ) -> HarmonicContext:
        degree, quality, function = _MAJOR_ROMANS.get(roman, _MAJOR_ROMANS["I"])
        root_pc = (tonic_pc + degree) % 12
        intervals = _CHORD_QUALITIES[quality]
        chord_tones = tuple((root_pc + interval) % 12 for interval in intervals)
        cadence_role = "CADENCE" if is_last else "CADENCE_PREP" if is_penultimate else function
        return HarmonicContext(
            tonic_pc=tonic_pc,
            roman=roman,
            function=function,
            root_pc=root_pc,
            quality=quality,
            chord_tones=chord_tones,
            cadence_role=cadence_role,
            bass_pc=root_pc,
        )

    @staticmethod
    def nearest_chord_pitch(
        pitch: int,
        harmony: Optional[Dict[str, Any]],
        prefer_root: bool = False,
    ) -> int:
        """Nearest pitch belonging to the current harmony."""
        if not harmony:
            return pitch
        chord_tones = [int(pc) % 12 for pc in harmony.get("chord_tones", [])]
        if not chord_tones:
            return pitch
        root_pc = int(harmony.get("root_pc", chord_tones[0])) % 12
        pcs = [root_pc] if prefer_root else chord_tones
        candidates = [
            pitch + offset
            for offset in range(-12, 13)
            if (pitch + offset) % 12 in pcs
        ]
        if not candidates and prefer_root:
            candidates = [
                pitch + offset
                for offset in range(-12, 13)
                if (pitch + offset) % 12 in chord_tones
            ]
        return min(candidates, key=lambda p: (abs(p - pitch), p)) if candidates else pitch

    @staticmethod
    def score_melody(
        notes: List[NoteEvent],
        harmony: Optional[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> float:
        """Soft score for melody/harmony agreement."""
        if not harmony:
            return 0.0
        cfg = config.get("harmony", {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return 0.0
        weights = cfg.get("scoring", {})
        if not isinstance(weights, dict):
            weights = {}

        melody = sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )
        if not melody:
            return -1.0

        chord_tones = set(int(pc) % 12 for pc in harmony.get("chord_tones", []))
        root_pc = int(harmony.get("root_pc", 0)) % 12
        cadence_role = str(harmony.get("cadence_role", ""))
        strong_beats = set(float(x) for x in cfg.get("strong_beats", [0.0, 2.0]))
        tolerance = float(cfg.get("strong_beat_tolerance", 0.08))

        chord_hits = sum(1 for n in melody if n.pitch % 12 in chord_tones)
        chord_ratio = chord_hits / max(1, len(melody))
        strong = [
            n for n in melody
            if any(abs((n.beat_offset % 4.0) - beat) <= tolerance for beat in strong_beats)
        ]
        strong_hits = sum(1 for n in strong if n.pitch % 12 in chord_tones)
        strong_ratio = strong_hits / max(1, len(strong))

        score = 0.0
        score += float(weights.get("chord_tone_ratio", 0.70)) * (chord_ratio - 0.55)
        score += float(weights.get("strong_beat_chord_tone", 1.10)) * (strong_ratio - 0.70)

        if cadence_role == "CADENCE":
            last = melody[-1]
            cadence_pc_ok = last.pitch % 12 in chord_tones
            root_bonus = 1.0 if last.pitch % 12 == root_pc else 0.0
            duration_bonus = min(1.0, last.duration_ql / 1.0)
            score += float(weights.get("cadence_fit", 1.25)) * (
                (0.75 if cadence_pc_ok else -0.75) + 0.25 * root_bonus + 0.25 * duration_bonus
            )

        score -= float(weights.get("unresolved_dissonance", 0.45)) * HarmonicPlanner._unresolved_cost(
            melody, chord_tones
        )
        score -= float(weights.get("non_chord_resolution", 0.65)) * HarmonicPlanner._non_chord_resolution_cost(
            melody, chord_tones
        )
        return float(score)

    @staticmethod
    def diagnostics(
        notes: List[NoteEvent],
        harmony: Optional[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return interpretable melody/harmony fit metrics for one bar."""
        if not harmony:
            return {
                "enabled": False,
                "chord_tone_ratio": None,
                "strong_beat_chord_tone_ratio": None,
                "unresolved_dissonance_cost": None,
                "score": 0.0,
            }
        melody = sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )
        bass = sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "bass"],
            key=lambda n: (n.beat_offset, n.pitch),
        )
        chord_tones = set(int(pc) % 12 for pc in harmony.get("chord_tones", []))
        root_pc = int(harmony.get("root_pc", 0)) % 12
        cfg = config.get("harmony", {})
        if not isinstance(cfg, dict):
            cfg = {}
        strong_beats = set(float(x) for x in cfg.get("strong_beats", [0.0, 2.0]))
        tolerance = float(cfg.get("strong_beat_tolerance", 0.08))
        strong = [
            n for n in melody
            if any(abs((n.beat_offset % 4.0) - beat) <= tolerance for beat in strong_beats)
        ]
        chord_ratio = (
            sum(1 for n in melody if n.pitch % 12 in chord_tones) / len(melody)
            if melody else None
        )
        strong_ratio = (
            sum(1 for n in strong if n.pitch % 12 in chord_tones) / len(strong)
            if strong else None
        )
        bass_pc = bass[0].pitch % 12 if bass else None
        return {
            "enabled": True,
            "roman": harmony.get("roman"),
            "function": harmony.get("function"),
            "root_pc": root_pc,
            "quality": harmony.get("quality"),
            "chord_tones": list(harmony.get("chord_tones", [])),
            "cadence_role": harmony.get("cadence_role"),
            "melody_note_count": len(melody),
            "bass_pitch": bass[0].pitch if bass else None,
            "bass_pc": bass_pc,
            "bass_is_root": bass_pc == root_pc if bass_pc is not None else None,
            "bass_is_root_or_fifth": (
                bass_pc in {root_pc, (root_pc + 7) % 12}
                if bass_pc is not None else None
            ),
            "chord_tone_ratio": chord_ratio,
            "strong_beat_chord_tone_ratio": strong_ratio,
            "unresolved_dissonance_cost": HarmonicPlanner._unresolved_cost(melody, chord_tones),
            "non_chord_resolution_cost": HarmonicPlanner._non_chord_resolution_cost(melody, chord_tones),
            "score": HarmonicPlanner.score_melody(notes, harmony, config),
        }

    @staticmethod
    def _unresolved_cost(melody: List[NoteEvent], chord_tones: set[int]) -> float:
        if len(melody) < 2:
            return 0.0
        cost = 0.0
        count = 0
        for current, nxt in zip(melody, melody[1:]):
            if current.pitch % 12 in chord_tones:
                continue
            count += 1
            if abs(nxt.pitch - current.pitch) <= 2:
                cost += 0.15
            else:
                cost += 1.0
        return cost / max(1, count)

    @staticmethod
    def _non_chord_resolution_cost(melody: List[NoteEvent], chord_tones: set[int]) -> float:
        """Penalty for non-chord tones that do not behave as passing tones.

        A non-chord tone is considered acceptable when it is approached and
        left by small steps, and at least one neighbouring tone is a chord
        tone.  This keeps expressive passing motion while discouraging random
        unresolved chromatic notes.
        """
        if len(melody) < 2:
            return 0.0
        cost = 0.0
        count = 0
        for i, current in enumerate(melody):
            if current.pitch % 12 in chord_tones:
                continue
            count += 1
            prev_note = melody[i - 1] if i > 0 else None
            next_note = melody[i + 1] if i + 1 < len(melody) else None
            prev_step = prev_note is not None and abs(current.pitch - prev_note.pitch) <= 2
            next_step = next_note is not None and abs(next_note.pitch - current.pitch) <= 2
            prev_chord = prev_note is not None and prev_note.pitch % 12 in chord_tones
            next_chord = next_note is not None and next_note.pitch % 12 in chord_tones
            if prev_step and next_step and (prev_chord or next_chord):
                cost += 0.10
            elif next_note is not None and next_step and next_chord:
                cost += 0.20
            elif prev_note is not None and prev_step and prev_chord and next_note is None:
                cost += 0.35
            else:
                cost += 1.0
        return cost / max(1, count)

    @staticmethod
    def add_bass_note(
        notes: List[NoteEvent],
        harmony: HarmonicContext,
        bar_length_ql: float,
        config: Dict[str, Any],
        rng: np.random.RandomState,
    ) -> None:
        """Append a bass note based on harmonic root/fifth targets."""
        cfg = config.get("harmony", {})
        bass_cfg = cfg.get("bass", {}) if isinstance(cfg, dict) else {}
        if not isinstance(bass_cfg, dict) or not bass_cfg.get("enabled", True):
            return

        low = int(bass_cfg.get("low", 36))
        high = int(bass_cfg.get("high", 60))
        root_weight = float(bass_cfg.get("root_weight", 0.78))
        bass_pc = harmony.bass_pc if rng.random() < root_weight else (harmony.root_pc + 7) % 12
        reference = int(bass_cfg.get("reference", 45))
        pitch = HarmonicPlanner._nearest_pitch(bass_pc, reference)
        while pitch < low:
            pitch += 12
        while pitch > high:
            pitch -= 12

        sustain = float(bass_cfg.get("sustain_fraction", 0.88))
        velocity_scale = float(bass_cfg.get("velocity_scale", 0.66))
        avg_vel = int(sum(n.velocity for n in notes if n.velocity > 0)
                      / max(1, sum(1 for n in notes if n.velocity > 0)))
        velocity = max(25, int(avg_vel * velocity_scale))
        notes.append(NoteEvent(
            pitch=int(pitch),
            duration_ql=max(0.25, min(bar_length_ql, bar_length_ql * sustain)),
            velocity=velocity,
            beat_offset=0.0,
            voice="bass",
        ))

    @staticmethod
    def _nearest_pitch(pc: int, reference: int) -> int:
        candidates = [reference + offset for offset in range(-12, 13) if (reference + offset) % 12 == pc]
        return min(candidates, key=lambda p: (abs(p - reference), p)) if candidates else reference
