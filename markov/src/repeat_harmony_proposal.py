#!/usr/bin/env python3
"""Harmony-aware proposals for early repeated motifs.

Early repeats occur before a second theme has been learned, so DualTheme cannot
help.  This module adds a few harmony-responsive candidates while preserving
the recalled motif's contour as much as possible.  It does not choose or repair
the final bar; the existing scoring stack still makes the decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from harmonic_planner import HarmonicPlanner
from hierarchical_types import BarGenerationTarget, NoteEvent


class RepeatHarmonyProposal:
    """Generate extra candidates for harmony-mismatched early repeats."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("early_repeat_adaptation", {})
        if not isinstance(cfg, dict):
            cfg = {}
        self.enabled = bool(cfg.get("enabled", True))
        self.extra_candidates = int(cfg.get("extra_candidates", 3))
        self.anchor_strength = float(cfg.get("anchor_strength", 0.72))
        self.weak_strength = float(cfg.get("weak_strength", 0.38))
        self.transpose_radius = int(cfg.get("transpose_radius", 3))
        self.max_adjust = int(cfg.get("max_adjust", 5))

    def count(
        self,
        role: str,
        target: BarGenerationTarget,
        partner_notes: Optional[List[NoteEvent]],
    ) -> int:
        if not self.enabled or role != "REPEAT" or not target.harmony:
            return 0
        if target.dual_theme or partner_notes:
            return 0
        return max(0, self.extra_candidates)

    def propose(
        self,
        base_notes: List[NoteEvent],
        target: BarGenerationTarget,
        rng: np.random.RandomState,
        proposal_index: int,
    ) -> List[NoteEvent]:
        result = self._clone(base_notes)
        melody_indices = self._melody_indices(result)
        if not melody_indices or not target.harmony:
            return result
        mode = self.kind(proposal_index)
        if mode == "anchor":
            self._project_structural_notes(result, melody_indices, target, self.anchor_strength)
        elif mode == "resolution":
            self._project_unresolved_notes(result, melody_indices, target, rng)
        elif mode == "transpose":
            self._transpose_for_harmony(result, melody_indices, target)
        elif mode == "combined":
            self._project_structural_notes(result, melody_indices, target, self.anchor_strength)
            self._project_unresolved_notes(result, melody_indices, target, rng)
        elif mode == "harmonic_contour":
            self._harmonic_contour(result, melody_indices, target)
        return result

    @staticmethod
    def kind(proposal_index: int) -> str:
        return ("anchor", "resolution", "transpose", "combined", "harmonic_contour")[proposal_index % 5]

    def _project_structural_notes(
        self,
        result: List[NoteEvent],
        melody_indices: List[int],
        target: BarGenerationTarget,
        strength: float,
    ) -> None:
        strong_beats = self._strong_beats(target)
        tolerance = self._strong_tolerance(target)
        for idx in melody_indices:
            note = result[idx]
            is_strong = any(abs((note.beat_offset % 4.0) - beat) <= tolerance for beat in strong_beats)
            is_final = idx == melody_indices[-1] and target.cadence_strength > 0.45
            if not is_strong and not is_final:
                continue
            chord_pitch = HarmonicPlanner.nearest_chord_pitch(
                note.pitch,
                target.harmony,
                prefer_root=is_final,
            )
            delta = chord_pitch - note.pitch
            if abs(delta) > self.max_adjust:
                continue
            pitch = int(round(note.pitch + delta * strength))
            result[idx] = NoteEvent(
                pitch=int(max(40, min(88, pitch))),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _project_unresolved_notes(
        self,
        result: List[NoteEvent],
        melody_indices: List[int],
        target: BarGenerationTarget,
        rng: np.random.RandomState,
    ) -> None:
        chord_tones = set(int(pc) % 12 for pc in target.harmony.get("chord_tones", []))
        if not chord_tones:
            return
        for pos, idx in enumerate(melody_indices):
            note = result[idx]
            if note.pitch % 12 in chord_tones:
                continue
            prev_note = result[melody_indices[pos - 1]] if pos > 0 else None
            next_note = result[melody_indices[pos + 1]] if pos + 1 < len(melody_indices) else None
            prev_step = prev_note is not None and abs(note.pitch - prev_note.pitch) <= 2
            next_step = next_note is not None and abs(next_note.pitch - note.pitch) <= 2
            prev_chord = prev_note is not None and prev_note.pitch % 12 in chord_tones
            next_chord = next_note is not None and next_note.pitch % 12 in chord_tones
            if prev_step and next_step and (prev_chord or next_chord):
                continue
            if rng.random() > 0.70:
                continue
            chord_pitch = HarmonicPlanner.nearest_chord_pitch(note.pitch, target.harmony)
            delta = chord_pitch - note.pitch
            if abs(delta) > self.max_adjust:
                continue
            pitch = int(round(note.pitch + delta * self.weak_strength))
            result[idx] = NoteEvent(
                pitch=int(max(40, min(88, pitch))),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _transpose_for_harmony(
        self,
        result: List[NoteEvent],
        melody_indices: List[int],
        target: BarGenerationTarget,
    ) -> None:
        best_shift = 0
        best_score = HarmonicPlanner.score_melody(result, target.harmony, {"harmony": {"enabled": True}})
        for shift in range(-self.transpose_radius, self.transpose_radius + 1):
            if shift == 0:
                continue
            candidate = self._clone(result)
            for idx in melody_indices:
                note = candidate[idx]
                candidate[idx] = NoteEvent(
                    pitch=int(max(40, min(88, note.pitch + shift))),
                    duration_ql=note.duration_ql,
                    velocity=note.velocity,
                    beat_offset=note.beat_offset,
                    voice=note.voice,
                )
            score = HarmonicPlanner.score_melody(candidate, target.harmony, {"harmony": {"enabled": True}})
            if score > best_score:
                best_score = score
                best_shift = shift
        if best_shift == 0:
            return
        for idx in melody_indices:
            note = result[idx]
            result[idx] = NoteEvent(
                pitch=int(max(40, min(88, note.pitch + best_shift))),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _harmonic_contour(
        self,
        result: List[NoteEvent],
        melody_indices: List[int],
        target: BarGenerationTarget,
    ) -> None:
        chord_tones = [int(pc) % 12 for pc in target.harmony.get("chord_tones", [])] if target.harmony else []
        if not chord_tones or len(melody_indices) < 2:
            return
        original = [result[idx] for idx in melody_indices]
        projected: List[int] = []
        first = original[0]
        projected.append(self._nearest_allowed_pitch(first.pitch, chord_tones, prefer_close=True))
        for pos in range(1, len(original)):
            prev_pitch = projected[-1]
            old = original[pos]
            old_prev = original[pos - 1]
            direction = 1 if old.pitch > old_prev.pitch else -1 if old.pitch < old_prev.pitch else 0
            chord_allowed = self._chord_pitches(old.pitch, chord_tones)
            allowed = self._allowed_pitches(old.pitch, chord_tones)
            if direction > 0:
                chord_directional = [p for p in chord_allowed if p >= prev_pitch]
                directional = [p for p in allowed if p >= prev_pitch]
            elif direction < 0:
                chord_directional = [p for p in chord_allowed if p <= prev_pitch]
                directional = [p for p in allowed if p <= prev_pitch]
            else:
                chord_directional = chord_allowed
                directional = allowed
            pool = chord_directional or chord_allowed or directional or allowed
            pitch = min(
                pool,
                key=lambda p: (
                    abs(p - old.pitch),
                    abs((p - prev_pitch) - (old.pitch - old_prev.pitch)),
                    abs(p - prev_pitch),
                ),
            )
            if abs(pitch - old.pitch) > self.max_adjust + 2:
                pitch = self._nearest_allowed_pitch(old.pitch, chord_tones, prefer_close=True)
            projected.append(int(max(40, min(88, pitch))))

        for idx, pitch in zip(melody_indices, projected):
            note = result[idx]
            result[idx] = NoteEvent(
                pitch=pitch,
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    @staticmethod
    def _chord_pitches(reference: int, chord_tones: List[int]) -> List[int]:
        return [
            pitch
            for pitch in range(max(40, reference - 9), min(88, reference + 9) + 1)
            if pitch % 12 in set(chord_tones)
        ]

    @staticmethod
    def _allowed_pitches(reference: int, chord_tones: List[int]) -> List[int]:
        allowed_pcs = set(chord_tones)
        # Passing/neighbor tones are allowed as candidates, but chord tones are
        # still favored by the distance sort and downstream harmony scorer.
        for pc in chord_tones:
            allowed_pcs.add((pc + 1) % 12)
            allowed_pcs.add((pc - 1) % 12)
        return [
            pitch
            for pitch in range(max(40, reference - 9), min(88, reference + 9) + 1)
            if pitch % 12 in allowed_pcs
        ]

    @staticmethod
    def _nearest_allowed_pitch(reference: int, chord_tones: List[int], prefer_close: bool = True) -> int:
        candidates = [
            pitch
            for pitch in range(max(40, reference - 12), min(88, reference + 12) + 1)
            if pitch % 12 in set(chord_tones)
        ]
        if not candidates:
            return reference
        return min(candidates, key=lambda p: (abs(p - reference), p))

    @staticmethod
    def _clone(notes: List[NoteEvent]) -> List[NoteEvent]:
        return [
            NoteEvent(n.pitch, n.duration_ql, n.velocity, n.beat_offset, n.voice)
            for n in notes
        ]

    @staticmethod
    def _melody_indices(notes: List[NoteEvent]) -> List[int]:
        return [
            idx for idx, n in sorted(
                enumerate(notes),
                key=lambda pair: (pair[1].beat_offset, pair[1].pitch),
            )
            if n.pitch >= 0 and n.voice == "melody"
        ]

    @staticmethod
    def _strong_beats(target: BarGenerationTarget) -> List[float]:
        return [0.0, 2.0]

    @staticmethod
    def _strong_tolerance(target: BarGenerationTarget) -> float:
        return 0.08
