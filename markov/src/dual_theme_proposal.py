#!/usr/bin/env python3
"""Partner-aware proposal generation for dual-theme development.

The proposal layer broadens the candidate pool.  It does not decide which bar
is best and it does not repair notes after selection; all proposals still pass
through the existing target fitting and scoring stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from harmonic_planner import HarmonicPlanner
from hierarchical_types import BarGenerationTarget, NoteEvent


class DualThemeProposal:
    """Create source/partner blended motif candidates."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config.get("dual_theme_development", {})
        if not isinstance(cfg, dict):
            cfg = {}
        proposal = cfg.get("proposal", {})
        if not isinstance(proposal, dict):
            proposal = {}
        self.enabled = bool(proposal.get("enabled", True))
        self.extra_candidates = int(proposal.get("extra_candidates", 3))
        self.contour_strength = float(proposal.get("contour_strength", 0.55))
        self.rhythm_strength = float(proposal.get("rhythm_strength", 0.35))
        self.register_strength = float(proposal.get("register_strength", 0.42))
        self.max_pitch_shift = int(proposal.get("max_pitch_shift", 7))
        self.harmony_guard_strength = float(proposal.get("harmony_guard_strength", 0.62))
        self.harmony_guard_max_adjust = int(proposal.get("harmony_guard_max_adjust", 4))

    def count(self, target: BarGenerationTarget, partner_notes: Optional[List[NoteEvent]]) -> int:
        if not self.enabled or not target.dual_theme or not partner_notes:
            return 0
        return max(0, self.extra_candidates)

    def propose(
        self,
        base_notes: List[NoteEvent],
        source_notes: List[NoteEvent],
        partner_notes: Optional[List[NoteEvent]],
        target: BarGenerationTarget,
        rng: np.random.RandomState,
        proposal_index: int,
    ) -> List[NoteEvent]:
        """Return one partner-aware candidate derived from ``base_notes``."""
        if not self.enabled or not target.dual_theme or not partner_notes:
            return list(base_notes)
        result = self._clone(base_notes)
        candidate = self._melody_indices(result)
        source = self._melody(source_notes)
        partner = self._melody(partner_notes)
        if len(candidate) < 2 or len(partner) < 2:
            return result

        desired_partner = self.desired_partner(target)
        current_partner = self._gesture_similarity_from_indices(result, candidate, partner)
        current_source = self._gesture_similarity_from_indices(result, candidate, source)
        gap = float(np.clip(desired_partner - current_partner, -0.25, 0.35))
        adaptive_partner = float(np.clip(desired_partner + 0.65 * max(0.0, gap), 0.12, 0.68))
        kind = self.kind(proposal_index)

        if kind in ("contour", "mixed", "harmony_guarded"):
            self._blend_contour(result, candidate, partner, adaptive_partner, rng)
        if kind in ("rhythm", "mixed", "harmony_guarded"):
            self._blend_rhythm(result, candidate, partner, adaptive_partner, rng)
        if kind == "mixed":
            self._blend_register(result, candidate, source, partner, adaptive_partner)
        if kind == "harmony_guarded":
            self._apply_harmony_guard(result, candidate, target)
        if current_source > 0.92 and gap >= -0.04:
            self._anti_copy_perturb(result, candidate, partner, rng, adaptive_partner)
        return result

    @staticmethod
    def kind(proposal_index: int) -> str:
        # Standalone register proposals often win by changing the local anchor
        # without producing real thematic development.  Keep register motion as
        # part of mixed proposals only.
        return ("contour", "rhythm", "contour", "harmony_guarded", "mixed")[proposal_index % 5]

    @staticmethod
    def desired_partner(target: BarGenerationTarget) -> float:
        relation = target.dual_theme or {}
        blend = float(relation.get("blend", 0.0))
        transform = float(relation.get("transform", 0.0))
        contrast = float(relation.get("contrast", 0.5))
        return float(np.clip(
            0.26 + 0.72 * blend + 0.42 * transform + 0.20 * contrast * max(blend, transform),
            0.12,
            0.68,
        ))

    def _blend_contour(
        self,
        result: List[NoteEvent],
        candidate_indices: List[int],
        partner: List[NoteEvent],
        desired_partner: float,
        rng: np.random.RandomState,
    ) -> None:
        strength = float(np.clip(self.contour_strength * desired_partner, 0.05, 0.55))
        length = min(len(candidate_indices), len(partner))
        if length < 2:
            return
        for pos in range(1, length):
            idx = candidate_indices[pos]
            prev_idx = candidate_indices[pos - 1]
            note = result[idx]
            prev = result[prev_idx]
            current_delta = note.pitch - prev.pitch
            partner_delta = partner[pos].pitch - partner[pos - 1].pitch
            blended_delta = int(round(current_delta * (1.0 - strength) + partner_delta * strength))
            if blended_delta == 0 and partner_delta != 0 and rng.random() < strength:
                blended_delta = 1 if partner_delta > 0 else -1
            pitch = prev.pitch + blended_delta
            result[idx] = NoteEvent(
                pitch=int(max(40, min(88, pitch))),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _blend_rhythm(
        self,
        result: List[NoteEvent],
        candidate_indices: List[int],
        partner: List[NoteEvent],
        desired_partner: float,
        rng: np.random.RandomState,
    ) -> None:
        strength = float(np.clip(self.rhythm_strength * desired_partner, 0.03, 0.38))
        length = min(len(candidate_indices), len(partner))
        for pos in range(length):
            if rng.random() > strength:
                continue
            idx = candidate_indices[pos]
            note = result[idx]
            partner_note = partner[pos]
            duration = float(
                np.clip(
                    note.duration_ql * (1.0 - strength) + partner_note.duration_ql * strength,
                    0.125,
                    2.0,
                )
            )
            result[idx] = NoteEvent(
                pitch=note.pitch,
                duration_ql=duration,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _blend_register(
        self,
        result: List[NoteEvent],
        candidate_indices: List[int],
        source: List[NoteEvent],
        partner: List[NoteEvent],
        desired_partner: float,
    ) -> None:
        if not candidate_indices or not partner:
            return
        candidate_mean = float(np.mean([result[idx].pitch for idx in candidate_indices]))
        source_mean = float(np.mean([n.pitch for n in source])) if source else candidate_mean
        partner_mean = float(np.mean([n.pitch for n in partner]))
        anchor = source_mean * (1.0 - desired_partner) + partner_mean * desired_partner
        shift = float(np.clip(
            (anchor - candidate_mean) * self.register_strength,
            -self.max_pitch_shift,
            self.max_pitch_shift,
        ))
        if abs(shift) < 0.25:
            return
        for idx in candidate_indices:
            note = result[idx]
            pitch = int(round(note.pitch + shift))
            result[idx] = NoteEvent(
                pitch=int(max(40, min(88, pitch))),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _anti_copy_perturb(
        self,
        result: List[NoteEvent],
        candidate_indices: List[int],
        partner: List[NoteEvent],
        rng: np.random.RandomState,
        desired_partner: float,
    ) -> None:
        if len(candidate_indices) < 3:
            return
        strength = float(np.clip(0.18 + 0.32 * desired_partner, 0.12, 0.40))
        editable = candidate_indices[1:-1]
        if not editable:
            editable = candidate_indices
        change_count = max(1, int(round(len(editable) * strength)))
        chosen = set(rng.choice(editable, size=min(len(editable), change_count), replace=False))
        for pos, idx in enumerate(candidate_indices):
            if idx not in chosen:
                continue
            note = result[idx]
            partner_pos = min(pos, len(partner) - 1)
            direction = partner[partner_pos].pitch - note.pitch
            if direction == 0:
                step = int(rng.choice([-1, 1]))
            else:
                step = 1 if direction > 0 else -1
            if rng.random() < 0.30:
                step *= 2
            result[idx] = NoteEvent(
                pitch=int(max(40, min(88, note.pitch + step))),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _apply_harmony_guard(
        self,
        result: List[NoteEvent],
        candidate_indices: List[int],
        target: BarGenerationTarget,
    ) -> None:
        if not target.harmony:
            return
        chord_tones = set(int(pc) % 12 for pc in target.harmony.get("chord_tones", []))
        if not chord_tones:
            return
        strong_beats = {0.0, 2.0}
        tolerance = 0.08
        for pos, idx in enumerate(candidate_indices):
            note = result[idx]
            is_strong = any(abs((note.beat_offset % 4.0) - beat) <= tolerance for beat in strong_beats)
            is_chord = note.pitch % 12 in chord_tones
            prev_note = result[candidate_indices[pos - 1]] if pos > 0 else None
            next_note = result[candidate_indices[pos + 1]] if pos + 1 < len(candidate_indices) else None
            prev_step = prev_note is not None and abs(note.pitch - prev_note.pitch) <= 2
            next_step = next_note is not None and abs(next_note.pitch - note.pitch) <= 2
            prev_chord = prev_note is not None and prev_note.pitch % 12 in chord_tones
            next_chord = next_note is not None and next_note.pitch % 12 in chord_tones
            resolved_passing = prev_step and next_step and (prev_chord or next_chord)
            if is_chord or (resolved_passing and not is_strong):
                continue
            if not is_strong and pos not in (0, len(candidate_indices) - 1):
                continue
            chord_pitch = HarmonicPlanner.nearest_chord_pitch(note.pitch, target.harmony)
            delta = chord_pitch - note.pitch
            if abs(delta) > self.harmony_guard_max_adjust:
                continue
            pitch = int(round(note.pitch + delta * self.harmony_guard_strength))
            result[idx] = NoteEvent(
                pitch=int(max(40, min(88, pitch))),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    @staticmethod
    def _clone(notes: List[NoteEvent]) -> List[NoteEvent]:
        return [
            NoteEvent(n.pitch, n.duration_ql, n.velocity, n.beat_offset, n.voice)
            for n in notes
        ]

    @staticmethod
    def _melody(notes: List[NoteEvent]) -> List[NoteEvent]:
        return sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )

    @staticmethod
    def _melody_indices(notes: List[NoteEvent]) -> List[int]:
        return [
            idx for idx, n in sorted(
                enumerate(notes),
                key=lambda pair: (pair[1].beat_offset, pair[1].pitch),
            )
            if n.pitch >= 0 and n.voice == "melody"
        ]

    def _gesture_similarity_from_indices(
        self,
        notes: List[NoteEvent],
        candidate_indices: List[int],
        partner: List[NoteEvent],
    ) -> float:
        candidate = [notes[idx] for idx in candidate_indices]
        if len(candidate) < 2 or len(partner) < 2:
            return 0.0
        contour = self._contour_similarity(self._intervals(candidate), self._intervals(partner))
        rhythm = self._rhythm_similarity(candidate, partner)
        register = float(np.exp(-abs(self._mean_pitch(candidate) - self._mean_pitch(partner)) / 10.0))
        return float(np.clip(0.50 * contour + 0.32 * rhythm + 0.18 * register, 0.0, 1.0))

    @staticmethod
    def _intervals(melody: List[NoteEvent]) -> List[int]:
        return [
            int(melody[i + 1].pitch - melody[i].pitch)
            for i in range(len(melody) - 1)
        ]

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
    def _mean_pitch(melody: List[NoteEvent]) -> float:
        if not melody:
            return 64.0
        return float(np.mean([n.pitch for n in melody]))
