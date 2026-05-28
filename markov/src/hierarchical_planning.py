#!/usr/bin/env python3
"""Structure graph, theme skeleton, and candidate scoring layer."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hierarchical_types import (
    BarGenerationTarget,
    BarSkeleton,
    CompositionPlan,
    NoteEvent,
    SectionAffect,
    StructureEdge,
    ThemeIdentity,
    ThemeSkeleton,
)
from harmonic_planner import HarmonicPlanner
from narrative_planner import NarrativePlanner

_USABLE_DUR_VALUES: List[float] = [4.0, 2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75]


def _stable_hash(*args: object) -> int:
    """Deterministic hash, same result across Python processes."""
    h = hashlib.md5()
    for a in args:
        h.update(str(a).encode())
    return int(h.hexdigest()[:8], 16)


class HierarchicalPlanningMixin:
    def _section_entropy(self, sec_labels: List[int]) -> float:
        """Average entropy across the cluster labels in a section."""
        centroids = self.model.clusterer.centroids
        indices = [c for c in sec_labels if 0 <= c < len(centroids)]
        if not indices:
            return 0.0
        return float(centroids[indices, 7].mean())

    @staticmethod
    def _build_structure_graph(
        measure_context: List[Tuple[str, int, str, int, int]],
    ) -> Dict[int, StructureEdge]:
        """Build a MELONS-style bar relation graph from section recurrence."""
        first_occurrence_bars: Dict[str, Dict[int, int]] = {}
        graph: Dict[int, StructureEdge] = {}
        for target_bar, (label, local_bar, role, _, _) in enumerate(measure_context):
            if role in ("FREE", "FLAT"):
                continue
            if label not in first_occurrence_bars:
                first_occurrence_bars[label] = {}
            if role == "NEW":
                first_occurrence_bars[label].setdefault(local_bar, target_bar)
                continue
            source_map = first_occurrence_bars.get(label, {})
            if not source_map:
                continue
            source_bar = source_map.get(local_bar)
            if source_bar is None:
                source_bar = source_map[min(source_map.keys(), key=lambda k: abs(k - local_bar))]
            if role == "REPEAT":
                relation = "SAME_THEME"
            elif role == "VARIANT":
                relation = "VARIATION"
            elif local_bar == measure_context[target_bar][4] - 1:
                relation = "CADENCE_RETURN"
            else:
                relation = "VARIATION"
            graph[target_bar] = StructureEdge(
                source_bar=source_bar,
                target_bar=target_bar,
                relation=relation,
                label=label,
            )
        return graph

    @staticmethod
    def _clone_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
        """Copy note events so motif memory can be reused without mutation."""
        return [
            NoteEvent(
                pitch=n.pitch,
                duration_ql=n.duration_ql,
                velocity=n.velocity,
                beat_offset=n.beat_offset,
                voice=n.voice,
            )
            for n in notes
        ]

    @staticmethod
    def _last_melody_pitch(notes: List[NoteEvent], fallback: Optional[int]) -> Optional[int]:
        sounding = [n for n in notes if n.pitch >= 0 and n.voice == "melody"]
        return sounding[-1].pitch if sounding else fallback

    def _update_theme_identity(
        self,
        identities: Dict[str, ThemeIdentity],
        label: str,
        section_notes: List[List[NoteEvent]],
        section_len: int,
    ) -> None:
        """Learn a theme identity once the first section has enough material."""
        if label in identities or len(section_notes) < section_len:
            return
        if any(not bar for bar in section_notes[:section_len]):
            return

        opening_notes = self._melody_events(section_notes[0])
        cadence_notes = self._melody_events(section_notes[section_len - 1])
        if len(opening_notes) < 2 or len(cadence_notes) < 2:
            return

        bar_intervals: List[Tuple[int, ...]] = []
        bar_durations: List[Tuple[float, ...]] = []
        bar_sizes: List[int] = []
        bar_means: List[float] = []
        for bar in section_notes[:section_len]:
            melody = self._melody_events(bar)
            gesture = melody[: min(4, len(melody))]
            bar_intervals.append(self._interval_signature(gesture))
            bar_durations.append(tuple(round(n.duration_ql, 3) for n in gesture))
            bar_sizes.append(len(gesture))
            bar_means.append(float(np.mean([n.pitch for n in melody])) if melody else 0.0)
        base_mean = bar_means[0] if bar_means else 0.0

        opening = opening_notes[: min(4, len(opening_notes))]
        cadence = cadence_notes[-min(4, len(cadence_notes)):]
        identities[label] = ThemeIdentity(
            label=label,
            bar_intervals=tuple(bar_intervals),
            bar_durations=tuple(bar_durations),
            bar_sizes=tuple(bar_sizes),
            bar_mean_offsets=tuple(round(m - base_mean, 3) for m in bar_means),
            opening_intervals=self._interval_signature(opening),
            opening_durations=tuple(round(n.duration_ql, 3) for n in opening),
            cadence_intervals=self._interval_signature(cadence),
            cadence_durations=tuple(round(n.duration_ql, 3) for n in cadence),
            cadence_pc=cadence[-1].pitch % 12,
            opening_size=len(opening),
            cadence_size=len(cadence),
        )

    @staticmethod
    def _melody_events(notes: List[NoteEvent]) -> List[NoteEvent]:
        return sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )

    @staticmethod
    def _interval_signature(notes: List[NoteEvent]) -> Tuple[int, ...]:
        return tuple(
            int(notes[i + 1].pitch - notes[i].pitch)
            for i in range(len(notes) - 1)
        )

    def _apply_theme_identity(
        self,
        notes: List[NoteEvent],
        identity: ThemeIdentity,
        bar_index: int,
        section_len: int,
        target_pitch: Optional[int],
        rng: np.random.RandomState,
    ) -> List[NoteEvent]:
        """Project opening/cadence gestures onto a returned section.

        The projection keeps the phrase recognisable without forcing an exact
        copy: intervals and coarse durations are retained, then transposed near
        the current tonal-emotional target.
        """
        melody = self._melody_events(notes)
        if not melody:
            return notes

        result = self._clone_notes(notes)
        melody_indices = [
            idx for idx, n in sorted(
                enumerate(result), key=lambda pair: (pair[1].beat_offset, pair[1].pitch),
            )
            if n.pitch >= 0 and n.voice == "melody"
        ]
        if not melody_indices:
            return result

        local_bar = min(bar_index, len(identity.bar_intervals) - 1)
        if local_bar >= 0 and local_bar < len(identity.bar_sizes):
            size = identity.bar_sizes[local_bar]
            if size >= 2:
                anchor = target_pitch
                if anchor is not None and local_bar < len(identity.bar_mean_offsets):
                    anchor = int(round(anchor + identity.bar_mean_offsets[local_bar] * 0.55))
                self._project_gesture(
                    result,
                    melody_indices[: min(size, len(melody_indices))],
                    identity.bar_intervals[local_bar],
                    identity.bar_durations[local_bar],
                    anchor_pitch=anchor,
                    rng=rng,
                    final_pc=None,
                    strength=0.78,
                )

        if bar_index == 0:
            self._project_gesture(
                result,
                melody_indices[: identity.opening_size],
                identity.opening_intervals,
                identity.opening_durations,
                anchor_pitch=target_pitch,
                rng=rng,
                final_pc=None,
                strength=0.92,
            )
        if bar_index >= section_len - 1:
            self._project_gesture(
                result,
                melody_indices[-identity.cadence_size:],
                identity.cadence_intervals,
                identity.cadence_durations,
                anchor_pitch=target_pitch,
                rng=rng,
                final_pc=identity.cadence_pc,
                strength=1.0,
            )
        return result

    def _project_gesture(
        self,
        notes: List[NoteEvent],
        indices: List[int],
        intervals: Tuple[int, ...],
        durations: Tuple[float, ...],
        anchor_pitch: Optional[int],
        rng: np.random.RandomState,
        final_pc: Optional[int],
        strength: float = 0.75,
    ) -> None:
        if not indices:
            return
        anchor = anchor_pitch if anchor_pitch is not None else notes[indices[0]].pitch
        start_pitch = int(max(40, min(84, anchor + int(rng.choice([-2, -1, 0, 1, 2])))))
        pitches = [start_pitch]
        for interval in intervals[: max(0, len(indices) - 1)]:
            pitches.append(int(max(40, min(84, pitches[-1] + interval))))
        if final_pc is not None and pitches:
            pitches[-1] = self._nearest_pitch(final_pc, pitches[-1])

        for local_i, note_idx in enumerate(indices):
            old = notes[note_idx]
            pitch = pitches[min(local_i, len(pitches) - 1)]
            pitch = int(round(old.pitch * (1.0 - strength) + pitch * strength))
            duration = old.duration_ql
            if local_i < len(durations) and old.duration_ql <= 1.0:
                duration = float(max(
                    0.1,
                    min(1.5, old.duration_ql * (1.0 - strength * 0.65)
                        + durations[local_i] * (strength * 0.65)),
                ))
            notes[note_idx] = NoteEvent(
                pitch=pitch,
                duration_ql=duration,
                velocity=old.velocity,
                beat_offset=old.beat_offset,
                voice=old.voice,
            )

    def _update_theme_skeleton(
        self,
        skeletons: Dict[str, ThemeSkeleton],
        label: str,
        section_notes: List[List[NoteEvent]],
        section_len: int,
        tonic_pc: int,
    ) -> None:
        """Extract a reusable skeleton from the first full theme statement."""
        if label in skeletons or len(section_notes) < section_len:
            return
        bars: List[BarSkeleton] = []
        for notes in section_notes[:section_len]:
            melody = self._melody_events(notes)
            if not melody:
                bars.append(BarSkeleton((), (), 0, 0, 60.0, 0.0, 0))
                continue
            gesture = melody[: min(4, len(melody))]
            durations = tuple(round(n.duration_ql, 3) for n in gesture)
            contour = self._interval_signature(gesture)
            mean_pitch = float(np.mean([n.pitch for n in melody]))
            pitch_range = float(max(n.pitch for n in melody) - min(n.pitch for n in melody))
            anchor_degree = (gesture[0].pitch - tonic_pc) % 12
            cadence_degree = (melody[-1].pitch - tonic_pc) % 12
            tension = float(np.clip((pitch_range / 16.0) + (len(melody) / 24.0), 0.0, 1.0))
            bars.append(BarSkeleton(
                rhythm_cell=durations,
                contour=contour,
                anchor_degree=int(anchor_degree),
                cadence_degree=int(cadence_degree),
                register_zone=mean_pitch,
                tension=tension,
                note_count=len(melody),
            ))
        skeletons[label] = ThemeSkeleton(label=label, bars=tuple(bars))

    def _build_bar_generation_target(
        self,
        bar_index: int,
        label: str,
        local_bar: int,
        target_pitch: Optional[int],
        affect: Optional[Dict[str, float]],
        structure_graph: Dict[int, StructureEdge],
        theme_skeletons: Dict[str, ThemeSkeleton],
        composition_plan: CompositionPlan,
    ) -> BarGenerationTarget:
        """Create the skeleton constraint used to render a single bar."""
        affect = affect or {}
        harmony = affect.get("harmony")
        if not isinstance(harmony, dict):
            harmony = None
        edge = structure_graph.get(bar_index)
        relation = edge.relation if edge is not None else "CONTRAST"
        source_bar = edge.source_bar if edge is not None else None
        development_role = composition_plan.development_roles.get(bar_index, "CONTRAST")
        role_cfg = self._theme_development_role_config(development_role)
        skeleton = theme_skeletons.get(label)
        bar_skel: Optional[BarSkeleton] = None
        if skeleton is not None and skeleton.bars:
            bar_skel = skeleton.bars[min(local_bar, len(skeleton.bars) - 1)]

        tonal_pc = int(affect.get("tonal_pc", composition_plan.global_tonic_pc))
        fallback_pitch = target_pitch if target_pitch is not None else self._nearest_pitch(tonal_pc, 64)
        role_register_shift = float(role_cfg.get("register_shift", 0.0))
        if bar_skel is not None:
            register_target = float(
                0.55 * float(affect.get("register_center", fallback_pitch))
                + 0.45 * bar_skel.register_zone
            ) + role_register_shift
            degree = bar_skel.cadence_degree if relation == "CADENCE_RETURN" else bar_skel.anchor_degree
            target_degree = (composition_plan.global_tonic_pc + degree) % 12
            target = self._nearest_pitch(target_degree, int(round(register_target)))
            rhythm_cell = bar_skel.rhythm_cell
            contour = bar_skel.contour
            cadence_strength = 0.85 if relation == "CADENCE_RETURN" else 0.35
            tension = float(0.5 * affect.get("tension", 0.35) + 0.5 * bar_skel.tension)
        else:
            target = int(fallback_pitch)
            target_degree = target % 12
            register_target = float(affect.get("register_center", target)) + role_register_shift
            rhythm_cell = ()
            contour = ()
            cadence_strength = 0.65 if local_bar >= 3 and local_bar % 4 == 3 else 0.25
            tension = float(affect.get("tension", 0.35))

        tension = float(np.clip(tension * float(role_cfg.get("tension_scale", 1.0)), 0.0, 1.0))
        cadence_strength = float(np.clip(
            cadence_strength + float(role_cfg.get("cadence_strength_add", 0.0)),
            0.0,
            1.0,
        ))
        exact_copy_penalty_scale = float(role_cfg.get("exact_copy_penalty_scale", 1.0))
        exact_copy_penalty = {
            "SAME_THEME": 0.35,
            "VARIATION": 0.70,
            "CADENCE_RETURN": 0.50,
            "CONTRAST": 0.10,
        }.get(relation, 0.4) * exact_copy_penalty_scale
        similarity_target = role_cfg.get("similarity_target", [0.0, 1.0])
        if not isinstance(similarity_target, list) or len(similarity_target) != 2:
            similarity_target = [0.0, 1.0]
        return BarGenerationTarget(
            relation=relation,
            source_bar=source_bar,
            development_role=development_role,
            rhythm_cell=rhythm_cell,
            contour=contour,
            target_pitch=int(max(40, min(84, target))),
            target_degree=int(target_degree),
            register_target=register_target,
            cadence_strength=cadence_strength,
            tension=tension,
            exact_copy_penalty=exact_copy_penalty,
            similarity_min=float(similarity_target[0]),
            similarity_max=float(similarity_target[1]),
            development_strength=float(role_cfg.get("strength_scale", 1.0)),
            target_attraction=float(role_cfg.get("target_attraction", 0.55)),
            rhythm_change_scale=float(role_cfg.get("rhythm_change_scale", 1.0)),
            harmony=harmony,
        )

    def _generate_scored_measure(
        self,
        cluster_label: int,
        time_signature: Tuple[int, int],
        is_section_end: bool,
        phrase_role: str,
        target_pitch: Optional[int],
        previous_pitch: Optional[int],
        affect: Optional[Dict[str, float]],
        bar_target: BarGenerationTarget,
        seed: int,
    ) -> List[NoteEvent]:
        """Generate several local textures and keep the best skeleton fit."""
        skeleton_cfg = self.config.get("skeleton", {})
        candidate_count = int(skeleton_cfg.get("candidate_count", 4))
        harmony_cfg = self.config.get("harmony", {})
        if isinstance(harmony_cfg, dict) and harmony_cfg.get("enabled", False) and bar_target.harmony:
            multiplier = float(harmony_cfg.get("candidate_count_multiplier", 1.5))
            if bar_target.cadence_strength > 0.5 or bar_target.harmony.get("cadence_role") == "CADENCE":
                multiplier = float(harmony_cfg.get("cadence_candidate_count_multiplier", 2.5))
            candidate_count = max(candidate_count, int(round(candidate_count * multiplier)))
        best_notes: Optional[List[NoteEvent]] = None
        best_score = -1e9
        for k in range(max(1, candidate_count)):
            candidate_seed = _stable_hash(seed, "candidate", k)
            rng = np.random.RandomState(_stable_hash(seed, "candidate-target", k))
            spread_base = float(skeleton_cfg.get("candidate_target_spread_base", 4))
            spread_tension = float(skeleton_cfg.get("candidate_target_spread_tension", 9))
            target_spread = int(round(spread_base + spread_tension * bar_target.tension))
            if bar_target.cadence_strength > 0.6:
                candidate_target = bar_target.target_pitch
            else:
                candidate_target = int(max(
                    40,
                    min(84, bar_target.target_pitch + rng.randint(-target_spread, target_spread + 1)),
                ))
            candidate = self.note_sampler.sample_measure(
                cluster_label=cluster_label,
                time_signature=time_signature,
                seed=candidate_seed,
                perturb=0.0,
                is_section_end=is_section_end,
                phrase_role=phrase_role,
                target_pitch=candidate_target,
                previous_pitch=previous_pitch,
                affect=affect,
            )
            rng = np.random.RandomState(_stable_hash(seed, "fit", k))
            fitted = self._fit_notes_to_bar_target(candidate, bar_target, rng)
            score = self._score_candidate(fitted, bar_target, cluster_label=cluster_label)
            if score > best_score:
                best_score = score
                best_notes = fitted
        return best_notes if best_notes is not None else []

    def _fit_notes_to_bar_target(
        self,
        notes: List[NoteEvent],
        target: BarGenerationTarget,
        rng: np.random.RandomState,
    ) -> List[NoteEvent]:
        """Project a generated texture onto the current bar skeleton."""
        result = self._clone_notes(notes)
        melody_indices = [
            idx for idx, n in sorted(
                enumerate(result), key=lambda pair: (pair[1].beat_offset, pair[1].pitch),
            )
            if n.pitch >= 0 and n.voice == "melody"
        ]
        if not melody_indices:
            return result

        contour_strength = {
            "STATEMENT": 0.55,
            "REPEAT": 0.68,
            "SEQUENCE_UP": 0.62,
            "SEQUENCE_DOWN": 0.62,
            "FRAGMENT": 0.35,
            "EXTENSION": 0.45,
            "INTENSIFY": 0.50,
            "RELAX": 0.42,
            "CADENTIAL": 0.75,
        }.get(target.development_role, 0.55)
        if target.contour and target.development_role != "FRAGMENT":
            size = min(len(melody_indices), len(target.contour) + 1)
            self._project_gesture(
                result,
                melody_indices[:size],
                target.contour,
                target.rhythm_cell,
                anchor_pitch=target.target_pitch,
                rng=rng,
                final_pc=None,
                strength=contour_strength,
            )

        # Register gravity over the whole bar, not only the motif head.
        skeleton_cfg = self.config.get("skeleton", {})
        register_strength = {
            "CONTRAST": float(skeleton_cfg.get("register_gravity_contrast", 0.06)),
            "SAME_THEME": float(skeleton_cfg.get("register_gravity_same_theme", 0.14)),
            "VARIATION": float(skeleton_cfg.get("register_gravity_variation", 0.15)),
            "CADENCE_RETURN": float(skeleton_cfg.get("register_gravity_cadence_return", 0.22)),
        }.get(target.relation, 0.14)
        for idx in melody_indices:
            n = result[idx]
            pitch = int(round(n.pitch * (1.0 - register_strength) + target.register_target * register_strength))
            if target.development_role == "SEQUENCE_UP":
                pitch += 1
            elif target.development_role == "SEQUENCE_DOWN":
                pitch -= 1
            elif target.development_role == "INTENSIFY":
                progress = melody_indices.index(idx) / max(1, len(melody_indices) - 1)
                pitch += int(round(progress * 2))
            elif target.development_role == "RELAX":
                pitch -= 1
            result[idx] = NoteEvent(
                pitch=max(40, min(88, pitch)),
                duration_ql=n.duration_ql,
                velocity=n.velocity,
                beat_offset=n.beat_offset,
                voice=n.voice,
            )

        if target.cadence_strength > 0.5:
            last_idx = melody_indices[-1]
            n = result[last_idx]
            pitch = self._nearest_pitch(target.target_degree, target.target_pitch)
            result[last_idx] = NoteEvent(
                pitch=max(40, min(88, pitch)),
                duration_ql=max(n.duration_ql, 0.75),
                velocity=max(35, n.velocity - int(10 * target.cadence_strength)),
                beat_offset=n.beat_offset,
                voice=n.voice,
            )
        self._fit_notes_to_harmony(result, melody_indices, target)
        return result

    def _fit_notes_to_harmony(
        self,
        result: List[NoteEvent],
        melody_indices: List[int],
        target: BarGenerationTarget,
    ) -> None:
        """Nudge structurally important melody notes toward chord tones."""
        if not target.harmony:
            return
        harmony_cfg = self.config.get("harmony", {})
        if not isinstance(harmony_cfg, dict) or not harmony_cfg.get("enabled", False):
            return
        realization_cfg = harmony_cfg.get("realization", {})
        if not isinstance(realization_cfg, dict):
            realization_cfg = {}
        strong_strength = float(realization_cfg.get("strong_beat_chord_tone_strength", 0.65))
        cadence_strength = float(realization_cfg.get("cadence_final_chord_tone_strength", 0.95))
        max_adjust = int(realization_cfg.get("max_pitch_adjust", 5))
        strong_beats = set(float(x) for x in harmony_cfg.get("strong_beats", [0.0, 2.0]))
        tolerance = float(harmony_cfg.get("strong_beat_tolerance", 0.08))
        cadence_role = str(target.harmony.get("cadence_role", ""))
        root_on_final = bool(realization_cfg.get("prefer_root_on_final_cadence", True))
        resolution_cfg = realization_cfg.get("non_chord_resolution", {})
        if not isinstance(resolution_cfg, dict):
            resolution_cfg = {}
        resolution_enabled = bool(resolution_cfg.get("enabled", True))
        weak_strength = float(resolution_cfg.get("weak_note_chord_tone_strength", 0.30))
        cadence_weak_strength = float(resolution_cfg.get("cadence_weak_note_strength", 0.50))
        weak_max_adjust = int(resolution_cfg.get("max_pitch_adjust", 3))
        preserve_passing = bool(resolution_cfg.get("preserve_stepwise_passing_tones", True))
        chord_tones = set(int(pc) % 12 for pc in target.harmony.get("chord_tones", []))

        for idx in melody_indices:
            note = result[idx]
            is_strong = any(abs((note.beat_offset % 4.0) - beat) <= tolerance for beat in strong_beats)
            is_final = idx == melody_indices[-1]
            strength = 0.0
            prefer_root = False
            if is_strong:
                strength = max(strength, strong_strength)
            if is_final and (target.cadence_strength > 0.5 or cadence_role == "CADENCE"):
                strength = max(strength, cadence_strength)
                prefer_root = root_on_final
            if strength <= 0.0:
                continue
            chord_pitch = HarmonicPlanner.nearest_chord_pitch(
                note.pitch,
                target.harmony,
                prefer_root=prefer_root,
            )
            delta = chord_pitch - note.pitch
            if abs(delta) > max_adjust:
                continue
            pitch = int(round(note.pitch + delta * strength))
            # Rounding can land between chord tones; snap once more when the
            # intent is a near-certain cadence.
            if is_final and strength >= 0.9:
                pitch = HarmonicPlanner.nearest_chord_pitch(
                    pitch,
                    target.harmony,
                    prefer_root=prefer_root,
                )
            result[idx] = NoteEvent(
                pitch=max(40, min(88, pitch)),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

        if not resolution_enabled or not chord_tones:
            return
        melody_pos = {idx: pos for pos, idx in enumerate(melody_indices)}
        is_cadence_bar = target.cadence_strength > 0.5 or cadence_role == "CADENCE"
        for idx in melody_indices:
            note = result[idx]
            if note.pitch % 12 in chord_tones:
                continue
            pos = melody_pos[idx]
            prev_note = result[melody_indices[pos - 1]] if pos > 0 else None
            next_note = result[melody_indices[pos + 1]] if pos + 1 < len(melody_indices) else None
            prev_step = prev_note is not None and abs(note.pitch - prev_note.pitch) <= 2
            next_step = next_note is not None and abs(next_note.pitch - note.pitch) <= 2
            prev_chord = prev_note is not None and prev_note.pitch % 12 in chord_tones
            next_chord = next_note is not None and next_note.pitch % 12 in chord_tones
            if preserve_passing and prev_step and next_step and (prev_chord or next_chord):
                continue
            strength = cadence_weak_strength if is_cadence_bar else weak_strength
            chord_pitch = HarmonicPlanner.nearest_chord_pitch(note.pitch, target.harmony)
            delta = chord_pitch - note.pitch
            if abs(delta) > weak_max_adjust:
                continue
            pitch = int(round(note.pitch + delta * strength))
            if pitch % 12 not in chord_tones and strength >= 0.5:
                pitch = HarmonicPlanner.nearest_chord_pitch(pitch, target.harmony)
            result[idx] = NoteEvent(
                pitch=max(40, min(88, pitch)),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

    def _repair_harmony_mismatch(
        self,
        notes: List[NoteEvent],
        target: BarGenerationTarget,
    ) -> List[NoteEvent]:
        """Build a stronger harmony-fit candidate for failed return motifs.

        This keeps onset/duration/velocity unchanged and only projects
        unresolved non-chord melody notes.  It is intended as a fallback
        candidate, not as the default rendering path.
        """
        if not target.harmony:
            return notes
        harmony_cfg = self.config.get("harmony", {})
        if not isinstance(harmony_cfg, dict) or not harmony_cfg.get("enabled", False):
            return notes
        repair_cfg = harmony_cfg.get("adaptive_repair", {})
        if not isinstance(repair_cfg, dict) or not repair_cfg.get("enabled", True):
            return notes

        diagnostics = HarmonicPlanner.diagnostics(notes, target.harmony, self.config)
        score = diagnostics.get("score")
        chord_ratio = diagnostics.get("chord_tone_ratio")
        resolution_cost = diagnostics.get("non_chord_resolution_cost")
        score_threshold = float(repair_cfg.get("score_threshold", -1.0))
        chord_threshold = float(repair_cfg.get("chord_tone_ratio_threshold", 0.30))
        resolution_threshold = float(repair_cfg.get("non_chord_resolution_threshold", 0.85))
        should_repair = (
            (score is not None and float(score) < score_threshold)
            or (chord_ratio is not None and float(chord_ratio) < chord_threshold)
            or (resolution_cost is not None and float(resolution_cost) > resolution_threshold)
        )
        if not should_repair:
            return notes

        result = self._clone_notes(notes)
        melody_indices = [
            idx for idx, n in sorted(
                enumerate(result), key=lambda pair: (pair[1].beat_offset, pair[1].pitch),
            )
            if n.pitch >= 0 and n.voice == "melody"
        ]
        if not melody_indices:
            return result

        chord_tones = set(int(pc) % 12 for pc in target.harmony.get("chord_tones", []))
        if not chord_tones:
            return result
        strength = float(repair_cfg.get("projection_strength", 0.82))
        max_adjust = int(repair_cfg.get("max_pitch_adjust", 5))
        preserve_passing = bool(repair_cfg.get("preserve_stepwise_passing_tones", True))
        strong_beats = set(float(x) for x in harmony_cfg.get("strong_beats", [0.0, 2.0]))
        tolerance = float(harmony_cfg.get("strong_beat_tolerance", 0.08))

        for pos, idx in enumerate(melody_indices):
            note = result[idx]
            is_chord = note.pitch % 12 in chord_tones
            is_strong = any(abs((note.beat_offset % 4.0) - beat) <= tolerance for beat in strong_beats)
            prev_note = result[melody_indices[pos - 1]] if pos > 0 else None
            next_note = result[melody_indices[pos + 1]] if pos + 1 < len(melody_indices) else None
            prev_step = prev_note is not None and abs(note.pitch - prev_note.pitch) <= 2
            next_step = next_note is not None and abs(next_note.pitch - note.pitch) <= 2
            prev_chord = prev_note is not None and prev_note.pitch % 12 in chord_tones
            next_chord = next_note is not None and next_note.pitch % 12 in chord_tones
            passing = prev_step and next_step and (prev_chord or next_chord)
            if is_chord or (preserve_passing and passing and not is_strong):
                continue
            chord_pitch = HarmonicPlanner.nearest_chord_pitch(note.pitch, target.harmony)
            delta = chord_pitch - note.pitch
            if abs(delta) > max_adjust:
                continue
            pitch = int(round(note.pitch + delta * strength))
            if pitch % 12 not in chord_tones:
                pitch = HarmonicPlanner.nearest_chord_pitch(pitch, target.harmony)
            result[idx] = NoteEvent(
                pitch=max(40, min(88, pitch)),
                duration_ql=note.duration_ql,
                velocity=note.velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )
        return result

    def _score_candidate(
        self,
        notes: List[NoteEvent],
        target: BarGenerationTarget,
        cluster_label: int = -1,
    ) -> float:
        """Score texture against skeleton, tonal tension, and copy penalty."""
        melody = self._melody_events(notes)
        if not melody:
            return -1e6
        pitches = [n.pitch for n in melody]
        durations = [n.duration_ql for n in melody]
        mean_pitch = float(np.mean(pitches))
        score = 0.0

        skeleton_cfg = self.config.get("skeleton", {})
        register_score_weight = float(skeleton_cfg.get("register_score_weight", 0.025))
        score -= abs(mean_pitch - target.register_target) * register_score_weight
        score -= abs(pitches[-1] - target.target_pitch) * target.cadence_strength * 0.12

        if target.contour and len(pitches) >= 2:
            contour = tuple(pitches[i + 1] - pitches[i] for i in range(min(len(pitches) - 1, len(target.contour))))
            contour_dist = sum(abs(a - b) for a, b in zip(contour, target.contour))
            score -= contour_dist * 0.10
            same_sign = sum(
                1 for a, b in zip(contour, target.contour)
                if (a == 0 and b == 0) or (a * b > 0)
            )
            score += same_sign * 0.20
            contour_similarity = same_sign / max(1, len(contour))
            if contour_similarity < target.similarity_min:
                score -= (target.similarity_min - contour_similarity) * 1.2
            elif contour_similarity > target.similarity_max:
                score -= (contour_similarity - target.similarity_max) * 1.6

        if target.rhythm_cell:
            rhythm_dist = 0.0
            for a, b in zip(durations, target.rhythm_cell):
                rhythm_dist += abs(a - b)
            score -= rhythm_dist * 0.12
            if durations:
                rhythm_similarity = 1.0 / (1.0 + rhythm_dist)
                if rhythm_similarity > target.similarity_max:
                    score -= (rhythm_similarity - target.similarity_max) * target.exact_copy_penalty

        # Tonal tension fit: high-tension bars may wander; low-tension bars
        # should stay near the target degree and cadence clearly.
        pc_distances = [
            min((p % 12 - target.target_degree) % 12, (target.target_degree - p % 12) % 12)
            for p in pitches
        ]
        avg_pc_distance = float(np.mean(pc_distances))
        score -= avg_pc_distance * (0.18 * (1.0 - target.tension))

        exactish = 0
        if target.contour and len(pitches) >= len(target.contour) + 1:
            contour = tuple(pitches[i + 1] - pitches[i] for i in range(len(target.contour)))
            if contour == target.contour:
                exactish += 1
        if target.rhythm_cell and tuple(round(d, 3) for d in durations[:len(target.rhythm_cell)]) == target.rhythm_cell:
            exactish += 1
        score -= exactish * target.exact_copy_penalty
        harmony_score = HarmonicPlanner.score_melody(notes, target.harmony, self.config)
        score += harmony_score
        conditional_model = getattr(self.model, "conditional_note_model", None)
        if conditional_model is not None:
            conditional_cfg = self.config.get("conditional_note_model", {})
            if isinstance(conditional_cfg, dict) and conditional_cfg.get("enabled", True):
                weight = float(conditional_cfg.get("candidate_score_weight", 0.85))
                score += weight * conditional_model.score_candidate(
                    notes,
                    target,
                    cluster_label,
                    self.config,
                )
        if target.development_role == "CADENTIAL":
            score -= abs(pitches[-1] - target.target_pitch) * 0.18
        elif target.development_role in ("SEQUENCE_UP", "INTENSIFY"):
            score += max(0.0, pitches[-1] - pitches[0]) * 0.04
        elif target.development_role in ("SEQUENCE_DOWN", "RELAX"):
            score += max(0.0, pitches[0] - pitches[-1]) * 0.04
        return score

    @staticmethod
    def _phrase_role(bar_index: int, section_len: int, structural_role: str) -> str:
        """Assign a simple intra-section role used by the note renderer."""
        if structural_role in ("FREE", "FLAT") or section_len <= 1:
            return "CONTINUATION"
        if bar_index == 0:
            return "OPENING"
        if bar_index == 1:
            return "ANSWER"
        if bar_index >= section_len - 1:
            return "CADENCE"
        if bar_index == section_len - 2:
            return "CADENCE_PREP"
        midpoint = max(2, section_len // 2)
        if bar_index >= midpoint:
            return "DEVELOPMENT"
        return "CONTINUATION"

    def _theme_development_config(self) -> Dict[str, Any]:
        """Return theme-development config with product-safe fallbacks."""
        cfg = self.config.get("theme_development", {})
        return cfg if isinstance(cfg, dict) else {}

    def _theme_development_role_config(self, role: str) -> Dict[str, Any]:
        cfg = self._theme_development_config()
        roles = cfg.get("roles", {})
        if not isinstance(roles, dict):
            roles = {}
        role_cfg = roles.get(role, {})
        return role_cfg if isinstance(role_cfg, dict) else {}

    def _choose_theme_development_role(
        self,
        structural_role: str,
        occurrence_index: int,
        section_pos: float,
        is_final_occurrence: bool,
    ) -> str:
        """Map a structural occurrence to a theme-development function.

        The sequence itself is config-driven.  The code only supplies stable
        musical defaults for first statements, literal repeats, and final
        cadences so different profiles can swap the development vocabulary.
        """
        cfg = self._theme_development_config()
        if structural_role == "NEW" or occurrence_index <= 0:
            return str(cfg.get("statement_role", "STATEMENT")).upper()
        if structural_role == "REPEAT":
            return str(cfg.get("repeat_role", "REPEAT")).upper()
        if is_final_occurrence:
            return str(cfg.get("final_role", "CADENTIAL")).upper()

        arc = cfg.get("return_arc", ["REPEAT", "SEQUENCE_UP", "FRAGMENT", "INTENSIFY", "CADENTIAL"])
        if not isinstance(arc, list) or not arc:
            arc = ["REPEAT", "SEQUENCE_UP", "FRAGMENT", "INTENSIFY", "CADENTIAL"]
        index = min(len(arc) - 1, max(0, occurrence_index - 1))
        role = str(arc[index]).upper()

        # Let late returns lean toward closure without forcing every final
        # section to be a cadence.
        late_threshold = float(cfg.get("late_cadential_threshold", 0.86))
        late_role = str(cfg.get("late_role", role)).upper()
        if section_pos >= late_threshold:
            return late_role
        return role

    def _build_composition_plan(
        self,
        labels: List[int],
        measure_context: List[Tuple[str, int, str, int, int]],
        base_seed: int,
    ) -> CompositionPlan:
        """Build a tonal-emotional plan for the whole generated piece.

        This layer gives separate sections a shared gravitational center.
        Section labels get stable tonal/register identities, RETURN sections
        reuse that identity, and each measure receives an affect target that
        shapes register, density, tension, and cadence behavior.
        """
        if not labels:
            return CompositionPlan(0, {}, {}, {}, {})

        pitch_mass = np.zeros(12, dtype=np.float64)
        for label in labels:
            pitch_mass += self.note_sampler._pitch_hists[label % self.note_sampler._n_clusters]
        global_tonic_pc = int(np.argmax(pitch_mass)) if pitch_mass.sum() > 0 else 0

        section_affects: Dict[int, SectionAffect] = {}
        measure_targets: Dict[int, int] = {}
        measure_affects: Dict[int, Dict[str, float]] = {}
        development_roles: Dict[int, str] = {}
        theme_identity: Dict[str, Tuple[int, int]] = {}
        label_order: Dict[str, int] = {}
        label_occurrence_count: Dict[str, int] = {}
        narrative_plan = NarrativePlanner(self.config).build(
            measure_context,
            labels,
            seed=_stable_hash(base_seed, "narrative"),
        )

        by_occurrence: Dict[int, List[int]] = {}
        for idx, (_, _, role, occurrence_id, _) in enumerate(measure_context):
            if role not in ("FREE", "FLAT"):
                by_occurrence.setdefault(occurrence_id, []).append(idx)

        total = max(1, len(measure_context) - 1)
        for occurrence_id, indices in by_occurrence.items():
            first_idx = indices[0]
            label, _, role, _, section_len = measure_context[first_idx]
            if label not in label_order:
                label_order[label] = len(label_order)
            occurrence_index = label_occurrence_count.get(label, 0)
            label_occurrence_count[label] = occurrence_index + 1

            if label in theme_identity:
                tonal_pc, register_center = theme_identity[label]
            else:
                order = label_order[label]
                offset_pool = [0, 7, 9, 5, 3, 2]
                tonal_pc = (global_tonic_pc + offset_pool[order % len(offset_pool)]) % 12
                rng = np.random.RandomState(_stable_hash(base_seed, label, "theme-identity"))
                register_center = int(rng.choice([60, 62, 64, 65, 67]))
                register_center = self._nearest_pitch(tonal_pc, register_center)
                theme_identity[label] = (tonal_pc, register_center)

            section_pos = first_idx / total
            is_final_occurrence = indices[-1] >= len(measure_context) - 2
            development_role = self._choose_theme_development_role(
                role,
                occurrence_index,
                section_pos,
                is_final_occurrence,
            )
            role_cfg = self._theme_development_role_config(development_role)
            arc = math.sin(math.pi * section_pos)
            role_tension = 0.18 if role in ("NEW", "REPEAT") else 0.10
            if role in ("RETURN", "VARIANT"):
                role_tension += 0.08
            tension = float(np.clip(
                (0.22 + 0.45 * arc + role_tension) * float(role_cfg.get("tension_scale", 1.0)),
                0.0,
                1.0,
            ))
            intensity = float(np.clip(
                (0.35 + 0.50 * arc + (0.08 if role == "VARIANT" else 0.0))
                * float(role_cfg.get("intensity_scale", 1.0)),
                0.0,
                1.0,
            ))
            if role == "RETURN":
                tension *= 0.82
                intensity *= 0.92

            section_affects[occurrence_id] = SectionAffect(
                label=label,
                role=role,
                occurrence_id=occurrence_id,
                tonal_pc=tonal_pc,
                register_center=register_center,
                intensity=intensity,
                tension=tension,
            )

            skeleton_cfg = self.config.get("skeleton", {})
            lift_base = float(skeleton_cfg.get("tension_register_lift_base", 5))
            lift_scale = float(skeleton_cfg.get("tension_register_lift_scale", 9))
            peak_pos = max(1, int(round((len(indices) - 1) * (0.50 + 0.15 * arc))))
            register_shift = float(role_cfg.get("register_shift", 0.0))
            high_pitch = min(88, register_center + int(round(lift_base + lift_scale * tension + register_shift)))
            cadence_pitch = self._nearest_pitch(tonal_pc, register_center - int(round(2 + 3 * (1.0 - tension))))

            for local_i, global_i in enumerate(indices):
                bar_development_role = development_role
                if local_i == len(indices) - 1 and bool(
                    self._theme_development_config().get("force_cadential_last_bar", True)
                ):
                    bar_development_role = "CADENTIAL"
                narrative = narrative_plan.get(global_i)
                if narrative is not None:
                    bar_development_role = narrative.development_role
                development_roles[global_i] = bar_development_role
                if len(indices) == 1:
                    target = cadence_pitch
                elif local_i <= peak_pos:
                    frac = local_i / max(1, peak_pos)
                    target = round(register_center + (high_pitch - register_center) * frac)
                else:
                    frac = (local_i - peak_pos) / max(1, len(indices) - 1 - peak_pos)
                    target = round(high_pitch + (cadence_pitch - high_pitch) * frac)
                if local_i == len(indices) - 1:
                    target = cadence_pitch
                if development_role == "SEQUENCE_UP":
                    target += int(round(local_i * float(role_cfg.get("sequence_step_per_bar", 1.0))))
                elif development_role == "SEQUENCE_DOWN":
                    target -= int(round(local_i * float(role_cfg.get("sequence_step_per_bar", 1.0))))
                elif development_role == "INTENSIFY":
                    target += int(round(2.0 * local_i / max(1, len(indices) - 1)))
                elif development_role == "RELAX":
                    target -= int(round(2.0 * local_i / max(1, len(indices) - 1)))

                phrase_frac = local_i / max(1, len(indices) - 1)
                local_tension = float(np.clip(tension * (0.75 + 0.45 * math.sin(math.pi * phrase_frac)), 0.0, 1.0))
                local_intensity = intensity
                if narrative is not None:
                    target += int(round(narrative.register_shift))
                    local_tension = float(np.clip(
                        0.42 * local_tension + 0.58 * narrative.tension,
                        0.0,
                        1.0,
                    ))
                    local_intensity = float(np.clip(
                        0.45 * local_intensity + 0.55 * narrative.intensity,
                        0.0,
                        1.0,
                    ))
                if local_i == len(indices) - 1:
                    local_tension *= 0.55
                final_release = max(0.0, (global_i - (len(measure_context) - 9)) / 8.0)
                if final_release > 0.0:
                    release_amount = min(1.0, final_release ** 0.75)
                    tonic_release_pitch = self._nearest_pitch(global_tonic_pc, 62)
                    target = round(target + (tonic_release_pitch - target) * release_amount)
                    local_tension *= 1.0 - 0.72 * release_amount
                    local_intensity *= 1.0 - 0.38 * release_amount
                measure_targets[global_i] = int(max(40, min(84, target)))
                measure_affects[global_i] = {
                    "tonal_pc": float(tonal_pc),
                    "register_center": float(register_center + (narrative.register_shift if narrative else 0.0)),
                    "intensity": local_intensity,
                    "tension": local_tension,
                }
                if narrative is not None:
                    measure_affects[global_i].update(narrative.to_affect())

        # FREE/FLAT bars still need tonal gravity, otherwise connective tissue
        # can drift away from the emotional premise of the piece.
        tonic_register = self._nearest_pitch(global_tonic_pc, 64)
        for i, (_, _, role, _, _) in enumerate(measure_context):
            if i in measure_affects:
                continue
            pos = i / total
            arc = math.sin(math.pi * pos)
            measure_targets[i] = tonic_register + int(round(2 * arc))
            narrative = narrative_plan.get(i)
            if narrative is not None:
                measure_targets[i] += int(round(narrative.register_shift))
            measure_affects[i] = {
                "tonal_pc": float(global_tonic_pc),
                "register_center": float(tonic_register + (narrative.register_shift if narrative else 0.0)),
                "intensity": float(
                    0.32 + 0.32 * arc if narrative is None else 0.45 * (0.32 + 0.32 * arc) + 0.55 * narrative.intensity
                ),
                "tension": float(
                    0.18 + 0.25 * arc if narrative is None else 0.42 * (0.18 + 0.25 * arc) + 0.58 * narrative.tension
                ),
            }
            if narrative is not None:
                measure_affects[i].update(narrative.to_affect())
                development_roles[i] = narrative.development_role

        return CompositionPlan(
            global_tonic_pc=global_tonic_pc,
            section_affects=section_affects,
            measure_targets=measure_targets,
            measure_affects=measure_affects,
            development_roles=development_roles,
        )

    def _build_melodic_skeleton(
        self,
        labels: List[int],
        measure_context: List[Tuple[str, int, str, int, int]],
        base_seed: int,
        composition_plan: Optional[CompositionPlan] = None,
    ) -> Dict[int, int]:
        """Create bar-level target pitches for phrase direction.

        This is intentionally lightweight: it does not force a melody, but it
        gives each section an opening point, a high point, and a cadence target.
        """
        if composition_plan is not None and composition_plan.measure_targets:
            return dict(composition_plan.measure_targets)

        skeleton: Dict[int, int] = {}
        by_occurrence: Dict[int, List[int]] = {}
        for idx, ctx in enumerate(measure_context):
            _, _, role, occurrence_id, _ = ctx
            if role not in ("FREE", "FLAT"):
                by_occurrence.setdefault(occurrence_id, []).append(idx)

        for occurrence_id, indices in by_occurrence.items():
            if not indices:
                continue
            rng = np.random.RandomState(_stable_hash(base_seed, "skeleton", occurrence_id))
            first_cluster = labels[indices[0]] % self.note_sampler._n_clusters
            last_cluster = labels[indices[-1]] % self.note_sampler._n_clusters
            first_pc = int(np.argmax(self.note_sampler._pitch_hists[first_cluster]))
            last_pc = int(np.argmax(self.note_sampler._pitch_hists[last_cluster]))
            start_pitch = self._nearest_pitch(first_pc, int(rng.choice([60, 62, 64, 65, 67])))
            cadence_pitch = self._nearest_pitch(last_pc, start_pitch)
            high_pitch = min(84, max(start_pitch, cadence_pitch) + int(rng.choice([3, 4, 5, 7])))
            peak_pos = max(1, int(round((len(indices) - 1) * rng.uniform(0.45, 0.65))))

            for local_i, global_i in enumerate(indices):
                if len(indices) == 1:
                    target = cadence_pitch
                elif local_i <= peak_pos:
                    frac = local_i / max(1, peak_pos)
                    target = round(start_pitch + (high_pitch - start_pitch) * frac)
                else:
                    frac = (local_i - peak_pos) / max(1, len(indices) - 1 - peak_pos)
                    target = round(high_pitch + (cadence_pitch - high_pitch) * frac)
                skeleton[global_i] = int(max(40, min(84, target)))
        return skeleton

    @staticmethod
    def _nearest_pitch(pc: int, reference: int) -> int:
        candidates = [reference + offset for offset in range(-12, 13) if (reference + offset) % 12 == pc]
        return min(candidates, key=lambda p: (abs(p - reference), p)) if candidates else reference

    def _build_return_variation_plan(
        self,
        section_len: int,
        role: str,
        rng: np.random.RandomState,
    ) -> List[str]:
        """Create a phrase-level plan for a returned section.

        The important product behavior is that a RETURN is recognisable as the
        same family without replaying every bar literally.  The plan assigns a
        distinct variation role per bar, so contour and cadence evolve across
        the phrase instead of each bar receiving an isolated random nudge.
        """
        section_len = max(1, int(section_len))
        if role == "REPEAT":
            palette = ["ANCHOR", "CONTOUR", "ANCHOR", "RHYTHM"]
        else:
            palette = ["ANCHOR", "CONTOUR", "DEVELOP", "RHYTHM"]

        plan: List[str] = []
        rotation = int(rng.randint(0, len(palette)))
        for i in range(section_len):
            if i == section_len - 1:
                plan.append("CADENCE")
            elif i == 0:
                plan.append("ANCHOR" if role == "REPEAT" else "CONTOUR")
            else:
                plan.append(palette[(i + rotation) % len(palette)])

        # Avoid long runs of exact anchors, which were the main cause of
        # repeated bar signatures in returned material.
        for i in range(1, len(plan) - 1):
            if plan[i - 1] == "ANCHOR" and plan[i] == "ANCHOR":
                plan[i] = "CONTOUR"
        return plan

    def _vary_return_motif(
        self,
        notes: List[NoteEvent],
        strength: float,
        rng: np.random.RandomState,
        target_pitch: Optional[int],
        variation_mode: str = "CONTOUR",
        phrase_role: str = "CONTINUATION",
        development_role: str = "REPEAT",
        target_attraction: Optional[float] = None,
        rhythm_change_scale: Optional[float] = None,
    ) -> List[NoteEvent]:
        """Convert a recalled motif into a same-family variant.

        Motif memory should preserve identity, but RETURN must not mean
        byte-for-byte repetition forever.  This function keeps onset rhythm
        stable and changes only a small number of melody pitches, velocities,
        and occasional short durations.  That gives recognisable recurrence
        without allowing long exact-copy loops to dominate the output.
        """
        varied = self._clone_notes(notes)
        melody_indices = [
            idx for idx, note in enumerate(varied)
            if note.pitch >= 0 and note.voice == "melody"
        ]
        if not melody_indices:
            return varied

        motif_cfg = self.config.get("motif_return", {})
        min_pitch = int(motif_cfg.get("min_pitch", 40))
        max_pitch = int(motif_cfg.get("max_pitch", 84))
        if target_attraction is None:
            target_attraction = float(motif_cfg.get("target_attraction", 0.55))
        velocity_change_prob = float(motif_cfg.get("velocity_change_prob", 0.35))
        velocity_jitter_std = float(motif_cfg.get("velocity_jitter_std", 3.0))
        if rhythm_change_scale is None:
            rhythm_change_scale = float(motif_cfg.get("rhythm_change_scale", 0.35))
        else:
            rhythm_change_scale *= float(motif_cfg.get("rhythm_change_scale", 0.35))
        mode = variation_mode.upper()
        development_role = development_role.upper()
        mode_scale = {
            "ANCHOR": 0.65,
            "CONTOUR": 1.15,
            "DEVELOP": 1.35,
            "RHYTHM": 1.05,
            "CADENCE": 1.10,
        }.get(mode, 1.0)
        development_scale = {
            "STATEMENT": 0.0,
            "REPEAT": 0.75,
            "SEQUENCE_UP": 1.20,
            "SEQUENCE_DOWN": 1.20,
            "FRAGMENT": 1.45,
            "EXTENSION": 1.15,
            "INTENSIFY": 1.35,
            "RELAX": 1.05,
            "CADENTIAL": 1.10,
        }.get(development_role, 1.0)
        effective_strength = min(0.85, max(0.02, strength * mode_scale * development_scale))

        max_changes = max(1, int(round(len(melody_indices) * effective_strength)))
        change_count = min(len(melody_indices), max_changes)
        selected = set(rng.choice(melody_indices, size=change_count, replace=False))
        if mode in ("CONTOUR", "DEVELOP", "CADENCE"):
            selected.add(melody_indices[-1])

        for local_pos, idx in enumerate(melody_indices):
            note = varied[idx]
            pitch = note.pitch
            duration = note.duration_ql
            velocity = note.velocity

            if idx in selected:
                step = int(rng.choice([-2, -1, 1, 2]))
                if development_role == "SEQUENCE_UP":
                    step = int(rng.choice([1, 2]))
                elif development_role == "SEQUENCE_DOWN":
                    step = -int(rng.choice([1, 2]))
                elif development_role == "INTENSIFY":
                    step = int(rng.choice([1, 2, 3]))
                elif development_role == "RELAX":
                    step = -int(rng.choice([1, 2]))
                elif mode == "DEVELOP":
                    step = int(rng.choice([-3, -2, 2, 3]))
                elif (mode == "CADENCE" or development_role == "CADENTIAL") and target_pitch is not None:
                    step = 1 if target_pitch > pitch else -1
                elif mode == "CONTOUR" and target_pitch is not None:
                    # Push later notes more strongly toward the section-level
                    # skeleton target, producing a phrase-level direction.
                    progress = local_pos / max(1, len(melody_indices) - 1)
                    if progress > 0.45:
                        step = 1 if target_pitch > pitch else -1
                if target_pitch is not None and rng.random() < target_attraction:
                    step = 1 if target_pitch > pitch else -1
                pitch = int(max(min_pitch, min(max_pitch, pitch + step)))

                # Keep rhythmic identity on returns; only very short gestures
                # may breathe a little so the bar is not mechanically copied.
                rhythm_prob = effective_strength * rhythm_change_scale
                if mode == "RHYTHM":
                    rhythm_prob = min(0.8, rhythm_prob * 2.0)
                if duration <= 0.75 and rng.random() < rhythm_prob:
                    candidates = [
                        d for d in _USABLE_DUR_VALUES
                        if 0.1 <= d <= 1.0 and abs(d - duration) <= 0.25
                    ]
                    if candidates:
                        duration = float(rng.choice(candidates))

            if development_role == "FRAGMENT" and local_pos >= max(1, len(melody_indices) // 2):
                # A fragment keeps the recognizable head and lets the tail
                # relax away from the exact recalled contour.
                pitch = int(max(min_pitch, min(max_pitch, pitch + rng.choice([-2, -1, 1, 2]))))
                velocity = max(35, int(velocity * 0.94))

            if (mode == "CADENCE" or development_role == "CADENTIAL") and idx == melody_indices[-1] and target_pitch is not None:
                pitch = int(max(min_pitch, min(max_pitch, target_pitch)))
                if phrase_role == "CADENCE":
                    duration = max(duration, 0.75)

            if velocity > 0 and rng.random() < velocity_change_prob:
                velocity = int(max(35, min(127, velocity + rng.normal(0, velocity_jitter_std))))

            varied[idx] = NoteEvent(
                pitch=pitch,
                duration_ql=duration,
                velocity=velocity,
                beat_offset=note.beat_offset,
                voice=note.voice,
            )

        return varied
