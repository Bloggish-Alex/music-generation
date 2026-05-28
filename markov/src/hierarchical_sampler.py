#!/usr/bin/env python3
"""Per-measure note sampling for hierarchical generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hierarchical_types import NoteEvent

# ---- melody / pitch walk ----
STEP_UPWARD_BIAS = 0.55
# Duration categories actually used (excludes sub-sixteenth and triplets for sanity)
_USABLE_DUR_VALUES: List[float] = [4.0, 2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75]
_USABLE_DUR_INDICES: List[int] = [0, 1, 2, 3, 4, 6, 7, 8]


class ClusterNoteSampler:
    """Generate per-measure notes from cluster centroids + pitch histograms.

    Each cluster's 8-D centroid determines rhythmic texture (density,
    duration preferences, rest probability, syncopation).  The 12-D
    pitch-class histogram determines which pitch classes are likely.

    Pitch continuity is enforced via random walk within a constrained
    register; velocity follows a phrase-level arc.
    """

    def __init__(
        self,
        centroids: np.ndarray,               # (n_clusters, 8)
        pitch_histograms: np.ndarray | None,  # (n_clusters, 12) or None
        step_histograms: np.ndarray | None = None,  # (n_clusters, 7) or None
        bass_histograms: np.ndarray | None = None,  # (n_clusters, 128) or None
        phrase_role_stats: Dict[int, Dict[str, Dict[str, float]]] | None = None,
        bass_config: Dict[str, Any] | None = None,
    ) -> None:
        if centroids.ndim != 2 or centroids.shape[1] < 8:
            raise ValueError("centroids must be (n_clusters, >=8)")
        self._centroids = centroids
        self._n_clusters = centroids.shape[0]

        if pitch_histograms is not None and pitch_histograms.shape[0] == self._n_clusters:
            self._pitch_hists = pitch_histograms
        else:
            self._pitch_hists = np.full(
                (self._n_clusters, 12), 1.0 / 12, dtype=np.float64,
            )

        if step_histograms is not None and step_histograms.shape == (self._n_clusters, 7):
            self._step_hists = step_histograms
        else:
            self._step_hists = np.array(
                [[0.20, 0.18, 0.18, 0.15, 0.12, 0.10, 0.07]] * self._n_clusters,
                dtype=np.float64,
            )

        if bass_histograms is not None and bass_histograms.shape == (self._n_clusters, 128):
            self._bass_hists = bass_histograms
        else:
            self._bass_hists = None

        self._bass_config = bass_config or {}
        self._bass_enabled = self._bass_config.get("enabled", True) and self._bass_hists is not None
        self._phrase_role_stats = phrase_role_stats or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_measure(
        self,
        cluster_label: int,
        time_signature: Tuple[int, int] = (4, 4),
        seed: int | None = None,
        perturb: float = 0.0,
        is_section_end: bool = False,
        phrase_role: str = "CONTINUATION",
        target_pitch: int | None = None,
        previous_pitch: int | None = None,
        affect: Optional[Dict[str, float]] = None,
    ) -> List[NoteEvent]:
        """Generate notes for one measure with melodic continuity.

        Args:
            cluster_label: Which cluster (0..k-1) this measure belongs to.
            time_signature: (numerator, denominator).
            seed: Per-measure random seed for reproducibility.
            perturb: 0.0 = exact, ~0.25 = mild variation (RETURN),
                ~0.5 = stronger variation (VARIANT).
            is_section_end: If True, ends the measure with a held note or
                rest to create breathing space at section boundaries.

        Returns:
            List of NoteEvent, whose durations sum to *bar_length_ql*.
        """
        rng = np.random.RandomState(seed)
        c = cluster_label % self._n_clusters
        centroid = self._centroids[c]
        pc_hist = self._pitch_hists[c]

        note_density = float(centroid[0])
        mean_dur = float(centroid[1])
        dur_var = float(centroid[2])
        short_ratio = float(centroid[3])
        silence_ratio = float(centroid[4])
        offbeat_ratio = float(centroid[5])
        # syncopation = centroid[6] — unused for now
        syncopation = float(centroid[6])
        entropy = float(centroid[7])
        cluster_pitch_mean = float(centroid[8]) if len(centroid) > 8 else 60.0
        cluster_pitch_slope = float(centroid[10]) if len(centroid) > 10 else 0.0
        cluster_pitch_range = float(centroid[9]) if len(centroid) > 9 else 8.0
        cluster_cadence = float(centroid[17]) if len(centroid) > 17 else 0.0

        role_profile = self._phrase_role_stats.get(c, {}).get(phrase_role, {})
        if role_profile:
            # These are learned corpus ratios for the same cluster under the
            # current phrase role.  Clamping keeps sparse roles from producing
            # extreme values while still letting the training data shape the
            # phrase direction.
            note_density *= float(np.clip(role_profile.get("density_scale", 1.0), 0.45, 1.8))
            mean_dur *= float(np.clip(role_profile.get("duration_scale", 1.0), 0.55, 1.8))
            entropy *= float(np.clip(role_profile.get("entropy_scale", 1.0), 0.45, 1.8))
            offbeat_ratio *= float(np.clip(role_profile.get("offbeat_scale", 1.0), 0.25, 2.0))

        affect = affect or {}
        intensity = float(np.clip(affect.get("intensity", 0.5), 0.0, 1.0))
        tension = float(np.clip(affect.get("tension", 0.35), 0.0, 1.0))
        tonal_pc = int(affect.get("tonal_pc", -1))
        register_center = affect.get("register_center")
        note_density *= 0.80 + 0.45 * intensity
        entropy *= 0.70 + 0.75 * tension
        offbeat_ratio *= 0.75 + 0.65 * tension

        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)

        if phrase_role == "OPENING":
            entropy *= 0.75
            offbeat_ratio *= 0.7
        elif phrase_role == "ANSWER":
            entropy *= 0.9
        elif phrase_role == "DEVELOPMENT":
            entropy *= 1.2
            syncopation *= 1.2
        elif phrase_role == "CADENCE_PREP":
            note_density *= 0.85
            offbeat_ratio *= 0.6
        elif phrase_role == "CADENCE":
            note_density *= 0.65
            entropy *= 0.6
            offbeat_ratio *= 0.25

        # --- rest probability ---
        rest_prob = float(np.clip(silence_ratio, 0.0, 0.6))
        if phrase_role == "CADENCE":
            rest_prob = min(0.75, rest_prob + 0.15)


        # --- note count ---
        # density is roughly onsets per quarter note; cap at 6 notes/quarter
        raw_count = max(2, int(note_density * bar_length_ql * 2.0))
        raw_count += int(rng.normal(0, entropy * 1.5))
        target_count = max(2, min(raw_count, 24))
        if phrase_role == "CADENCE":
            target_count = min(target_count, 6)
        elif phrase_role == "OPENING":
            target_count = min(target_count, 10)

        # --- pick 2-4 "preferred" durations for rhythmic coherence ---
        pref_durs = self._pick_preferred_durations(
            mean_dur, dur_var, short_ratio, rng,
        )

        # --- choose a register for this measure ---
        # A melodic skeleton target keeps repeated sections directionally
        # coherent; without one, fall back to the cluster pitch-class histogram.
        if target_pitch is not None:
            centre_pitch = int(target_pitch)
        elif register_center is not None:
            centre_pitch = int(register_center)
        else:
            pc_centre = int(rng.choice(12, p=pc_hist))
            centre_octave = int(rng.choice([3, 3, 4, 4, 4, 5]))  # weighted toward middle
            centre_pitch = 12 * (centre_octave + 1) + pc_centre
            if len(centroid) > 8:
                centre_pitch = int(round(centre_pitch * 0.55 + cluster_pitch_mean * 0.45))
        if role_profile:
            centre_pitch = int(round(centre_pitch + role_profile.get("pitch_offset", 0.0)))
        centre_pitch = max(40, min(84, centre_pitch))
        window = int(max(7, min(14, cluster_pitch_range * 0.65 + 5)))
        lo_pitch = max(28, centre_pitch - window)
        hi_pitch = min(96, centre_pitch + window)

        # Start pitch near centre
        if previous_pitch is not None:
            current_pitch = int(round(previous_pitch * 0.55 + centre_pitch * 0.45))
        else:
            current_pitch = int(centre_pitch + rng.randint(-4, 5))
        current_pitch = max(lo_pitch, min(hi_pitch, current_pitch))

        # --- velocity arc ---
        vel_peak = int(rng.normal(100, 8))  # peak velocity this measure
        vel_trough = int(rng.normal(55, 10))
        vel_peak_frac = rng.uniform(0.25, 0.55)  # when the peak hits in the bar

        # --- generate notes ---
        notes: List[NoteEvent] = []
        remaining = bar_length_ql
        beat = 0.0
        note_idx = 0

        while remaining > 0.03 and note_idx < target_count:
            note_idx += 1

            # --- rest? ---
            if note_idx > 1 and rng.random() < rest_prob * 0.5:
                # Insert a short rest
                rest_choices = [d for d in pref_durs if d <= remaining + 0.01]
                if not rest_choices:
                    break
                rest_dur = float(rng.choice(rest_choices))
                rest_dur = float(min(rest_dur, remaining))
                notes.append(NoteEvent(
                    pitch=-1, duration_ql=rest_dur,
                    velocity=0, beat_offset=beat,
                ))
                beat += rest_dur
                remaining -= rest_dur
                if remaining <= 0.03:
                    break

            # --- duration ---
            eligible = [(d, d) for d in pref_durs if d <= remaining + 0.01]
            if not eligible:
                dur = remaining
            else:
                # Weight by inverse of how often each dur has been used
                used_counts = {d: 0 for d in pref_durs}
                for n in notes:
                    if n.duration_ql in used_counts:
                        used_counts[n.duration_ql] += 1
                w = np.array([1.0 / (1.0 + used_counts.get(d, 0))
                               for d, _ in eligible], dtype=np.float64)
                w /= w.sum()
                dur = float(eligible[int(rng.choice(len(eligible), p=w))][0])

            dur = float(min(dur, remaining))
            if dur < 0.25:
                dur = 0.25  # no shorter than sixteenth

            # --- pitch (random walk with pitch-class constraint) ---
            if note_idx == 1:
                pitch = current_pitch
            else:
                # Step size: from per-cluster interval distribution (learned from data)
                step_weights = self._step_hists[c]
                step_dir = 1 if rng.random() < STEP_UPWARD_BIAS else -1
                max_step = int(rng.choice(len(step_weights), p=step_weights))
                step = rng.randint(0, max_step + 3) * step_dir

                # Bias toward pitch classes in the histogram
                candidate_pitch = current_pitch + step
                candidate_pc = candidate_pitch % 12
                if rng.random() > pc_hist[candidate_pc] * 4:  # unlikely PC → reject leap
                    candidate_pitch = current_pitch + int(step * 0.5)
                    candidate_pc = candidate_pitch % 12

                if tonal_pc >= 0 and rng.random() < 0.28 + 0.35 * (1.0 - tension):
                    pc_distance = min(
                        (candidate_pc - tonal_pc) % 12,
                        (tonal_pc - candidate_pc) % 12,
                    )
                    if pc_distance > 2:
                        direction = 1 if ((tonal_pc - candidate_pc) % 12) < 6 else -1
                        candidate_pitch += direction

                if target_pitch is not None:
                    delta = int(target_pitch - candidate_pitch)
                    if abs(delta) > 2 and rng.random() < 0.45:
                        candidate_pitch += int(np.sign(delta) * min(abs(delta), 3))
                elif role_profile:
                    role_slope = float(role_profile.get("pitch_slope", cluster_pitch_slope))
                    if abs(role_slope) > 0.2 and rng.random() < 0.35:
                        candidate_pitch += int(np.sign(role_slope))

                current_pitch = max(lo_pitch, min(hi_pitch, candidate_pitch))

            pitch = int(current_pitch)
            if target_pitch is not None and phrase_role == "CADENCE" and remaining - dur <= 0.03:
                pitch = max(lo_pitch, min(hi_pitch, int(target_pitch)))
                current_pitch = pitch
            elif phrase_role == "CADENCE" and role_profile and remaining - dur <= 0.03:
                closure = float(role_profile.get("cadence_closure", cluster_cadence))
                if closure > 0.08:
                    pitch = self._nearest_pitch_to_pc(int(np.argmax(pc_hist)), centre_pitch)
                    current_pitch = pitch

            # --- velocity (phrase arc) ---
            progress = beat / bar_length_ql if bar_length_ql > 0 else 0
            vel_peak_local = int(vel_peak * (0.85 + 0.30 * intensity))
            vel_trough_local = int(vel_trough * (0.90 + 0.20 * intensity))
            if progress < vel_peak_frac:
                # Build toward peak
                frac = progress / vel_peak_frac
                vel = int(vel_trough_local + (vel_peak_local - vel_trough_local) * frac)
            else:
                # Decay after peak
                frac = (progress - vel_peak_frac) / (1.0 - vel_peak_frac)
                vel = int(vel_peak_local + (vel_trough_local - vel_peak_local) * frac)
            vel += int(rng.normal(0, 5))  # small jitter
            vel = max(35, min(127, vel))

            # --- beat offset (subtle swing for offbeat clusters) ---
            if rng.random() < offbeat_ratio * 0.3:
                beat_offset = beat + rng.uniform(0.02, 0.12)
            else:
                beat_offset = beat

            notes.append(NoteEvent(
                pitch=pitch,
                duration_ql=dur,
                velocity=vel,
                beat_offset=beat_offset,
            ))

            beat += dur
            remaining -= dur

        self._apply_breathing(notes, is_section_end, bar_length_ql)
        self._apply_perturbation(notes, perturb, lo_pitch, hi_pitch, seed)

        return notes

    @staticmethod
    def _nearest_pitch_to_pc(pc: int, reference: int) -> int:
        """Nearest pitch with pitch class ``pc`` around ``reference``."""
        candidates = [
            reference + offset
            for offset in range(-12, 13)
            if (reference + offset) % 12 == pc
        ]
        return min(candidates, key=lambda p: (abs(p - reference), p)) if candidates else reference

    @staticmethod
    def _apply_breathing(
        notes: List[NoteEvent],
        is_section_end: bool,
        bar_length_ql: float,
    ) -> None:
        """Shorten the last note to create a gap at section boundaries."""
        if not is_section_end or not notes:
            return
        gap = 0.5
        for idx in range(len(notes) - 1, -1, -1):
            if notes[idx].pitch >= 0:
                n = notes[idx]
                new_dur = bar_length_ql - gap - n.beat_offset
                if new_dur > 0:
                    notes[idx] = NoteEvent(
                        pitch=n.pitch,
                        duration_ql=new_dur,
                        velocity=max(30, n.velocity - 15),
                        beat_offset=n.beat_offset,
                        voice=n.voice,
                    )
                else:
                    del notes[idx]
                    continue
                del notes[idx + 1:]
                break

    @staticmethod
    def _apply_perturbation(
        notes: List[NoteEvent],
        perturb: float,
        lo_pitch: int,
        hi_pitch: int,
        seed: int | None,
    ) -> None:
        """Post-generation random perturbation for RETURN / VARIANT."""
        if perturb <= 0.0 or not notes:
            return
        rng = np.random.RandomState(
            None if seed is None else seed + 999999,
        )
        for idx in range(len(notes)):
            n = notes[idx]
            if n.pitch < 0:
                continue
            if rng.random() < perturb:
                shift = int(rng.choice([-2, -1, 1, 2]))
                new_pitch = max(lo_pitch, min(hi_pitch, n.pitch + shift))
                dur = n.duration_ql
                if rng.random() < perturb * 0.5:
                    neighbors = [
                        d for d in _USABLE_DUR_VALUES
                        if abs(d - dur) < dur * 0.6 and d != dur
                    ]
                    if neighbors:
                        dur = float(rng.choice(neighbors))
                vel = n.velocity
                if vel > 0:
                    vel = max(35, min(127, int(vel + rng.normal(0, 5))))
                notes[idx] = NoteEvent(
                    pitch=new_pitch,
                    duration_ql=dur,
                    velocity=vel,
                    beat_offset=n.beat_offset,
                    voice=n.voice,
                )

    def _add_bass_note(
        self,
        notes: List[NoteEvent],
        cluster_label: int,
        bar_length_ql: float,
        rng: np.random.RandomState,
    ) -> None:
        """Append a bass anchor sampled from the trained bass distribution.

        The bass register and sustain are tunable. Defaults avoid two common
        failure modes: every bar using the same sustain and the bass collapsing
        into an impractically low register.
        """
        hist = self._bass_hists[cluster_label]
        pitch = int(rng.choice(128, p=hist))
        octave_shift = self._bass_config.get("octave_shift", 1)
        pitch = pitch - 12 * octave_shift
        low = int(self._bass_config.get("low", 36))
        high = int(self._bass_config.get("high", 60))
        pitch = max(low, min(high, pitch))

        sustain = float(self._bass_config.get("sustain_fraction", 0.78))
        full_bar_probability = float(self._bass_config.get("full_bar_probability", 0.22))
        if rng.random() < full_bar_probability:
            sustain = 1.0
        vel_scale = self._bass_config.get("velocity_scale", 0.7)
        avg_vel = int(sum(n.velocity for n in notes if n.velocity > 0)
                      / max(1, sum(1 for n in notes if n.velocity > 0)))
        vel = max(25, int(avg_vel * vel_scale))

        notes.append(NoteEvent(
            pitch=pitch,
            duration_ql=bar_length_ql * sustain,
            velocity=vel,
            beat_offset=0.0,
            voice="bass",
        ))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_preferred_durations(
        self, mean_dur: float, dur_var: float, short_ratio: float,
        rng: np.random.RandomState,
    ) -> List[float]:
        """Pick 2–4 preferred duration values for this measure.

        Selects durations close to *mean_dur*, with spread controlled by
        *dur_var* and a bias toward shorter values when *short_ratio* is
        high.  Returning a small set per measure creates rhythmic motifs.
        """
        scored = []
        for dv in _USABLE_DUR_VALUES:
            z = (dv - mean_dur) / max(dur_var, 0.08)
            score = np.exp(-0.5 * z * z)
            if dv <= 0.5:
                score += short_ratio * 1.5
            scored.append((dv, score))

        scored.sort(key=lambda x: -x[1])

        # Take top 2–4
        n_pick = int(rng.choice([2, 2, 3, 3, 4]))
        picked = [dv for dv, _ in scored[:n_pick]]
        return sorted(picked)
