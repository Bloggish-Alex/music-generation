#!/usr/bin/env python3
"""
Rhythmic Template Variation — strategy 1 for per-measure note generation.

Keeps the exact durations and onset positions from a real stored measure
(rhythmic template), then regenerates pitches via random walk constrained
by the cluster's pitch-class histogram.  Rhythmic identity is preserved;
melodic content is new but stylistically plausible.

Usage::

    from measure_clustering import MeasureClusterer
    from rhythmic_template_variation import RhythmicTemplateVariation

    clusterer = MeasureClusterer.load("./models/my_model/clusterer.pkl")
    strategy = RhythmicTemplateVariation(clusterer)
    notes = strategy.sample_measure(cluster_label=3, time_signature=(4, 4), seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from measure_clustering import MeasureClusterer

# ---------------------------------------------------------------------------
# NoteEvent — shared output type for all generation strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteEvent:
    """One note or rest within a measure."""

    pitch: int           # MIDI pitch 0–127, or -1 for rest
    duration_ql: float   # quarterLength
    velocity: int        # 1–127, 0 for rests
    beat_offset: float   # position within the bar (0.0 = downbeat)


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

_USABLE_DURATIONS = np.array([4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25], dtype=np.float64)


def _quantize_duration(ql: float) -> int:
    """Return index into _USABLE_DURATIONS closest to *ql*."""
    return int(np.argmin(np.abs(_USABLE_DURATIONS - ql)))


# ---------------------------------------------------------------------------
# RhythmicTemplateVariation
# ---------------------------------------------------------------------------


class RhythmicTemplateVariation:
    """Generate notes by transplanting new pitches onto stored rhythmic templates.

    For a given cluster, samples a real stored measure, preserves its duration
    and onset structure exactly, and regenerates pitches via a random walk
    biased toward the cluster's pitch-class distribution.
    """

    def __init__(self, clusterer: MeasureClusterer) -> None:
        n_clusters = clusterer.centroids.shape[0] if clusterer.centroids is not None else 0
        if n_clusters == 0:
            raise ValueError("Clusterer has no centroids — is it fitted?")
        self._clusterer = clusterer
        self._n_clusters = n_clusters
        self._pitch_hists = self._compute_pitch_histograms()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_measure(
        self,
        cluster_label: int,
        time_signature: Tuple[int, int] = (4, 4),
        seed: Optional[int] = None,
        is_section_end: bool = False,
    ) -> List[NoteEvent]:
        """Generate notes for one measure.

        Args:
            cluster_label: Which cluster (0..k-1).
            time_signature: (numerator, denominator).
            seed: Per-measure seed for reproducibility.
            is_section_end: If True, shortens the last note and appends a rest.

        Returns:
            List of NoteEvent whose durations sum to bar_length_ql.
        """
        rng = np.random.RandomState(seed)
        c = cluster_label % self._n_clusters

        # 1. Sample a stored measure as the rhythmic template
        template = self._clusterer.sample_measure(c, seed=seed)
        if template is None or not template.notes:
            return self._fallback(c, time_signature, rng)

        pc_hist = self._pitch_hists[c]

        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)

        # 2. Determine register from the template's pitch range
        pitches = [nd["pitch"] for nd in template.notes]
        lo_pitch = max(28, min(pitches) - 4)
        hi_pitch = min(96, max(pitches) + 4)
        centre = int(np.median(pitches)) if pitches else 60

        # Start pitch near the template's first pitch or median
        if pitches:
            start_pc = pitches[0] % 12
            # Keep same pitch class, adjust octave to centre
            current_pitch = (centre // 12) * 12 + start_pc
            if abs(current_pitch - centre) > 6:
                current_pitch += 12 if current_pitch < centre else -12
        else:
            current_pitch = centre
        current_pitch = max(lo_pitch, min(hi_pitch, current_pitch))

        # 3. Velocity baseline from template
        vel_baseline = 80

        # 4. Regenerate pitches on top of the rhythmic template
        notes: List[NoteEvent] = []
        for i, nd in enumerate(template.notes):
            dur = float(nd["quarterLength"])
            onset = float(nd.get("onset_in_measure", 0.0))

            # Clamp onset within bar bounds
            if onset < 0:
                onset = 0.0
            if onset >= bar_length_ql:
                onset = onset % bar_length_ql

            # Clamp duration so the note doesn't exceed bar boundary
            dur = min(dur, bar_length_ql - onset)
            if dur < 0.03:
                continue

            # --- Random walk for pitch ---
            if i == 0:
                pitch = current_pitch
            else:
                # Prefer small steps (≤ 2 semitones), rarely leap
                step_probs = np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])
                max_step = int(rng.choice(len(step_probs), p=step_probs))
                step = rng.randint(0, max_step + 3)
                direction = 1 if rng.random() < 0.55 else -1
                candidate = current_pitch + step * direction

                # Reject if unlikely pitch class
                candidate_pc = candidate % 12
                if rng.random() > pc_hist[candidate_pc] * 5:
                    candidate = current_pitch + int(step * 0.4) * direction

                current_pitch = max(lo_pitch, min(hi_pitch, candidate))
            pitch = int(current_pitch)

            # --- Velocity with small jitter ---
            vel = max(40, min(127, int(vel_baseline + rng.normal(0, 8))))

            notes.append(NoteEvent(
                pitch=pitch,
                duration_ql=dur,
                velocity=vel,
                beat_offset=onset,
            ))

        # 5. Section-end breathing
        if is_section_end and notes:
            for idx in range(len(notes) - 1, -1, -1):
                if notes[idx].pitch >= 0:
                    n = notes[idx]
                    breath = 0.5
                    new_dur = max(0.25, n.duration_ql - breath)
                    notes[idx] = NoteEvent(
                        pitch=n.pitch,
                        duration_ql=new_dur,
                        velocity=max(40, n.velocity - 15),
                        beat_offset=n.beat_offset,
                    )
                    notes.append(NoteEvent(
                        pitch=-1,
                        duration_ql=breath,
                        velocity=0,
                        beat_offset=n.beat_offset + new_dur,
                    ))
                    break

        return notes

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_pitch_histograms(self) -> np.ndarray:
        """(n_clusters, 12) pitch-class distributions from stored measures."""
        hists = np.full((self._n_clusters, 12), 1.0 / 12, dtype=np.float64)
        for c in range(self._n_clusters):
            measures = self._clusterer.get_cluster_measures(c)
            counts = np.zeros(12, dtype=np.float64)
            for m in measures:
                for nd in m.notes:
                    counts[nd["pitch"] % 12] += 1.0
            if counts.sum() > 0:
                hists[c] = counts / counts.sum()
        return hists

    def _fallback(
        self, cluster_label: int,
        time_signature: Tuple[int, int],
        rng: np.random.RandomState,
    ) -> List[NoteEvent]:
        """Minimal fallback when no stored measures exist for the cluster."""
        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)
        pc_hist = self._pitch_hists[cluster_label]
        centre_pc = int(rng.choice(12, p=pc_hist))
        centre_pitch = 60 + centre_pc

        dur = min(1.0, bar_length_ql)
        notes = [NoteEvent(
            pitch=centre_pitch,
            duration_ql=dur,
            velocity=80,
            beat_offset=0.0,
        )]
        remaining = bar_length_ql - dur
        if remaining > 0.03:
            notes.append(NoteEvent(
                pitch=-1,
                duration_ql=remaining,
                velocity=0,
                beat_offset=dur,
            ))
        return notes
