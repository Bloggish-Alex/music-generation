#!/usr/bin/env python3
"""
Hierarchical Generator — three-tier music generation from a trained MusicModel.

Tier 1 (Section):  SectionGrammar → section label sequence + FREE placeholders
Tier 2 (State):    Prototypes fill SECTION blocks; PhraseGenerator fills FREE blocks
Tier 3 (Note):     ClusterNoteSampler generates pitches, durations, velocities
                   per measure from cluster centroids + pitch-class histograms

Usage::

    from music_model import MusicModel
    from hierarchical_generator import HierarchicalGenerator

    model = MusicModel.load("./models/corelli")
    gen = HierarchicalGenerator(model)
    gen.generate_midi("output.mid", target_measures=120, seed=42)

    # With user-specified starting bars:
    gen.generate_midi("output.mid", target_measures=120,
                      start_states=[2, 2, 2, 0, 0], seed=42)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from music_model import MusicModel
from phrase_generator import PhraseGenerator


def _stable_hash(*args: object) -> int:
    """Deterministic hash — same result across Python processes."""
    h = hashlib.md5()
    for a in args:
        h.update(str(a).encode())
    return int(h.hexdigest()[:8], 16)

log = logging.getLogger("hierarchical_generator")

# ---------------------------------------------------------------------------
# Duration categories (quarterLength), same as measure_clustering.py
# ---------------------------------------------------------------------------

_DUR_VALUES: List[float] = [
    4.0, 2.0, 1.0, 0.5, 0.25, 0.125,
    3.0, 1.5, 0.75, 2.0 / 3.0, 1.0 / 3.0,
]

_SHORT_DUR_INDICES = {3, 4, 5, 8, 9, 10}  # <= 0.5 ql

# ---------------------------------------------------------------------------
# Musical Style Parameters
# ---------------------------------------------------------------------------
# These are co-adapted through empirical tuning on the Corelli corpus.
# Changing any single value may require re-tuning others.  Each parameter
# is documented with its musical meaning and the expected range.
#
# To adapt to a new corpus: adjust here, regenerate, evaluate with the
# same metrics (16-bar block similarity, cadence gaps, polyphony).

# ---- breathing / cadence ----

CADENCE_GAP = 0.5          # beats of silence at section boundaries (4/4 eighth-note)

# ---- melody / pitch walk ----

# Step-size distribution: now derived per-cluster from training data.
# See clusterer.step_histograms — computed during MusicModel.fit().
STEP_UPWARD_BIAS = 0.55          # probability of ascending vs descending
REGISTER_WINDOW = 10             # semi-octave range per measure (±10 from centre)
REGISTER_LO = 28                 # MIDI E2 — absolute floor
REGISTER_HI = 96                 # MIDI C7 — absolute ceiling
REGISTER_CENTRE_LO = 40          # MIDI E2 — centre pitch minimum
REGISTER_CENTRE_HI = 84          # MIDI C6 — centre pitch maximum
OCTAVE_WEIGHTS = [3, 3, 4, 4, 4, 5]  # weight toward middle register (octave 4)
PC_REJECT_THRESHOLD = 4.0        # if histogram[pc] * this < random, reduce leap
PC_REJECT_SCALE = 0.5            # how much to reduce rejected leap

# ---- rhythm / note generation ----

NOTE_DENSITY_SCALE = 2.0          # density → raw note count multiplier
ENTROPY_JITTER_SCALE = 1.5        # entropy → standard deviation of note count
MIN_NOTES_PER_MEASURE = 2
MAX_NOTES_PER_MEASURE = 24        # ~6 notes/beat in 4/4
MIN_DURATION = 0.25               # sixteenth note — shortest allowed
MIN_REMAINING = 0.03              # stop generating when less than this remains
DURATION_COUNT_WEIGHTS = [2, 2, 3, 3, 4]  # 2–4 preferred durations per measure
REST_PROBABILITY_SCALE = 0.5      # multiplier on silence_ratio for actual rests
MAX_REST_PROB = 0.6               # cap on rest probability per note

# ---- velocity / dynamics ----

VELOCITY_PEAK_MEAN, VELOCITY_PEAK_STD = 100, 8
VELOCITY_TROUGH_MEAN, VELOCITY_TROUGH_STD = 55, 10
VELOCITY_PEAK_FRAC_MIN, VELOCITY_PEAK_FRAC_MAX = 0.25, 0.55
VELOCITY_JITTER = 5               # Gaussian std added to velocity arc

# ---- offbeat / swing ----

OFFBEAT_SCALE = 0.3               # offbeat_ratio → probability of swing placement
SWING_MIN, SWING_MAX = 0.02, 0.12 # beat_offset jitter range (quarterLength)

# ---- variation / transforms ----

MAX_ENTROPY = 3.5                 # normalizer for centroid entropy (theoretical max)
MIN_VARIATION_STRENGTH = 0.02     # strength below this → skip transforms
VARIATION_PROGRESSION_DENOM = 3   # k/(k+N) curve steepness

# ---- section structure / grid ----

GRID_SIZE = 4                     # snap to this many bars
MAX_GRID_PAD = 3                  # max bars padded to reach grid boundary
MULTI_FAMILY_PREFERENCE = 0.7     # probability of using multi-family template
GRID_WEIGHTS = [7, 2, 1]          # grid-aligned, mostly-grid, irregular template ratio

# ---- MIDI output ----

TICKS_PER_BEAT = 480

# ---------------------------------------------------------------------------
# NoteEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteEvent:
    """One note or rest within a measure."""

    pitch: int           # MIDI pitch 0–127, or -1 for rest
    duration_ql: float   # quarterLength
    velocity: int        # 1–127, 0 for rests
    beat_offset: float   # position within the bar (0.0 = downbeat)


# ---------------------------------------------------------------------------
# ClusterNoteSampler
# ---------------------------------------------------------------------------


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
            # Fallback: uniform + slight bias toward smaller steps
            self._step_hists = np.array(
                [[0.20, 0.18, 0.18, 0.15, 0.12, 0.10, 0.07]] * self._n_clusters,
                dtype=np.float64,
            )

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
        entropy = float(centroid[7])

        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)

        # --- rest probability ---
        rest_prob = float(np.clip(silence_ratio, 0.0, 0.6))

        # --- note count ---
        # density is roughly onsets per quarter note; cap at 6 notes/quarter
        raw_count = max(2, int(note_density * bar_length_ql * 2.0))
        raw_count += int(rng.normal(0, entropy * 1.5))
        target_count = max(2, min(raw_count, 24))

        # --- pick 2-4 "preferred" durations for rhythmic coherence ---
        pref_durs = self._pick_preferred_durations(
            mean_dur, dur_var, short_ratio, rng,
        )

        # --- choose a register for this measure ---
        # Pick a centre pitch from the pitch histogram, in octave 3 or 4
        pc_centre = int(rng.choice(12, p=pc_hist))
        centre_octave = int(rng.choice([3, 3, 4, 4, 4, 5]))  # weighted toward middle
        centre_pitch = 12 * (centre_octave + 1) + pc_centre
        centre_pitch = max(40, min(84, centre_pitch))
        lo_pitch = max(28, centre_pitch - 10)  # ~1.5 octave window
        hi_pitch = min(96, centre_pitch + 10)

        # Start pitch near centre
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
                rest_dur = float(rng.choice([d for d in pref_durs if d <= remaining + 0.01]))
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

                current_pitch = max(lo_pitch, min(hi_pitch, candidate_pitch))

            pitch = int(current_pitch)

            # --- velocity (phrase arc) ---
            progress = beat / bar_length_ql if bar_length_ql > 0 else 0
            if progress < vel_peak_frac:
                # Build toward peak
                frac = progress / vel_peak_frac
                vel = int(vel_trough + (vel_peak - vel_trough) * frac)
            else:
                # Decay after peak
                frac = (progress - vel_peak_frac) / (1.0 - vel_peak_frac)
                vel = int(vel_peak + (vel_trough - vel_peak) * frac)
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

        # --- section-end breathing space ---
        # Force the last note to end *gap* beats before the bar boundary.
        # The natural silence between note_off and the next bar's note_on
        # creates the audible gap.  No rest is needed with mido output.
        if is_section_end and notes:
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
                        )
                    else:
                        # Note starts too late — remove it and keep searching
                        del notes[idx]
                        continue
                    del notes[idx + 1:]
                    break

        # --- post-generation perturbation (for RETURN / VARIANT) ---
        if perturb > 0.0 and notes:
            perturb_rng = np.random.RandomState(
                None if seed is None else seed + 999999,
            )
            for idx in range(len(notes)):
                n = notes[idx]
                if n.pitch < 0:
                    continue  # don't perturb rests

                # Pitch: shift by ±1-2 semitones with probability = perturb
                if perturb_rng.random() < perturb:
                    shift = int(perturb_rng.choice([-2, -1, 1, 2]))
                    new_pitch = max(lo_pitch, min(hi_pitch, n.pitch + shift))
                    # Duration: sometimes adjust to a neighbor
                    dur = n.duration_ql
                    if perturb_rng.random() < perturb * 0.5:
                        neighbors = [
                            d for d in _USABLE_DUR_VALUES
                            if abs(d - dur) < dur * 0.6 and d != dur
                        ]
                        if neighbors:
                            dur = float(perturb_rng.choice(neighbors))
                    # Velocity: small jitter
                    vel = n.velocity
                    if vel > 0:
                        vel = max(35, min(127, int(vel + perturb_rng.normal(0, 5))))

                    notes[idx] = NoteEvent(
                        pitch=new_pitch,
                        duration_ql=dur,
                        velocity=vel,
                        beat_offset=n.beat_offset,
                    )

        return notes

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


# ---------------------------------------------------------------------------
# HierarchicalGenerator
# ---------------------------------------------------------------------------


def _load_style_config(path: str | Path | None) -> Dict[str, Any]:
    """Load style parameters from a YAML file, falling back to defaults."""
    import yaml
    default_path = Path(__file__).resolve().parent.parent / "config" / "style_defaults.yaml"
    if path:
        with open(path) as f:
            return yaml.safe_load(f)
    if default_path.exists():
        with open(default_path) as f:
            return yaml.safe_load(f)
    return {}


class HierarchicalGenerator:
    """Three-tier music generator.

    Combines SectionGrammar (macro form), PhraseGenerator (FREE block
    states), and ClusterNoteSampler (per-measure notes) into a single
    end-to-end pipeline.

    Args:
        model: Trained MusicModel.
        config_path: Optional path to a YAML style config file.
            See ``config/style_defaults.yaml`` for available parameters.
            Falls back to the defaults if not provided.
    """

    def __init__(self, model: MusicModel, config_path: str | Path | None = None) -> None:
        self.model = model
        self.phrase_gen = PhraseGenerator(model)

        centroids = model.clusterer.centroids
        if centroids is None:
            raise ValueError("Model clusterer has no centroids — is it trained?")

        pitch_hists = getattr(model.clusterer, "pitch_histograms", None)
        step_hists = getattr(model.clusterer, "step_histograms", None)
        self.note_sampler = ClusterNoteSampler(centroids, pitch_hists, step_hists)
        self._current_variation_profile: Optional[List] = None
        self._max_entropy = float(centroids[:, 7].max()) if centroids is not None else 3.5

        # Load style config (defaults or from file)
        self.config = _load_style_config(config_path)

    @property
    def grammar(self):
        return self.model.grammar

    # ------------------------------------------------------------------
    # Timeline generation
    # ------------------------------------------------------------------

    def generate_timeline(
        self,
        target_measures: int,
        start_states: Optional[List[int]] = None,
        template_file: Optional[Union[int, str]] = None,
        variation_strength: float = 0.3,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Generate a full cluster-label timeline.

        Args:
            target_measures: Desired total length in measures.
            start_states: Optional cluster labels to prepend.
            template_file: Index or filename stem for section template.
            variation_strength: RETURN/VARIANT deviation from prototype.
            seed: Random seed.

        Returns:
            labels: Full cluster label sequence.
            event_log: Description of each segment (SECTION, FREE, USER_START).
        """
        rng = np.random.RandomState(seed)
        labels: List[int] = []
        event_log: List[Dict[str, Any]] = []

        # Phase 1: User-specified start bars
        if start_states:
            labels.extend(start_states)
            event_log.append({
                "kind": "USER_START",
                "length": len(start_states),
                "labels": list(start_states),
            })

        remaining = target_measures - len(labels)
        if remaining <= 0:
            return labels[:target_measures], event_log

        # Phase 2: Section grammar + PhraseGenerator
        if self.grammar is not None and self.grammar.files:
            labels, event_log = self._generate_with_grammar(
                target_measures, labels, event_log,
                template_file, variation_strength, rng,
            )
        else:
            log.info("No section grammar — using flat phrase generation.")
            extra = self.phrase_gen.generate(
                remaining, seed=int(rng.randint(0, 2 ** 31 - 1)),
            )
            labels.extend(extra)
            event_log.append({
                "kind": "FLAT",
                "length": len(extra),
                "labels": extra,
            })

        if len(labels) > target_measures:
            labels = labels[:target_measures]

        log.info(
            "Timeline: %d measures, %d events.",
            len(labels), len(event_log),
        )
        return labels, event_log

    # ------------------------------------------------------------------
    # MIDI generation
    # ------------------------------------------------------------------

    def generate_midi(
        self,
        output_path: Union[str, Path],
        target_measures: int = 120,
        start_states: Optional[List[int]] = None,
        template_file: Optional[Union[int, str]] = None,
        variation_strength: float = 0.3,
        time_signature: Tuple[int, int] = (4, 4),
        tempo: int = 120,
        seed: Optional[int] = None,
        enable_variation: bool = True,
    ) -> List[int]:
        """Generate a full MIDI file.

        Args:
            enable_variation: If True (default), apply controlled transforms
                to non-NEW section occurrences.  Set to False for
                exact repeats only.

        Returns the cluster label timeline used.
        """
        # 1. Timeline
        labels, event_log = self.generate_timeline(
            target_measures=target_measures,
            start_states=start_states,
            template_file=template_file,
            variation_strength=variation_strength,
            seed=seed,
        )

        # 2. Build measure-level context:
        #    (section_label, bar_in_section, role) for each measure
        measure_context: List[Tuple[str, int, str]] = []
        for event in event_log:
            length = event["length"]
            if event["kind"] == "SECTION":
                for bar_in_sec in range(length):
                    measure_context.append(
                        (event["label"], bar_in_sec, event["role"]))
            elif event["kind"] == "FREE":
                for _ in range(length):
                    measure_context.append(("FREE", 0, "FREE"))
            else:
                for _ in range(length):
                    measure_context.append(("FLAT", 0, "FLAT"))

        # 2b. Pre-compute breathing points:
        #     a bar breathes if it is the last bar of a SECTION, or a
        #     FREE/FLAT bar immediately before a SECTION start.
        #     The final bar of the piece never breathes.
        n = len(measure_context)
        breathing = [False] * n
        for i in range(n):
            sl, bi, role = measure_context[i]
            if role not in ("FREE", "FLAT") and bi > 0:
                # Find section length for this label
                sec_len = sum(1 for j in range(i - bi, n)
                              if measure_context[j][0] == sl)
                if bi == sec_len - 1:  # last bar of this section
                    breathing[i] = True
            elif role in ("FREE", "FLAT") and i + 1 < n:
                next_role = measure_context[i + 1][2]
                if next_role not in ("FREE", "FLAT"):
                    breathing[i] = True
        if breathing:
            breathing[-1] = False  # end of piece never breathes

        # 3. Per-measure note generation
        all_notes: List[List[NoteEvent]] = []
        base_seed = seed if seed is not None else 0
        occurrence_count: Dict[str, int] = {}

        _select = _apply = None
        if enable_variation:
            from note_transform import select_transforms, apply_variation
            _select = select_transforms
            _apply = apply_variation

        for i, cluster_id in enumerate(labels):
            sl, bi, role = measure_context[i]
            is_end = breathing[i]

            # Seed from section identity (FREE/FLAT use per-measure seed)
            if role in ("FREE", "FLAT"):
                measure_seed = _stable_hash(base_seed, "FREE", i)
            else:
                measure_seed = _stable_hash(base_seed, sl, bi)

            notes = self.note_sampler.sample_measure(
                cluster_label=cluster_id,
                time_signature=time_signature,
                seed=measure_seed,
                perturb=0.0,
                is_section_end=is_end,
            )

            # Controlled variation: non-NEW sections get transforms.
            # Strength s is data-driven: average cluster entropy scaled
            # by an information-theoretic progression k/(k+1).
            if (_select is not None and _apply is not None
                    and role not in ("FREE", "FLAT", "NEW")):
                if bi == 0:
                    k = occurrence_count.get(sl, 0)
                    if k > 0:
                        sec_len = sum(1 for j in range(i, n)
                                      if measure_context[j][0] == sl)
                        sec_labels = labels[i:i + sec_len]
                        s = (self._section_entropy(sec_labels) / self._max_entropy
                             ) * (k / (k + VARIATION_PROGRESSION_DENOM))
                        self._current_variation_profile = _select(
                            self.model.clusterer.centroids,
                            sec_labels, s,
                        ) if s > 0.02 else None
                    else:
                        self._current_variation_profile = None
                if self._current_variation_profile:
                    notes = _apply(notes, self._current_variation_profile)
            if role not in ("FREE", "FLAT") and bi == 0:
                occurrence_count[sl] = occurrence_count.get(sl, 0) + 1

            all_notes.append(notes)

        # 3c. Rendering: clamp overlaps on every measure
        self._clamp_overlaps(all_notes)

        # 4. Write MIDI via mido (silence = absence of note events)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_midi(all_notes, output_path, tempo, time_signature)
        log.info("Wrote MIDI to %s (%d measures).", output_path, len(all_notes))

        # 5. Save structure visualization
        plot_path = output_path.with_suffix('.png')
        self.plot_structure(labels, event_log, plot_path)

        return labels

    def _section_entropy(self, sec_labels: List[int]) -> float:
        """Average entropy across the cluster labels in a section."""
        centroids = self.model.clusterer.centroids
        indices = [c for c in sec_labels if 0 <= c < len(centroids)]
        if not indices:
            return 0.0
        return float(centroids[indices, 7].mean())

    def _clamp_overlaps(self, all_notes: List[List[NoteEvent]]) -> None:
        """Rendering constraint: clamp per-measure note durations so
        consecutive notes do not overlap beyond the configured maximum."""
        max_ov = self.config.get("monophonic", {}).get("max_overlap", 0.15)
        _min_dur = 0.1
        for mi in range(len(all_notes)):
            sounding = sorted(
                [n for n in all_notes[mi] if n.pitch >= 0],
                key=lambda n: n.beat_offset,
            )
            for i in range(len(sounding) - 1):
                if sounding[i + 1].beat_offset - sounding[i].beat_offset < 0.02:
                    continue
                cur_end = sounding[i].beat_offset + sounding[i].duration_ql
                allowed = sounding[i + 1].beat_offset + max_ov
                if cur_end > allowed:
                    sounding[i] = NoteEvent(
                        pitch=sounding[i].pitch,
                        duration_ql=max(_min_dur,
                                        allowed - sounding[i].beat_offset),
                        velocity=sounding[i].velocity,
                        beat_offset=sounding[i].beat_offset,
                    )
            rests = [n for n in all_notes[mi] if n.pitch < 0]
            all_notes[mi] = sorted(sounding + rests,
                                   key=lambda n: n.beat_offset)

    # ------------------------------------------------------------------
    # Structure visualization
    # ------------------------------------------------------------------

    def plot_structure(
        self,
        labels: List[int],
        event_log: List[Dict[str, Any]],
        save_path: Union[str, Path],
    ) -> None:
        """Render the generated structure as a color-coded form diagram.

        Each measure is a colored square; section blocks are grouped and
        labeled (A, B, FREE).  Cadence boundaries are marked with gaps.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch

        n_clusters = self.model.n_clusters
        n_measures = len(labels)
        cmap = plt.get_cmap("tab10")
        cluster_colors = [cmap(i % 10) for i in range(n_clusters)]

        # Layout: rows of measures, 16 per row
        measures_per_row = 16
        n_rows = (n_measures + measures_per_row - 1) // measures_per_row
        square_w = 0.35
        square_h = 0.30
        gap = 0.03
        row_gap = 0.08
        section_label_offset = 0.12

        fig_w = measures_per_row * (square_w + gap) + 3.0
        fig_h = n_rows * (square_h + row_gap + section_label_offset) + 1.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(0, measures_per_row * (square_w + gap) + 2.5)
        ax.set_ylim(-n_rows * (square_h + row_gap + section_label_offset) - 0.5, 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Build measure → section mapping
        measure_section: List[Optional[str]] = [None] * n_measures
        measure_role: List[Optional[str]] = [None] * n_measures
        pos = 0
        for event in event_log:
            length = event["length"]
            if event["kind"] == "SECTION":
                for k in range(length):
                    if pos + k < n_measures:
                        measure_section[pos + k] = event["label"]
                        measure_role[pos + k] = event.get("role", "")
            elif event["kind"] == "FREE":
                for k in range(length):
                    if pos + k < n_measures:
                        measure_section[pos + k] = "FREE"
            pos += length

        # Draw measures row by row
        for row in range(n_rows):
            y_base = -row * (square_h + row_gap + section_label_offset)
            for col in range(measures_per_row):
                idx = row * measures_per_row + col
                if idx >= n_measures:
                    break
                x = col * (square_w + gap)
                y = y_base
                c = labels[idx]
                color = cluster_colors[c]
                rect = Rectangle(
                    (x, y), square_w, square_h,
                    facecolor=color, edgecolor="white", linewidth=0.3,
                )
                ax.add_patch(rect)

                # Section start marker
                if idx == 0 or (idx > 0 and measure_section[idx] != measure_section[idx - 1]):
                    ax.plot(
                        [x - gap, x - gap],
                        [y - 0.05, y + square_h + 0.05],
                        color="#333333", linewidth=1.0,
                    )

            # Section labels above each row
            prev_sec = None
            for col in range(measures_per_row):
                idx = row * measures_per_row + col
                if idx >= n_measures:
                    break
                sec = measure_section[idx]
                if sec is not None and sec != prev_sec:
                    x_start = col * (square_w + gap)
                    # Span to end of this contiguous block
                    end_col = col
                    while (end_col + 1 < measures_per_row and
                           row * measures_per_row + end_col + 1 < n_measures and
                           measure_section[row * measures_per_row + end_col + 1] == sec):
                        end_col += 1
                    x_end = end_col * (square_w + gap) + square_w
                    x_mid = (x_start + x_end) / 2
                    label_text = f"{sec}"
                    if measure_role[idx] and measure_role[idx] not in ("NEW",):
                        label_text = f"{sec}'" if measure_role[idx] == "RETURN" else sec
                    ax.text(
                        x_mid, y_base + square_h + 0.06,
                        label_text, ha="center", va="bottom",
                        fontsize=7, color="#333333", fontweight="bold",
                    )
                    prev_sec = sec

        # 4-bar and 8-bar vertical grid lines
        for col in [4, 8, 12]:
            x = col * (square_w + gap) - gap
            ax.plot([x, x], [0.2, -n_rows * (square_h + row_gap + section_label_offset)],
                    color="#aaaaaa", linewidth=0.5, linestyle="--", alpha=0.5)
        for col in [8]:
            x = col * (square_w + gap) - gap
            ax.plot([x, x], [0.2, -n_rows * (square_h + row_gap + section_label_offset)],
                    color="#666666", linewidth=0.8, linestyle="--")

        # Bar number labels (every 4 bars)
        for row in range(n_rows):
            for col in range(0, measures_per_row, 4):
                idx = row * measures_per_row + col
                if idx < n_measures:
                    x = col * (square_w + gap) + square_w / 2
                    y = -row * (square_h + row_gap + section_label_offset)
                    ax.text(
                        x, y - 0.06, str(idx + 1),
                        ha="center", va="top", fontsize=5, color="#999999",
                    )

        # Legend
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=cluster_colors[c], edgecolor="white")
            for c in range(n_clusters)
        ]
        legend_labels = [f"Cluster {c}" for c in range(n_clusters)]
        ax.legend(
            legend_handles, legend_labels,
            loc="upper right", fontsize=6, ncol=min(4, n_clusters),
            markerscale=0.6, framealpha=0.8,
        )

        # Title
        n_sections = sum(1 for e in event_log if e["kind"] == "SECTION")
        n_free = sum(1 for e in event_log if e["kind"] == "FREE")
        ax.set_title(
            f"Generated Structure — {n_measures} measures, "
            f"{n_sections} sections, {n_free} FREE blocks",
            fontsize=11, pad=6,
        )

        fig.tight_layout(pad=0.5)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Internal: grammar-aware timeline
    # ------------------------------------------------------------------

    def _generate_with_grammar(
        self,
        target_measures: int,
        labels: List[int],
        event_log: List[Dict[str, Any]],
        template_file: Optional[Union[int, str]],
        variation_strength: float,
        rng: np.random.RandomState,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        grammar = self.grammar

        if template_file is None:
            # Prefer multi-family templates with grid-aligned section lengths
            multi = [f for f in grammar.files if f.n_families >= 2]
            # Also prefer templates whose sections are multiples of 4 bars
            by_grid = {0: [], 1: [], 2: []}  # 0=grid, 1=mostly-grid, 2=irregular
            for f in (multi if multi and rng.random() < 0.7 else grammar.files):
                lengths = [len(seq) for seq in f.prototypes.values()]
                aligned = sum(1 for L in lengths if L % 4 == 0)
                if aligned == len(lengths) and lengths:
                    by_grid[0].append(f)
                elif aligned >= len(lengths) // 2:
                    by_grid[1].append(f)
                else:
                    by_grid[2].append(f)
            # Weighted draw: 70% grid-aligned, 20% mostly-grid, 10% irregular
            pool = (by_grid[0] * 7 + by_grid[1] * 2 + by_grid[2]) or grammar.files
            fs = pool[rng.randint(0, len(pool))]
        elif isinstance(template_file, int):
            fs = grammar.files[template_file % len(grammar.files)]
        else:
            match = next(
                (f for f in grammar.files
                 if f.filename == template_file
                 or f.filename.endswith(template_file)
                 or Path(f.filename).stem == template_file),
                None,
            )
            if match is None:
                raise KeyError(f"No file matching '{template_file}'")
            fs = match

        label_seen: set[str] = set()
        cycle = 0
        section_labels = fs.label_sequence
        n_gaps = len(section_labels) - 1

        while len(labels) < target_measures:
            free_lengths = grammar._sample_free_lengths(n_gaps, rng)

            for i, sec_label in enumerate(section_labels):
                if len(labels) >= target_measures:
                    break

                if cycle == 0 and i == 0:
                    role = "NEW"
                    vary = False
                elif (
                    (i > 0 and sec_label == section_labels[i - 1])
                    or (i == 0 and sec_label == section_labels[-1])
                ):
                    role = "REPEAT"
                    vary = False  # exact repeat — same cluster labels, same notes
                elif sec_label in label_seen:
                    role = "RETURN"
                    vary = variation_strength > 0
                else:
                    role = "NEW"
                    vary = False

                content = grammar.generate_section_content(
                    sec_label, fs, vary=vary,
                    variation_strength=variation_strength,
                    seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
                )

                labels.extend(content)
                event_log.append({
                    "kind": "SECTION",
                    "label": sec_label,
                    "role": role,
                    "cycle": cycle,
                    "length": len(content),
                    "labels": content,
                })

                if i < len(section_labels) - 1 and len(labels) < target_measures:
                    free_len = free_lengths[i] if i < len(free_lengths) else 4
                    free_labels = self.phrase_gen.generate(
                        free_len,
                        seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
                    )
                    labels.extend(free_labels)
                    event_log.append({
                        "kind": "FREE",
                        "length": free_len,
                        "labels": free_labels,
                    })

                    # Snap to next 4-bar grid, max 3 bars padding
                    _GRID = 4
                    remainder = len(labels) % _GRID
                    if 0 < remainder <= 3:
                        pad = _GRID - remainder
                        pad_labels = self.phrase_gen.generate(
                            pad,
                            seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
                        )
                        labels.extend(pad_labels)
                        event_log.append({
                            "kind": "FREE",
                            "length": pad,
                            "labels": pad_labels,
                            "grid_pad": True,
                        })

                label_seen.add(sec_label)

            cycle += 1

        return labels, event_log

    # ------------------------------------------------------------------
    # Internal: MIDI output via mido
    # ------------------------------------------------------------------

    @staticmethod
    def _write_midi(
        measures: List[List[NoteEvent]],
        output_path: Path,
        tempo: int,
        time_signature: Tuple[int, int],
    ) -> None:
        """Write MIDI via mido — direct tick-level control.

        Silence is simply the absence of note events.  No rests needed.
        """
        import mido

        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)
        tpb = 480
        us_per_beat = int(60_000_000 / tempo)

        mid = mido.MidiFile(ticks_per_beat=tpb)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
        track.append(mido.MetaMessage(
            'time_signature', numerator=ts_num, denominator=ts_den,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
        ))

        # Collect all (tick, note_on/off) events and sort
        events: List[Tuple[int, str, int, int]] = []

        for measure_idx, nev_list in enumerate(measures):
            bar_base_ticks = measure_idx * bar_length_ql * tpb
            for nev in nev_list:
                if nev.pitch < 0:
                    continue
                start_tick = int(bar_base_ticks + nev.beat_offset * tpb)
                end_tick = int(start_tick + nev.duration_ql * tpb)
                events.append((start_tick, 'on', nev.pitch, nev.velocity))
                events.append((end_tick, 'off', nev.pitch, 0))

        events.sort(key=lambda e: e[0])

        prev_tick = 0
        for tick, etype, pitch, velocity in events:
            delta = tick - prev_tick
            if etype == 'on':
                track.append(mido.Message('note_on', note=pitch,
                                          velocity=velocity, time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch,
                                          velocity=0, time=delta))
            prev_tick = tick

        mid.save(str(output_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hierarchical Generator — three-tier music generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="./models/corelli",
        help="Path to trained MusicModel directory.",
    )
    parser.add_argument(
        "--output", "-o",
        default="generated/hierarchical_output.mid",
        help="Output MIDI path.",
    )
    parser.add_argument(
        "--target-measures", "-n",
        type=int,
        default=120,
        help="Target number of measures.",
    )
    parser.add_argument(
        "--start-states",
        default=None,
        help="Comma-separated cluster labels for the first N bars "
        "(e.g. '2,2,2,0,0').",
    )
    parser.add_argument(
        "--template", "-t",
        default=None,
        help="Section template: file index or name stem.",
    )
    parser.add_argument(
        "--variation", "-v",
        type=float,
        default=0.3,
        help="Variation strength for RETURN sections (0–1).",
    )
    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="Time signature (e.g. '4/4', '3/4').",
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=120,
        help="Tempo in BPM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--no-variation",
        action="store_true",
        help="Disable controlled variation transforms (exact repeats only).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML style config (default: config/style_defaults.yaml).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading model from %s ...", args.model_dir)
    model = MusicModel.load(args.model_dir)
    print()
    print(model.summary())

    ts_parts = args.time_signature.split("/")
    time_sig = (int(ts_parts[0]), int(ts_parts[1]))

    start_states = None
    if args.start_states:
        start_states = [
            int(x.strip()) for x in args.start_states.split(",") if x.strip()
        ]

    gen = HierarchicalGenerator(model, config_path=args.config)

    log.info(
        "Generating %d measures (template=%s, variation=%.2f) ...",
        args.target_measures, args.template or "random", args.variation,
    )
    labels = gen.generate_midi(
        output_path=args.output,
        target_measures=args.target_measures,
        start_states=start_states,
        template_file=args.template,
        variation_strength=args.variation,
        time_signature=time_sig,
        tempo=args.tempo,
        seed=args.seed,
        enable_variation=not args.no_variation,
    )

    print(f"\nGenerated {len(labels)} measures → {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
