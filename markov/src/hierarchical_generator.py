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
    voice: str = "melody"


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
        bass_histograms: np.ndarray | None = None,  # (n_clusters, 128) or None
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
        else:
            pc_centre = int(rng.choice(12, p=pc_hist))
            centre_octave = int(rng.choice([3, 3, 4, 4, 4, 5]))  # weighted toward middle
            centre_pitch = 12 * (centre_octave + 1) + pc_centre
        centre_pitch = max(40, min(84, centre_pitch))
        lo_pitch = max(28, centre_pitch - 10)  # ~1.5 octave window
        hi_pitch = min(96, centre_pitch + 10)

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

                if target_pitch is not None:
                    delta = int(target_pitch - candidate_pitch)
                    if abs(delta) > 2 and rng.random() < 0.45:
                        candidate_pitch += int(np.sign(delta) * min(abs(delta), 3))

                current_pitch = max(lo_pitch, min(hi_pitch, candidate_pitch))

            pitch = int(current_pitch)
            if target_pitch is not None and phrase_role == "CADENCE" and remaining - dur <= 0.03:
                pitch = max(lo_pitch, min(hi_pitch, int(target_pitch)))
                current_pitch = pitch

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

        self._apply_breathing(notes, is_section_end, bar_length_ql)
        self._apply_perturbation(notes, perturb, lo_pitch, hi_pitch, seed)

        return notes

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

        # Load style config first — needed by sampler
        self.config = _load_style_config(config_path)

        centroids = model.clusterer.centroids
        if centroids is None:
            raise ValueError("Model clusterer has no centroids — is it trained?")

        pitch_hists = getattr(model.clusterer, "pitch_histograms", None)
        step_hists = getattr(model.clusterer, "step_histograms", None)
        bass_hists = getattr(model.clusterer, "bass_histograms", None)
        self.note_sampler = ClusterNoteSampler(
            centroids, pitch_hists, step_hists, bass_hists,
            bass_config=self.config.get("bass", {}),
        )
        self._current_variation_profile: Optional[List] = None
        self._max_entropy = float(centroids[:, 7].max()) if centroids is not None else 3.5

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

        # 2. Build measure-level context and breathing points
        measure_context = self._build_measure_context(event_log)
        breathing = self._compute_breathing(measure_context)
        n = len(measure_context)

        # 3. Per-measure note generation
        all_notes: List[List[NoteEvent]] = []
        base_seed = seed if seed is not None else 0
        occurrence_count: Dict[str, int] = {}
        motif_memory: Dict[str, List[List[NoteEvent]]] = {}
        melodic_skeleton = self._build_melodic_skeleton(
            labels, measure_context, base_seed,
        )
        previous_melody_pitch: Optional[int] = None

        _select = _apply = None
        if enable_variation:
            from note_transform import select_transforms, apply_variation
            _select = select_transforms
            _apply = apply_variation

        for i, cluster_id in enumerate(labels):
            sl, bi, role, occurrence_id, section_len = measure_context[i]
            is_end = breathing[i]

            phrase_role = self._phrase_role(bi, section_len, role)
            target_pitch = melodic_skeleton.get(i)

            # Seed from section identity (FREE/FLAT use per-measure seed)
            if role in ("FREE", "FLAT"):
                measure_seed = _stable_hash(base_seed, "FREE", i)
            else:
                measure_seed = _stable_hash(base_seed, sl, bi)

            reused_motif = False
            if role not in ("FREE", "FLAT", "NEW") and sl in motif_memory and bi < len(motif_memory[sl]):
                notes = self._clone_notes(motif_memory[sl][bi])
                reused_motif = True
            else:
                notes = self.note_sampler.sample_measure(
                    cluster_label=cluster_id,
                    time_signature=time_signature,
                    seed=measure_seed,
                    perturb=0.0,
                    is_section_end=is_end,
                    phrase_role=phrase_role,
                    target_pitch=target_pitch,
                    previous_pitch=previous_melody_pitch,
                )
                if role == "NEW":
                    motif_memory.setdefault(sl, [])
                    while len(motif_memory[sl]) <= bi:
                        motif_memory[sl].append([])
                    motif_memory[sl][bi] = self._clone_notes(notes)

            # Controlled variation: non-NEW sections get transforms.
            # Strength s is data-driven: average cluster entropy scaled
            # by an information-theoretic progression k/(k+1).
            if (_select is not None and _apply is not None
                    and role not in ("FREE", "FLAT", "NEW", "REPEAT")):
                if bi == 0:
                    k = occurrence_count.get(sl, 0)
                    if k > 0:
                        sec_labels = labels[i:i + section_len]
                        variation_rng = np.random.RandomState(
                            _stable_hash(base_seed, sl, occurrence_id, "variation")
                        )
                        s = (self._section_entropy(sec_labels) / self._max_entropy
                             ) * (k / (k + VARIATION_PROGRESSION_DENOM))
                        self._current_variation_profile = _select(
                            self.model.clusterer.centroids,
                            sec_labels, s, rng=variation_rng,
                        ) if s > 0.02 else None
                    else:
                        self._current_variation_profile = None
                if self._current_variation_profile:
                    variation_rng = np.random.RandomState(
                        _stable_hash(base_seed, sl, occurrence_id, bi, "variation")
                    )
                    notes = _apply(notes, self._current_variation_profile, rng=variation_rng)
            if reused_motif and enable_variation and role in ("RETURN", "VARIANT"):
                k = occurrence_count.get(sl, 1)
                motif_cfg = self.config.get("motif_return", {})
                base_strength = float(motif_cfg.get("base_strength", 0.10))
                occurrence_growth = float(motif_cfg.get("occurrence_growth", 0.06))
                max_strength = float(motif_cfg.get("max_strength", 0.38))
                variation_rng = np.random.RandomState(
                    _stable_hash(base_seed, sl, occurrence_id, bi, "motif-return")
                )
                notes = self._vary_return_motif(
                    notes,
                    strength=min(max_strength, base_strength + occurrence_growth * k),
                    rng=variation_rng,
                    target_pitch=target_pitch,
                )
            if role not in ("FREE", "FLAT") and bi == 0:
                occurrence_count[sl] = occurrence_count.get(sl, 0) + 1

            all_notes.append(notes)
            previous_melody_pitch = self._last_melody_pitch(notes, previous_melody_pitch)

        # 3c. Add bass line (after transforms, so bass is unaffected)
        if self.note_sampler._bass_enabled:
            bl_ql = time_signature[0] * (4.0 / time_signature[1])
            for mi in range(len(all_notes)):
                cid = labels[mi]
                self.note_sampler._add_bass_note(
                    all_notes[mi], cid, bl_ql,
                    np.random.RandomState(_stable_hash(base_seed, "bass", mi)),
                )

        # 3d. Rendering: clamp overlaps on every measure (melody only — bass is
        # added after transforms so it never participates in clamping).
        self._clamp_overlaps(all_notes)
        self._ensure_final_bar_end(all_notes, time_signature)

        # 4. Write MIDI via mido (silence = absence of note events)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_midi(all_notes, output_path, tempo, time_signature)
        log.info("Wrote MIDI to %s (%d measures).", output_path, len(all_notes))

        # 5. Save structure visualization
        from structure_plotter import StructurePlotter
        plot_path = output_path.with_suffix('.png')
        StructurePlotter.plot(labels, event_log, self.model.n_clusters, plot_path)

        return labels

    def _section_entropy(self, sec_labels: List[int]) -> float:
        """Average entropy across the cluster labels in a section."""
        centroids = self.model.clusterer.centroids
        indices = [c for c in sec_labels if 0 <= c < len(centroids)]
        if not indices:
            return 0.0
        return float(centroids[indices, 7].mean())

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

    def _build_melodic_skeleton(
        self,
        labels: List[int],
        measure_context: List[Tuple[str, int, str, int, int]],
        base_seed: int,
    ) -> Dict[int, int]:
        """Create bar-level target pitches for phrase direction.

        This is intentionally lightweight: it does not force a melody, but it
        gives each section an opening point, a high point, and a cadence target.
        """
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

    def _vary_return_motif(
        self,
        notes: List[NoteEvent],
        strength: float,
        rng: np.random.RandomState,
        target_pitch: Optional[int],
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
        target_attraction = float(motif_cfg.get("target_attraction", 0.55))
        velocity_change_prob = float(motif_cfg.get("velocity_change_prob", 0.35))
        velocity_jitter_std = float(motif_cfg.get("velocity_jitter_std", 3.0))
        rhythm_change_scale = float(motif_cfg.get("rhythm_change_scale", 0.35))

        max_changes = max(1, int(round(len(melody_indices) * strength)))
        change_count = min(len(melody_indices), max_changes)
        selected = set(rng.choice(melody_indices, size=change_count, replace=False))

        for idx in melody_indices:
            note = varied[idx]
            pitch = note.pitch
            duration = note.duration_ql
            velocity = note.velocity

            if idx in selected:
                step = int(rng.choice([-2, -1, 1, 2]))
                if target_pitch is not None and rng.random() < target_attraction:
                    step = 1 if target_pitch > pitch else -1
                pitch = int(max(min_pitch, min(max_pitch, pitch + step)))

                # Keep rhythmic identity on returns; only very short gestures
                # may breathe a little so the bar is not mechanically copied.
                if duration <= 0.75 and rng.random() < strength * rhythm_change_scale:
                    candidates = [
                        d for d in _USABLE_DUR_VALUES
                        if 0.1 <= d <= 1.0 and abs(d - duration) <= 0.25
                    ]
                    if candidates:
                        duration = float(rng.choice(candidates))

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

    def _clamp_overlaps(self, all_notes: List[List[NoteEvent]]) -> None:
        """Rendering constraint: clamp per-measure note durations so
        consecutive notes do not overlap beyond the configured maximum."""
        max_ov = self.config.get("monophonic", {}).get("max_overlap", 0.15)
        _min_dur = 0.1
        for mi in range(len(all_notes)):
            sounding = sorted(
                [n for n in all_notes[mi] if n.pitch >= 0 and n.voice == "melody"],
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
                        voice=sounding[i].voice,
                    )
            others = [n for n in all_notes[mi] if n.pitch < 0 or n.voice != "melody"]
            all_notes[mi] = sorted(sounding + others,
                                   key=lambda n: n.beat_offset)

    @staticmethod
    def _ensure_final_bar_end(
        all_notes: List[List[NoteEvent]],
        time_signature: Tuple[int, int],
    ) -> None:
        """Guarantee the rendered MIDI reaches the requested final bar.

        This is a rendering invariant, not a stylistic hack: the structure
        layer asks for N measures, so the MIDI should end at exactly N bars
        unless the user explicitly requests a pickup or fade-out.
        """
        if not all_notes:
            return
        final_measure = all_notes[-1]
        sounding = [
            (idx, note) for idx, note in enumerate(final_measure)
            if note.pitch >= 0
        ]
        if not sounding:
            return

        bar_length_ql = time_signature[0] * (4.0 / time_signature[1])
        idx, note = max(
            sounding,
            key=lambda pair: pair[1].beat_offset + pair[1].duration_ql,
        )
        end = note.beat_offset + note.duration_ql
        if end >= bar_length_ql - 1e-6:
            return

        final_measure[idx] = NoteEvent(
            pitch=note.pitch,
            duration_ql=max(0.1, bar_length_ql - note.beat_offset),
            velocity=note.velocity,
            beat_offset=note.beat_offset,
            voice=note.voice,
        )

    @staticmethod
    def _build_measure_context(
        event_log: List[Dict[str, Any]],
    ) -> List[Tuple[str, int, str, int, int]]:
        """Flatten event_log into per-measure section context.

        Each tuple is ``(label, bar_index, role, occurrence_id, section_len)``.
        The occurrence id prevents repeated labels such as A ... A ... A from
        being merged when calculating phrase length or cadences.
        """
        ctx: List[Tuple[str, int, str, int, int]] = []
        occurrence_id = 0
        for event in event_log:
            length = event["length"]
            if event["kind"] == "SECTION":
                for bar_in_sec in range(length):
                    ctx.append((event["label"], bar_in_sec, event["role"], occurrence_id, length))
                occurrence_id += 1
            elif event["kind"] == "FREE":
                for _ in range(length):
                    ctx.append(("FREE", 0, "FREE", occurrence_id, length))
                occurrence_id += 1
            else:
                for _ in range(length):
                    ctx.append(("FLAT", 0, "FLAT", occurrence_id, length))
                occurrence_id += 1
        return ctx

    @staticmethod
    def _compute_breathing(
        measure_context: List[Tuple[str, int, str, int, int]],
    ) -> List[bool]:
        """Mark bars that should breathe at structural boundaries.

        A boundary is detected both from the local bar index and from the next
        occurrence id.  The second check matters in product use because short
        sections, truncated templates, or mixed FREE/SECTION timelines can
        make bar-index-only cadence detection miss a phrase ending.
        """
        n = len(measure_context)
        breathing = [False] * n
        for i in range(n):
            sl, bi, role, occurrence_id, section_len = measure_context[i]
            if role not in ("FREE", "FLAT"):
                if bi == section_len - 1:
                    breathing[i] = True
            elif role in ("FREE", "FLAT") and i + 1 < n:
                if measure_context[i + 1][2] not in ("FREE", "FLAT"):
                    breathing[i] = True
            if i + 1 < n:
                next_role = measure_context[i + 1][2]
                next_occurrence_id = measure_context[i + 1][3]
                if role not in ("FREE", "FLAT") and next_occurrence_id != occurrence_id:
                    breathing[i] = True
                elif role in ("FREE", "FLAT") and next_role not in ("FREE", "FLAT"):
                    breathing[i] = True
        if breathing:
            breathing[-1] = False
        return breathing

    # ------------------------------------------------------------------
    # Structure visualization
    # ------------------------------------------------------------------

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

        fs = self._select_template(grammar, template_file, rng)

        label_seen: set[str] = set()
        cycle = 0
        section_labels = fs.label_sequence
        n_gaps = len(section_labels) - 1

        while len(labels) < target_measures:
            free_lengths = grammar._sample_free_lengths(n_gaps, rng)

            for i, sec_label in enumerate(section_labels):
                if len(labels) >= target_measures:
                    break

                role, vary = self._assign_role(
                    sec_label, i, cycle, section_labels,
                    label_seen, variation_strength,
                )

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

        self._end_with_return(labels, event_log, fs, variation_strength, rng)
        return labels, event_log

    @staticmethod
    def _select_template(
        grammar: Any,
        template_file: Optional[Union[int, str]],
        rng: np.random.RandomState,
    ) -> Any:
        """Pick a template file, preferring multi-family and grid-aligned."""
        if template_file is not None:
            if isinstance(template_file, int):
                return grammar.files[template_file % len(grammar.files)]
            match = next(
                (f for f in grammar.files
                 if f.filename == template_file
                 or f.filename.endswith(template_file)
                 or Path(f.filename).stem == template_file),
                None,
            )
            if match is None:
                raise KeyError(f"No file matching '{template_file}'")
            return match

        multi = [f for f in grammar.files if f.n_families >= 2]
        by_grid = {0: [], 1: [], 2: []}
        candidates = multi if multi and rng.random() < 0.7 else grammar.files
        for f in candidates:
            lengths = [len(seq) for seq in f.prototypes.values()]
            aligned = sum(1 for L in lengths if L % 4 == 0)
            if aligned == len(lengths) and lengths:
                by_grid[0].append(f)
            elif aligned >= len(lengths) // 2:
                by_grid[1].append(f)
            else:
                by_grid[2].append(f)
        pool = (by_grid[0] * 7 + by_grid[1] * 2 + by_grid[2]) or grammar.files
        return pool[rng.randint(0, len(pool))]

    @staticmethod
    def _assign_role(
        sec_label: str,
        i: int,
        cycle: int,
        section_labels: List[str],
        label_seen: set[str],
        variation_strength: float,
    ) -> Tuple[str, bool]:
        """Determine the structural role of a section occurrence."""
        if cycle == 0 and i == 0:
            return "NEW", False
        if (i > 0 and sec_label == section_labels[i - 1]) \
                or (i == 0 and sec_label == section_labels[-1]):
            return "REPEAT", False
        if sec_label in label_seen:
            return "RETURN", variation_strength > 0
        return "NEW", False

    def _end_with_return(
        self,
        labels: List[int],
        event_log: List[Dict[str, Any]],
        fs: Any,
        variation_strength: float,
        rng: np.random.RandomState,
    ) -> None:
        """Replace trailing FREE blocks with a RETURN of the primary theme."""
        if not event_log or event_log[-1]["kind"] != "FREE":
            return

        grammar = self.grammar
        primary_label = fs.label_sequence[0]

        # Count and remove trailing FREE
        free_len = 0
        while event_log and event_log[-1]["kind"] == "FREE":
            free_len += event_log[-1]["length"]
            event_log.pop()
        del labels[-free_len:]

        # Fill with RETURN of the primary section
        content: List[int] = []
        while len(content) < free_len:
            content.extend(grammar.generate_section_content(
                primary_label, fs, vary=True,
                variation_strength=variation_strength,
                seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
            ))
        labels.extend(content[:free_len])
        event_log.append({
            "kind": "SECTION",
            "label": primary_label,
            "role": "RETURN",
            "length": free_len,
            "labels": content[:free_len],
        })

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

        # Collect all (tick, note_on/off) events and sort.  At identical ticks,
        # note_off must come before note_on to avoid accidental stuck overlaps.
        events: List[Tuple[int, str, int, int, int]] = []

        for measure_idx, nev_list in enumerate(measures):
            bar_base_ticks = measure_idx * bar_length_ql * tpb
            for nev in nev_list:
                if nev.pitch < 0:
                    continue
                start_tick = int(bar_base_ticks + nev.beat_offset * tpb)
                end_tick = int(start_tick + nev.duration_ql * tpb)
                if end_tick <= start_tick:
                    continue
                channel = 1 if nev.voice == "bass" else 0
                events.append((start_tick, 'on', nev.pitch, nev.velocity, channel))
                events.append((end_tick, 'off', nev.pitch, 0, channel))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

        prev_tick = 0
        for tick, etype, pitch, velocity, channel in events:
            delta = tick - prev_tick
            if etype == 'on':
                track.append(mido.Message('note_on', note=pitch,
                                          velocity=velocity, channel=channel,
                                          time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch,
                                          velocity=0, channel=channel,
                                          time=delta))
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
        "--no-bass",
        action="store_true",
        help="Disable bass line generation.",
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
    if args.no_bass:
        gen.note_sampler._bass_enabled = False

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
