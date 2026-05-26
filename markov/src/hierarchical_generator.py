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
                # Step size: prefer small steps (≤ 2 semitones), allow small leaps
                step_weights = [0.35, 0.25, 0.15, 0.15, 0.05, 0.03, 0.02]  # 0..6+ semitones
                step_dir = 1 if rng.random() < 0.55 else -1  # slight upward bias
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

        # --- section-end cadence (grid-aligned) ---
        if is_section_end and notes:
            bar_end = bar_length_ql
            # Pick the last sounding note that starts early enough to hold
            # ≥1.5 beats without crossing the bar boundary.  Fall back to
            # any sounding note if none qualify.
            cadence_idx = -1
            for idx in range(len(notes) - 1, -1, -1):
                n = notes[idx]
                if n.pitch >= 0:
                    if n.beat_offset <= bar_end - 1.5:
                        cadence_idx = idx
                        break
                    if cadence_idx < 0:
                        cadence_idx = idx  # fallback to any sounding note
            if cadence_idx >= 0:
                del notes[cadence_idx + 1:]
                n = notes[cadence_idx]
                # Duration always ends exactly at the bar boundary
                cadence_dur = bar_end - n.beat_offset
                notes[cadence_idx] = NoteEvent(
                    pitch=n.pitch,
                    duration_ql=cadence_dur,
                    velocity=max(20, int(n.velocity * 0.4)),
                    beat_offset=n.beat_offset,
                )
                notes.append(NoteEvent(
                    pitch=-1,
                    duration_ql=1.5,
                    velocity=0,
                    beat_offset=bar_end,
                ))

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


class HierarchicalGenerator:
    """Three-tier music generator.

    Combines SectionGrammar (macro form), PhraseGenerator (FREE block
    states), and ClusterNoteSampler (per-measure notes) into a single
    end-to-end pipeline.
    """

    def __init__(self, model: MusicModel) -> None:
        self.model = model
        self.phrase_gen = PhraseGenerator(model)

        centroids = model.clusterer.centroids
        if centroids is None:
            raise ValueError("Model clusterer has no centroids — is it trained?")

        pitch_hists = getattr(model.clusterer, "pitch_histograms", None)
        self.note_sampler = ClusterNoteSampler(centroids, pitch_hists)

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
    ) -> List[int]:
        """Generate a full MIDI file.

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

        # 2. Build section-context map: measure_idx → (section_label, bar_in_section, role)
        measure_context: List[Tuple[str, int, str]] = []
        measure_idx = 0
        for event in event_log:
            length = event["length"]
            if event["kind"] == "SECTION":
                label = event["label"]
                role = event["role"]
                for bar_in_sec in range(length):
                    measure_context.append((label, bar_in_sec, role))
            elif event["kind"] == "FREE":
                for bar_in_free in range(length):
                    measure_context.append(("FREE", measure_idx, "FREE"))
            else:
                for k in range(length):
                    measure_context.append(("FLAT", measure_idx + k, "FLAT"))
            measure_idx += length

        # 3. Per-measure notes, seeded by section identity
        all_notes: List[NoteEvent] = []
        base_seed = seed if seed is not None else 0

        for i, cluster_id in enumerate(labels):
            if i < len(measure_context):
                sec_label, bar_in_sec, role = measure_context[i]
            else:
                sec_label, bar_in_sec, role = "FLAT", i, "FLAT"

            # Derive seed from section identity — same seed for all roles
            # so RETURN/VARIANT preserve the original's contour.
            if sec_label in ("FREE", "FLAT"):
                measure_seed = _stable_hash(base_seed, sec_label, i)
                _perturb = 0.0
            else:
                measure_seed = _stable_hash(base_seed, sec_label, bar_in_sec)
                if role in ("NEW", "REPEAT"):
                    _perturb = 0.0
                elif role == "RETURN":
                    _perturb = 0.25
                elif role == "VARIANT":
                    _perturb = 0.5
                else:
                    _perturb = 0.0

            # Cadence at every 16th bar, not at section boundaries
            is_grid_cadence = ((i + 1) % 16 == 0)

            notes = self.note_sampler.sample_measure(
                cluster_label=cluster_id,
                time_signature=time_signature,
                seed=measure_seed,
                perturb=_perturb,
                is_section_end=is_grid_cadence,
            )
            all_notes.extend(notes)

        # 4. Write MIDI
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_midi(all_notes, output_path, tempo, time_signature)
        log.info("Wrote MIDI to %s (%d notes).", output_path, len(all_notes))

        return labels

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
    # Internal: MIDI output via music21
    # ------------------------------------------------------------------

    @staticmethod
    def _write_midi(
        notes: List[NoteEvent],
        output_path: Path,
        tempo: int,
        time_signature: Tuple[int, int],
    ) -> None:
        """Convert NoteEvent list to a music21 Score and write MIDI."""
        from music21 import stream, tempo as m21tempo, meter as m21meter
        from music21 import note as m21note, instrument, midi as m21midi

        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)

        score = stream.Score()
        score.append(m21tempo.MetronomeMark(number=tempo))
        score.append(m21meter.TimeSignature(f"{ts_num}/{ts_den}"))

        part = stream.Part()
        part.append(instrument.Instrument(midiProgram=0))  # piano

        current_time = 0.0
        bar_start = 0.0
        measure_notes: List[Tuple[float, m21note.Note | m21note.Rest]] = []

        for nev in notes:
            # Advance bar boundary
            if nev.beat_offset > 0 and nev.beat_offset < bar_length_ql:
                current_time = bar_start + nev.beat_offset

            if nev.pitch >= 0:
                n = m21note.Note(pitch=nev.pitch)
                n.duration.quarterLength = nev.duration_ql
                n.volume.velocity = nev.velocity
                part.append(n)
            else:
                r = m21note.Rest()
                r.duration.quarterLength = nev.duration_ql
                part.append(r)

            current_time += nev.duration_ql

            # Check bar boundary
            bar_elapsed = current_time - bar_start
            if bar_elapsed >= bar_length_ql - 0.01:
                bar_start += bar_length_ql
                current_time = bar_start

        score.append(part)
        mf = m21midi.translate.music21ObjectToMidiFile(score)
        mf.open(str(output_path), "wb")
        mf.write()
        mf.close()


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

    gen = HierarchicalGenerator(model)

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
    )

    print(f"\nGenerated {len(labels)} measures → {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
