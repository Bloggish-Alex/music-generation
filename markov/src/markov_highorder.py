"""
Music Markov Model: Higher-Order Markov Chain for MIDI Learning and Generation.

Models pitch, duration, beat position, velocity, tempo, time signature,
program change, and control change. Features duration Markov chain and
velocity humanization. Built on music21, numpy/pandas, seaborn/matplotlib.
"""

from __future__ import annotations

import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from music21 import converter, instrument, midi, note, chord, tempo, meter, key
from music21 import harmony as _m21harmony

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DURATION_CATEGORIES = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
    "thirtysecond": 0.125,
    "dotted_half": 3.0,
    "dotted_quarter": 1.5,
    "dotted_eighth": 0.75,
    "triplet_quarter": 2.0 / 3.0,
    "triplet_eighth": 1.0 / 3.0,
}

VELOCITY_CATEGORIES = {
    "ppp": 20,
    "pp": 35,
    "p": 50,
    "mp": 64,
    "mf": 74,
    "f": 90,
    "ff": 105,
    "fff": 120,
}

BEAT_DIVISIONS = 16  # 16th-note resolution within a beat
BEATS_PER_BAR_DEFAULT = 4

# Chord-analysis constants
_ROOT_TO_PC: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

_CHORD_QUALITY_INTERVALS: Dict[str, List[int]] = {
    "":       [0, 4, 7],       # major triad
    "m":      [0, 3, 7],       # minor triad
    "dim":    [0, 3, 6],       # diminished
    "aug":    [0, 4, 8],       # augmented
    "7":      [0, 4, 7, 10],   # dominant 7th
    "maj7":   [0, 4, 7, 11],   # major 7th
    "m7":     [0, 3, 7, 10],   # minor 7th
    "dim7":   [0, 3, 6, 9],    # diminished 7th
    "m7b5":   [0, 3, 6, 10],   # half-diminished
    "sus4":   [0, 5, 7],       # suspended 4th
    "sus2":   [0, 2, 7],       # suspended 2nd
    "6":      [0, 4, 7, 9],    # major 6th
    "m6":     [0, 3, 7, 9],    # minor 6th
}

# Diatonic scale degrees for key/mode constraint
_KEY_DIATONIC: Dict[str, Set[int]] = {
    "C":  {0, 2, 4, 5, 7, 9, 11},
    "G":  {0, 2, 4, 5, 7, 9, 11},  # G major = F#
    "D":  {0, 2, 4, 5, 7, 9, 11},  # D major = F#,C#
    "A":  {0, 2, 4, 5, 7, 9, 11},  # A major = F#,C#,G#
    "E":  {0, 2, 4, 5, 7, 9, 11},  # E major = F#,C#,G#,D#
    "B":  {0, 2, 4, 5, 7, 9, 11},  # B major = F#,C#,G#,D#,A#
    "F#": {0, 2, 4, 5, 7, 9, 11},  # F# major = all sharps
    "F":  {0, 2, 4, 5, 7, 9, 11},  # F major = Bb
    "Bb": {0, 2, 4, 5, 7, 9, 11},  # Bb major = Bb,Eb
    "Eb": {0, 2, 4, 5, 7, 9, 11},  # Eb major = Bb,Eb,Ab
    "Ab": {0, 2, 4, 5, 7, 9, 11},  # Ab major = Bb,Eb,Ab,Db
    "Db": {0, 2, 4, 5, 7, 9, 11},  # Db major = Bb,Eb,Ab,Db,Gb
    "Gb": {0, 2, 4, 5, 7, 9, 11},  # Gb major = Bb,Eb,Ab,Db,Gb,Cb
    "Am": {0, 2, 4, 5, 7, 9, 11},  # natural minor = same as C major
    "Em": {0, 2, 4, 5, 7, 9, 11},
    "Bm": {0, 2, 4, 5, 7, 9, 11},
    "F#m":{0, 2, 4, 5, 7, 9, 11},
    "Dm": {0, 2, 4, 5, 7, 9, 11},
    "Gm": {0, 2, 4, 5, 7, 9, 11},
    "Cm": {0, 2, 4, 5, 7, 9, 11},
}

# Cadence patterns: (approach_pc, target_pc) — approach leans toward target
_CADENCE_APPROACHES: Dict[int, List[int]] = {
    # Leading tone → tonic
    11: [0],   # B → C
    # Supertonic → tonic
    2: [0],    # D → C
    # Dominant → tonic resolution
    7: [0],    # G → C
    # Subdominant → mediant
    5: [4],    # F → E
    # Leading tone → tonic (minor)
    10: [0],   # Bb → C (in minor context)
}

# Common cadence patterns as pitch-class sequences
_CADENCE_PATTERNS: List[List[int]] = [
    [7, 0],    # V-I bass: G→C
    [11, 0],   # leading tone → tonic: B→C
    [2, 0],    # ii-I: D→C
    [5, 4],    # IV-iii: F→E
    [7, 11, 0],  # V-viio-I
    [2, 7, 0],   # ii-V-I
]

# MIDI pitch for middle C — used for register calculations
_MIDDLE_C = 60

_PC_TO_CANONICAL_KEY: Dict[int, str] = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}

_MAJOR_KEY_BY_SHARPS: Dict[int, str] = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}

_MAJOR_INTERVALS: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)
_MINOR_INTERVALS: Tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)


def _parse_key_name(key_name: str, mode: Optional[str] = None) -> Tuple[str, str, int]:
    """Parse key strings such as C, G major, Am, A minor into root/mode/root_pc."""
    text = (key_name or "C").strip()
    if not text:
        text = "C"

    parts = text.replace("_", " ").split()
    root = parts[0]
    parsed_mode = mode.lower() if mode else None

    if root.lower().endswith("minor"):
        root = root[:-5]
        parsed_mode = "minor"
    elif root.lower().endswith("major"):
        root = root[:-5]
        parsed_mode = "major"
    elif root.endswith("m") and len(root) > 1:
        root = root[:-1]
        parsed_mode = "minor"

    if len(parts) > 1:
        if parts[1].lower().startswith("min"):
            parsed_mode = "minor"
        elif parts[1].lower().startswith("maj"):
            parsed_mode = "major"

    root = root[0].upper() + root[1:]
    if root not in _ROOT_TO_PC:
        raise ValueError(f"Unsupported key root: {key_name!r}")

    parsed_mode = parsed_mode or "major"
    if parsed_mode not in ("major", "minor"):
        raise ValueError(f"Unsupported key mode: {mode!r}")

    return root, parsed_mode, _ROOT_TO_PC[root]


def _format_key_name(root: str, mode: str) -> str:
    return f"{root}m" if mode == "minor" else root


def _key_to_pitch_classes(key_name: str) -> Set[int]:
    """Map a key name like 'C', 'G', 'Am' to its diatonic pitch classes."""
    # Parse root and mode
    key_name = key_name.strip()
    if not key_name:
        return set(range(12))

    root_str = key_name[0].upper()
    i = 1
    if i < len(key_name) and key_name[i] in ("#", "b"):
        root_str += key_name[i]
        i += 1
    mode = key_name[i:]  # "" or "m"

    root_pc = _ROOT_TO_PC.get(root_str, 0)
    # Major scale intervals or natural minor
    if mode == "m":
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]
    return {(root_pc + iv) % 12 for iv in intervals}


def chord_to_pitch_classes(chord_symbol: str) -> Set[int]:
    """Parse a chord symbol to the set of MIDI pitch classes (0–11).

    Examples: ``"C"`` → {0,4,7}, ``"G7"`` → {7,11,2,5}, ``"Am"`` → {9,0,4}.
    Returns ``set(range(12))`` (all allowed) when parsing fails.
    """
    if not chord_symbol:
        return set(range(12))
    root = chord_symbol[0]
    i = 1
    if i < len(chord_symbol) and chord_symbol[i] in ("#", "b"):
        root += chord_symbol[i]
        i += 1
    quality = chord_symbol[i:]
    root_pc = _ROOT_TO_PC.get(root)
    if root_pc is None:
        return set(range(12))
    intervals = _CHORD_QUALITY_INTERVALS.get(quality, [0, 4, 7])
    return {(root_pc + iv) % 12 for iv in intervals}


class ScaleDegreeCodec:
    """Convert between absolute MIDI pitches and key-relative scale degrees."""

    def __init__(self, key_name: str = "C", mode: Optional[str] = None, tonic_octave: int = 4):
        self.root, self.mode, self.root_pc = _parse_key_name(key_name, mode)
        self.intervals = _MINOR_INTERVALS if self.mode == "minor" else _MAJOR_INTERVALS
        self.tonic_midi = 12 * (tonic_octave + 1) + self.root_pc
        self.tonic_octave_bucket = self.tonic_midi // 12

    @property
    def key_name(self) -> str:
        return _format_key_name(self.root, self.mode)

    @staticmethod
    def _signed_delta(target_pc: int, base_pc: int) -> int:
        delta = (target_pc - base_pc) % 12
        if delta > 6:
            delta -= 12
        return delta

    def pitch_to_degree(self, pitch: int) -> Tuple[int, int, int]:
        """Return (degree, accidental, octave_offset) for a MIDI pitch."""
        relative_pc = (pitch - self.root_pc) % 12
        octave = (pitch // 12) - self.tonic_octave_bucket

        best_degree = 1
        best_accidental = 0
        best_distance = 99
        for idx, base_pc in enumerate(self.intervals, start=1):
            accidental = self._signed_delta(relative_pc, base_pc)
            distance = abs(accidental)
            if distance < best_distance or (
                distance == best_distance and abs(accidental) < abs(best_accidental)
            ):
                best_degree = idx
                best_accidental = accidental
                best_distance = distance

        # Keep chromatic spelling compact. In ordinary 12-TET material the
        # nearest diatonic degree is always within +/-1 semitone.
        best_accidental = max(-2, min(2, best_accidental))
        return best_degree, best_accidental, octave

    def degree_to_pitch(self, degree: int, accidental: int, octave: int) -> int:
        """Map a scale degree back to an absolute MIDI pitch in this codec's key."""
        if degree <= 0:
            return -1
        interval = self.intervals[(degree - 1) % 7] + accidental
        pitch = self.tonic_midi + octave * 12 + interval
        return max(0, min(127, int(pitch)))

    def event_to_tonal(self, event: MusicEvent) -> TonalEvent:
        if event.pitch < 0:
            return TonalEvent(
                degree=0, accidental=0, octave=0,
                duration_idx=event.duration_idx,
                beat_position=event.beat_position,
                velocity_idx=event.velocity_idx,
                program=event.program,
            )
        degree, accidental, octave = self.pitch_to_degree(event.pitch)
        return TonalEvent(
            degree=degree,
            accidental=accidental,
            octave=octave,
            duration_idx=event.duration_idx,
            beat_position=event.beat_position,
            velocity_idx=event.velocity_idx,
            program=event.program,
        )

    def tonal_to_event(self, tonal: TonalEvent) -> MusicEvent:
        return MusicEvent(
            pitch=self.degree_to_pitch(tonal.degree, tonal.accidental, tonal.octave),
            duration_idx=tonal.duration_idx,
            beat_position=tonal.beat_position,
            velocity_idx=tonal.velocity_idx,
            program=tonal.program,
        )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MusicEvent:
    """A single discretized musical event."""

    pitch: int  # MIDI pitch 0..127, or -1 for rest
    duration_idx: int  # index into DURATION_CATEGORIES
    beat_position: int  # 0 .. BEAT_DIVISIONS-1
    velocity_idx: int  # index into VELOCITY_CATEGORIES
    program: int  # MIDI program number 0..127

    def as_token(self) -> str:
        return f"p{self.pitch}_d{self.duration_idx}_b{self.beat_position}_v{self.velocity_idx}_pg{self.program}"

    @staticmethod
    def from_token(token: str) -> "MusicEvent":
        parts = token.split("_")
        return MusicEvent(
            pitch=int(parts[0][1:]),
            duration_idx=int(parts[1][1:]),
            beat_position=int(parts[2][1:]),
            velocity_idx=int(parts[3][1:]),
            program=int(parts[4][2:]),
        )


@dataclass(frozen=True, slots=True)
class TonalEvent:
    """A key-relative event used by the tonal Markov chain.

    degree is 1..7 for scale degrees and 0 for rests. accidental is -1/0/+1
    relative to the active mode. octave is relative to the tonic octave.
    """

    degree: int
    accidental: int
    octave: int
    duration_idx: int
    beat_position: int
    velocity_idx: int
    program: int

    def as_token(self) -> str:
        return (
            f"dg{self.degree}_a{self.accidental}_o{self.octave}_"
            f"d{self.duration_idx}_b{self.beat_position}_v{self.velocity_idx}_pg{self.program}"
        )

    @staticmethod
    def from_token(token: str) -> "TonalEvent":
        parts = token.split("_")
        return TonalEvent(
            degree=int(parts[0][2:]),
            accidental=int(parts[1][1:]),
            octave=int(parts[2][1:]),
            duration_idx=int(parts[3][1:]),
            beat_position=int(parts[4][1:]),
            velocity_idx=int(parts[5][1:]),
            program=int(parts[6][2:]),
        )


@dataclass
class MidiMetadata:
    """Metadata extracted from a MIDI file."""

    tempos: List[Tuple[float, float]] = field(default_factory=list)  # (offset_seconds, bpm)
    time_signatures: List[Tuple[float, int, int]] = field(default_factory=list)  # (offset, num, den)
    key_signatures: List[Tuple[float, int]] = field(default_factory=list)  # (offset, sharps)
    chord_symbols: List[Tuple[float, str]] = field(default_factory=list)  # (offset, chord_name)
    program_changes: Dict[int, List[Tuple[float, int]]] = field(default_factory=dict)  # channel -> [(offset, program)]
    control_changes: Dict[int, List[Tuple[float, int, int]]] = field(default_factory=dict)  # channel -> [(offset, cc_num, value)]


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def _quantize_duration(ql: float) -> int:
    """Map a music21 quarterLength to the closest DURATION_CATEGORIES index."""
    candidates = list(DURATION_CATEGORIES.values())
    idx = int(np.argmin([abs(ql - v) for v in candidates]))
    return idx


def _quantize_velocity(vel: int) -> int:
    """Map a raw MIDI velocity to the closest VELOCITY_CATEGORIES index."""
    candidates = list(VELOCITY_CATEGORIES.values())
    idx = int(np.argmin([abs(vel - v) for v in candidates]))
    return idx


def _beat_position(offset: float, ts_num: int, ts_den: int) -> int:
    """Return 0..BEAT_DIVISIONS-1 position within a bar."""
    if ts_den == 0:
        ts_den = 4
    beat_length = 4.0 / ts_den
    bar_length = ts_num * beat_length
    if bar_length <= 0:
        return 0
    pos_in_bar = offset % bar_length
    return int(pos_in_bar / bar_length * BEAT_DIVISIONS) % BEAT_DIVISIONS


# ---------------------------------------------------------------------------
# MIDI Parser (music21-based)
# ---------------------------------------------------------------------------

class MidiParser:
    """Parse MIDI and ABC files using music21, extracting comprehensive event data.

    Both MIDI (.mid, .midi) and ABC (.abc) formats are supported transparently
    via music21's ``converter.parse()``.  The internal extraction logic
    (time-signature tracking, duration quantisation, chord expansion) is
    format-agnostic.
    """

    def __init__(self, beat_divisions: int = BEAT_DIVISIONS):
        self.beat_divisions = beat_divisions

    def parse(self, midi_path: Union[str, Path]) -> Tuple[List[MusicEvent], MidiMetadata]:
        """Parse a MIDI file into a sequence of MusicEvents and metadata.

        Returns:
            (events, metadata) tuple
        """
        score = converter.parse(str(midi_path))
        metadata = MidiMetadata()
        events: List[MusicEvent] = []

        # --- extract metadata ---
        for el in score.flatten():
            if isinstance(el, tempo.MetronomeMark):
                offset_sec = el.offset
                bpm = el.getQuarterBPM()
                metadata.tempos.append((float(offset_sec), float(bpm)))
            elif isinstance(el, meter.TimeSignature):
                metadata.time_signatures.append((float(el.offset), el.numerator, el.denominator))
            elif isinstance(el, key.KeySignature):
                metadata.key_signatures.append((float(el.offset), el.sharps))
            elif isinstance(el, _m21harmony.ChordSymbol) and el.figure:
                metadata.chord_symbols.append((float(el.offset), el.figure.strip()))

        # Fallback: infer chord symbols from sounding chords (MIDI without labels)
        if not metadata.chord_symbols:
            for el in score.flatten().notes:
                if hasattr(el, "isChord") and el.isChord and len(el.pitches) >= 3:
                    try:
                        cs = _m21harmony.chordSymbolFromChord(el)
                        if cs.figure:
                            metadata.chord_symbols.append((float(el.offset), cs.figure.strip()))
                    except Exception:
                        pass

        # --- extract per-part events ---
        for part in score.parts:
            prog = 0
            for inst_obj in part.recurse().getElementsByClass(instrument.Instrument):
                prog = inst_obj.midiProgram
                break

            current_ts = (4, 4)
            ts_events = sorted(metadata.time_signatures, key=lambda x: x[0])
            ts_idx = 0

            for el in part.flatten().notesAndRests:
                while ts_idx < len(ts_events) and ts_events[ts_idx][0] <= el.offset:
                    _, num, den = ts_events[ts_idx]
                    current_ts = (num, den)
                    ts_idx += 1

                offset = float(el.offset)
                bp = _beat_position(offset, current_ts[0], current_ts[1])

                if el.isRest:
                    dur_idx = _quantize_duration(float(el.quarterLength))
                    events.append(MusicEvent(
                        pitch=-1, duration_idx=dur_idx,
                        beat_position=bp, velocity_idx=0,
                        program=prog,
                    ))
                elif el.isNote:
                    midi_pitch = el.pitch.midi if el.pitch else 60
                    dur_idx = _quantize_duration(float(el.quarterLength))
                    vel = el.volume.velocity if el.volume.velocity is not None else 80
                    vel_idx = _quantize_velocity(int(vel))
                    events.append(MusicEvent(
                        pitch=int(midi_pitch), duration_idx=dur_idx,
                        beat_position=bp, velocity_idx=vel_idx,
                        program=prog,
                    ))
                elif el.isChord:
                    for p in el.pitches:
                        midi_pitch = p.midi
                        dur_idx = _quantize_duration(float(el.quarterLength))
                        vel = el.volume.velocity if el.volume.velocity is not None else 80
                        vel_idx = _quantize_velocity(int(vel))
                        events.append(MusicEvent(
                            pitch=int(midi_pitch), duration_idx=dur_idx,
                            beat_position=bp, velocity_idx=vel_idx,
                            program=prog,
                        ))

        return events, metadata


# ---------------------------------------------------------------------------
# Higher-Order Markov Chain
# ---------------------------------------------------------------------------

class HigherOrderMarkovChain:
    """Higher-order Markov chain with back-off smoothing and additive smoothing.

    Parameters:
        order: Maximum history length (1 to k).
        alpha: Additive-smoothing strength.  0.0 = no smoothing (pure empirical).
            Values around 0.05 give a gentle unigram- prior pull that prevents
            zero-probability tokens in sparse contexts.  Equivalent to adding
            ``alpha`` pseudo-counts drawn from the unigram distribution to every
            transition row before normalisation.
        min_unigram_count: Minimum occurrence count for a token to be included
            in the unigram prior.  Tokens below this threshold are excluded.
    """

    def __init__(self, order: int = 3, alpha: float = 0.0, min_unigram_count: int = 3):
        self.order = order
        self.alpha = alpha
        self.min_unigram_count = min_unigram_count
        self._chains: Dict[int, Dict[Tuple[str, ...], Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        self._counts: Dict[int, Dict[Tuple[str, ...], Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._unigram_prior: Dict[str, float] = {}

    def fit(self, sequences: List[List[str]]):
        """Train the Markov chain on token sequences."""
        self._counts.clear()
        self._chains.clear()
        self._unigram_prior.clear()

        for seq in sequences:
            if len(seq) < 2:
                continue
            for o in range(1, self.order + 1):
                for i in range(len(seq) - o):
                    history = tuple(seq[i : i + o])
                    nxt = seq[i + o]
                    self._counts[o][history][nxt] += 1

        for o, trans in self._counts.items():
            for history, next_counts in trans.items():
                total = sum(next_counts.values())
                for nxt, count in next_counts.items():
                    self._chains[o][history][nxt] = count / total

        if self.alpha > 0 and 1 in self._counts:
            unigram_counts: Dict[str, float] = defaultdict(float)
            for (_tok,), next_counts in self._counts[1].items():
                for nxt, cnt in next_counts.items():
                    unigram_counts[nxt] += cnt
            total = sum(unigram_counts.values())
            if total > 0:
                for tok, cnt in unigram_counts.items():
                    if cnt >= self.min_unigram_count:
                        self._unigram_prior[tok] = cnt / total
                prior_sum = sum(self._unigram_prior.values())
                if prior_sum > 0:
                    for tok in list(self._unigram_prior):
                        self._unigram_prior[tok] /= prior_sum

    def sample_next(self, history: List[str], temperature: float = 1.0) -> Optional[str]:
        """Sample the next token given a history, with back-off and additive smoothing."""
        for o in range(min(self.order, len(history)), 0, -1):
            key = tuple(history[-o:])
            if key in self._chains[o] and self._chains[o][key]:
                candidates = list(self._chains[o][key].keys())
                probs = np.array([self._chains[o][key][c] for c in candidates], dtype=np.float64)

                if self.alpha > 0 and self._unigram_prior:
                    total_count = sum(self._counts[o][key].values())
                    alpha_eff = self.alpha / (self.alpha + total_count)
                    for i, tok in enumerate(candidates):
                        prior = self._unigram_prior.get(tok, 0.0)
                        probs[i] = probs[i] * (1.0 - alpha_eff) + prior * alpha_eff
                    probs /= probs.sum()

                if temperature != 1.0 and temperature > 0:
                    probs = np.power(probs, 1.0 / temperature)
                    probs /= probs.sum()

                return str(np.random.choice(candidates, p=probs))

        return None

    def generate(self, seed: List[str], length: int, temperature: float = 1.0) -> List[str]:
        """Generate a sequence of tokens."""
        if len(seed) < self.order:
            raise ValueError(f"Seed length {len(seed)} < order {self.order}")

        result = list(seed)
        for _ in range(length):
            nxt = self.sample_next(result[-self.order :], temperature)
            if nxt is None:
                break
            result.append(nxt)
        return result


# ---------------------------------------------------------------------------
# Duration Markov Chain
# ---------------------------------------------------------------------------

class DurationMarkovChain:
    """Specialized Markov chain for rhythm/duration patterns."""

    def __init__(self, order: int = 4):
        self.order = order
        self._transitions: Dict[Tuple[int, ...], Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def fit(self, duration_sequences: List[List[int]]):
        """Train on sequences of duration indices."""
        counts: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for seq in duration_sequences:
            if len(seq) < self.order + 1:
                continue
            for i in range(len(seq) - self.order):
                history = tuple(seq[i : i + self.order])
                nxt = seq[i + self.order]
                counts[history][nxt] += 1

        for history, next_counts in counts.items():
            total = sum(next_counts.values())
            for nxt, cnt in next_counts.items():
                self._transitions[history][nxt] = cnt / total

    def sample_next(self, history: List[int], temperature: float = 1.0) -> Optional[int]:
        """Sample next duration with optional temperature for diversity."""
        key = tuple(history[-self.order :])
        if key in self._transitions and self._transitions[key]:
            candidates = list(self._transitions[key].keys())
            probs = np.array([self._transitions[key][c] for c in candidates], dtype=np.float64)
            if temperature != 1.0 and temperature > 0:
                probs = np.power(probs, 1.0 / temperature)
                probs /= probs.sum()
            return int(np.random.choice(candidates, p=probs))
        return None

    def generate(self, seed: List[int], length: int, temperature: float = 1.0) -> List[int]:
        """Generate duration sequence with optional temperature."""
        result = list(seed)
        for _ in range(length):
            nxt = self.sample_next(result[-self.order :], temperature)
            if nxt is None:
                nxt = random.choice(list(DURATION_CATEGORIES.keys()))
                nxt = list(DURATION_CATEGORIES.keys()).index(nxt) if isinstance(nxt, str) else nxt
            result.append(nxt)
        return result

    def generate_bars(
        self,
        seed: List[int],
        num_bars: int,
        time_signature: Tuple[int, int] = (4, 4),
        max_attempts: int = 200,
        tolerance: float = 0.005,
        temperature: float = 1.0,
    ) -> List[int]:
        """Generate duration sequence respecting bar-length constraints."""
        dur_values = list(DURATION_CATEGORIES.values())
        bar_length_ql = (time_signature[0] / time_signature[1]) * 4.0

        result = list(seed)
        for _ in range(num_bars):
            remain = bar_length_ql
            bar_ok = False
            for _ in range(max_attempts):
                candidates = [
                    idx for idx, ql in enumerate(dur_values)
                    if 0 < ql <= remain + tolerance
                ]
                if not candidates:
                    break

                key = tuple(result[-self.order :])
                if key not in self._transitions:
                    idx = random.choice(candidates)
                else:
                    probs = np.zeros(len(dur_values), dtype=np.float64)
                    for c in candidates:
                        probs[c] = self._transitions[key].get(c, 0.0)
                    if probs.sum() <= 0:
                        idx = random.choice(candidates)
                    else:
                        probs /= probs.sum()
                        if temperature != 1.0 and temperature > 0:
                            probs = np.power(probs, 1.0 / temperature)
                            probs /= probs.sum()
                        idx = int(np.random.choice(len(dur_values), p=probs))

                result.append(idx)
                remain -= dur_values[idx]
                if remain <= tolerance:
                    bar_ok = True
                    break

            if not bar_ok and remain > tolerance:
                for fill_idx in (10, 4):
                    while remain > tolerance and dur_values[fill_idx] <= remain + tolerance:
                        result.append(fill_idx)
                        remain -= dur_values[fill_idx]
                    if remain <= tolerance:
                        break

        return result


# ---------------------------------------------------------------------------
# Chord Markov Chain
# ---------------------------------------------------------------------------


class ChordMarkovChain:
    """Markov chain for chord progressions (harmonic syntax)."""

    def __init__(self, order: int = 2):
        self.order = order
        self._transitions: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def fit(self, chord_sequences: List[List[str]]):
        """Train on sequences of chord symbols (strings)."""
        counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for seq in chord_sequences:
            if len(seq) < self.order + 1:
                continue
            for i in range(len(seq) - self.order):
                history = tuple(seq[i : i + self.order])
                nxt = seq[i + self.order]
                counts[history][nxt] += 1

        for history, next_counts in counts.items():
            total = sum(next_counts.values())
            for nxt, cnt in next_counts.items():
                self._transitions[history][nxt] = cnt / total

    def sample_next(self, history: List[str]) -> Optional[str]:
        """Sample the next chord given a history."""
        key = tuple(history[-self.order :])
        if key in self._transitions and self._transitions[key]:
            candidates = list(self._transitions[key].keys())
            probs = [self._transitions[key][c] for c in candidates]
            return str(np.random.choice(candidates, p=probs))
        return None

    def generate(self, seed: List[str], num_bars: int) -> List[str]:
        """Generate a chord progression of *num_bars* chords."""
        result = list(seed)
        for _ in range(num_bars - len(seed)):
            nxt = self.sample_next(result[-self.order :])
            if nxt is None:
                nxt = random.choice(list(result))
            result.append(nxt)
        return result


# ---------------------------------------------------------------------------
# Pitch Markov Chain (factorized auxiliary chain)
# ---------------------------------------------------------------------------


class PitchMarkovChain:
    """Higher-order Markov chain for pure pitch sequences.

    Supports chord-constrained sampling, key/mode filtering, and cadence bias.
    """

    def __init__(self, order: int = 4):
        self.order = order
        self._transitions: Dict[Tuple[int, ...], Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def fit(self, pitch_sequences: List[List[int]]):
        """Train on pitch sequences (integers, rests excluded)."""
        counts: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for seq in pitch_sequences:
            if len(seq) < self.order + 1:
                continue
            for i in range(len(seq) - self.order):
                history = tuple(seq[i : i + self.order])
                nxt = seq[i + self.order]
                counts[history][nxt] += 1

        for history, next_counts in counts.items():
            total = sum(next_counts.values())
            for nxt, cnt in next_counts.items():
                self._transitions[history][nxt] = cnt / total

    def sample_next(
        self,
        history: List[int],
        allowed_pitch_classes: Optional[Set[int]] = None,
        key_pitch_classes: Optional[Set[int]] = None,
        cadence_target: Optional[int] = None,
        key_strength: float = 0.0,
    ) -> Optional[int]:
        """Sample the next pitch with optional constraints.

        Args:
            history: Previous pitch values.
            allowed_pitch_classes: Hard-filter to these PC's (chord constraint).
            key_pitch_classes: Boost probabilities for these PC's (key constraint).
            cadence_target: Target MIDI pitch for cadence resolution.
            key_strength: 0.0-1.0 weight for key-based probability boost.
        """
        key = tuple(history[-self.order :])
        if key not in self._transitions:
            return None

        candidates = list(self._transitions[key].keys())

        # Hard filter: chord constraint
        if allowed_pitch_classes is not None:
            constrained = [p for p in candidates if p % 12 in allowed_pitch_classes]
            if constrained:
                candidates = constrained

        probs = np.array([self._transitions[key][c] for c in candidates], dtype=np.float64)

        # Soft boost: key/mode constraint — boost diatonic pitches
        if key_pitch_classes is not None and key_strength > 0:
            for i, p in enumerate(candidates):
                if p % 12 in key_pitch_classes:
                    probs[i] *= (1.0 + key_strength)

        # Soft bias: cadence target
        if cadence_target is not None:
            for i, p in enumerate(candidates):
                dist = abs(p - cadence_target)
                if dist <= 2:
                    probs[i] *= (2.0 - dist * 0.5)

        probs /= probs.sum()
        return int(np.random.choice(candidates, p=probs))

    def generate(
        self,
        seed: List[int],
        length: int,
        allowed_pitch_classes: Optional[Set[int]] = None,
        key_pitch_classes: Optional[Set[int]] = None,
        key_strength: float = 0.0,
    ) -> List[int]:
        """Generate a pitch sequence with optional constraints."""
        result = list(seed)
        for _ in range(length):
            nxt = self.sample_next(
                result[-self.order :],
                allowed_pitch_classes=allowed_pitch_classes,
                key_pitch_classes=key_pitch_classes,
                key_strength=key_strength,
            )
            if nxt is None:
                nxt = random.randint(48, 84)
            result.append(nxt)
        return result


# ---------------------------------------------------------------------------
# Tonal rhythm Markov chain (relative scale degree + duration + beat)
# ---------------------------------------------------------------------------


class TonalRhythmMarkovChain:
    """Higher-order chain over TonalEvent tokens.

    This chain keeps melody and rhythm coupled while representing pitch as
    scale degree, so patterns learned in one key can be generated in another.
    """

    def __init__(self, order: int = 3):
        self.order = order
        self._transitions: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._unigram: Dict[str, float] = {}
        self._seed_windows: List[List[str]] = []

    def fit(self, tonal_sequences: List[List[str]]):
        counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        unigram_counts: Dict[str, int] = defaultdict(int)
        self._seed_windows.clear()

        for seq in tonal_sequences:
            if len(seq) < 2:
                continue
            for token in seq:
                unigram_counts[token] += 1
            if len(seq) >= self.order:
                for i in range(0, len(seq) - self.order + 1):
                    self._seed_windows.append(seq[i:i + self.order])
            for o in range(1, self.order + 1):
                if len(seq) < o + 1:
                    continue
                for i in range(len(seq) - o):
                    history = tuple(seq[i:i + o])
                    nxt = seq[i + o]
                    counts[history][nxt] += 1

        self._transitions.clear()
        for history, next_counts in counts.items():
            total = sum(next_counts.values())
            for nxt, count in next_counts.items():
                self._transitions[history][nxt] = count / total

        total_unigrams = sum(unigram_counts.values())
        self._unigram = {
            token: count / total_unigrams
            for token, count in unigram_counts.items()
        } if total_unigrams else {}

    @staticmethod
    def _token_pitch_signature(token: str) -> Tuple[int, int, int]:
        ev = TonalEvent.from_token(token)
        return ev.degree, ev.accidental, ev.octave

    def random_seed(self) -> List[str]:
        if self._seed_windows:
            return list(random.choice(self._seed_windows))
        if self._unigram:
            seed = [random.choice(list(self._unigram.keys())) for _ in range(self.order)]
            return seed
        return [TonalEvent(1, 0, 0, 2, 0, 4, 0).as_token()] * self.order

    def sample_next(
        self,
        history: List[str],
        temperature: float = 1.0,
        chromatic_strictness: float = 0.8,
        repeat_penalty: float = 0.35,
    ) -> Optional[str]:
        candidates: List[str] = []
        probs: np.ndarray

        for o in range(min(self.order, len(history)), 0, -1):
            key = tuple(history[-o:])
            if key in self._transitions and self._transitions[key]:
                candidates = list(self._transitions[key].keys())
                probs = np.array([self._transitions[key][c] for c in candidates], dtype=np.float64)
                break
        else:
            if not self._unigram:
                return None
            candidates = list(self._unigram.keys())
            probs = np.array([self._unigram[c] for c in candidates], dtype=np.float64)

        strictness = max(0.0, min(1.0, chromatic_strictness))
        if strictness > 0:
            for i, token in enumerate(candidates):
                ev = TonalEvent.from_token(token)
                if ev.degree > 0 and ev.accidental != 0:
                    probs[i] *= max(0.03, 1.0 - strictness)

        if history and repeat_penalty < 1.0:
            last_signature = self._token_pitch_signature(history[-1])
            run_length = 1
            for prev in reversed(history[:-1]):
                if self._token_pitch_signature(prev) != last_signature:
                    break
                run_length += 1
            for i, token in enumerate(candidates):
                if self._token_pitch_signature(token) == last_signature:
                    probs[i] *= max(0.02, repeat_penalty ** run_length)

        if probs.sum() <= 0:
            probs = np.ones(len(candidates), dtype=np.float64) / len(candidates)
        else:
            probs /= probs.sum()

        if temperature != 1.0 and temperature > 0:
            probs = np.power(probs, 1.0 / temperature)
            probs /= probs.sum()

        return str(np.random.choice(candidates, p=probs))

    def generate(
        self,
        seed: Optional[List[str]],
        length: int,
        temperature: float = 1.0,
        chromatic_strictness: float = 0.8,
        repeat_penalty: float = 0.35,
    ) -> List[str]:
        result = list(seed) if seed else self.random_seed()
        if len(result) < self.order:
            result = self.random_seed()

        for _ in range(length):
            nxt = self.sample_next(
                result[-self.order:],
                temperature=temperature,
                chromatic_strictness=chromatic_strictness,
                repeat_penalty=repeat_penalty,
            )
            if nxt is None:
                break
            result.append(nxt)
        return result


# ---------------------------------------------------------------------------
# Humanizer
# ---------------------------------------------------------------------------

class Humanizer:
    """Adds human-like random perturbations to velocities."""

    def __init__(
        self,
        velocity_jitter: float = 8.0,
        timing_jitter_sec: float = 0.015,
        seed: Optional[int] = None,
    ):
        self.velocity_jitter = velocity_jitter
        self.timing_jitter_sec = timing_jitter_sec
        self._rng = np.random.RandomState(seed)

    def perturb_velocity(self, velocity: int) -> int:
        """Add Gaussian jitter to a velocity value, clamped to [1, 127]."""
        delta = self._rng.normal(0, self.velocity_jitter)
        return max(1, min(127, int(round(velocity + delta))))

    def perturb_timing(self, time_sec: float) -> float:
        """Add Gaussian jitter to a timing value, clamped >= 0."""
        delta = self._rng.normal(0, self.timing_jitter_sec)
        return max(0.0, time_sec + delta)


# ---------------------------------------------------------------------------
# MIDI Generator (music21-based output)
# ---------------------------------------------------------------------------

class MidiGenerator:
    """Generate MIDI files from event sequences using music21."""

    def __init__(self, humanizer: Optional[Humanizer] = None):
        self.humanizer = humanizer or Humanizer()

    def events_to_score(self, events: List[MusicEvent], metadata: Optional[MidiMetadata] = None) -> "music21.stream.Score":
        """Convert MusicEvent sequence to a music21 Score."""
        from music21 import stream, tempo as m21tempo, meter as m21meter

        dur_values = list(DURATION_CATEGORIES.values())
        vel_values = list(VELOCITY_CATEGORIES.values())

        prog_events: Dict[int, List[MusicEvent]] = defaultdict(list)
        for ev in events:
            prog_events[ev.program].append(ev)

        score = stream.Score()

        if metadata:
            if metadata.tempos:
                offset, bpm = metadata.tempos[0]
                score.append(m21tempo.MetronomeMark(number=bpm))
            if metadata.time_signatures:
                _, num, den = metadata.time_signatures[0]
                score.append(m21meter.TimeSignature(f"{num}/{den}"))

        for prog, ev_list in prog_events.items():
            part = stream.Part()
            part.append(instrument.Instrument(midiProgram=prog))

            current_time = 0.0
            for ev in ev_list:
                ql = dur_values[ev.duration_idx] if ev.duration_idx < len(dur_values) else 1.0

                if ev.pitch >= 0:
                    vel = vel_values[ev.velocity_idx] if ev.velocity_idx < len(vel_values) else 80
                    vel = self.humanizer.perturb_velocity(vel)
                    n = note.Note(pitch=ev.pitch)
                    n.duration.quarterLength = ql
                    n.volume.velocity = vel
                    n.storedInstrument = instrument.Instrument(midiProgram=ev.program)
                    part.append(n)
                else:
                    r = note.Rest()
                    r.duration.quarterLength = ql
                    part.append(r)

                current_time += ql

            score.append(part)

        return score

    def write_midi(self, events: List[MusicEvent], output_path: Union[str, Path],
                   metadata: Optional[MidiMetadata] = None):
        """Write events to a MIDI file."""
        score = self.events_to_score(events, metadata)
        mf = midi.translate.music21ObjectToMidiFile(score)
        mf.open(str(output_path), "wb")
        mf.write()
        mf.close()

    def write_abc(self, events: List[MusicEvent], output_path: Union[str, Path],
                  metadata: Optional[MidiMetadata] = None):
        """Write events to an ABC notation file."""
        score = self.events_to_score(events, metadata)
        score.write(fmt="abc", fp=str(output_path))


# ---------------------------------------------------------------------------
# Chord-sequence extraction helper
# ---------------------------------------------------------------------------


def _build_chord_sequences(
    all_event_tokens: List[List[str]],
    all_metadata: List[MidiMetadata],
    all_duration_seqs: List[List[int]],
) -> List[List[str]]:
    """Build deduplicated bar-level chord sequences for training."""
    result: List[List[str]] = []
    dur_values = list(DURATION_CATEGORIES.values())

    for tokens, meta, dur_seq in zip(all_event_tokens, all_metadata, all_duration_seqs):
        if not meta.chord_symbols:
            continue

        chord_symbols = sorted(meta.chord_symbols, key=lambda x: x[0])
        default_ts = (4, 4)
        if meta.time_signatures:
            default_ts = (meta.time_signatures[0][1], meta.time_signatures[0][2])
        bar_length_ql = (default_ts[0] / default_ts[1]) * 4.0

        bar_chords: List[str] = []
        current_offset = 0.0
        chord_idx = 0

        for di in dur_seq:
            ql = dur_values[di] if di < len(dur_values) else 1.0
            bar_midpoint = current_offset + ql / 2.0

            active = ""
            while chord_idx < len(chord_symbols) and chord_symbols[chord_idx][0] <= bar_midpoint:
                active = chord_symbols[chord_idx][1]
                chord_idx += 1
            if not active and chord_idx < len(chord_symbols):
                active = chord_symbols[chord_idx][1]

            current_bar = int(current_offset / bar_length_ql) if bar_length_ql > 0 else 0
            if not bar_chords or current_bar >= len(bar_chords):
                bar_chords.append(active)

            current_offset += ql

        if bar_chords:
            deduped = [bar_chords[0]]
            for c in bar_chords[1:]:
                if c != deduped[-1]:
                    deduped.append(c)
            deduped = [c for c in deduped if c]
            if len(deduped) >= 2:
                result.append(deduped)

    return result


# ---------------------------------------------------------------------------
# Main System
# ---------------------------------------------------------------------------

class MusicMarkovSystem:
    """Complete music learning and generation system.

    Usage::

        system = MusicMarkovSystem(markov_order=3, duration_order=4)
        system.train("path/to/midi/dir")
        events = system.generate(num_events=500, temperature=1.0)
        system.save_midi(events, "output.mid")
        system.visualize(events, save_prefix="analysis")
    """

    def __init__(
        self,
        markov_order: int = 3,
        duration_order: int = 4,
        chord_order: int = 2,
        pitch_order: int = 4,
        velocity_jitter: float = 8.0,
        timing_jitter_sec: float = 0.015,
        random_seed: Optional[int] = None,
        use_chord_chain: bool = False,
        use_pitch_chain: bool = False,
        use_tonal_chain: bool = False,
        tonal_order: int = 3,
        markov_alpha: float = 0.0,
        min_unigram_count: int = 3,
    ):
        self.markov_order = markov_order
        self.duration_order = duration_order
        self.chord_order = chord_order
        self.pitch_order = pitch_order
        self.tonal_order = tonal_order

        self.parser = MidiParser()
        self.markov_chain = HigherOrderMarkovChain(
            order=markov_order, alpha=markov_alpha,
            min_unigram_count=min_unigram_count,
        )
        self.duration_chain = DurationMarkovChain(order=duration_order)
        self.chord_chain: Optional[ChordMarkovChain] = (
            ChordMarkovChain(order=chord_order) if use_chord_chain else None
        )
        self.pitch_chain: Optional[PitchMarkovChain] = (
            PitchMarkovChain(order=pitch_order) if use_pitch_chain else None
        )
        self.tonal_chain: Optional[TonalRhythmMarkovChain] = (
            TonalRhythmMarkovChain(order=tonal_order) if use_tonal_chain else None
        )
        self.humanizer = Humanizer(
            velocity_jitter=velocity_jitter,
            timing_jitter_sec=timing_jitter_sec,
            seed=random_seed,
        )
        self.generator = MidiGenerator(humanizer=self.humanizer)
        from music_analyzer import MusicVisualizer
        self.visualizer = MusicVisualizer()

        self._metadata: Optional[MidiMetadata] = None
        self._trained = False
        self._chord_trained: bool = False
        self._pitch_trained: bool = False
        self._tonal_trained: bool = False

        # New: inferred key and auto-enable state
        self._detected_key: Optional[str] = None
        self._has_chord_data: bool = False
        self._last_generated_key: Optional[str] = None

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, music_dir: Union[str, Path],
              file_patterns: Optional[Union[str, Sequence[str]]] = None) -> "MusicMarkovSystem":
        """Train the system on all MIDI and/or ABC files in a directory.

        Args:
            music_dir: Path to directory containing music files.
            file_patterns: Glob pattern(s) for music files.
                Defaults to ``["*.mid", "*.midi", "*.abc"]``.

        Returns:
            self (for chaining).
        """
        if file_patterns is None:
            file_patterns = ["*.mid", "*.midi", "*.abc"]
        elif isinstance(file_patterns, str):
            file_patterns = [file_patterns]

        music_dir = Path(music_dir)
        music_paths: List[Path] = []
        for pat in file_patterns:
            music_paths.extend(sorted(music_dir.glob(pat)))
        music_paths = sorted(set(music_paths))

        if not music_paths:
            raise FileNotFoundError(
                f"No music files matching {list(file_patterns)} in {music_dir}"
            )

        all_event_tokens: List[List[str]] = []
        all_duration_seqs: List[List[int]] = []
        all_pitch_seqs: List[List[int]] = []
        all_tonal_seqs: List[List[str]] = []
        all_metadata: List[MidiMetadata] = []

        for mp in music_paths:
            try:
                events, meta = self.parser.parse(mp)
                if not events:
                    continue
                if self._metadata is None:
                    self._metadata = meta
                all_metadata.append(meta)

                tokens = [ev.as_token() for ev in events]
                all_event_tokens.append(tokens)

                dur_seq = [ev.duration_idx for ev in events]
                all_duration_seqs.append(dur_seq)

                pitch_seq = [ev.pitch for ev in events if ev.pitch >= 0]
                if pitch_seq:
                    all_pitch_seqs.append(pitch_seq)

                if self.tonal_chain is not None and pitch_seq:
                    file_key = self._key_from_metadata(meta) or self._infer_key(pitch_seq)
                    codec = ScaleDegreeCodec(file_key)
                    all_tonal_seqs.append([codec.event_to_tonal(ev).as_token() for ev in events])
            except Exception:
                continue

        if not all_event_tokens:
            raise RuntimeError("No valid event sequences extracted from music files.")

        self.markov_chain.fit(all_event_tokens)
        self.duration_chain.fit(all_duration_seqs)
        self._trained = True

        # Auto-detect key from training data
        all_pitches = [p for seq in all_pitch_seqs for p in seq]
        if all_pitches:
            self._detected_key = self._infer_key(all_pitches)

        # Check if any training files have chord data
        self._has_chord_data = any(
            bool(meta.chord_symbols) for meta in all_metadata
        )

        # Auto-enable chord chain if chord data is present but chains not explicitly set
        if self._has_chord_data and self.chord_chain is None:
            self.chord_chain = ChordMarkovChain(order=self.chord_order)
        if self.chord_chain is not None:
            chord_seqs = _build_chord_sequences(all_event_tokens, all_metadata, all_duration_seqs)
            if chord_seqs:
                self.chord_chain.fit(chord_seqs)
                self._chord_trained = True

        # Auto-enable pitch chain if chord chain is active
        if self.pitch_chain is None and self._chord_trained:
            self.pitch_chain = PitchMarkovChain(order=self.pitch_order)
        if self.pitch_chain is not None and all_pitch_seqs:
            self.pitch_chain.fit(all_pitch_seqs)
            self._pitch_trained = True

        if self.tonal_chain is not None and all_tonal_seqs:
            self.tonal_chain.fit(all_tonal_seqs)
            self._tonal_trained = bool(self.tonal_chain._transitions)

        return self

    @staticmethod
    def _key_from_metadata(meta: MidiMetadata) -> Optional[str]:
        """Return a major key name from metadata when a key signature is present."""
        if not meta.key_signatures:
            return None
        _, sharps = sorted(meta.key_signatures, key=lambda x: x[0])[0]
        return _MAJOR_KEY_BY_SHARPS.get(int(sharps))

    def _infer_key(self, pitches: List[int]) -> str:
        """Infer key from pitch-class distribution using simple Krumhansl-Schmuckler."""
        major_profiles = {
            "C": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        }
        pc_counts = np.zeros(12)
        for p in pitches:
            pc_counts[p % 12] += 1
        if pc_counts.sum() == 0:
            return "C"
        pc_dist = pc_counts / pc_counts.sum()

        tonic_pc = int(np.argmax(pc_counts))
        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        # Check if minor (more emphasis on Eb/Ab)
        minor_indicators = pc_counts[3] + pc_counts[8]  # Eb + Ab
        major_indicators = pc_counts[4] + pc_counts[9]  # E + A
        is_minor = minor_indicators > major_indicators * 1.3

        key_name = pc_names[tonic_pc]
        if is_minor and key_name not in ("C", "G", "D", "A", "E", "B", "F#", "F"):
            # Map to a more common minor key
            pass
        if is_minor:
            key_name += "m"

        return key_name

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        num_events: int = 500,
        temperature: float = 1.0,
        seed_events: Optional[List[MusicEvent]] = None,
        use_duration_chain: bool = True,
        use_bar_constraint: bool = False,
        use_chord_constraint: Optional[bool] = None,
        use_pitch_chain: Optional[bool] = None,
        time_signature: Tuple[int, int] = (4, 4),
        # New generation parameters
        phrase_length_beats: int = 16,
        insert_phrase_rests: bool = True,
        duration_temperature: float = 1.0,
        velocity_shaping: bool = True,
        motif_repetition: bool = False,
        key_constraint: bool = False,
        key_strength: float = 0.5,
        register_arc: bool = False,
        cadence_ending: bool = True,
        use_tonal_chain: bool = False,
        target_key: Optional[str] = None,
        random_key: bool = False,
        tonal_strictness: float = 0.8,
        repeat_penalty: float = 0.35,
    ) -> List[MusicEvent]:
        """Generate a new sequence of MusicEvents.

        Args:
            num_events: Number of events to generate.
            temperature: Sampling temperature for main Markov chain.
            seed_events: Optional seed; randomly sampled from training if None.
            use_duration_chain: Override durations with the duration chain.
            use_bar_constraint: Generate bar-constrained rhythms.
            use_chord_constraint: Generate chord-constrained pitches.
            use_pitch_chain: Override pitches with the pitch chain.
            time_signature: Time signature for bar-constrained generation.
            phrase_length_beats: Length of a phrase in beats (for rests, velocity shaping).
            insert_phrase_rests: Insert rests at phrase boundaries.
            duration_temperature: Temperature for duration chain sampling (>1 = more variety).
            velocity_shaping: Apply phrase-level velocity envelopes.
            motif_repetition: Enable motif memory for repeated patterns.
            key_constraint: Constrain pitches to the detected key.
            key_strength: 0.0-1.0 strength of key constraint.
            register_arc: Shape pitch register across the piece.
            cadence_ending: Apply cadence pattern at the end of the piece.
            use_tonal_chain: Generate from the relative scale-degree + duration chain.
            target_key: Output key for tonal generation (for example, "C", "G", "Am").
            random_key: Randomly choose an output key for tonal generation.
            tonal_strictness: 0.0-1.0 penalty for chromatic scale degrees.
            repeat_penalty: Probability penalty for repeated same pitch states.

        Returns:
            List of MusicEvent objects.
        """
        if not self._trained:
            raise RuntimeError("System not trained. Call .train() first.")

        if use_tonal_chain:
            if not self._tonal_trained or self.tonal_chain is None:
                raise RuntimeError(
                    "Tonal chain is not trained. Initialize MusicMarkovSystem with "
                    "use_tonal_chain=True and call .train() first."
                )
            events = self._generate_tonal_events(
                num_events=num_events,
                temperature=temperature,
                target_key=target_key,
                random_key=random_key,
                tonal_strictness=tonal_strictness,
                repeat_penalty=repeat_penalty,
            )
            events = self._apply_beat_alignment(events, time_signature)
            if cadence_ending:
                events = self._apply_tonal_cadence(
                    events, self._last_generated_key or target_key or self._detected_key or "C"
                )
            if insert_phrase_rests:
                events = self._insert_phrase_rests(events, phrase_length_beats, time_signature)
            if velocity_shaping:
                events = self._apply_velocity_shaping(events, phrase_length_beats, time_signature)
            return events

        # Auto-enable chord/pitch constraint when chains are trained and user didn't opt out
        if use_chord_constraint is None:
            use_chord_constraint = self._chord_trained and self._pitch_trained
        elif use_chord_constraint and not self._chord_trained:
            use_chord_constraint = False
        if use_pitch_chain is None:
            use_pitch_chain = self._pitch_trained and not use_chord_constraint
        elif use_pitch_chain and not self._pitch_trained:
            use_pitch_chain = False

        # build seed tokens
        if seed_events:
            seed_tokens = [ev.as_token() for ev in seed_events]
        else:
            seed_tokens = self._random_seed_tokens()

        token_seq = self.markov_chain.generate(seed_tokens, num_events, temperature)
        events = [MusicEvent.from_token(t) for t in token_seq]

        # Duration override — bar-constrained or free
        if use_bar_constraint:
            events = self._apply_bar_constrained_duration(events, time_signature, duration_temperature)
        elif use_duration_chain:
            events = self._apply_duration_chain(events, duration_temperature)

        # Beat-position alignment after duration changes
        events = self._apply_beat_alignment(events, time_signature)

        # Pitch override — chord-constrained or free
        key_pcs: Optional[Set[int]] = None
        if key_constraint and self._detected_key:
            key_pcs = _key_to_pitch_classes(self._detected_key)

        if use_chord_constraint and self._chord_trained and self._pitch_trained:
            events = self._apply_chord_constrained_pitch(
                events, time_signature,
                key_pitch_classes=key_pcs,
                key_strength=key_strength if key_constraint else 0.0,
            )
        elif use_pitch_chain and self._pitch_trained:
            events = self._apply_pitch_chain(
                events,
                key_pitch_classes=key_pcs,
                key_strength=key_strength if key_constraint else 0.0,
            )

        # Register arc shaping
        if register_arc:
            events = self._apply_register_arc(events)

        # Motif repetition
        if motif_repetition:
            events = self._apply_motif_repetition(events)

        # Cadence ending
        if cadence_ending:
            events = self._apply_cadence(events, time_signature)

        # Phrase boundary rests
        if insert_phrase_rests:
            events = self._insert_phrase_rests(events, phrase_length_beats, time_signature)

        # Velocity shaping (after rests, so velocities apply to notes only)
        if velocity_shaping:
            events = self._apply_velocity_shaping(events, phrase_length_beats, time_signature)

        return events

    def generate_multi_voice(
        self,
        num_events: int = 500,
        temperature: float = 1.0,
        time_signature: Tuple[int, int] = (4, 4),
        **kwargs,
    ) -> List[MusicEvent]:
        """Generate a two-voice piece (melody + bass).

        First generates a chord progression, then melody constrained by chords,
        then a bass line on a lower program (pizzicato strings or acoustic bass).
        """
        if not self._chord_trained or not self._pitch_trained:
            # Fallback: single voice generation
            return self.generate(num_events=num_events, temperature=temperature,
                                 time_signature=time_signature, **kwargs)

        dur_values = list(DURATION_CATEGORIES.values())
        bar_length_ql = (time_signature[0] / time_signature[1]) * 4.0

        # 1. Generate chord progression for the whole piece
        seed_chord = [random.choice(list(self.chord_chain._transitions.keys()))[0]]
        # Estimate bars from num_events (assume avg 8 events/bar)
        num_bars = max(4, num_events // 8)
        chord_seq = self.chord_chain.generate(seed_chord, num_bars)

        # 2. Generate melody (standard generation with chord constraint)
        melody_events = self.generate(
            num_events=num_events, temperature=temperature,
            use_duration_chain=True, use_bar_constraint=False,
            use_chord_constraint=True, use_pitch_chain=True,
            time_signature=time_signature,
            program=0,  # piano for melody
            velocity_shaping=True,
            insert_phrase_rests=True,
            cadence_ending=True,
            **{k: v for k, v in kwargs.items()
               if k in ("duration_temperature", "key_constraint", "key_strength")},
        )

        # 3. Generate bass line — longer notes, lower register, chord roots
        bass_events: List[MusicEvent] = []
        bar_offset = 0.0
        for bar_idx in range(num_bars):
            chord = chord_seq[min(bar_idx, len(chord_seq) - 1)]
            root_pc = chord_to_pitch_classes(chord)
            # Pick bass note: root or fifth, 1-2 octaves below middle C
            root_pc_list = sorted(root_pc)
            if root_pc_list:
                bass_pc = root_pc_list[0]
                # Place bass in octave 2-3 (MIDI 36-48)
                bass_pitch = bass_pc + 36
                if bass_pitch < 36:
                    bass_pitch += 12
            else:
                bass_pitch = 36

            # Long duration: half note or dotted half
            bass_dur_idx = random.choice([1, 8])  # half or dotted_half
            bass_vel_idx = 4  # mf

            bass_events.append(MusicEvent(
                pitch=bass_pitch,
                duration_idx=bass_dur_idx,
                beat_position=0,
                velocity_idx=bass_vel_idx,
                program=44,  # Contrabass
            ))

            bar_offset += bar_length_ql

        # Merge melody and bass, sorted by time
        # Reconstruct timing
        all_events: List[Tuple[float, MusicEvent]] = []
        current_time = 0.0
        for ev in melody_events:
            all_events.append((current_time, ev))
            current_time += dur_values[ev.duration_idx]

        bass_time = 0.0
        for ev in bass_events:
            all_events.append((bass_time, ev))
            bass_time += bar_length_ql  # one bass note per bar

        all_events.sort(key=lambda x: x[0])
        return [ev for _, ev in all_events]

    def generate_sections(
        self,
        num_events: int = 500,
        temperature: float = 1.0,
        time_signature: Tuple[int, int] = (4, 4),
        **kwargs,
    ) -> List[MusicEvent]:
        """Generate music in A/B/A' sectional form.

        A section: lower temperature (more faithful to training style).
        B section: higher temperature + higher register (contrast).
        A' section: return with variation (different seed, same temperature as A).
        """
        events_per_section = num_events // 3

        # A section — conservative
        events_a = self.generate(
            num_events=events_per_section,
            temperature=min(temperature, 1.0),
            time_signature=time_signature,
            register_arc=False,
            **{k: v for k, v in kwargs.items() if k != "register_arc"},
        )

        # B section — more adventurous, higher register
        b_temp = temperature * 1.3 if temperature < 1.5 else temperature
        events_b = self.generate(
            num_events=events_per_section,
            temperature=b_temp,
            time_signature=time_signature,
            register_arc=False,
            **{k: v for k, v in kwargs.items() if k != "register_arc"},
        )
        # Shift B section up by 5 semitones (a fourth)
        events_b = [
            MusicEvent(
                pitch=ev.pitch + 5 if ev.pitch >= 0 else -1,
                duration_idx=ev.duration_idx,
                beat_position=ev.beat_position,
                velocity_idx=ev.velocity_idx,
                program=ev.program,
            )
            for ev in events_b
        ]

        # A' section — return to original register, slightly varied
        events_a2 = self.generate(
            num_events=num_events - 2 * events_per_section,
            temperature=min(temperature, 1.0),
            time_signature=time_signature,
            register_arc=False,
            **{k: v for k, v in kwargs.items() if k != "register_arc"},
        )

        return events_a + events_b + events_a2

    def _random_seed_tokens(self) -> List[str]:
        """Pick a random seed from the first-order chain."""
        if 1 in self.markov_chain._chains:
            all_keys = list(self.markov_chain._chains[1].keys())
            if all_keys:
                start = random.choice(all_keys)
                result = [start[0]]
                for _ in range(self.markov_order - 1):
                    nxt = self.markov_chain.sample_next(result)
                    if nxt is None:
                        break
                    result.append(nxt)
                return result
        return ["p60_d2_b0_v3_pg0"] * self.markov_order

    @staticmethod
    def _random_output_key(mode_hint: Optional[str] = None) -> str:
        roots = ["C", "G", "D", "A", "E", "F", "Bb", "Eb"]
        mode = mode_hint if mode_hint in ("major", "minor") else random.choice(["major", "minor"])
        root = random.choice(roots)
        return _format_key_name(root, mode)

    def _generate_tonal_events(
        self,
        num_events: int,
        temperature: float,
        target_key: Optional[str],
        random_key: bool,
        tonal_strictness: float,
        repeat_penalty: float,
    ) -> List[MusicEvent]:
        """Generate events using the key-relative joint melody/rhythm chain."""
        if self.tonal_chain is None:
            return []

        detected_key = self._detected_key or "C"
        _, detected_mode, _ = _parse_key_name(detected_key)
        key_name = (
            self._random_output_key(detected_mode)
            if random_key
            else (target_key or detected_key)
        )
        self._last_generated_key = key_name
        codec = ScaleDegreeCodec(key_name)
        seed = self.tonal_chain.random_seed()
        tokens = self.tonal_chain.generate(
            seed=seed,
            length=max(0, num_events - len(seed)),
            temperature=temperature,
            chromatic_strictness=tonal_strictness,
            repeat_penalty=repeat_penalty,
        )
        return [codec.tonal_to_event(TonalEvent.from_token(token)) for token in tokens[:num_events]]

    @staticmethod
    def _apply_tonal_cadence(events: List[MusicEvent], target_key: str) -> List[MusicEvent]:
        """Resolve the final notes to the requested key's tonic."""
        note_indices = [i for i, ev in enumerate(events) if ev.pitch >= 0]
        if len(note_indices) < 3:
            return events

        root, mode, root_pc = _parse_key_name(target_key)
        result = list(events)
        final_idx = note_indices[-1]
        previous_idx = note_indices[-2]
        final_pitch = result[final_idx].pitch
        tonic = (final_pitch // 12) * 12 + root_pc
        if abs(tonic - final_pitch) > 6:
            tonic += 12 if tonic < final_pitch else -12

        leading_pc = (root_pc + 11) % 12 if mode == "major" else (root_pc + 11) % 12
        leading = (tonic // 12) * 12 + leading_pc
        if leading > tonic:
            leading -= 12

        result[previous_idx] = MusicEvent(
            pitch=max(0, min(127, leading)),
            duration_idx=result[previous_idx].duration_idx,
            beat_position=result[previous_idx].beat_position,
            velocity_idx=result[previous_idx].velocity_idx,
            program=result[previous_idx].program,
        )
        result[final_idx] = MusicEvent(
            pitch=max(0, min(127, tonic)),
            duration_idx=2,
            beat_position=result[final_idx].beat_position,
            velocity_idx=3,
            program=result[final_idx].program,
        )
        return result

    # ------------------------------------------------------------------
    # Phase 1: Quick wins
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_phrase_rests(
        events: List[MusicEvent],
        phrase_length_beats: int,
        time_signature: Tuple[int, int],
    ) -> List[MusicEvent]:
        """Insert rests at phrase boundaries (every N beats)."""
        if phrase_length_beats <= 0:
            return events

        dur_values = list(DURATION_CATEGORIES.values())
        phrase_ql = float(phrase_length_beats)
        result: List[MusicEvent] = []
        current_ql = 0.0

        for ev in events:
            ql = dur_values[ev.duration_idx] if ev.duration_idx < len(dur_values) else 1.0

            # Detect phrase boundary crossed by this event
            phrase_before = int(current_ql / phrase_ql) if phrase_ql > 0 else 0
            phrase_after = int((current_ql + ql) / phrase_ql) if phrase_ql > 0 else 0

            if phrase_after > phrase_before and ev.pitch >= 0:
                # We crossed a phrase boundary — insert a rest
                # Use an eighth rest so it's noticeable but not too long
                rest_idx = 3  # eighth note
                bp = _beat_position(current_ql, time_signature[0], time_signature[1])
                rest_ev = MusicEvent(
                    pitch=-1, duration_idx=rest_idx,
                    beat_position=bp,
                    velocity_idx=0, program=ev.program,
                )
                if not result or result[-1].pitch >= 0:
                    result.append(rest_ev)

            result.append(ev)
            current_ql += ql

        return result

    @staticmethod
    def _apply_velocity_shaping(
        events: List[MusicEvent],
        phrase_length_beats: int,
        time_signature: Tuple[int, int],
    ) -> List[MusicEvent]:
        """Apply crescendo-decrescendo velocity envelopes per phrase."""
        dur_values = list(DURATION_CATEGORIES.values())
        vel_values = list(VELOCITY_CATEGORIES.values())

        phrase_ql = phrase_length_beats if phrase_length_beats > 0 else 16.0

        result = list(events)
        current_ql = 0.0
        phrase_start_idx = 0
        phrase_note_indices: List[int] = []

        for i, ev in enumerate(events):
            ql = dur_values[ev.duration_idx] if ev.duration_idx < len(dur_values) else 1.0
            phrase_idx = int(current_ql / phrase_ql) if phrase_ql > 0 else 0
            next_phrase_idx = int((current_ql + ql) / phrase_ql) if phrase_ql > 0 else 0

            if phrase_idx != next_phrase_idx and phrase_note_indices and ev.pitch >= 0:
                # End of phrase — apply envelope
                n_notes = len(phrase_note_indices)
                for j, ni in enumerate(phrase_note_indices):
                    # Envelope: ramp from mp to f and back to mp
                    frac = j / max(n_notes - 1, 1)
                    # Arch shape: sin curve 0→π
                    import math
                    envelope = math.sin(frac * math.pi)
                    # Map to velocity indices: mp(3) to f(5) peak
                    vel_idx = int(3 + envelope * 3)  # range 3-6 (mp to ff)
                    vel_idx = max(0, min(len(vel_values) - 1, vel_idx))
                    result[ni] = MusicEvent(
                        pitch=result[ni].pitch,
                        duration_idx=result[ni].duration_idx,
                        beat_position=result[ni].beat_position,
                        velocity_idx=vel_idx,
                        program=result[ni].program,
                    )
                phrase_note_indices = []

            if ev.pitch >= 0:
                phrase_note_indices.append(i)

            current_ql += ql

        # Handle final partial phrase
        if phrase_note_indices:
            n_notes = len(phrase_note_indices)
            import math
            for j, ni in enumerate(phrase_note_indices):
                frac = j / max(n_notes - 1, 1)
                envelope = math.sin(frac * math.pi)
                vel_idx = int(3 + envelope * 3)
                vel_idx = max(0, min(len(vel_values) - 1, vel_idx))
                result[ni] = MusicEvent(
                    pitch=result[ni].pitch,
                    duration_idx=result[ni].duration_idx,
                    beat_position=result[ni].beat_position,
                    velocity_idx=vel_idx,
                    program=result[ni].program,
                )

        return result

    @staticmethod
    def _apply_beat_alignment(
        events: List[MusicEvent],
        time_signature: Tuple[int, int],
    ) -> List[MusicEvent]:
        """Recalculate beat_positions so they align with the bar grid after duration changes."""
        dur_values = list(DURATION_CATEGORIES.values())
        bar_length_ql = (time_signature[0] / time_signature[1]) * 4.0

        result = []
        current_ql = 0.0
        for ev in events:
            ql = dur_values[ev.duration_idx] if ev.duration_idx < len(dur_values) else 1.0
            pos_in_bar = current_ql % bar_length_ql if bar_length_ql > 0 else 0.0
            bp = int(pos_in_bar / bar_length_ql * BEAT_DIVISIONS) % BEAT_DIVISIONS
            result.append(MusicEvent(
                pitch=ev.pitch,
                duration_idx=ev.duration_idx,
                beat_position=bp,
                velocity_idx=ev.velocity_idx,
                program=ev.program,
            ))
            current_ql += ql

        return result

    # ------------------------------------------------------------------
    # Phase 2: Structural coherence
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_motif_repetition(events: List[MusicEvent]) -> List[MusicEvent]:
        """Extract and repeat short pitch/duration motifs for coherence."""
        if len(events) < 40:
            return events

        # Extract a few 4-note motifs from the first third of the piece
        motifs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        note_events = [ev for ev in events if ev.pitch >= 0]
        if len(note_events) < 12:
            return events

        for _ in range(3):
            start = random.randint(0, max(0, len(note_events) // 3 - 4))
            motif_pitches = tuple(ev.pitch for ev in note_events[start:start + 4])
            motif_durs = tuple(ev.duration_idx for ev in note_events[start:start + 4])
            motifs.append((motif_pitches, motif_durs))

        # Re-insert motifs at later positions with variation
        note_positions = [i for i, ev in enumerate(events) if ev.pitch >= 0]
        insertion_points = random.sample(
            note_positions[len(note_positions)//2:],
            min(len(motifs), len(note_positions)//2)
        )

        result = list(events)
        for motif, ins_pos in zip(motifs, insertion_points):
            motif_pitches, motif_durs = motif
            for j, (mp, md) in enumerate(zip(motif_pitches, motif_durs)):
                idx = ins_pos + j
                if idx < len(result) and result[idx].pitch >= 0:
                    # Variation: 50% chance to transpose by ±2
                    pitch_var = mp + random.choice([0, 0, 2, -2])
                    result[idx] = MusicEvent(
                        pitch=pitch_var,
                        duration_idx=md,
                        beat_position=result[idx].beat_position,
                        velocity_idx=result[idx].velocity_idx,
                        program=result[idx].program,
                    )

        return result

    @staticmethod
    def _apply_register_arc(events: List[MusicEvent]) -> List[MusicEvent]:
        """Shape the overall pitch register into an arc: low→high→low."""
        note_indices = [i for i, ev in enumerate(events) if ev.pitch >= 0]
        n = len(note_indices)
        if n < 20:
            return events

        result = list(events)
        import math

        for rank, idx in enumerate(note_indices):
            # Arc: sin curve 0→π mapped to ±6 semitone offset
            frac = rank / max(n - 1, 1)
            offset = int(round(6.0 * math.sin(frac * math.pi)))
            old_pitch = result[idx].pitch
            new_pitch = max(21, min(108, old_pitch + offset - 3))
            result[idx] = MusicEvent(
                pitch=new_pitch,
                duration_idx=result[idx].duration_idx,
                beat_position=result[idx].beat_position,
                velocity_idx=result[idx].velocity_idx,
                program=result[idx].program,
            )

        return result

    @staticmethod
    def _apply_cadence(
        events: List[MusicEvent],
        time_signature: Tuple[int, int],
    ) -> List[MusicEvent]:
        """Apply a cadence pattern to the last few notes for a sense of closure."""
        note_indices = [i for i, ev in enumerate(events) if ev.pitch >= 0]
        if len(note_indices) < 8:
            return events

        result = list(events)
        # Target: last note should approach tonic from leading tone or supertonic
        last_notes = note_indices[-4:]
        if len(last_notes) < 3:
            return result

        # Determine tonic from first/last pitch class
        tonic_pc = result[note_indices[0]].pitch % 12
        # Find the octave of the final notes
        last_pitch = result[last_notes[-1]].pitch
        tonic_in_octave = (last_pitch // 12) * 12 + tonic_pc
        if abs(tonic_in_octave - last_pitch) > 6:
            tonic_in_octave = (last_pitch // 12 + 1) * 12 + tonic_pc
            if abs(tonic_in_octave - last_pitch) > 6:
                tonic_in_octave = (last_pitch // 12 - 1) * 12 + tonic_pc

        # Penultimate notes: move toward leading tone or supertonic
        leading_tone = (tonic_pc + 11) % 12
        supertonic = (tonic_pc + 2) % 12
        lt_in_octave = (tonic_in_octave // 12) * 12 + leading_tone
        if lt_in_octave > tonic_in_octave:
            lt_in_octave -= 12

        st_in_octave = (tonic_in_octave // 12) * 12 + supertonic
        if st_in_octave > tonic_in_octave:
            st_in_octave -= 12

        approach = lt_in_octave if abs(lt_in_octave - tonic_in_octave) <= 2 else st_in_octave

        # Second-to-last note: approach tone
        if len(last_notes) >= 2:
            idx = last_notes[-2]
            result[idx] = MusicEvent(
                pitch=approach,
                duration_idx=result[idx].duration_idx,
                beat_position=result[idx].beat_position,
                velocity_idx=result[idx].velocity_idx,
                program=result[idx].program,
            )

        # Last note: tonic
        idx = last_notes[-1]
        # Longer duration for final note
        final_dur_idx = 2  # quarter note
        result[idx] = MusicEvent(
            pitch=tonic_in_octave,
            duration_idx=final_dur_idx,
            beat_position=result[idx].beat_position,
            velocity_idx=3,  # mp — gentle ending
            program=result[idx].program,
        )

        return result

    # ------------------------------------------------------------------
    # Duration helpers
    # ------------------------------------------------------------------

    def _apply_duration_chain(self, events: List[MusicEvent], duration_temperature: float = 1.0) -> List[MusicEvent]:
        """Replace durations using the duration Markov chain for rhythmic coherence."""
        current_durs = [ev.duration_idx for ev in events]
        if len(current_durs) < self.duration_order:
            return events

        seed = current_durs[:self.duration_order]
        new_durs = self.duration_chain.generate(seed, len(events) - self.duration_order, duration_temperature)

        result = []
        for i, ev in enumerate(events):
            new_ev = MusicEvent(
                pitch=ev.pitch,
                duration_idx=new_durs[i] if i < len(new_durs) else ev.duration_idx,
                beat_position=ev.beat_position,
                velocity_idx=ev.velocity_idx,
                program=ev.program,
            )
            result.append(new_ev)
        return result

    def _apply_bar_constrained_duration(
        self,
        events: List[MusicEvent],
        time_signature: Tuple[int, int],
        duration_temperature: float = 1.0,
    ) -> List[MusicEvent]:
        """Replace durations with bar-constrained generation."""
        dur_values = list(DURATION_CATEGORIES.values())
        current_durs = [ev.duration_idx for ev in events]
        if len(current_durs) < self.duration_order:
            return events

        total_ql = sum(dur_values[d] for d in current_durs)
        bar_length_ql = (time_signature[0] / time_signature[1]) * 4.0
        num_bars = max(1, int(total_ql / bar_length_ql + 0.5))

        seed = current_durs[:self.duration_order]
        new_durs = self.duration_chain.generate_bars(
            seed, num_bars, time_signature, temperature=duration_temperature,
        )

        result = []
        for i, ev in enumerate(events):
            new_ev = MusicEvent(
                pitch=ev.pitch,
                duration_idx=new_durs[i] if i < len(new_durs) else ev.duration_idx,
                beat_position=ev.beat_position,
                velocity_idx=ev.velocity_idx,
                program=ev.program,
            )
            result.append(new_ev)
        return result

    def _apply_pitch_chain(
        self,
        events: List[MusicEvent],
        key_pitch_classes: Optional[Set[int]] = None,
        key_strength: float = 0.0,
    ) -> List[MusicEvent]:
        """Override pitches using the factorized pitch chain (no chord constraint)."""
        if self.pitch_chain is None:
            return events
        current_pitches = [ev.pitch for ev in events if ev.pitch >= 0]
        if len(current_pitches) < self.pitch_order:
            return events

        seed = current_pitches[:self.pitch_order]
        new_pitches = self.pitch_chain.generate(
            seed, len(current_pitches) - self.pitch_order,
            key_pitch_classes=key_pitch_classes,
            key_strength=key_strength,
        )

        pitch_idx = 0
        result = []
        for ev in events:
            if ev.pitch >= 0:
                new_p = (new_pitches[pitch_idx]
                         if pitch_idx < len(new_pitches) else ev.pitch)
                pitch_idx += 1
                result.append(MusicEvent(
                    pitch=new_p, duration_idx=ev.duration_idx,
                    beat_position=ev.beat_position, velocity_idx=ev.velocity_idx,
                    program=ev.program,
                ))
            else:
                result.append(ev)
        return result

    def _apply_chord_constrained_pitch(
        self,
        events: List[MusicEvent],
        time_signature: Tuple[int, int],
        key_pitch_classes: Optional[Set[int]] = None,
        key_strength: float = 0.0,
    ) -> List[MusicEvent]:
        """Override pitches using chord-constrained pitch chain generation."""
        if self.chord_chain is None or self.pitch_chain is None:
            return events

        dur_values = list(DURATION_CATEGORIES.values())
        bar_length_ql = (time_signature[0] / time_signature[1]) * 4.0

        seed_chord = ["C"]
        if self.chord_chain._transitions:
            seed_chord = [random.choice(list(self.chord_chain._transitions.keys()))[0]]
        total_ql = sum(dur_values[ev.duration_idx] for ev in events)
        num_bars = max(1, int(total_ql / bar_length_ql + 0.5))
        chord_seq = self.chord_chain.generate(seed_chord, num_bars)

        pitches = [ev.pitch for ev in events if ev.pitch >= 0]
        if len(pitches) < self.pitch_order:
            return events

        seed = pitches[:self.pitch_order]
        current_bar = 0
        bar_offset_ql = 0.0
        pitch_idx = self.pitch_order

        all_new_pitches = list(seed)
        for ev in events:
            dur_ql = dur_values[ev.duration_idx] if ev.duration_idx < len(dur_values) else 1.0
            bar_offset_ql += dur_ql
            while bar_offset_ql >= bar_length_ql and current_bar < num_bars - 1:
                current_bar += 1
                bar_offset_ql -= bar_length_ql

            if ev.pitch >= 0 and pitch_idx > 0:
                pitch_idx -= 1
                continue

            if ev.pitch >= 0:
                chord = chord_seq[min(current_bar, len(chord_seq) - 1)]
                allowed = chord_to_pitch_classes(chord)
                is_first_in_bar = (bar_offset_ql - dur_ql < 0.01)
                actual_allowed = allowed if is_first_in_bar else None
                nxt = self.pitch_chain.sample_next(
                    all_new_pitches[-self.pitch_order:],
                    allowed_pitch_classes=actual_allowed,
                    key_pitch_classes=key_pitch_classes,
                    key_strength=key_strength,
                )
                if nxt is None:
                    nxt = random.choice(pitches) if pitches else 60
                all_new_pitches.append(nxt)

        # Reconstruct events with new pitches
        result = []
        note_idx = 0
        for ev in events:
            if ev.pitch >= 0:
                new_p = (all_new_pitches[note_idx]
                         if note_idx < len(all_new_pitches) else ev.pitch)
                note_idx += 1
                result.append(MusicEvent(
                    pitch=new_p, duration_idx=ev.duration_idx,
                    beat_position=ev.beat_position, velocity_idx=ev.velocity_idx,
                    program=ev.program,
                ))
            else:
                result.append(ev)
        return result

    # ------------------------------------------------------------------
    # MIDI output
    # ------------------------------------------------------------------

    def save_midi(self, events: List[MusicEvent], output_path: Union[str, Path],
                  metadata: Optional[MidiMetadata] = None):
        """Save generated events as a MIDI file."""
        self.generator.write_midi(events, output_path, metadata or self._metadata)

    def save_abc(self, events: List[MusicEvent], output_path: Union[str, Path],
                 metadata: Optional[MidiMetadata] = None):
        """Save generated events as an ABC notation file."""
        self.generator.write_abc(events, output_path, metadata or self._metadata)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, events: List[MusicEvent], save_prefix: Optional[str] = None,
                  include_melody_dashboard: bool = False, **melody_kwargs: Any):
        """Generate standard visualization plots."""
        pitch_path = f"{save_prefix}_pitch.png" if save_prefix else None
        dur_path = f"{save_prefix}_duration.png" if save_prefix else None
        vel_path = f"{save_prefix}_velocity_heatmap.png" if save_prefix else None
        dash_path = f"{save_prefix}_dashboard.png" if save_prefix else None

        self.visualizer.plot_pitch_distribution(events, save_path=pitch_path)
        self.visualizer.plot_duration_distribution(events, save_path=dur_path)
        self.visualizer.plot_velocity_heatmap(events, save_path=vel_path)
        self.visualizer.plot_summary_dashboard(
            events,
            transition_matrix=self.markov_chain._chains.get(1),
            save_path=dash_path,
        )

        if include_melody_dashboard:
            try:
                from music_analyzer import MelodyAnalyzer
                melody_path = f"{save_prefix}_melody.png" if save_prefix else None
                MelodyAnalyzer.plot_melody_dashboard(
                    events, save_path=melody_path, **melody_kwargs
                )
            except ImportError as exc:
                print(f"Melody analyzer not available: {exc}")

    # ------------------------------------------------------------------
    # Persistence (pickle-based for simplicity)
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Save the entire system state to disk."""
        import pickle
        has_chord = self.chord_chain is not None
        has_pitch = self.pitch_chain is not None
        has_tonal = self.tonal_chain is not None
        state = {
            "version": 6,
            "markov_order": self.markov_order,
            "duration_order": self.duration_order,
            "chord_order": self.chord_order,
            "pitch_order": self.pitch_order,
            "tonal_order": self.tonal_order,
            "alpha": self.markov_chain.alpha,
            "min_unigram_count": self.markov_chain.min_unigram_count,
            "unigram_prior": dict(self.markov_chain._unigram_prior),
            "markov_chains": dict(self.markov_chain._chains),
            "markov_counts": dict(self.markov_chain._counts),
            "duration_transitions": dict(self.duration_chain._transitions),
            "chord_transitions": (
                dict(self.chord_chain._transitions) if has_chord else None
            ),
            "chord_trained": self._chord_trained,
            "pitch_transitions": (
                dict(self.pitch_chain._transitions) if has_pitch else None
            ),
            "pitch_trained": self._pitch_trained,
            "tonal_transitions": (
                dict(self.tonal_chain._transitions) if has_tonal else None
            ),
            "tonal_unigram": (
                dict(self.tonal_chain._unigram) if has_tonal else None
            ),
            "tonal_seed_windows": (
                list(self.tonal_chain._seed_windows) if has_tonal else None
            ),
            "tonal_trained": self._tonal_trained,
            "metadata": self._metadata,
            "trained": self._trained,
            "detected_key": self._detected_key,
            "has_chord_data": self._has_chord_data,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MusicMarkovSystem":
        """Load a saved system from disk."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)

        version = state.get("version", 1)
        if version >= 2:
            alpha = state.get("alpha", 0.0)
            min_unigram_count = state.get("min_unigram_count", 3)
        else:
            alpha = 0.0
            min_unigram_count = 3

        use_chord = version >= 3 and state.get("chord_transitions") is not None
        use_pitch = version >= 4 and state.get("pitch_transitions") is not None
        use_tonal = version >= 6 and state.get("tonal_transitions") is not None
        obj = cls(
            markov_order=state["markov_order"],
            duration_order=state["duration_order"],
            chord_order=state.get("chord_order", 2),
            pitch_order=state.get("pitch_order", 4),
            tonal_order=state.get("tonal_order", 3),
            use_chord_chain=use_chord,
            use_pitch_chain=use_pitch,
            use_tonal_chain=use_tonal,
        )
        obj.markov_chain.alpha = alpha
        obj.markov_chain.min_unigram_count = min_unigram_count
        if version >= 2 and "unigram_prior" in state:
            obj.markov_chain._unigram_prior = state["unigram_prior"]
        for o, trans in state["markov_chains"].items():
            obj.markov_chain._chains[o] = trans
        for o, cnt in state["markov_counts"].items():
            obj.markov_chain._counts[o] = cnt
        obj.duration_chain._transitions = state["duration_transitions"]
        if use_chord and obj.chord_chain is not None:
            obj.chord_chain._transitions = state["chord_transitions"]
            obj._chord_trained = state.get("chord_trained", True)
        if use_pitch and obj.pitch_chain is not None:
            obj.pitch_chain._transitions = state["pitch_transitions"]
            obj._pitch_trained = state.get("pitch_trained", True)
        if use_tonal and obj.tonal_chain is not None:
            obj.tonal_chain._transitions = state["tonal_transitions"]
            obj.tonal_chain._unigram = state.get("tonal_unigram", {}) or {}
            obj.tonal_chain._seed_windows = state.get("tonal_seed_windows", []) or []
            obj._tonal_trained = state.get("tonal_trained", True)
        obj._metadata = state["metadata"]
        obj._trained = state["trained"]
        if version >= 5:
            obj._detected_key = state.get("detected_key")
            obj._has_chord_data = state.get("has_chord_data", False)
        return obj


# ---------------------------------------------------------------------------
# Convenience functions for direct import
# ---------------------------------------------------------------------------

def train_and_generate(
    midi_dir: str,
    output_path: str = "generated_music.mid",
    num_events: int = 500,
    markov_order: int = 3,
    temperature: float = 1.0,
    visualize: bool = True,
) -> MusicMarkovSystem:
    """One-shot: train on a MIDI directory and generate a new piece.

    Args:
        midi_dir: Path to directory of MIDI files.
        output_path: Where to write the generated MIDI.
        num_events: Number of events to generate.
        markov_order: Order of the Markov chain.
        temperature: Sampling temperature.
        visualize: Whether to produce analysis plots.

    Returns:
        Trained MusicMarkovSystem instance.
    """
    system = MusicMarkovSystem(markov_order=markov_order)
    system.train(midi_dir)
    events = system.generate(num_events=num_events, temperature=temperature)
    system.save_midi(events, output_path)

    if visualize:
        prefix = str(Path(output_path).with_suffix(""))
        system.visualize(events, save_prefix=prefix)

    return system


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Music Markov Model — learn from MIDI/ABC and generate new music.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python music_markov_model.py ./music_files
  python music_markov_model.py ./music_files -o out.mid -n 1000 -t 1.2
  python music_markov_model.py ./music_files --order 4 --format abc -o out.abc
  python music_markov_model.py ./music_files --multi-voice --sections --key-constraint
        """,
    )
    parser.add_argument("-d", "--music-dir", default="../../datasets/",
                        help="Directory containing music files (.mid, .midi, .abc, .krn)")
    parser.add_argument("-o", "--output", default="../../output/highorder_music.mid",
                        help="Output file path")
    parser.add_argument("--format", choices=["midi", "abc"], default=None,
                        help="Output format.  Guessed from --output extension if omitted.")
    parser.add_argument("-n", "--num-events", type=int, default=500,
                        help="Number of events to generate")
    parser.add_argument("--order", type=int, default=2, help="Markov chain order")
    parser.add_argument("--duration-order", type=int, default=4,
                        help="Duration Markov chain order")
    parser.add_argument("--tonal-order", type=int, default=3,
                        help="Order for the joint scale-degree/duration Markov chain")
    parser.add_argument("-t", "--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization plots")
    parser.add_argument("--melody-dashboard", action="store_true",
                        help="Also generate the 5-panel melody analysis dashboard")
    parser.add_argument("--save-model", default=None,
                        help="Save trained model to a .pkl file")
    parser.add_argument("--load-model", default=None,
                        help="Load a pre-trained model from a .pkl file")

    # Phase 1: Quick wins
    parser.add_argument("--no-rests", action="store_true",
                        help="Disable phrase-boundary rests")
    parser.add_argument("--dur-temp", type=float, default=1.3,
                        help="Duration temperature (>1 = more rhythmic variety)")
    parser.add_argument("--no-velocity-shaping", action="store_true",
                        help="Disable phrase-level velocity envelopes")
    parser.add_argument("--phrase-length", type=int, default=16,
                        help="Phrase length in beats for rests and velocity shaping")

    # Phase 2: Structural
    parser.add_argument("--motifs", action="store_true",
                        help="Enable motif memory for pattern repetition")
    parser.add_argument("--sections", action="store_true",
                        help="Generate in A/B/A-prime sectional form")
    parser.add_argument("--key-constraint", action="store_true",
                        help="Constrain pitches to detected key")
    parser.add_argument("--key-strength", type=float, default=0.5,
                        help="Key constraint strength (0-1)")
    parser.add_argument("--register-arc", action="store_true",
                        help="Shape pitch register in an arc across the piece")

    # Phase 3: Harmonic depth
    parser.add_argument("--no-chord-constraint", action="store_true",
                        help="Disable automatic chord-constrained pitch generation")
    parser.add_argument("--multi-voice", action="store_true",
                        help="Generate two-voice piece with bass line")
    parser.add_argument("--no-cadence", action="store_true",
                        help="Disable cadence resolution at ending")

    # Tonal degree generation
    parser.add_argument("--tonal-degree-chain", action="store_true",
                        help="Use the joint scale-degree + duration + beat Markov chain")
    parser.add_argument("--target-key", default=None,
                        help="Output key for tonal generation, e.g. C, G, D, Am, A minor")
    parser.add_argument("--random-key", action="store_true",
                        help="Randomly choose an output key for tonal generation")
    parser.add_argument("--tonal-strictness", type=float, default=0.8,
                        help="Penalty for chromatic scale degrees in tonal generation (0-1)")
    parser.add_argument("--repeat-penalty", type=float, default=0.35,
                        help="Penalty multiplier for repeated same tonal pitch states")

    args = parser.parse_args()

    # Add timestamp to output filename and resolve relative paths
    from datetime import datetime
    out_path = Path(args.output).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_path.parent / f"{out_path.stem}_{timestamp}{out_path.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(out_path)

    # Resolve output format
    if args.format is not None:
        out_fmt = args.format
    else:
        ext = Path(args.output).suffix.lower()
        if ext in (".abc",):
            out_fmt = "abc"
        else:
            out_fmt = "midi"

    if args.load_model:
        print(f"Loading model from {args.load_model} ...")
        system = MusicMarkovSystem.load(args.load_model)
        print(f"Detected key: {system._detected_key or 'unknown'}")
        print(f"Chord data available: {system._has_chord_data}")
        print(f"Chord chain trained: {system._chord_trained}")
        print(f"Pitch chain trained: {system._pitch_trained}")
        print(f"Tonal chain trained: {system._tonal_trained}")
    else:
        system = MusicMarkovSystem(
            markov_order=args.order,
            duration_order=args.duration_order,
            tonal_order=args.tonal_order,
            use_tonal_chain=args.tonal_degree_chain,
            random_seed=args.seed,
        )
        print(f"Training on music files in {args.music_dir} ...")
        system.train(args.music_dir)
        print(f"Detected key: {system._detected_key or 'unknown'}")
        print(f"Chord chain trained: {system._chord_trained}")
        print(f"Pitch chain trained: {system._pitch_trained}")
        print(f"Tonal chain trained: {system._tonal_trained}")

    # Determine generation mode — auto-enable chord/pitch when trained, unless opted out
    generation_kwargs = dict(
        num_events=args.num_events,
        temperature=args.temperature,
        use_duration_chain=True,
        use_bar_constraint=False,
        use_chord_constraint=False if args.no_chord_constraint else None,
        use_pitch_chain=False,
        phrase_length_beats=args.phrase_length,
        insert_phrase_rests=not args.no_rests,
        duration_temperature=args.dur_temp,
        velocity_shaping=not args.no_velocity_shaping,
        motif_repetition=args.motifs,
        key_constraint=args.key_constraint,
        key_strength=args.key_strength,
        register_arc=args.register_arc,
        cadence_ending=not args.no_cadence,
        use_tonal_chain=args.tonal_degree_chain,
        target_key=args.target_key,
        random_key=args.random_key,
        tonal_strictness=args.tonal_strictness,
        repeat_penalty=args.repeat_penalty,
    )

    print(f"Generating {args.num_events} events (temperature={args.temperature}) ...")
    if args.sections:
        print("Using A/B/A' sectional form ...")
        events = system.generate_sections(**generation_kwargs)
    elif args.multi_voice:
        print("Using multi-voice generation ...")
        events = system.generate_multi_voice(**generation_kwargs)
    else:
        events = system.generate(**generation_kwargs)

    if out_fmt == "abc":
        print(f"Writing ABC to {args.output} ...")
        system.save_abc(events, args.output)
    else:
        print(f"Writing MIDI to {args.output} ...")
        system.save_midi(events, args.output)

    if not args.no_viz:
        prefix = str(Path(args.output).with_suffix(""))
        print(f"Saving visualizations with prefix '{prefix}' ...")
        system.visualize(events, save_prefix=prefix,
                         include_melody_dashboard=args.melody_dashboard)

    if args.save_model:
        print(f"Saving model to {args.save_model} ...")
        system.save(args.save_model)

    print("Done.")
