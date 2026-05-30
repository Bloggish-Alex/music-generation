#!/usr/bin/env python3
"""
Measure Clustering System — KMeans-based classification of musical measures.

Reads MIDI/ABC/KRN files, extracts per-measure feature vectors (note density,
duration statistics, offbeat ratio, syncopation, entropy), clusters them with
KMeans, and provides APIs for centroid retrieval and new-measure classification.

Usage (library)::

    from measure_clustering import MeasureExtractor, MeasureClusterer

    extractor = MeasureExtractor()
    vectors = extractor.extract_all("path/to/music/dir")
    clusterer = MeasureClusterer().fit(vectors, n_clusters=8)
    label = clusterer.predict(some_vector)

Usage (CLI)::

    python measure_clustering.py --music-dir ../../datasets/corelli --n-clusters 5 --viz
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from music21 import converter, instrument, meter, note as m21note

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger("measure_clustering")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DURATION_CATEGORIES: Dict[str, float] = {
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

# Duration bins for rhythmic entropy (edges in quarterLength)
_DURATION_BIN_EDGES: np.ndarray = np.array(
    [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, float("inf")]
)

SHORT_NOTE_THRESHOLD: float = 0.5  # quarterLength < 0.5 = "short" (shorter than eighth)

# Canonical shape exemplars (8-point, normalised [0, 1]) for curve comparison
_SHAPE_EXEMPLARS: Dict[str, np.ndarray] = {
    "rising":  np.linspace(0.0, 1.0, 8),
    "falling": np.linspace(1.0, 0.0, 8),
    "arch":    np.sin(np.pi * np.linspace(0.0, 1.0, 8)),
    "valley":  1.0 - np.sin(np.pi * np.linspace(0.0, 1.0, 8)),
    "flat":    np.full(8, 0.5),
}
_SHAPE_NAMES: Tuple[str, ...] = ("rising", "falling", "arch", "valley", "flat")

# -- Key transposition constants ------------------------------------------------

_MAJOR_SCALE: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE: Tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)

_ROOT_TO_PC: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MeasureInfo:
    """Raw data for one musical measure (bar)."""

    notes: List[Dict[str, Any]] = field(default_factory=list)
    # Each dict: {"pitch": int, "quarterLength": float,
    #              "onset_in_measure": float, "duration": float}

    beats: float = 4.0           # measure length in quarterLength
    time_signature: str = "4/4"
    file_path: str = ""
    measure_index: int = 0
    total_measures: int = 0     # number of measures in the source file


@dataclass
class MeasureVector:
    """19-dimensional feature vector for one measure."""

    note_density: float = 0.0
    mean_duration: float = 0.0
    duration_variance: float = 0.0
    short_note_ratio: float = 0.0
    silence_ratio: float = 0.0
    offbeat_ratio: float = 0.0
    syncopation_score: float = 0.0
    entropy: float = 0.0
    relative_position: float = 0.0

    # Pitch shape distances to 5 canonical exemplars (0 = identical, 1 = opposite)
    pitch_shape_rising: float = 0.0
    pitch_shape_falling: float = 0.0
    pitch_shape_arch: float = 0.0     # rise then fall
    pitch_shape_valley: float = 0.0   # fall then rise
    pitch_shape_flat: float = 0.0     # all same

    # Duration shape distances to 5 canonical exemplars
    dur_shape_rising: float = 0.0
    dur_shape_falling: float = 0.0
    dur_shape_arch: float = 0.0
    dur_shape_valley: float = 0.0
    dur_shape_flat: float = 0.0

    measure_length: float = 0.0  # measure.beats / 16, normalised bar length

    file_path: str = ""
    measure_index: int = 0
    time_signature: str = "4/4"
    cluster_label: int = -1

    def as_array(self) -> np.ndarray:
        """Return the 20 features as a numpy array (float64).

        Order: 10 rhythm features then 10 contour features (matches
        the concatenation order in MeasureExtractor.vectorize).
        """
        return np.array([
            self.note_density,
            self.mean_duration,
            self.duration_variance,
            self.short_note_ratio,
            self.silence_ratio,
            self.offbeat_ratio,
            self.syncopation_score,
            self.entropy,
            self.relative_position,
            self.measure_length,
            self.pitch_shape_rising,
            self.pitch_shape_falling,
            self.pitch_shape_arch,
            self.pitch_shape_valley,
            self.pitch_shape_flat,
            self.dur_shape_rising,
            self.dur_shape_falling,
            self.dur_shape_arch,
            self.dur_shape_valley,
            self.dur_shape_flat,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray, file_path: str = "",
                   measure_index: int = 0, time_signature: str = "4/4",
                   cluster_label: int = -1) -> "MeasureVector":
        """Construct from a numpy array (for centroid reconstruction)."""
        n = len(arr)

        def _f(i: int) -> float:
            return float(arr[i]) if i < n else 0.0

        return cls(
            note_density=_f(0),
            mean_duration=_f(1),
            duration_variance=_f(2),
            short_note_ratio=_f(3),
            silence_ratio=_f(4),
            offbeat_ratio=_f(5),
            syncopation_score=_f(6),
            entropy=_f(7),
            relative_position=_f(8),
            measure_length=_f(9),
            pitch_shape_rising=_f(10),
            pitch_shape_falling=_f(11),
            pitch_shape_arch=_f(12),
            pitch_shape_valley=_f(13),
            pitch_shape_flat=_f(14),
            dur_shape_rising=_f(15),
            dur_shape_falling=_f(16),
            dur_shape_arch=_f(17),
            dur_shape_valley=_f(18),
            dur_shape_flat=_f(19),
            file_path=file_path,
            measure_index=measure_index,
            time_signature=time_signature,
            cluster_label=cluster_label,
        )


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


class RhythmFeatureExtractor:
    """Extract rhythmic / structural features from a single measure.

    Features (10): note_density, mean_duration, duration_variance,
    short_note_ratio, silence_ratio, offbeat_ratio, syncopation_score,
    entropy, relative_position, measure_length.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "note_density",
        "mean_duration",
        "duration_variance",
        "short_note_ratio",
        "silence_ratio",
        "offbeat_ratio",
        "syncopation_score",
        "entropy",
        "relative_position",
        "measure_length",
    )

    def __init__(self) -> None:
        self._dur_bins = _DURATION_BIN_EDGES

    def features(self, measure: MeasureInfo) -> Dict[str, float]:
        """Return a dict keyed by feature name for *measure*."""
        notes = measure.notes
        n = len(notes)

        denom = max(1, measure.total_measures - 1)
        rel_pos = measure.measure_index / denom

        if n == 0:
            return {
                "note_density": 0.0,
                "mean_duration": 0.0,
                "duration_variance": 0.0,
                "short_note_ratio": 0.0,
                "silence_ratio": 0.0,
                "offbeat_ratio": 0.0,
                "syncopation_score": 0.0,
                "entropy": 0.0,
                "relative_position": rel_pos,
                "measure_length": measure.beats / 16.0,
            }

        durations = np.array([nd["quarterLength"] for nd in notes], dtype=np.float64)
        onsets = np.array([nd["onset_in_measure"] for nd in notes], dtype=np.float64)

        density = n / measure.beats if measure.beats > 0 else 0.0
        mean_dur = float(np.mean(durations))
        dur_var = float(np.var(durations))
        short_ratio = float(np.sum(durations < SHORT_NOTE_THRESHOLD) / n)
        total_sounding = float(np.sum(durations))
        silence = 1.0 - (total_sounding / measure.beats) if measure.beats > 0 else 0.0
        silence = max(0.0, min(1.0, silence))
        offbeat = float(np.sum(onsets % 1.0 > 1e-9) / n)
        sync_count = self._count_syncopations(notes)
        ent = self._rhythmic_entropy(durations)

        return {
            "note_density": density,
            "mean_duration": mean_dur,
            "duration_variance": dur_var,
            "short_note_ratio": short_ratio,
            "silence_ratio": silence,
            "offbeat_ratio": offbeat,
            "syncopation_score": sync_count / n,
            "entropy": ent,
            "relative_position": rel_pos,
            "measure_length": measure.beats * 10,
        }

    def vector(self, feats: Dict[str, float]) -> np.ndarray:
        """Return the feature values in canonical order as a numpy array."""
        return np.array([feats[n] for n in self.FEATURE_NAMES], dtype=np.float64)

    @staticmethod
    def _is_onbeat(onset: float) -> bool:
        return abs(onset % 1.0) < 1e-9

    @staticmethod
    def _count_syncopations(notes: List[Dict[str, Any]]) -> int:
        sorted_notes = sorted(notes, key=lambda nd: nd["onset_in_measure"])
        count = 0
        prev_onbeat_dur = 0.0
        for nd in sorted_notes:
            onset = nd["onset_in_measure"]
            dur = nd["quarterLength"]
            if RhythmFeatureExtractor._is_onbeat(onset):
                prev_onbeat_dur = dur
            elif dur > prev_onbeat_dur and prev_onbeat_dur > 0:
                count += 1
        return count

    def _rhythmic_entropy(self, durations: np.ndarray) -> float:
        hist, _ = np.histogram(durations, bins=self._dur_bins)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return 0.0
        probs = hist / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))


class ContourFeatureExtractor:
    """Extract contour / shape features from a single measure.

    Features (10): distances from the measure's pitch and duration curves
    to 5 canonical shape exemplars (rising, falling, arch, valley, flat).
    Lower = more similar to that shape (0 = identical, 1 = opposite).
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "pitch_shape_rising",
        "pitch_shape_falling",
        "pitch_shape_arch",
        "pitch_shape_valley",
        "pitch_shape_flat",
        "dur_shape_rising",
        "dur_shape_falling",
        "dur_shape_arch",
        "dur_shape_valley",
        "dur_shape_flat",
    )

    def features(self, measure: MeasureInfo) -> Dict[str, float]:
        """Return a dict keyed by feature name for *measure*."""
        notes = sorted(measure.notes, key=lambda nd: nd["onset_in_measure"])
        pitches = np.array([nd["pitch"] for nd in notes], dtype=np.float64)
        durs = np.array([nd["quarterLength"] for nd in notes], dtype=np.float64)

        p = self._shape_distances(pitches)
        d = self._shape_distances(durs)

        return {
            "pitch_shape_rising":  p[0],
            "pitch_shape_falling": p[1],
            "pitch_shape_arch":    p[2],
            "pitch_shape_valley":  p[3],
            "pitch_shape_flat":    p[4],
            "dur_shape_rising":    d[0],
            "dur_shape_falling":   d[1],
            "dur_shape_arch":      d[2],
            "dur_shape_valley":    d[3],
            "dur_shape_flat":      d[4],
        }

    def vector(self, feats: Dict[str, float]) -> np.ndarray:
        """Return the feature values in canonical order as a numpy array."""
        return np.array([feats[n] for n in self.FEATURE_NAMES], dtype=np.float64)

    @staticmethod
    def _shape_distances(values: np.ndarray) -> np.ndarray:
        """Resample *values* to 8 points, normalise, return (1−r)/2 to each exemplar."""
        result = np.full(5, 0.5, dtype=np.float64)
        n = len(values)
        if n < 2:
            return result

        indices = np.linspace(0, n - 1, 8)
        lo = np.clip(np.floor(indices).astype(int), 0, n - 1)
        hi = np.clip(np.ceil(indices).astype(int), 0, n - 1)
        frac = indices - lo
        curve = values[lo] * (1.0 - frac) + values[hi] * frac

        c_min, c_max = curve.min(), curve.max()
        span = c_max - c_min
        curve = (curve - c_min) / span if span > 1e-9 else np.zeros(8)

        with np.errstate(invalid="ignore"):
            for i, name in enumerate(_SHAPE_NAMES):
                exemplar = _SHAPE_EXEMPLARS[name]
                r = np.corrcoef(curve, exemplar)[0, 1]
                result[i] = (1.0 - (0.0 if np.isnan(r) else r)) / 2.0

        return result


# Combined feature names (order matches MeasureVector.as_array)
FEATURE_NAMES: Tuple[str, ...] = (
    RhythmFeatureExtractor.FEATURE_NAMES + ContourFeatureExtractor.FEATURE_NAMES
)

# ---------------------------------------------------------------------------
# Measure extractor
# ---------------------------------------------------------------------------


class MeasureExtractor:
    """Parse music files and extract per-measure feature vectors.

    Args:
        transpose_to_common_key: When True (default), detect each file's key
            and transpose all pitches to C major so that measures from
            different source keys are harmonically compatible.
    """

    def __init__(self, transpose_to_common_key: bool = True) -> None:
        self._rhythm = RhythmFeatureExtractor()
        self._contour = ContourFeatureExtractor()
        self.transpose_to_common_key = transpose_to_common_key

    # -- file-level parsing --------------------------------------------------

    def extract(self, file_path: Union[str, Path]) -> List[MeasureInfo]:
        """Parse a single music file into a list of MeasureInfo objects.

        Uses music21's stream.Measure for robust bar detection — handles
        pickup bars, time-signature changes, and incomplete measures.
        Chords are expanded: each pitch becomes a separate note entry.
        """
        from music21 import stream

        file_path = Path(file_path)
        score = converter.parse(str(file_path))

        if hasattr(score, "scores"):
            scores = score.scores
            if not scores:
                return []
            score = scores[0]

        if not score.parts:
            return []
        part = score.parts[0]

        # Use music21's own measure detection
        measures = list(part.getElementsByClass(stream.Measure))
        if not measures:
            return []

        result: List[MeasureInfo] = []
        for bar_idx, measure in enumerate(measures):
            bar_offset = float(measure.offset)
            bar_length = float(measure.duration.quarterLength)
            if bar_length <= 0:
                bar_length = 4.0

            # Time signature for this measure
            ts_objs = list(measure.getElementsByClass(meter.TimeSignature))
            ts_str = f"{ts_objs[0].numerator}/{ts_objs[0].denominator}" if ts_objs else "4/4"

            notes_data: List[Dict[str, Any]] = []
            for el in measure.flatten().notesAndRests:
                onset_in_measure = float(el.offset) - bar_offset
                if el.isRest:
                    continue
                if el.isNote:
                    midi = el.pitch.midi if el.pitch else 60
                    notes_data.append({
                        "pitch": int(midi),
                        "quarterLength": float(el.quarterLength),
                        "onset_in_measure": onset_in_measure,
                    })
                elif el.isChord:
                    for p in el.pitches:
                        notes_data.append({
                            "pitch": p.midi,
                            "quarterLength": float(el.quarterLength),
                            "onset_in_measure": onset_in_measure,
                        })

            if not notes_data:
                continue

            result.append(MeasureInfo(
                notes=notes_data,
                beats=bar_length,
                time_signature=ts_str,
                file_path=str(file_path),
                measure_index=bar_idx,
            ))

        # Stamp total measure count on each MeasureInfo
        total = len(result)
        for mi in result:
            mi.total_measures = total

        # Detect key via music21 and transpose to C major
        if self.transpose_to_common_key and result:
            try:
                key_obj = score.analyze("key")
                tonic_pc = key_obj.tonic.midi % 12
                mode = key_obj.mode  # "major" or "minor"
            except Exception:
                tonic_pc, mode = 0, "major"  # fallback

            if tonic_pc != 0 or mode != "major":
                for mi in result:
                    for nd in mi.notes:
                        nd["pitch"] = self._transpose_pitch(
                            nd["pitch"], tonic_pc, mode, 0, "major",
                        )

        return result

    # -- key transposition ---------------------------------------------------

    @staticmethod
    def parse_key(key_str: str) -> Tuple[int, str]:
        """Parse a key string like ``"C"``, ``"G"``, ``"Am"`` into (tonic_pc, mode)."""
        text = key_str.strip()
        if not text:
            return 0, "major"
        mode = "minor" if text.endswith("m") else "major"
        root = text[:-1] if mode == "minor" else text
        tonic_pc = _ROOT_TO_PC.get(root)
        if tonic_pc is None:
            raise ValueError(f"Unrecognised key: {key_str!r}")
        return tonic_pc, mode

    @staticmethod
    def transpose_notes(
        notes: List[Any],
        from_tonic: int,
        from_mode: str,
        to_tonic: int,
        to_mode: str,
    ) -> List[Dict[str, Any]]:
        """Transpose a list of NoteEvent-like objects between keys.

        Returns new dicts ``{pitch, duration_ql, velocity, beat_offset}``.
        Notes with pitch < 0 (rests) pass through unchanged.
        """
        result: List[Dict[str, Any]] = []
        for n in notes:
            pitch = getattr(n, "pitch", n[0] if isinstance(n, tuple) else -1)
            if pitch >= 0:
                pitch = MeasureExtractor._transpose_pitch(
                    pitch, from_tonic, from_mode, to_tonic, to_mode,
                )
            result.append({
                "pitch": pitch,
                "duration_ql": getattr(n, "duration_ql", 0.0),
                "velocity": getattr(n, "velocity", 80),
                "beat_offset": getattr(n, "beat_offset", 0.0),
            })
        return result

    @staticmethod
    def _transpose_pitch(
        pitch: int,
        from_tonic: int,
        from_mode: str,
        to_tonic: int,
        to_mode: str,
    ) -> int:
        """Transpose a MIDI pitch between keys via scale-degree space.

        Converts *pitch* to a key-relative (degree, accidental, octave) in
        the source key, then maps it back to MIDI pitch in the target key.
        """
        from_scale = _MAJOR_SCALE if from_mode == "major" else _MINOR_SCALE
        to_scale = _MAJOR_SCALE if to_mode == "major" else _MINOR_SCALE

        pc = pitch % 12
        octave = pitch // 12
        rel_pc = (pc - from_tonic) % 12

        # Find nearest diatonic degree in source key
        best_deg, best_acc, best_dist = 1, 0, 99
        for deg, base in enumerate(from_scale, start=1):
            acc = (rel_pc - base + 6) % 12 - 6  # signed delta [-6, 6]
            dist = abs(acc)
            if dist < best_dist or (dist == best_dist and abs(acc) < abs(best_acc)):
                best_deg, best_acc, best_dist = deg, acc, dist

        # Map to target key: same degree, same accidental, same octave
        base_pc = to_scale[(best_deg - 1) % 7]
        new_pitch = octave * 12 + to_tonic + base_pc + best_acc
        return max(0, min(127, int(new_pitch)))

    # -- vectorization -------------------------------------------------------

    def vectorize(self, measure: MeasureInfo) -> MeasureVector:
        """Compute the full feature vector for a single measure."""
        arr = np.concatenate([
            self._rhythm.vector(self._rhythm.features(measure)),
            # self._contour.vector(self._contour.features(measure)),
            []
        ])
        return MeasureVector.from_array(
            arr,
            file_path=measure.file_path,
            measure_index=measure.measure_index,
            time_signature=measure.time_signature,
        )

    # -- batch extraction ----------------------------------------------------

    def extract_all(
        self,
        music_dir: Union[str, Path],
        file_patterns: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[MeasureVector]:
        """Walk *music_dir* recursively, parse every matching file, and return
        a flat list of MeasureVectors.

        Args:
            music_dir: Root directory containing music files.
            file_patterns: Glob pattern(s).  Defaults to ``["*.mid","*.midi","*.abc","*.krn"]``.

        Returns:
            All extracted measure vectors.
        """
        if file_patterns is None:
            file_patterns = ["*.mid", "*.midi", "*.abc", "*.krn"]
        elif isinstance(file_patterns, str):
            file_patterns = [file_patterns]

        music_dir = Path(music_dir)
        music_paths: List[Path] = []
        for pat in file_patterns:
            music_paths.extend(sorted(music_dir.rglob(pat)))
        music_paths = sorted(set(music_paths))

        if not music_paths:
            raise FileNotFoundError(
                f"No music files matching {list(file_patterns)} found in {music_dir}"
            )

        all_vectors: List[MeasureVector] = []
        success = 0
        for mp in music_paths:
            try:
                measures = self.extract(mp)
                for m in measures:
                    all_vectors.append(self.vectorize(m))
                success += 1
                log.info("Parsed %s: %d measures", mp.name, len(measures))
            except Exception as exc:
                log.warning("Skipping %s: %s", mp, exc)

        log.info(
            "Extracted %d measure vectors from %d/%d files.",
            len(all_vectors), success, len(music_paths),
        )
        if not all_vectors:
            raise RuntimeError(
                f"No valid measures extracted from {len(music_paths)} files "
                f"in {music_dir}"
            )
        return all_vectors


# ---------------------------------------------------------------------------
# Measure clusterer (KMeans)
# ---------------------------------------------------------------------------


class MeasureClusterer:
    """KMeans clustering over MeasureVector features.

    Usage::

        clusterer = MeasureClusterer()
        clusterer.fit(vectors, n_clusters=8)
        centroids = clusterer.centroids         # np.ndarray (n_clusters × 8)
        label     = clusterer.predict(new_vec)  # int
    """

    def __init__(self) -> None:
        self._scaler: Any = None        # sklearn StandardScaler
        self._kmeans: Any = None        # sklearn KMeans
        self._centroids_raw: Optional[np.ndarray] = None  # in original space
        self._inertia: float = 0.0
        self._labels: Optional[np.ndarray] = None
        self._cluster_measures: Dict[int, List[MeasureInfo]] = {}
        self._feature_weights: Optional[np.ndarray] = None

    # -- properties ----------------------------------------------------------

    @property
    def centroids(self) -> Optional[np.ndarray]:
        """Cluster centroids in the original (unscaled) feature space."""
        return self._centroids_raw

    @property
    def inertia(self) -> float:
        """Within-cluster sum-of-squares (KMeans inertia)."""
        return self._inertia

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Cluster labels from the last fit call."""
        return self._labels

    # -- fit -----------------------------------------------------------------

    def fit(
        self,
        vectors: List[MeasureVector],
        n_clusters: int = 8,
        random_seed: int = 42,
        feature_weights: Optional[Sequence[float]] = None,
    ) -> "MeasureClusterer":
        """Normalize vectors with StandardScaler, fit KMeans, and store model.

        Args:
            vectors: Training data.
            n_clusters: Number of KMeans clusters.
            random_seed: Reproducibility seed.
            feature_weights: Per-feature multiplier applied after scaling
                (default 1.0 for all).  Use < 1.0 to dampen a feature
                (e.g. 0.15 for ``relative_position``), > 1.0 to amplify.

        Returns:
            self (for chaining).
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = np.stack([v.as_array() for v in vectors], axis=0)
        log.info("Feature matrix shape: %s", X.shape)

        n_feat = X.shape[1]
        if feature_weights is None:
            weights = np.ones(n_feat, dtype=np.float64)
        else:
            weights = np.asarray(feature_weights, dtype=np.float64)
            if len(weights) != n_feat:
                raise ValueError(
                    f"feature_weights length {len(weights)} != {n_feat} features"
                )
        self._feature_weights = weights

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_weighted = X_scaled * weights

        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init=10
        )
        self._labels = self._kmeans.fit_predict(X_weighted)
        self._inertia = float(self._kmeans.inertia_)

        # Recover centroids in original (unscaled, unweighted) space
        centroids_weighted = self._kmeans.cluster_centers_
        centroids_scaled = centroids_weighted / weights
        self._centroids_raw = self._scaler.inverse_transform(centroids_scaled)

        # Tag vectors with their labels
        for v, lbl in zip(vectors, self._labels):
            v.cluster_label = int(lbl)

        log.info(
            "KMeans fit: n_clusters=%d, inertia=%.3f, samples=%d, weights=%s",
            n_clusters, self._inertia, len(vectors),
            ", ".join(f"{w:.2f}" for w in weights),
        )
        return self

    # -- predict -------------------------------------------------------------

    def predict(self, vector: MeasureVector) -> int:
        """Classify a single MeasureVector. Returns cluster label (0..n-1)."""
        self._require_fit()
        X = vector.as_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        weights = getattr(self, "_feature_weights", None)
        if weights is not None:
            X_scaled = X_scaled * weights
        return int(self._kmeans.predict(X_scaled)[0])

    def predict_many(self, vectors: List[MeasureVector]) -> np.ndarray:
        """Classify multiple MeasureVectors. Returns array of cluster labels."""
        self._require_fit()
        X = np.stack([v.as_array() for v in vectors], axis=0)
        X_scaled = self._scaler.transform(X)
        weights = getattr(self, "_feature_weights", None)
        if weights is not None:
            X_scaled = X_scaled * weights
        return self._kmeans.predict(X_scaled)

    # -- measure storage -----------------------------------------------------

    def store_measures(
        self,
        measure_infos: List[MeasureInfo],
        labels: Union[np.ndarray, List[int]],
    ) -> None:
        """Store MeasureInfo objects grouped by cluster label.

        Args:
            measure_infos: MeasureInfo objects from extraction.
            labels: Cluster label for each measure (same length as *measure_infos*).
        """
        labels_arr = np.asarray(labels)
        if len(measure_infos) != len(labels_arr):
            raise ValueError(
                f"Length mismatch: {len(measure_infos)} measures vs "
                f"{len(labels_arr)} labels"
            )
        self._cluster_measures.clear()
        for mi, lbl in zip(measure_infos, labels_arr):
            c = int(lbl)
            self._cluster_measures.setdefault(c, []).append(mi)

    def get_cluster_measures(self, cluster_label: int) -> List[MeasureInfo]:
        """Return all MeasureInfo objects assigned to *cluster_label*."""
        self._require_fit()
        return list(self._cluster_measures.get(cluster_label, []))

    def sample_measure(
        self, cluster_label: int, seed: Optional[int] = None
    ) -> Optional[MeasureInfo]:
        """Randomly sample one MeasureInfo from *cluster_label*.

        Returns None if the cluster has no stored measures.
        """
        measures = self._cluster_measures.get(cluster_label, [])
        if not measures:
            return None
        rng = np.random.RandomState(seed)
        return measures[int(rng.randint(0, len(measures)))]

    def print_sample_measures(self, seed: int = 42) -> None:
        """Print sample measures from a randomly chosen cluster.

        Picks a cluster with > 3 stored measures, prints 5 random measures'
        notes as ``(pitch, quarterLength, onset_in_measure)`` tuples.
        """
        eligible = [
            (c, measures) for c, measures in self._cluster_measures.items()
            if len(measures) > 3
        ]
        if not eligible:
            print("\nNo stored measures to display (train with updated MusicModel).")
            return

        rng = np.random.RandomState(seed)
        c, measures = eligible[rng.randint(0, len(eligible))]
        n_show = min(5, len(measures))
        indices = rng.choice(len(measures), n_show, replace=False)

        extractor = MeasureExtractor()

        print(f"\n{'=' * 72}")
        print(f"SAMPLE MEASURES — Cluster {c} ({len(measures)} stored)")
        print(f"{'=' * 72}")

        for i, idx in enumerate(indices):
            mi = measures[int(idx)]
            vec = extractor.vectorize(mi)
            source = Path(mi.file_path).name if mi.file_path else "?"
            vec_str = ", ".join(f"{v:.2f}" for v in vec.as_array())
            print(f"\n  Measure {i + 1}  [file={source}, bar={mi.measure_index}, "
                  f"ts={mi.time_signature}, beats={mi.beats:.1f}], "
                  f"vector: [{vec_str}]")
            for nd in mi.notes:
                print(f"    (pitch={nd['pitch']:>4}, dur={nd['quarterLength']:.2f}, "
                      f"onset={nd.get('onset_in_measure', 0):.2f})")

        print()

    # -- stats ---------------------------------------------------------------

    def cluster_stats(self, vectors: List[MeasureVector]) -> List[Dict[str, Any]]:
        """Return per-cluster statistics: size, centroid, std per feature.

        Returns:
            List of dicts with keys ``cluster``, ``size``, ``centroid``
            (MeasureVector), ``std`` (np.ndarray), ``feature_means`` (dict).
        """
        self._require_fit()
        labels_arr = self._labels
        if labels_arr is None:
            return []
        X = np.stack([v.as_array() for v in vectors], axis=0)
        n_clusters = self._centroids_raw.shape[0] if self._centroids_raw is not None else 0
        stats = []
        for c in range(n_clusters):
            mask = labels_arr == c
            cluster_x = X[mask]
            centroid_vec = MeasureVector.from_array(
                self._centroids_raw[c],
                cluster_label=c,
            )
            n_feat = self._centroids_raw.shape[1]
            std_arr = np.std(cluster_x, axis=0) if len(cluster_x) > 0 else np.zeros(n_feat)
            feature_means = {
                FEATURE_NAMES[i]: float(self._centroids_raw[c, i]) for i in range(n_feat)
            }
            stats.append({
                "cluster": c,
                "size": int(mask.sum()),
                "centroid": centroid_vec,
                "std": std_arr,
                "feature_means": feature_means,
            })
        return stats

    def _require_fit(self) -> None:
        if self._kmeans is None:
            raise RuntimeError("MeasureClusterer is not fitted. Call .fit() first.")

    # -- persistence ---------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted clusterer to disk (pickle)."""
        self._require_fit()
        state = {
            "scaler": self._scaler,
            "kmeans": self._kmeans,
            "centroids_raw": self._centroids_raw,
            "inertia": self._inertia,
            "labels": self._labels,
            "cluster_measures": self._cluster_measures,
            "feature_weights": getattr(self, "_feature_weights", None),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved clusterer to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MeasureClusterer":
        """Load a fitted clusterer from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._scaler = state["scaler"]
        obj._kmeans = state["kmeans"]
        obj._centroids_raw = state["centroids_raw"]
        obj._inertia = state.get("inertia", 0.0)
        obj._labels = state.get("labels")
        obj._cluster_measures = state.get("cluster_measures", {})
        obj._feature_weights = state.get("feature_weights")
        return obj


# ---------------------------------------------------------------------------
# Measure classifier (thin wrapper)
# ---------------------------------------------------------------------------


class MeasureClassifier:
    """Convenience wrapper for classifying measures with a fitted clusterer.

    Usage::

        classifier = MeasureClassifier(clusterer)
        label = classifier.classify(measure_vector)
    """

    def __init__(self, clusterer: MeasureClusterer) -> None:
        self._clusterer = clusterer

    def classify(self, vector: MeasureVector) -> int:
        """Return the cluster label for a single measure vector."""
        return self._clusterer.predict(vector)

    def classify_many(self, vectors: List[MeasureVector]) -> np.ndarray:
        """Return cluster labels for multiple measure vectors."""
        return self._clusterer.predict_many(vectors)

    @property
    def centroids(self) -> Optional[np.ndarray]:
        return self._clusterer.centroids

    @property
    def clusterer(self) -> MeasureClusterer:
        return self._clusterer


# ---------------------------------------------------------------------------
# Shared classification pipeline
# ---------------------------------------------------------------------------


def classify_files(
    music_dir: Union[str, Path],
    clusterer: MeasureClusterer,
    extractor: Optional[MeasureExtractor] = None,
    file_patterns: Optional[Union[str, Sequence[str]]] = None,
    return_measures: bool = False,
) -> Union[
    List[List[int]],
    Tuple[List[List[int]], List[MeasureInfo], List[int]],
]:
    """Extract measures from all music files, classify, return per-file labels.

    File boundaries are preserved — each inner list corresponds to one file's
    ordered cluster labels.

    Args:
        music_dir: Root directory containing music files.
        clusterer: Fitted MeasureClusterer.
        extractor: Optional pre-created MeasureExtractor. Created if None.
        file_patterns: Glob patterns. Defaults to standard music formats.
        return_measures: When True, also returns a flat list of all
            MeasureInfo objects with their cluster labels set.

    Returns:
        List of per-file label sequences. Files with <2 measures are skipped.
        If *return_measures* is True, returns ``(file_labels, all_measures)``.
    """
    if extractor is None:
        extractor = MeasureExtractor()

    if file_patterns is None:
        file_patterns = ["*.mid", "*.midi", "*.abc", "*.krn"]
    elif isinstance(file_patterns, str):
        file_patterns = [file_patterns]

    music_dir = Path(music_dir)
    file_paths: List[Path] = []
    for pat in file_patterns:
        file_paths.extend(sorted(music_dir.rglob(pat)))
    file_paths = sorted(set(file_paths))

    if not file_paths:
        raise FileNotFoundError(
            f"No music files matching {list(file_patterns)} "
            f"found in {music_dir}"
        )

    file_labels: List[List[int]] = []
    all_measures: List[MeasureInfo] = []
    all_labels_flat: List[int] = []
    success = 0
    skipped = 0
    for fp in file_paths:
        try:
            measures = extractor.extract(fp)
            if len(measures) < 2:
                log.debug(
                    "%s: %d measure(s); need >= 2, skipping.",
                    fp.name, len(measures),
                )
                skipped += 1
                continue
            measures.sort(key=lambda m: m.measure_index)
            labels: List[int] = []
            for m in measures:
                vec = extractor.vectorize(m)
                lbl = clusterer.predict(vec)
                labels.append(lbl)
            file_labels.append(labels)
            if return_measures:
                all_measures.extend(measures)
                all_labels_flat.extend(labels)
            success += 1
        except Exception as exc:
            log.warning("Skipping %s: %s", fp, exc)

    log.info(
        "Classified %d files (%d skipped, %d total labels).",
        success, skipped, sum(len(l) for l in file_labels),
    )
    if return_measures:
        return file_labels, all_measures, all_labels_flat
    return file_labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


class ClusterVisualizer:
    """Generate diagnostic plots for measure clustering results."""

    def __init__(self, vectors: List[MeasureVector], clusterer: MeasureClusterer) -> None:
        self._vectors = vectors
        self._clusterer = clusterer
        self._X = np.stack([v.as_array() for v in vectors], axis=0)
        self._labels = clusterer.labels
        if self._labels is None:
            self._labels = np.zeros(len(vectors), dtype=int)
        self._centroids = clusterer.centroids

    def plot_all(self, save_prefix: str = "cluster") -> None:
        """Generate and save all diagnostic plots."""
        import matplotlib
        matplotlib.use("Agg")

        self._plot_summary(f"{save_prefix}_summary.png")
        self._plot_silhouette(f"{save_prefix}_silhouette.png")
        self._plot_radar(f"{save_prefix}_radar.png")
        self._plot_sizes(f"{save_prefix}_sizes.png")
        log.info("Saved cluster plots with prefix '%s'", save_prefix)

    # -- summary figure -------------------------------------------------------

    def _plot_summary(self, save_path: str) -> None:
        """Combined 2×2 summary: t-SNE, sizes, feature heatmap, silhouette."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # --- Top-left: t-SNE ---
        ax_tsne = axes[0, 0]
        if len(self._X) >= 2:
            X_sub = self._X
            labels_sub = self._labels
            if len(X_sub) > 3000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_sub), 3000, replace=False)
                X_sub = X_sub[idx]
                labels_sub = labels_sub[idx]
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sub) - 1))
            X_2d = tsne.fit_transform(X_sub)
            sc = ax_tsne.scatter(
                X_2d[:, 0], X_2d[:, 1], c=labels_sub,
                cmap="tab10", alpha=0.4, s=8,
            )
            ax_tsne.set_title("t-SNE Projection", fontsize=12)
            ax_tsne.legend(*sc.legend_elements(), title="Cluster", fontsize=7)
        else:
            ax_tsne.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            ax_tsne.set_title("t-SNE Projection")

        # --- Top-right: cluster sizes ---
        ax_sizes = axes[0, 1]
        unique, counts = np.unique(self._labels, return_counts=True)
        colors = plt.get_cmap("tab10")(unique)
        bars = ax_sizes.bar(unique, counts, color=colors)
        ax_sizes.set_xlabel("Cluster")
        ax_sizes.set_ylabel("Measures")
        ax_sizes.set_title("Cluster Sizes", fontsize=12)
        for bar, count in zip(bars, counts):
            ax_sizes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                          str(count), ha="center", fontsize=9)

        # --- Bottom-left: feature heatmap (clusters × features) ---
        ax_heat = axes[1, 0]
        if self._centroids is not None and self._centroids.shape[0] > 0:
            n_feat = self._centroids.shape[1]
            # Z-score across clusters per feature for contrast
            c_mean = self._centroids.mean(axis=0)
            c_std = self._centroids.std(axis=0)
            c_std[c_std == 0] = 1.0
            c_norm = (self._centroids - c_mean) / c_std
            im = ax_heat.imshow(c_norm.T, aspect="auto", cmap="RdBu_r",
                                interpolation="nearest")
            ax_heat.set_xticks(range(self._centroids.shape[0]))
            ax_heat.set_xticklabels([f"C{i}" for i in range(self._centroids.shape[0])])
            ax_heat.set_xlabel("Cluster")
            ax_heat.set_yticks(range(min(n_feat, len(FEATURE_NAMES))))
            ax_heat.set_yticklabels(
                [FEATURE_NAMES[i] for i in range(min(n_feat, len(FEATURE_NAMES)))],
                fontsize=7,
            )
            ax_heat.set_title("Cluster Centroids (z-score by feature)", fontsize=12)
            plt.colorbar(im, ax=ax_heat, shrink=0.8)
        else:
            ax_heat.text(0.5, 0.5, "No centroids", ha="center", va="center")

        # --- Bottom-right: silhouette ---
        ax_sil = axes[1, 1]
        try:
            from sklearn.metrics import silhouette_samples
            if len(self._X) >= 2 and len(unique) >= 2:
                sil_vals = silhouette_samples(self._X, self._labels)
                y_pos = 0
                for c in sorted(unique):
                    cluster_sil = sil_vals[self._labels == c]
                    cluster_sil.sort()
                    ax_sil.fill_betweenx(
                        np.arange(y_pos, y_pos + len(cluster_sil)),
                        0, cluster_sil,
                        alpha=0.6, color=colors[c], label=f"C{c}"
                    )
                    y_pos += len(cluster_sil)
                ax_sil.axvline(x=sil_vals.mean(), color="red", linestyle="--", linewidth=1)
                ax_sil.set_xlabel("Silhouette score")
                ax_sil.set_title("Silhouette by Cluster", fontsize=12)
                ax_sil.legend(fontsize=7, loc="lower left")
            else:
                ax_sil.text(0.5, 0.5, "Need >= 2 samples and clusters",
                            ha="center", va="center")
        except ImportError:
            ax_sil.text(0.5, 0.5, "scikit-learn not available",
                        ha="center", va="center")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved summary plot to %s", save_path)

    # -- silhouette plot ------------------------------------------------------

    def _plot_silhouette(self, save_path: str) -> None:
        """Silhouette scores per cluster (bar chart)."""
        import matplotlib.pyplot as plt

        unique = np.unique(self._labels)
        if len(unique) < 2:
            log.info("Silhouette: need >= 2 clusters, skipping.")
            return
        if len(self._X) < 2:
            return

        try:
            from sklearn.metrics import silhouette_samples, silhouette_score
        except ImportError:
            log.info("scikit-learn silhouette not available.")
            return

        sil_vals = silhouette_samples(self._X, self._labels)
        overall = silhouette_score(self._X, self._labels)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.get_cmap("tab10")(unique)

        cluster_means = []
        for c in sorted(unique):
            cluster_sil = sil_vals[self._labels == c]
            cluster_means.append(cluster_sil.mean())
        bars = ax.bar(unique, cluster_means, color=colors)
        ax.axhline(y=overall, color="red", linestyle="--", linewidth=1.5,
                   label=f"Overall ({overall:.3f})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Mean Silhouette Score")
        ax.set_title(f"Silhouette Scores by Cluster (overall={overall:.3f})")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, cluster_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=9)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved silhouette plot to %s", save_path)

    # -- pairplot ------------------------------------------------------------

    def _plot_pairplot(self, save_path: str) -> None:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        n = min(len(FEATURE_NAMES), self._X.shape[1])
        cols = [f"{FEATURE_NAMES[i]}" for i in range(n)]
        df = pd.DataFrame(self._X[:, :n], columns=cols)
        df["cluster"] = [f"Cluster {lbl}" for lbl in self._labels]

        g = sns.pairplot(
            df, hue="cluster", diag_kind="hist",
            palette="tab10", plot_kws={"alpha": 0.4, "s": 12},
        )
        g.fig.suptitle("Measure Feature Pairplot by Cluster", y=1.02, fontsize=16)
        g.fig.tight_layout()
        g.fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(g.fig)

    # -- radar chart ---------------------------------------------------------

    def _plot_radar(self, save_path: str) -> None:
        if self._centroids is None:
            return
        import matplotlib.pyplot as plt

        n_features = min(len(FEATURE_NAMES), self._centroids.shape[1])
        centroids = self._centroids[:, :n_features]
        # Normalise each feature to [0, 1] across centroids for radar readability
        mins = centroids.min(axis=0)
        maxs = centroids.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        normed = (centroids - mins) / ranges

        angles = np.linspace(0, 2 * math.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        cmap = plt.get_cmap("tab10")
        for c in range(centroids.shape[0]):
            values = normed[c].tolist() + [normed[c][0]]
            ax.fill(angles, values, alpha=0.1, color=cmap(c))
            ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {c}", color=cmap(c))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([FEATURE_NAMES[i] for i in range(n_features)], fontsize=9)
        ax.set_title("Cluster Centroid Radar (normalised per feature)", pad=30, fontsize=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -- cluster sizes -------------------------------------------------------

    def _plot_sizes(self, save_path: str) -> None:
        import matplotlib.pyplot as plt

        unique, counts = np.unique(self._labels, return_counts=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(unique, counts, color=plt.get_cmap("tab10")(unique))
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Measures")
        ax.set_title("Cluster Sizes")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -- t-SNE projection ----------------------------------------------------

    def _plot_tsne(self, save_path: str) -> None:
        if len(self._X) < 2:
            return
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # Subsample if very large
        X_sub = self._X
        labels_sub = self._labels
        if len(X_sub) > 5000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_sub), 5000, replace=False)
            X_sub = X_sub[idx]
            labels_sub = labels_sub[idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sub) - 1))
        X_2d = tsne.fit_transform(X_sub)

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1], c=labels_sub,
            cmap="tab10", alpha=0.5, s=12,
        )
        ax.set_title("t-SNE Projection of Measure Vectors by Cluster")
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend1)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse
    from _argparse_utils import (
        add_clustering_args,
        add_model_io_args,
        add_music_source_args,
    )

    parser = argparse.ArgumentParser(
        description="Measure Clustering — KMeans classification of musical measures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_music_source_args(parser)
    add_clustering_args(parser)
    add_model_io_args(parser)
    parser.add_argument(
        "--viz", action="store_true",
        help="Generate cluster visualization plots.",
    )
    parser.add_argument(
        "--viz-prefix", default="cluster",
        help="Filename prefix for visualization output.",
    )
    return parser


def _interpret_clusters(
    centroids: np.ndarray, stats: List[Dict[str, Any]]
) -> None:
    """Print a human-readable interpretation of each cluster's musical character.

    Describes each cluster in terms of note density, duration profile,
    rhythmic complexity, syncopation, and silence — derived from the
    centroid values relative to the min/max range across all clusters.
    """

    def _rel(val: float, i: int) -> str:
        """Classify *val* for feature index *i* as low / medium / high
        relative to the range across all centroids."""
        lo = float(centroids[:, i].min())
        hi = float(centroids[:, i].max())
        span = hi - lo if hi > lo else 1.0
        t = (val - lo) / span  # 0..1
        if t < 0.33:
            return "low"
        elif t < 0.67:
            return "medium"
        return "high"

    def _build_description(
        density: float, mean_dur: float, dur_var: float,
        short_r: float, silence_r: float, offbeat_r: float,
        sync_score: float, entropy: float,
    ) -> str:
        parts: List[str] = []

        # Density + mean duration → overall character
        d_rel = _rel(density, 0)
        m_rel = _rel(mean_dur, 1)
        if d_rel == "high" and m_rel == "low":
            parts.append("dense rapid-note passages (florid runs / figuration)")
        elif d_rel == "low" and m_rel == "high":
            parts.append("sparse long-note lines (sustained / chorale style)")
        elif d_rel == "low":
            parts.append("low-density texture with longer note values")
        elif d_rel == "high":
            parts.append("busy texture with predominantly short notes")
        else:
            parts.append("moderate rhythmic activity")

        # Duration variance
        if _rel(dur_var, 2) == "high":
            parts.append("highly varied note durations")
        elif _rel(dur_var, 2) == "low":
            parts.append("uniform note durations")

        # Short-note ratio
        if _rel(short_r, 3) == "high":
            parts.append("abundant ornamentation / short-note flourishes")
        elif _rel(short_r, 3) == "low" and d_rel != "low":
            parts.append("few very short notes")

        # Silence ratio
        if _rel(silence_r, 4) == "high":
            parts.append("frequent rests / breathing space")
        elif _rel(silence_r, 4) == "low":
            parts.append("continuous (legato) phrasing")

        # Offbeat + syncopation
        o_rel = _rel(offbeat_r, 5)
        s_rel = _rel(sync_score, 6)
        if s_rel == "high":
            parts.append("strongly syncopated")
        elif o_rel == "high":
            parts.append("off-beat emphasis / rhythmic displacement")
        elif o_rel == "low":
            parts.append("square, on-the-beat rhythm")

        # Entropy
        if _rel(entropy, 7) == "high":
            parts.append("high rhythmic diversity")
        elif _rel(entropy, 7) == "low":
            parts.append("simple, predictable rhythmic pattern")

        return "; ".join(parts) + "."

    print("\n" + "=" * 72)
    print("CLUSTER INTERPRETATION")
    print("=" * 72)

    for s in stats:
        c = s["cluster"]
        fm = s["feature_means"]
        desc = _build_description(
            fm["note_density"], fm["mean_duration"], fm["duration_variance"],
            fm["short_note_ratio"], fm["silence_ratio"], fm["offbeat_ratio"],
            fm["syncopation_score"], fm["entropy"],
        )
        print(f"\n  Cluster {c:>2}  ({s['size']:>5} measures)")
        print(f"         {desc}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Resolve file patterns
    patterns = [p.strip() for p in args.file_patterns.split(",") if p.strip()]

    vectors: List[MeasureVector] = []

    if args.load_model:
        log.info("Loading clusterer from %s ...", args.load_model)
        clusterer = MeasureClusterer.load(args.load_model)
    else:
        log.info("Extracting measures from %s ...", args.music_dir)
        extractor = MeasureExtractor()
        vectors = extractor.extract_all(args.music_dir, file_patterns=patterns)
        log.info("Extracted %d measure vectors.", len(vectors))

        log.info("Fitting KMeans (k=%d, seed=%d) ...", args.n_clusters, args.seed)
        clusterer = MeasureClusterer()
        clusterer.fit(vectors, n_clusters=args.n_clusters, random_seed=args.seed)

        if args.save_model:
            clusterer.save(args.save_model)

        if args.viz:
            log.info("Generating cluster visualizations ...")
            viz = ClusterVisualizer(vectors, clusterer)
            viz.plot_all(save_prefix=args.viz_prefix)

    # Print cluster statistics
    centroids = clusterer.centroids
    if centroids is not None:
        print(f"\nKMeans inertia: {clusterer.inertia:.3f}")
        print(f"Number of clusters: {centroids.shape[0]}\n")
        print("Cluster centroids (original feature space):")
        print("-" * 90)
        header = f"{'Cluster':>8}" + "".join(f"{name:>13}" for name in FEATURE_NAMES)
        print(header)
        print("-" * 90)
        for c in range(centroids.shape[0]):
            vals = "".join(f"{centroids[c, i]:13.4f}" for i in range(centroids.shape[1]))
            print(f"{c:>8}{vals}")
        print("-" * 90)

        # Interpret clusters if we have vectors (fit path), otherwise from centroids alone
        if vectors:
            stats = clusterer.cluster_stats(vectors)
        else:
            # Build minimal stats from centroids for loaded models
            stats = []
            for c in range(centroids.shape[0]):
                fm = {FEATURE_NAMES[i]: float(centroids[c, i]) for i in range(centroids.shape[1])}
                stats.append({"cluster": c, "size": -1, "feature_means": fm})
        _interpret_clusters(centroids, stats)
        clusterer.print_sample_measures()

    print("Done.")


if __name__ == "__main__":
    main()
