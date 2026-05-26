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
from collections import defaultdict
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

FEATURE_NAMES: Tuple[str, ...] = (
    "note_density",
    "mean_duration",
    "duration_variance",
    "short_note_ratio",
    "silence_ratio",
    "offbeat_ratio",
    "syncopation_score",
    "entropy",
)

# music21 time-signature → quarterLength per bar
_TS_BAR_LENGTH: Dict[str, float] = {
    "4/4": 4.0, "3/4": 3.0, "2/4": 2.0, "1/4": 1.0,
    "6/8": 3.0, "3/8": 1.5, "2/2": 4.0, "6/4": 6.0,
    "9/8": 4.5, "12/8": 6.0, "5/4": 5.0, "7/8": 3.5,
}


def _bar_length_ql(ts_str: str) -> float:
    """Return the quarterLength of one bar for a time-signature string."""
    cached = _TS_BAR_LENGTH.get(ts_str)
    if cached is not None:
        return cached
    parts = ts_str.split("/")
    if len(parts) == 2:
        num, den = int(parts[0]), int(parts[1])
        return num * (4.0 / den)
    return 4.0


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


@dataclass
class MeasureVector:
    """8-D texture + 12-D pitch-class feature vector for one measure."""

    note_density: float = 0.0
    mean_duration: float = 0.0
    duration_variance: float = 0.0
    short_note_ratio: float = 0.0
    silence_ratio: float = 0.0
    offbeat_ratio: float = 0.0
    syncopation_score: float = 0.0
    entropy: float = 0.0

    # Duration-weighted pitch-class histogram (12 semitones, normalised to 1).
    pitch_class_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.float64),
    )

    file_path: str = ""
    measure_index: int = 0
    cluster_label: int = -1

    def as_array(self) -> np.ndarray:
        """Return the 8 texture features as a numpy array (float64).

        This is the vector used for KMeans clustering — it intentionally
        excludes the pitch-class histogram so clusters describe *how* the
        music is played, not *which* notes are played.
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
        ], dtype=np.float64)

    def as_full_array(self) -> np.ndarray:
        """Return 20-D array: 8 texture features + 12 pitch-class histogram.

        Used by the section miner for transposition-invariant similarity.
        """
        return np.concatenate([self.as_array(), self.pitch_class_histogram])

    @classmethod
    def from_array(cls, arr: np.ndarray, file_path: str = "",
                   measure_index: int = 0, cluster_label: int = -1) -> "MeasureVector":
        """Construct from a numpy array (for centroid reconstruction).

        Handles both 8-D (legacy) and 20-D (with pitch-class) arrays.
        """
        pc_hist = np.zeros(12, dtype=np.float64)
        if len(arr) >= 20:
            pc_hist = arr[8:20].astype(np.float64)
        return cls(
            note_density=float(arr[0]),
            mean_duration=float(arr[1]),
            duration_variance=float(arr[2]),
            short_note_ratio=float(arr[3]),
            silence_ratio=float(arr[4]),
            offbeat_ratio=float(arr[5]),
            syncopation_score=float(arr[6]),
            entropy=float(arr[7]),
            pitch_class_histogram=pc_hist,
            file_path=file_path,
            measure_index=measure_index,
            cluster_label=cluster_label,
        )


# ---------------------------------------------------------------------------
# Measure extractor
# ---------------------------------------------------------------------------


class MeasureExtractor:
    """Parse music files and extract per-measure feature vectors."""

    def __init__(self) -> None:
        self._dur_bins = _DURATION_BIN_EDGES

    # -- file-level parsing --------------------------------------------------

    def extract(self, file_path: Union[str, Path]) -> List[MeasureInfo]:
        """Parse a single music file into a list of MeasureInfo objects.

        Supports .mid, .midi, .abc, .krn via music21's converter.
        Chords are expanded: each pitch becomes a separate note entry sharing
        the same onset and duration.
        """
        file_path = Path(file_path)
        score = converter.parse(str(file_path))

        # Handle Opus (multi-tune ABC) by taking the first score
        if hasattr(score, "scores"):
            scores = score.scores
            if not scores:
                return []
            score = scores[0]

        # Determine primary time signature
        ts_str = self._primary_time_signature(score)

        # Collect all notes/rests from all parts
        all_notes: List[Dict[str, Any]] = []  # {offset, quarterLength, pitch, is_rest}
        for part in score.parts:
            for el in part.flatten().notesAndRests:
                offset = float(el.offset)
                ql = float(el.quarterLength)
                if el.isRest:
                    all_notes.append({
                        "offset": offset, "quarterLength": ql,
                        "pitch": -1, "is_rest": True,
                    })
                elif el.isNote:
                    midi = el.pitch.midi if el.pitch else 60
                    all_notes.append({
                        "offset": offset, "quarterLength": ql,
                        "pitch": int(midi), "is_rest": False,
                    })
                elif el.isChord:
                    for p in el.pitches:
                        all_notes.append({
                            "offset": offset, "quarterLength": ql,
                            "pitch": p.midi, "is_rest": False,
                        })

        # Group by measure
        bar_length = _bar_length_ql(ts_str)
        if bar_length <= 0:
            bar_length = 4.0

        measure_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for n in all_notes:
            bar_idx = int(n["offset"] / bar_length)
            measure_map[bar_idx].append(n)

        # Build MeasureInfo per bar
        result: List[MeasureInfo] = []
        for bar_idx in sorted(measure_map.keys()):
            entries = measure_map[bar_idx]
            bar_offset = bar_idx * bar_length
            notes_data: List[Dict[str, Any]] = []

            for e in entries:
                onset_in_measure = e["offset"] - bar_offset
                if e["is_rest"]:
                    continue  # rests tracked via silence_ratio
                notes_data.append({
                    "pitch": e["pitch"],
                    "quarterLength": e["quarterLength"],
                    "onset_in_measure": onset_in_measure,
                })

            if not notes_data:
                continue  # skip empty measures

            result.append(MeasureInfo(
                notes=notes_data,
                beats=bar_length,
                time_signature=ts_str,
                file_path=str(file_path),
                measure_index=bar_idx,
            ))

        return result

    @staticmethod
    def _primary_time_signature(score) -> str:
        """Extract the first time signature from a music21 score."""
        for el in score.flatten():
            if isinstance(el, meter.TimeSignature):
                return f"{el.numerator}/{el.denominator}"
        return "4/4"

    # -- vectorization -------------------------------------------------------

    def vectorize(self, measure: MeasureInfo) -> MeasureVector:
        """Compute the 8-dimensional feature vector for a single measure."""
        notes = measure.notes
        n = len(notes)
        if n == 0:
            return MeasureVector(
                file_path=measure.file_path,
                measure_index=measure.measure_index,
            )

        durations = np.array([nd["quarterLength"] for nd in notes], dtype=np.float64)
        onsets = np.array([nd["onset_in_measure"] for nd in notes], dtype=np.float64)

        # 1. note_density — onsets per beat
        density = n / measure.beats if measure.beats > 0 else 0.0

        # 2. mean_duration
        mean_dur = float(np.mean(durations))

        # 3. duration_variance
        dur_var = float(np.var(durations))

        # 4. short_note_ratio
        short_ratio = float(np.sum(durations < SHORT_NOTE_THRESHOLD) / n)

        # 5. silence_ratio
        total_sounding = float(np.sum(durations))
        silence = 1.0 - (total_sounding / measure.beats) if measure.beats > 0 else 0.0
        silence = max(0.0, min(1.0, silence))

        # 6. offbeat_ratio — onset not on quarter-beat boundary
        offbeat = float(np.sum(onsets % 1.0 > 1e-9) / n)

        # 7. syncopation_score — offbeat note where duration exceeds previous on-beat note
        sync_count = self._count_syncopations(notes)

        # 8. rhythmic entropy
        ent = self._rhythmic_entropy(durations)

        # 9. pitch-class histogram (duration-weighted, 12-bin, normalised)
        pc_hist = self._pitch_class_histogram(notes)

        return MeasureVector(
            note_density=density,
            mean_duration=mean_dur,
            duration_variance=dur_var,
            short_note_ratio=short_ratio,
            silence_ratio=silence,
            offbeat_ratio=offbeat,
            syncopation_score=sync_count / n,
            entropy=ent,
            pitch_class_histogram=pc_hist,
            file_path=measure.file_path,
            measure_index=measure.measure_index,
        )

    @staticmethod
    def _is_onbeat(onset: float) -> bool:
        """Return True if *onset* falls on a quarter-beat boundary."""
        return abs(onset % 1.0) < 1e-9

    def _count_syncopations(self, notes: List[Dict[str, Any]]) -> int:
        """Count syncopations: offbeat notes longer than the preceding on-beat note.

        Walk through notes sorted by onset.  Track the most recent on-beat
        noteʼs duration.  When we encounter an offbeat note whose duration
        exceeds that value, count it as syncopated.
        """
        sorted_notes = sorted(notes, key=lambda nd: nd["onset_in_measure"])
        count = 0
        prev_onbeat_dur = 0.0
        for nd in sorted_notes:
            onset = nd["onset_in_measure"]
            dur = nd["quarterLength"]
            if self._is_onbeat(onset):
                prev_onbeat_dur = dur
            else:
                if dur > prev_onbeat_dur and prev_onbeat_dur > 0:
                    count += 1
        return count

    def _rhythmic_entropy(self, durations: np.ndarray) -> float:
        """Shannon entropy of the duration distribution (binned)."""
        hist, _ = np.histogram(durations, bins=self._dur_bins)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return 0.0
        probs = hist / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _pitch_class_histogram(notes: List[Dict[str, Any]]) -> np.ndarray:
        """Duration-weighted pitch-class histogram (12 bins, normalised to 1).

        Each note contributes its quarterLength to its pitch class bin, so
        longer notes have more weight than short ornamental ones.  Returns
        a zero vector when there are no pitched notes.
        """
        hist = np.zeros(12, dtype=np.float64)
        total = 0.0
        for nd in notes:
            dur = nd.get("quarterLength", 0.0)
            pitch = nd.get("pitch", -1)
            if pitch < 0 or dur <= 0:
                continue
            hist[pitch % 12] += dur
            total += dur
        if total > 0:
            hist /= total
        return hist

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
        file_map = self.extract_file_map(music_dir, file_patterns)
        all_vectors: List[MeasureVector] = []
        for vectors in file_map.values():
            all_vectors.extend(vectors)
        log.info(
            "Extracted %d measure vectors from %d files.",
            len(all_vectors), len(file_map),
        )
        return all_vectors

    def extract_file_map(
        self,
        music_dir: Union[str, Path],
        file_patterns: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, List[MeasureVector]]:
        """Walk *music_dir*, parse every matching file, return per-file vectors.

        Args:
            music_dir: Root directory containing music files.
            file_patterns: Glob pattern(s).  Defaults to ``["*.mid","*.midi","*.abc","*.krn"]``.

        Returns:
            Dict mapping ``str(file_path)`` → ``List[MeasureVector]``.
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

        result: Dict[str, List[MeasureVector]] = {}
        success = 0
        for mp in music_paths:
            try:
                measures = self.extract(mp)
                if not measures:
                    continue
                vectors = [self.vectorize(m) for m in measures]
                result[str(mp)] = vectors
                success += 1
                log.info("Parsed %s: %d measures", mp.name, len(measures))
            except Exception as exc:
                log.warning("Skipping %s: %s", mp, exc)

        if not result:
            raise RuntimeError(
                f"No valid measures extracted from {len(music_paths)} files "
                f"in {music_dir}"
            )
        log.info(
            "Extracted per-file vectors: %d/%d files succeeded.",
            success, len(music_paths),
        )
        return result


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
        self.pitch_histograms: Optional[np.ndarray] = None  # (n_clusters, 12)

    # -- properties ----------------------------------------------------------

    @property
    def centroids(self) -> Optional[np.ndarray]:
        """Cluster centroids in the original (unscaled) feature space (n_clusters × 8)."""
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
    ) -> "MeasureClusterer":
        """Normalize vectors with StandardScaler, fit KMeans, and store model.

        Args:
            vectors: Training data.
            n_clusters: Number of KMeans clusters.
            random_seed: Reproducibility seed.

        Returns:
            self (for chaining).
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = np.stack([v.as_array() for v in vectors], axis=0)
        log.info("Feature matrix shape: %s", X.shape)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init="auto",
        )
        self._labels = self._kmeans.fit_predict(X_scaled)
        self._inertia = float(self._kmeans.inertia_)

        # Store centroids in original (unscaled) space
        centroids_scaled = self._kmeans.cluster_centers_
        self._centroids_raw = self._scaler.inverse_transform(centroids_scaled)

        # Tag vectors with their labels
        for v, lbl in zip(vectors, self._labels):
            v.cluster_label = int(lbl)

        log.info(
            "KMeans fit: n_clusters=%d, inertia=%.3f, samples=%d",
            n_clusters, self._inertia, len(vectors),
        )
        return self

    # -- predict -------------------------------------------------------------

    def predict(self, vector: MeasureVector) -> int:
        """Classify a single MeasureVector. Returns cluster label (0..n-1)."""
        self._require_fit()
        X = vector.as_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        return int(self._kmeans.predict(X_scaled)[0])

    def predict_many(self, vectors: List[MeasureVector]) -> np.ndarray:
        """Classify multiple MeasureVectors. Returns array of cluster labels."""
        self._require_fit()
        X = np.stack([v.as_array() for v in vectors], axis=0)
        X_scaled = self._scaler.transform(X)
        return self._kmeans.predict(X_scaled)

    def compute_pitch_histograms(
        self,
        file_map: Dict[str, List["MeasureVector"]],
        file_labels: List[List[int]],
    ) -> np.ndarray:
        """Compute per-cluster average pitch-class histograms.

        For each measure in the training corpus, accumulates its 12-D
        pitch-class histogram into the bucket for its assigned cluster,
        then normalizes each cluster's histogram to sum to 1.

        Args:
            file_map: filename → list of MeasureVector (in order).
            file_labels: list of label lists, same file order as file_map.

        Returns:
            Array of shape (n_clusters, 12).  Also stored in
            ``self.pitch_histograms``.
        """
        self._require_fit()
        n_clusters = self._centroids_raw.shape[0] if self._centroids_raw is not None else 0
        if n_clusters == 0:
            self.pitch_histograms = np.zeros((0, 12), dtype=np.float64)
            return self.pitch_histograms

        accum = np.zeros((n_clusters, 12), dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.int64)

        for vecs, labels in zip(file_map.values(), file_labels):
            for vec, label in zip(vecs, labels):
                if 0 <= label < n_clusters:
                    accum[label] += vec.pitch_class_histogram
                    counts[label] += 1

        # Normalize each row
        for c in range(n_clusters):
            if counts[c] > 0:
                accum[c] /= counts[c]
                total = float(accum[c].sum())
                if total > 0:
                    accum[c] /= total

        self.pitch_histograms = accum
        return accum

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
            std_arr = np.std(cluster_x, axis=0) if len(cluster_x) > 0 else np.zeros(8)
            feature_means = {
                FEATURE_NAMES[i]: float(self._centroids_raw[c, i]) for i in range(8)
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
            "pitch_histograms": self.pitch_histograms,
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
        obj.pitch_histograms = state.get("pitch_histograms")
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
) -> List[List[int]]:
    """Extract measures from all music files, classify, return per-file labels.

    This is the shared extraction+classification step used by all three
    builders.  File boundaries are preserved — each inner list corresponds
    to one file's ordered cluster labels.

    Args:
        music_dir: Root directory containing music files.
        clusterer: Fitted MeasureClusterer.
        extractor: Optional pre-created MeasureExtractor. Created if None.
        file_patterns: Glob patterns. Defaults to standard music formats.

    Returns:
        List of per-file label sequences. Files with <2 measures are skipped.
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
                labels.append(clusterer.predict(vec))
            file_labels.append(labels)
            success += 1
        except Exception as exc:
            log.warning("Skipping %s: %s", fp, exc)

    log.info(
        "Classified %d files (%d skipped, %d total labels).",
        success, skipped, sum(len(l) for l in file_labels),
    )
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

        self._plot_pairplot(f"{save_prefix}_pairplot.png")
        self._plot_radar(f"{save_prefix}_radar.png")
        self._plot_sizes(f"{save_prefix}_sizes.png")
        self._plot_tsne(f"{save_prefix}_tsne.png")
        log.info("Saved cluster plots with prefix '%s'", save_prefix)

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

    print("Done.")


if __name__ == "__main__":
    main()
