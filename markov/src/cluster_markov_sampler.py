#!/usr/bin/env python3
"""
Cluster-Constrained Markov Sampler — strategy 3 for per-measure note generation.

Trains a small per-cluster Markov chain over ``(pitch_class, duration_idx)``
tokens extracted from stored measures.  Generation walks the chain note-by-note
within the bar, filling gaps with rests and resolving pitch class to MIDI pitch
via the cluster's octave profile.

Usage::

    from measure_clustering import MeasureClusterer
    from cluster_markov_sampler import ClusterMarkovSampler

    clusterer = MeasureClusterer.load("./models/my_model/clusterer.pkl")
    strategy = ClusterMarkovSampler(clusterer, order=2)
    notes = strategy.sample_measure(cluster_label=3, time_signature=(4, 4), seed=42)
"""

from __future__ import annotations

from collections import defaultdict
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


def _duration_from_idx(idx: int) -> float:
    """Return quarterLength for a duration index."""
    return float(_USABLE_DURATIONS[min(idx, len(_USABLE_DURATIONS) - 1)])


# ---------------------------------------------------------------------------
# Token codec
# ---------------------------------------------------------------------------


def _note_to_token(nd: dict) -> str:
    """Encode a note dict as ``pcN_dM``."""
    pc = nd["pitch"] % 12
    di = _quantize_duration(float(nd["quarterLength"]))
    return f"pc{pc}_d{di}"


def _token_to_pc_dur(token: str) -> Tuple[int, int]:
    """Decode ``pcN_dM`` → (pitch_class, duration_idx)."""
    pc_part, d_part = token.split("_")
    return int(pc_part[2:]), int(d_part[1:])


# ---------------------------------------------------------------------------
# ClusterMarkovSampler
# ---------------------------------------------------------------------------


class ClusterMarkovSampler:
    """Per-cluster Markov chain over ``(pitch_class, duration_idx)`` tokens.

    Each cluster gets its own chain trained on stored measures.  Generation
    walks the chain note-by-note, building a bar-length sequence.
    """

    def __init__(self, clusterer: MeasureClusterer, order: int = 2) -> None:
        n_clusters = clusterer.centroids.shape[0] if clusterer.centroids is not None else 0
        if n_clusters == 0:
            raise ValueError("Clusterer has no centroids — is it fitted?")
        self._clusterer = clusterer
        self._n_clusters = n_clusters
        self._order = max(1, order)

        # Per-cluster data
        self._chains: Dict[int, Dict[Tuple[str, ...], Dict[str, float]]] = {}
        self._counts: Dict[int, Dict[Tuple[str, ...], Dict[str, int]]] = {}
        self._starters: Dict[int, List[List[str]]] = {}  # observed starting n-grams
        self._octaves: Dict[int, List[int]] = {}          # all octaves seen in cluster
        self._pitch_hists: np.ndarray = np.full((n_clusters, 12), 1.0 / 12, dtype=np.float64)

        self._train()

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
        """Generate notes for one measure by walking the cluster's Markov chain.

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

        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)

        chain = self._chains.get(c)
        starters_cluster = self._starters.get(c, [])
        if chain is None or not starters_cluster:
            return self._fallback(c, bar_length_ql, rng)

        # Seed history from a random observed starter
        seed_seq = list(starters_cluster[rng.randint(0, len(starters_cluster))])
        if not seed_seq:
            return self._fallback(c, bar_length_ql, rng)

        # Token→NoteEvent accumulator
        tokens: List[str] = []
        current_beat = 0.0
        history: List[str] = []

        # Determine octave profile for this cluster
        octaves = self._octaves.get(c, [4])
        median_octave = int(np.median(octaves)) if octaves else 4

        # Generate tokens until bar is filled
        for _ in range(48):  # safety cap: max 48 notes per bar
            if current_beat >= bar_length_ql - 0.03:
                break

            # Sample next token
            nxt = self._sample_next(chain, history[-self._order:], rng)
            if nxt is None:
                break

            tokens.append(nxt)
            history.append(nxt)

            pc, di = _token_to_pc_dur(nxt)
            dur = _duration_from_idx(di)

            current_beat += dur

        # Convert tokens to NoteEvents
        notes: List[NoteEvent] = []
        beat = 0.0
        vel_base = 80

        for token in tokens:
            if beat >= bar_length_ql - 0.03:
                break

            pc, di = _token_to_pc_dur(token)
            dur = _duration_from_idx(di)

            # Clamp to bar boundary
            if beat + dur > bar_length_ql + 0.03:
                dur = max(0.25, bar_length_ql - beat)
            if dur < 0.03:
                break

            # Assign octave: use median ± small random offset
            octave = median_octave + rng.choice([-1, 0, 0, 0, 0, 1])
            pitch = octave * 12 + pc
            pitch = max(28, min(96, pitch))

            vel = max(40, min(127, int(vel_base + rng.normal(0, 8))))

            notes.append(NoteEvent(
                pitch=pitch,
                duration_ql=dur,
                velocity=vel,
                beat_offset=beat,
            ))
            beat += dur

        # 4. Section-end breathing
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
    # Internal: training
    # ------------------------------------------------------------------

    def _train(self) -> None:
        """Train one Markov chain per cluster from stored measures."""
        for c in range(self._n_clusters):
            measures = self._clusterer.get_cluster_measures(c)
            if not measures:
                continue

            token_seqs: List[List[str]] = []
            octave_list: List[int] = []
            pc_counts = np.zeros(12, dtype=np.float64)

            for m in measures:
                notes = sorted(m.notes, key=lambda nd: nd.get("onset_in_measure", 0.0))
                seq = [_note_to_token(nd) for nd in notes]
                if len(seq) >= self._order + 1:
                    token_seqs.append(seq)
                elif seq:
                    # Single-note or short measure — still useful for starters/unigrams
                    token_seqs.append(seq)

                for nd in notes:
                    pc_counts[nd["pitch"] % 12] += 1.0
                    octave_list.append(nd["pitch"] // 12)

            if pc_counts.sum() > 0:
                self._pitch_hists[c] = pc_counts / pc_counts.sum()
            self._octaves[c] = octave_list if octave_list else [4]

            if not token_seqs:
                continue

            # Build Markov chain counts
            counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            starters: List[List[str]] = []

            for seq in token_seqs:
                if len(seq) >= self._order:
                    for i in range(len(seq) - self._order + 1):
                        starters.append(seq[i:i + self._order])
                elif seq:
                    starters.append(seq)

                for i in range(len(seq) - self._order):
                    key = tuple(seq[i:i + self._order])
                    nxt = seq[i + self._order]
                    counts[key][nxt] += 1

            # Convert counts to probabilities
            chain: Dict[Tuple[str, ...], Dict[str, float]] = {}
            for key, nxt_counts in counts.items():
                total = sum(nxt_counts.values())
                chain[key] = {tok: cnt / total for tok, cnt in nxt_counts.items()}

            self._counts[c] = dict(counts)
            self._chains[c] = chain
            self._starters[c] = starters

    # ------------------------------------------------------------------
    # Internal: sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_next(
        chain: Dict[Tuple[str, ...], Dict[str, float]],
        history: List[str],
        rng: np.random.RandomState,
    ) -> Optional[str]:
        """Sample the next token given a history, with back-off.

        Tries: full history → last-1 → unigram → None.
        """
        for o in range(len(history), 0, -1):
            key = tuple(history[-o:])
            if key in chain and chain[key]:
                candidates = list(chain[key].keys())
                probs = np.array([chain[key][tok] for tok in candidates], dtype=np.float64)
                return str(rng.choice(candidates, p=probs))

        # Fallback: pick any token from the chain
        all_tokens: List[str] = []
        for nxt_map in chain.values():
            all_tokens.extend(nxt_map.keys())
        if all_tokens:
            return str(rng.choice(all_tokens))
        return None

    # ------------------------------------------------------------------
    # Internal: fallback
    # ------------------------------------------------------------------

    def _fallback(
        self, cluster_label: int,
        bar_length_ql: float,
        rng: np.random.RandomState,
    ) -> List[NoteEvent]:
        """Minimal fallback when no chain exists for the cluster."""
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
