#!/usr/bin/env python3
"""
Section Miner — discover recurring section structure from bar-level feature vectors.

Uses Self-Similarity Matrix (SSM) to find candidate section pairs, then Minimum
Description Length (MDL) to select the families that best compress the sequence.
Supports transposition-invariant matching via pitch-class histogram cyclic shifts.

The miner is intentionally independent from MIDI parsing and clustering.  Feed it
bar-level numeric vectors (e.g. MeasureVector.as_full_array()) and it discovers
repeated sections regardless of key.

Usage (library)::

    from section_miner import SectionMiner, SectionMinerConfig

    miner = SectionMiner(SectionMinerConfig(
        input_mode="vector",
        transposition_invariant=True,
        pc_dims=12,
    ))
    result = miner.analyze(bar_vectors)

Usage (CLI — JSON input)::

    python section_miner.py --input bars.json --pretty

Usage (CLI — music files)::

    python section_miner.py --music-dir ../../datasets/corelli --pretty
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (Any, Dict, Hashable, Iterable, List, Literal, Optional,
                    Sequence, Tuple, TypeVar)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

NormalizedToken = Tuple[Any, ...]

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class SectionMinerConfig:
    """Configuration for SSM candidate mining and MDL family selection.

    All similarity thresholds are in [0, 1] where 1 is identical.
    """

    # -- input mode -----------------------------------------------------------
    input_mode: Literal["auto", "vector", "token"] = "auto"
    vector_similarity: Literal["cosine", "rbf"] = "cosine"
    vector_distance_scale: float = 1.0

    # -- transposition invariance ---------------------------------------------
    transposition_invariant: bool = False
    pc_dims: int = 12
    ti_alpha: float = 0.5

    # -- candidate mining -----------------------------------------------------
    min_len: int = 3
    max_len: int = 12
    min_gap: int = 5
    similarity_threshold: float = 0.75
    same_state_similarity: float = 0.70

    # -- MDL selection --------------------------------------------------------
    token_cost: float = 1.0
    section_token_cost: float = 1.0
    mdl_alpha: float = 1.5
    variant_penalty: float = 0.50
    min_occurrences: int = 2
    allow_zero_gain: bool = False
    max_overlap_fraction: float = 0.0
    return_threshold: float = 0.85
    adjacency_gap: int = 1

    def validate(self) -> None:
        if self.input_mode not in {"auto", "vector", "token"}:
            raise ValueError("input_mode must be auto, vector, or token")
        if self.vector_similarity not in {"cosine", "rbf"}:
            raise ValueError("vector_similarity must be cosine or rbf")
        if self.vector_distance_scale <= 0:
            raise ValueError("vector_distance_scale must be > 0")
        if self.min_len < 1:
            raise ValueError("min_len must be >= 1")
        if self.max_len < self.min_len:
            raise ValueError("max_len must be >= min_len")
        if self.min_gap < 0:
            raise ValueError("min_gap must be >= 0")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if not 0 <= self.same_state_similarity <= 1:
            raise ValueError("same_state_similarity must be in [0, 1]")
        if self.token_cost <= 0:
            raise ValueError("token_cost must be > 0")
        if self.section_token_cost <= 0:
            raise ValueError("section_token_cost must be > 0")
        if self.mdl_alpha < 0:
            raise ValueError("mdl_alpha must be >= 0")
        if self.variant_penalty < 0:
            raise ValueError("variant_penalty must be >= 0")
        if self.min_occurrences < 2:
            raise ValueError("min_occurrences must be >= 2")
        if not 0 <= self.max_overlap_fraction <= 1:
            raise ValueError("max_overlap_fraction must be in [0, 1]")
        if not 0 <= self.return_threshold <= 1:
            raise ValueError("return_threshold must be in [0, 1]")
        if self.adjacency_gap < 0:
            raise ValueError("adjacency_gap must be >= 0")
        if self.transposition_invariant:
            if self.pc_dims < 1:
                raise ValueError("pc_dims must be >= 1 when transposition_invariant=True")
            if not 0 <= self.ti_alpha <= 1:
                raise ValueError("ti_alpha must be in [0, 1]")


@dataclass(frozen=True)
class CandidatePair:
    """Two similar same-length windows discovered from the SSM."""

    start_a: int
    start_b: int
    length: int
    similarity: float

    @property
    def end_a(self) -> int:
        return self.start_a + self.length

    @property
    def end_b(self) -> int:
        return self.start_b + self.length


@dataclass(frozen=True)
class Occurrence:
    """One occurrence of a selected section family."""

    start: int
    length: int
    similarity_to_prototype: float
    role: Literal["NEW", "REPEAT", "RETURN", "VARIANT"] = "NEW"

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class SectionFamily:
    """A selected section family after MDL compression."""

    label: str
    length: int
    prototype_start: int
    prototype_tokens: Tuple[NormalizedToken, ...]
    occurrences: Tuple[Occurrence, ...]
    avg_pair_similarity: float
    define_cost: float
    encoded_saving: float
    variant_cost: float
    mdl_gain: float


@dataclass(frozen=True)
class EncodedItem:
    """One item in the mixed section/state sequence."""

    kind: Literal["SECTION", "FREE"]
    start: int
    end: int
    label: Optional[str] = None
    role: Optional[str] = None
    tokens: Tuple[NormalizedToken, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SectionStructureResult:
    """Full output for one analyzed song or sequence."""

    sequence_length: int
    tokens: Tuple[NormalizedToken, ...]
    families: Tuple[SectionFamily, ...]
    mixed_sequence: Tuple[EncodedItem, ...]
    candidate_pair_count: int
    rejected_family_count: int
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "tokens": [list(token) for token in self.tokens],
            "families": [
                {
                    "label": f.label,
                    "length": f.length,
                    "prototype_start": f.prototype_start,
                    "prototype_tokens": [list(t) for t in f.prototype_tokens],
                    "occurrences": [
                        {
                            "start": o.start,
                            "end": o.end,
                            "length": o.length,
                            "similarity_to_prototype": o.similarity_to_prototype,
                            "role": o.role,
                        }
                        for o in f.occurrences
                    ],
                    "avg_pair_similarity": f.avg_pair_similarity,
                    "define_cost": f.define_cost,
                    "encoded_saving": f.encoded_saving,
                    "variant_cost": f.variant_cost,
                    "mdl_gain": f.mdl_gain,
                }
                for f in self.families
            ],
            "mixed_sequence": [
                {
                    "kind": item.kind,
                    "start": item.start,
                    "end": item.end,
                    "label": item.label,
                    "role": item.role,
                    "tokens": [list(t) for t in item.tokens],
                }
                for item in self.mixed_sequence
            ],
            "candidate_pair_count": self.candidate_pair_count,
            "rejected_family_count": self.rejected_family_count,
            "config": dict(self.config),
        }


# ---------------------------------------------------------------------------
# Token normalization & similarity
# ---------------------------------------------------------------------------


def normalize_token(token: Any) -> NormalizedToken:
    """Convert supported token shapes to a hashable normalized token.

    Supported input examples::

        2
        "2:M"
        ("2", "M")
        [2, "M"]
        {"state": 2, "duration": "M"}
    """
    if isinstance(token, dict):
        if "state" in token:
            state = token["state"]
            duration = token.get("duration", token.get("duration_class"))
            return (state,) if duration is None else (state, duration)
        return tuple(sorted(token.items()))
    if isinstance(token, (list, tuple)):
        return tuple(token)
    if isinstance(token, str) and ":" in token:
        left, right = token.split(":", 1)
        return (left, right)
    return (token,)


def normalize_sequence(tokens: Iterable[Any]) -> Tuple[NormalizedToken, ...]:
    return tuple(normalize_token(t) for t in tokens)


def is_numeric_vector(token: NormalizedToken) -> bool:
    return bool(token) and all(isinstance(v, (int, float)) for v in token)


# -- vector similarity --------------------------------------------------------


def vector_similarity_score(
    left: NormalizedToken,
    right: NormalizedToken,
    *,
    metric: Literal["cosine", "rbf"] = "cosine",
    distance_scale: float = 1.0,
) -> float:
    if len(left) != len(right):
        raise ValueError("Vector tokens must have the same dimensionality")
    if metric == "cosine":
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        norm_left = math.sqrt(sum(float(a) * float(a) for a in left))
        norm_right = math.sqrt(sum(float(b) * float(b) for b in right))
        if norm_left == 0.0 and norm_right == 0.0:
            return 1.0
        if norm_left == 0.0 or norm_right == 0.0:
            return 0.0
        return max(0.0, min(1.0, (dot / (norm_left * norm_right) + 1.0) / 2.0))
    sq_dist = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    return math.exp(-sq_dist / (2.0 * distance_scale * distance_scale))


def _transposition_invariant_similarity(
    left: NormalizedToken,
    right: NormalizedToken,
    *,
    pc_dims: int,
    ti_alpha: float,
    texture_metric: Literal["cosine", "rbf"] = "cosine",
    distance_scale: float = 1.0,
) -> float:
    """Token similarity with transposition-invariant pitch-class matching.

    Splits the vector into texture features (all but last *pc_dims*) and
    pitch-class histogram (last *pc_dims*).  Texture similarity uses the
    standard metric.  Pitch-class similarity tries all cyclic shifts and
    keeps the best match, so a section transposed up a fifth still scores
    high.
    """
    n = len(left)
    tex_dim = n - pc_dims
    tex_left = left[:tex_dim]
    tex_right = right[:tex_dim]
    pc_left = left[tex_dim:]
    pc_right = right[tex_dim:]

    tex_sim = vector_similarity_score(
        tex_left, tex_right, metric=texture_metric, distance_scale=distance_scale,
    )

    best_pc = 0.0
    for shift in range(pc_dims):
        rotated = pc_right[shift:] + pc_right[:shift]
        dot = sum(float(a) * float(b) for a, b in zip(pc_left, rotated))
        nl = math.sqrt(sum(float(a) * float(a) for a in pc_left))
        nr = math.sqrt(sum(float(b) * float(b) for b in rotated))
        if nl > 0 and nr > 0:
            sim = max(0.0, (dot / (nl * nr) + 1.0) / 2.0)
            best_pc = max(best_pc, sim)
        elif nl == 0 and nr == 0:
            best_pc = max(best_pc, 1.0)

    return ti_alpha * best_pc + (1.0 - ti_alpha) * tex_sim


# -- token / segment similarity -----------------------------------------------


def token_similarity(
    left: NormalizedToken,
    right: NormalizedToken,
    *,
    input_mode: Literal["auto", "vector", "token"] = "auto",
    vector_metric: Literal["cosine", "rbf"] = "cosine",
    vector_distance_scale: float = 1.0,
    same_state_similarity: float = 0.70,
    transposition_invariant: bool = False,
    pc_dims: int = 12,
    ti_alpha: float = 0.5,
) -> float:
    """Similarity for two bar tokens (vector or symbolic)."""

    if input_mode in {"auto", "vector"} and is_numeric_vector(left) and is_numeric_vector(right):
        if transposition_invariant and len(left) > pc_dims:
            return _transposition_invariant_similarity(
                left, right,
                pc_dims=pc_dims, ti_alpha=ti_alpha,
                texture_metric=vector_metric, distance_scale=vector_distance_scale,
            )
        return vector_similarity_score(
            left, right, metric=vector_metric, distance_scale=vector_distance_scale,
        )
    if input_mode == "vector":
        raise ValueError("input_mode='vector' requires every bar item to be a numeric vector")
    if left == right:
        return 1.0
    if left and right and left[0] == right[0]:
        return same_state_similarity
    return 0.0


def segment_similarity(
    tokens: Sequence[NormalizedToken],
    start_a: int,
    start_b: int,
    length: int,
    *,
    input_mode: Literal["auto", "vector", "token"] = "auto",
    vector_metric: Literal["cosine", "rbf"] = "cosine",
    vector_distance_scale: float = 1.0,
    same_state_similarity: float = 0.70,
    transposition_invariant: bool = False,
    pc_dims: int = 12,
    ti_alpha: float = 0.5,
) -> float:
    """Average token similarity over a same-length window pair."""
    if length <= 0:
        raise ValueError("length must be > 0")
    total = 0.0
    for offset in range(length):
        total += token_similarity(
            tokens[start_a + offset],
            tokens[start_b + offset],
            input_mode=input_mode,
            vector_metric=vector_metric,
            vector_distance_scale=vector_distance_scale,
            same_state_similarity=same_state_similarity,
            transposition_invariant=transposition_invariant,
            pc_dims=pc_dims,
            ti_alpha=ti_alpha,
        )
    return total / length


# ---------------------------------------------------------------------------
# Union-Find (disjoint-set)
# ---------------------------------------------------------------------------


class UnionFind:
    """Small deterministic disjoint-set implementation."""

    def __init__(self, values: Iterable[T] = ()) -> None:
        self._parent: Dict[T, T] = {}
        self._rank: Dict[T, int] = {}
        for value in values:
            self.add(value)

    def add(self, value: T) -> None:
        if value not in self._parent:
            self._parent[value] = value
            self._rank[value] = 0

    def find(self, value: T) -> T:
        if value not in self._parent:
            self.add(value)
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: T, right: T) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self._rank[root_left]
        rank_right = self._rank[root_right]
        if rank_left < rank_right:
            self._parent[root_left] = root_right
        elif rank_left > rank_right:
            self._parent[root_right] = root_left
        else:
            self._parent[root_right] = root_left
            self._rank[root_left] += 1

    def components(self) -> List[List[T]]:
        groups: Dict[T, List[T]] = {}
        for value in self._parent:
            groups.setdefault(self.find(value), []).append(value)
        return [sorted(values) for values in groups.values()]


# ---------------------------------------------------------------------------
# SectionMiner
# ---------------------------------------------------------------------------


class SectionMiner:
    """Discover section families with a self-similarity matrix and MDL.

    The miner is intentionally independent from MIDI parsing and clustering.
    Feed it bar-level numeric vectors and it finds repeated sections.

    Parameters
    ----------
    config : SectionMinerConfig
        All tuning parameters.  See :class:`SectionMinerConfig` for defaults.
    """

    def __init__(self, config: SectionMinerConfig | None = None) -> None:
        self.config = config or SectionMinerConfig()
        self.config.validate()

    # -- public API -----------------------------------------------------------

    def analyze(self, tokens: Iterable[Any]) -> SectionStructureResult:
        """Discover section structure from a sequence of bar-level tokens.

        Parameters
        ----------
        tokens : iterable
            Bar-level items.  In vector mode each item is a numeric vector
            (list/tuple of floats).  In token mode each item is a symbolic
            label such as ``"2:M"`` or ``("2", "M")``.

        Returns
        -------
        SectionStructureResult
        """
        normalized = normalize_sequence(tokens)
        if not normalized:
            return SectionStructureResult(
                sequence_length=0,
                tokens=(),
                families=(),
                mixed_sequence=(),
                candidate_pair_count=0,
                rejected_family_count=0,
                config=asdict(self.config),
            )

        ssm = self.build_ssm(normalized)
        pairs = self.find_candidate_pairs(ssm)
        family_specs = self.build_candidate_families(normalized, pairs)
        selected, rejected_count = self.select_families(normalized, family_specs)
        encoded = self.encode_mixed_sequence(normalized, selected)
        return SectionStructureResult(
            sequence_length=len(normalized),
            tokens=normalized,
            families=tuple(selected),
            mixed_sequence=tuple(encoded),
            candidate_pair_count=len(pairs),
            rejected_family_count=rejected_count,
            config=asdict(self.config),
        )

    # -- SSM ------------------------------------------------------------------

    def build_ssm(self, tokens: Sequence[NormalizedToken]) -> List[List[float]]:
        """Build a bar-to-bar self-similarity matrix.

        A returning section appears as a high-valued diagonal stripe: if bars
        i:i+L resemble j:j+L, then SSM[i+k][j+k] is high for every offset k.
        """
        n = len(tokens)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = token_similarity(
                    tokens[i], tokens[j],
                    input_mode=self.config.input_mode,
                    vector_metric=self.config.vector_similarity,
                    vector_distance_scale=self.config.vector_distance_scale,
                    same_state_similarity=self.config.same_state_similarity,
                    transposition_invariant=self.config.transposition_invariant,
                    pc_dims=self.config.pc_dims,
                    ti_alpha=self.config.ti_alpha,
                )
                matrix[i][j] = sim
                matrix[j][i] = sim
        return matrix

    # -- candidate mining -----------------------------------------------------

    def find_candidate_pairs(self, ssm: Sequence[Sequence[float]]) -> List[CandidatePair]:
        """Find same-length non-local similar windows using diagonal prefix sums.

        Each SSM diagonal compares windows separated by a fixed temporal gap.
        Prefix sums let us compute the average similarity of any L-bar window
        pair on that diagonal in O(1), after O(N) setup.
        """
        n = len(ssm)
        max_len = min(self.config.max_len, n)
        pairs: List[CandidatePair] = []
        for diagonal in range(max(self.config.min_gap, 1), n):
            diagonal_len = n - diagonal
            values = [ssm[i][i + diagonal] for i in range(diagonal_len)]
            prefix = [0.0]
            for value in values:
                prefix.append(prefix[-1] + value)

            for length in range(self.config.min_len, max_len + 1):
                if length > diagonal_len:
                    break
                for i in range(0, diagonal_len - length + 1):
                    avg = (prefix[i + length] - prefix[i]) / length
                    if avg >= self.config.similarity_threshold:
                        pairs.append(
                            CandidatePair(
                                start_a=i,
                                start_b=i + diagonal,
                                length=length,
                                similarity=avg,
                            )
                        )
        return pairs

    # -- family building ------------------------------------------------------

    def build_candidate_families(
        self,
        tokens: Sequence[NormalizedToken],
        pairs: Sequence[CandidatePair],
    ) -> List[Dict[str, Any]]:
        """Merge candidate pairs into same-length connected components."""

        by_length: Dict[int, List[CandidatePair]] = {}
        for pair in pairs:
            by_length.setdefault(pair.length, []).append(pair)

        families: List[Dict[str, Any]] = []
        for length, length_pairs in sorted(by_length.items()):
            starts = sorted({p.start_a for p in length_pairs} | {p.start_b for p in length_pairs})
            uf = UnionFind(starts)
            pair_lookup: Dict[Tuple[int, int], float] = {}
            for pair in length_pairs:
                uf.union(pair.start_a, pair.start_b)
                key = (min(pair.start_a, pair.start_b), max(pair.start_a, pair.start_b))
                pair_lookup[key] = max(pair_lookup.get(key, 0.0), pair.similarity)

            for component in uf.components():
                if len(component) < self.config.min_occurrences:
                    continue
                # Filter overlapping starts within the component.  The
                # Union-Find can transitively connect overlapping windows
                # (sliding-window artifact), so we greedily select a
                # non-overlapping subset ordered by centrality.
                filtered = self._filter_overlapping_starts(
                    component, length, pair_lookup,
                )
                if len(filtered) < self.config.min_occurrences:
                    continue
                prototype_start, avg_sim = self._choose_medoid(tokens, filtered, length)
                occurrence_sims = {
                    start: segment_similarity(
                        tokens, prototype_start, start, length,
                        input_mode=self.config.input_mode,
                        vector_metric=self.config.vector_similarity,
                        vector_distance_scale=self.config.vector_distance_scale,
                        same_state_similarity=self.config.same_state_similarity,
                        transposition_invariant=self.config.transposition_invariant,
                        pc_dims=self.config.pc_dims,
                        ti_alpha=self.config.ti_alpha,
                    )
                    for start in filtered
                }
                families.append(
                    {
                        "length": length,
                        "starts": tuple(filtered),
                        "prototype_start": prototype_start,
                        "prototype_tokens": tuple(tokens[prototype_start:prototype_start + length]),
                        "avg_pair_similarity": avg_sim,
                        "occurrence_sims": occurrence_sims,
                    }
                )
        return families

    # -- MDL selection --------------------------------------------------------

    def select_families(
        self,
        tokens: Sequence[NormalizedToken],
        family_specs: Sequence[Dict[str, Any]],
    ) -> Tuple[List[SectionFamily], int]:
        """Greedy MDL selection with non-overlap constraints."""

        scored = [self._score_family(spec) for spec in family_specs]
        scored.sort(
            key=lambda item: (
                item["mdl_gain"],
                item["length"],
                len(item["starts"]),
                item["avg_pair_similarity"],
            ),
            reverse=True,
        )

        selected: List[SectionFamily] = []
        covered: List[Tuple[int, int]] = []
        rejected = 0

        for spec in scored:
            valid_starts = [
                start
                for start in spec["starts"]
                if self._interval_is_available(start, start + spec["length"], covered)
            ]
            if len(valid_starts) < self.config.min_occurrences:
                rejected += 1
                continue

            rescore_spec = dict(spec)
            rescore_spec["starts"] = tuple(valid_starts)
            rescore_spec["occurrence_sims"] = {
                start: spec["occurrence_sims"][start] for start in valid_starts
            }
            rescored = self._score_family(rescore_spec)
            gain_ok = (
                rescored["mdl_gain"] >= 0
                if self.config.allow_zero_gain
                else rescored["mdl_gain"] > 0
            )
            if not gain_ok:
                rejected += 1
                continue

            label = self._family_label(len(selected))
            occurrences = self._label_occurrence_roles(rescored)
            family = SectionFamily(
                label=label,
                length=rescored["length"],
                prototype_start=rescored["prototype_start"],
                prototype_tokens=rescored["prototype_tokens"],
                occurrences=tuple(occurrences),
                avg_pair_similarity=rescored["avg_pair_similarity"],
                define_cost=rescored["define_cost"],
                encoded_saving=rescored["encoded_saving"],
                variant_cost=rescored["variant_cost"],
                mdl_gain=rescored["mdl_gain"],
            )
            selected.append(family)
            for occ in occurrences:
                covered.append((occ.start, occ.end))

        selected.sort(key=lambda family: min(occ.start for occ in family.occurrences))
        return selected, rejected

    # -- mixed sequence -------------------------------------------------------

    def encode_mixed_sequence(
        self,
        tokens: Sequence[NormalizedToken],
        families: Sequence[SectionFamily],
    ) -> List[EncodedItem]:
        """Convert selected families into a mixed SECTION/FREE sequence."""

        by_start: Dict[int, Tuple[SectionFamily, Occurrence]] = {}
        for family in families:
            for occurrence in family.occurrences:
                by_start[occurrence.start] = (family, occurrence)

        encoded: List[EncodedItem] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i in by_start:
                family, occurrence = by_start[i]
                encoded.append(
                    EncodedItem(
                        kind="SECTION",
                        start=occurrence.start,
                        end=occurrence.end,
                        label=family.label,
                        role=occurrence.role,
                        tokens=tuple(tokens[occurrence.start:occurrence.end]),
                    )
                )
                i = occurrence.end
                continue

            free_start = i
            while i < n and i not in by_start:
                i += 1
            encoded.append(
                EncodedItem(
                    kind="FREE",
                    start=free_start,
                    end=i,
                    tokens=tuple(tokens[free_start:i]),
                )
            )
        return encoded

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _filter_overlapping_starts(
        starts: List[int],
        length: int,
        pair_lookup: Dict[Tuple[int, int], float],
    ) -> List[int]:
        """Remove overlapping starts from a connected component.

        Greedily selects non-overlapping starts ordered by descending
        centrality (sum of edge similarities to other starts in the
        component), so the best-connected windows anchor the family.
        """
        if not starts:
            return []
        sorted_starts = sorted(starts)
        n = len(sorted_starts)
        # Score each start by its total pairwise similarity to others
        scores = {s: 0.0 for s in sorted_starts}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sorted_starts[i], sorted_starts[j]
                key = (min(a, b), max(a, b))
                sim = pair_lookup.get(key, 0.0)
                scores[a] += sim
                scores[b] += sim
        # Greedy non-overlapping selection by descending score
        selected: List[int] = []
        covered_ends: List[int] = []  # sorted list of end positions
        for start in sorted(sorted_starts, key=lambda s: scores[s], reverse=True):
            end = start + length
            # Check overlap with already-selected starts
            ok = True
            for sel in selected:
                sel_end = sel + length
                overlap = max(0, min(end, sel_end) - max(start, sel))
                if overlap > 0:
                    ok = False
                    break
            if ok:
                selected.append(start)
        return sorted(selected)

    def _choose_medoid(
        self,
        tokens: Sequence[NormalizedToken],
        starts: Sequence[int],
        length: int,
    ) -> Tuple[int, float]:
        best_start = starts[0]
        best_avg = -1.0
        for candidate in starts:
            similarities = [
                segment_similarity(
                    tokens, candidate, other, length,
                    same_state_similarity=self.config.same_state_similarity,
                    input_mode=self.config.input_mode,
                    vector_metric=self.config.vector_similarity,
                    vector_distance_scale=self.config.vector_distance_scale,
                    transposition_invariant=self.config.transposition_invariant,
                    pc_dims=self.config.pc_dims,
                    ti_alpha=self.config.ti_alpha,
                )
                for other in starts
                if other != candidate
            ]
            avg = sum(similarities) / len(similarities) if similarities else 1.0
            if avg > best_avg or (avg == best_avg and candidate < best_start):
                best_avg = avg
                best_start = candidate
        return best_start, best_avg

    def _score_family(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        starts = tuple(sorted(spec["starts"]))
        length = spec["length"]
        occurrence_sims = spec["occurrence_sims"]
        k = len(starts)
        define_cost = length * self.config.token_cost + self.config.mdl_alpha
        raw_replacement_saving = (
            k * length * self.config.token_cost - k * self.config.section_token_cost
        )
        variant_cost = sum(
            (1.0 - occurrence_sims[start]) * length * self.config.variant_penalty
            for start in starts
        )
        encoded_saving = raw_replacement_saving - variant_cost
        mdl_gain = encoded_saving - define_cost
        rescored = dict(spec)
        rescored.update(
            {
                "starts": starts,
                "define_cost": define_cost,
                "encoded_saving": encoded_saving,
                "variant_cost": variant_cost,
                "mdl_gain": mdl_gain,
            }
        )
        return rescored

    def _label_occurrence_roles(self, spec: Dict[str, Any]) -> List[Occurrence]:
        starts = tuple(sorted(spec["starts"]))
        first_start = starts[0]
        previous_end = None
        occurrences: List[Occurrence] = []
        for start in starts:
            sim = spec["occurrence_sims"][start]
            if start == first_start:
                role = "NEW"
            else:
                gap = start - (previous_end if previous_end is not None else first_start)
                if gap <= self.config.adjacency_gap:
                    role = "REPEAT"
                elif sim >= self.config.return_threshold:
                    role = "RETURN"
                else:
                    role = "VARIANT"
            occurrences.append(
                Occurrence(
                    start=start,
                    length=spec["length"],
                    similarity_to_prototype=sim,
                    role=role,
                )
            )
            previous_end = start + spec["length"]
        return occurrences

    def _interval_is_available(
        self,
        start: int,
        end: int,
        covered: Sequence[Tuple[int, int]],
    ) -> bool:
        length = end - start
        for covered_start, covered_end in covered:
            overlap = max(0, min(end, covered_end) - max(start, covered_start))
            if overlap <= 0:
                continue
            if overlap / length > self.config.max_overlap_fraction:
                return False
        return True

    @staticmethod
    def _family_label(index: int) -> str:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if index < len(alphabet):
            return alphabet[index]
        return f"S{index + 1}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    cfg = SectionMinerConfig()

    parser = argparse.ArgumentParser(
        description="Discover section structure with MDL + self-similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python section_miner.py --music-dir ../../datasets/corelli --pretty
  python section_miner.py --input bars.json --transposition-invariant
  python section_miner.py --music-dir data/ --ti --pc-alpha 0.6 --min-len 4
        """,
    )

    # Input sources (mutually exclusive in spirit; --input takes precedence)
    parser.add_argument(
        "--music-dir",
        default=None,
        help="Directory containing music files (recursive).",
    )
    parser.add_argument(
        "--file-patterns",
        default="*.mid,*.midi,*.abc,*.krn",
        help="Comma-separated glob patterns for music files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSON file (bar vectors). Overrides --music-dir.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output.",
    )

    # Config flags
    parser.add_argument("--min-len", type=int, default=cfg.min_len)
    parser.add_argument("--max-len", type=int, default=cfg.max_len)
    parser.add_argument("--min-gap", type=int, default=cfg.min_gap)
    parser.add_argument("--similarity-threshold", type=float, default=cfg.similarity_threshold)
    parser.add_argument("--mdl-alpha", type=float, default=cfg.mdl_alpha)
    parser.add_argument("--variant-penalty", type=float, default=cfg.variant_penalty)
    parser.add_argument("--min-occurrences", type=int, default=cfg.min_occurrences)
    parser.add_argument(
        "--allow-zero-gain", action="store_true",
        help="Keep zero-gain families.",
    )
    parser.add_argument(
        "--transposition-invariant", "--ti", action="store_true",
        dest="transposition_invariant",
        help="Enable transposition-invariant pitch-class matching.",
    )
    parser.add_argument(
        "--pc-dims", type=int, default=cfg.pc_dims,
        help="Number of pitch-class dimensions at the end of each vector.",
    )
    parser.add_argument(
        "--pc-alpha", type=float, default=cfg.ti_alpha,
        help="Weight of pitch-class vs texture similarity when --ti is on.",
    )
    return parser


def _load_json_input(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return [{"id": f"sequence-{idx + 1}", "tokens": tokens}
                for idx, tokens in enumerate(data)]
    if "songs" in data:
        return list(data["songs"])
    if "tokens" in data:
        return [{"id": data.get("id", path.stem), "tokens": data["tokens"]}]
    raise ValueError("JSON must be a token list, {'tokens': ...}, or {'songs': [...]}")


def _load_music_files(
    music_dir: str, file_patterns: str,
) -> List[Dict[str, Any]]:
    """Extract bar vectors from music files using MeasureExtractor."""
    from measure_clustering import MeasureExtractor

    patterns = [p.strip() for p in file_patterns.split(",") if p.strip()]
    extractor = MeasureExtractor()

    music_paths: List[Path] = []
    music_dir_path = Path(music_dir)
    for pat in patterns:
        music_paths.extend(sorted(music_dir_path.rglob(pat)))
    music_paths = sorted(set(music_paths))

    if not music_paths:
        raise FileNotFoundError(
            f"No music files matching {patterns} found in {music_dir}"
        )

    songs: List[Dict[str, Any]] = []
    for mp in music_paths:
        try:
            measures = extractor.extract(mp)
            if len(measures) < 2:
                continue
            measures.sort(key=lambda m: m.measure_index)
            vectors = [extractor.vectorize(m) for m in measures]
            # Use full array (texture + pitch-class histogram) for richer analysis
            songs.append({
                "id": mp.stem,
                "tokens": [v.as_full_array().tolist() for v in vectors],
            })
        except Exception as exc:
            print(f"Warning: skipping {mp.name}: {exc}", file=sys.stderr)

    if not songs:
        raise RuntimeError(f"No valid music files in {music_dir}")
    return songs


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    config = SectionMinerConfig(
        min_len=args.min_len,
        max_len=args.max_len,
        min_gap=args.min_gap,
        similarity_threshold=args.similarity_threshold,
        mdl_alpha=args.mdl_alpha,
        variant_penalty=args.variant_penalty,
        min_occurrences=args.min_occurrences,
        allow_zero_gain=args.allow_zero_gain,
        transposition_invariant=args.transposition_invariant,
        pc_dims=args.pc_dims,
        ti_alpha=args.pc_alpha,
    )

    if args.input:
        jobs = _load_json_input(args.input)
    elif args.music_dir:
        print(f"Extracting measures from {args.music_dir} ...", file=sys.stderr)
        jobs = _load_music_files(args.music_dir, args.file_patterns)
        print(f"Loaded {len(jobs)} songs.", file=sys.stderr)
    else:
        print("Error: either --input or --music-dir is required.", file=sys.stderr)
        return 1

    miner = SectionMiner(config)
    output = {
        "input": str(args.input or args.music_dir),
        "results": [
            {
                "id": job.get("id", f"sequence-{idx + 1}"),
                "analysis": miner.analyze(job["tokens"]).to_dict(),
            }
            for idx, job in enumerate(jobs)
        ],
    }

    text = json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
