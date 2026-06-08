#!/usr/bin/env python3
"""Bar clustering strategies and observation vocabulary construction."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from bar_density import TokenDensityAnalyzer
from config_loader import ConfigLoader, ConfigView
from core_data import BarRecord, ObservationVocab, SongRecord
from edit_distance import EditDistanceCalculator, EditDistanceDiagnosticsAnalyzer
from generation_data import CodebookCandidate, CodebookEntry


log = logging.getLogger("bar_clustering")


@dataclass(frozen=True)
class GlobalCodebookConfig:
    song_distance_threshold: float = 0.25
    song_max_clusters: int = 48
    codebook_size: int = 1024
    codebook_clustering_strategy: str = "default"
    codebook_filter_min_bar_count: int = 1
    codebook_distance_threshold: Optional[float] = None
    assignment_distance_threshold: Optional[float] = None
    expand_codebook_on_assignment_miss: bool = False
    song_linkage: str = "average"
    codebook_linkage: str = "average"
    criterion: str = "distance"


@dataclass(frozen=True)
class KMeansFeatureConfig:
    enabled: bool = False
    n_clusters: int = 8
    random_seed: int = 42


@dataclass(frozen=True)
class ObservationVocabConfig:
    strategy: str = "composite"
    position_strategy: str = "period_role"
    position_modulo: int = 8
    position_source: str = "bar_index"
    key_format: str = "structured"


@dataclass(frozen=True)
class SongClusterResult:
    labels: List[int]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class MedoidCandidate:
    bar: BarRecord
    represented_bar_count: int
    member_bar_ids: tuple[int, ...]


@dataclass(frozen=True)
class CodebookBuildResult:
    codebook: List[BarRecord]
    diagnostics: Dict[str, Any]
    forced_rare_bar_ids: Optional[set[int]] = None
    reserved_rare_cluster_id: Optional[int] = None


class SongAgglomerativeClusterer:
    """Cluster bars inside one song with distance threshold plus max-cluster cap."""

    def __init__(self, config: GlobalCodebookConfig) -> None:
        self.config = config

    def cluster(self, matrix: np.ndarray) -> SongClusterResult:
        if matrix.shape[0] <= 1:
            labels = [0 for _ in range(matrix.shape[0])]
            return SongClusterResult(labels, {
                "mode": "single_bar",
                "threshold_cluster_count": int(len(set(map(int, labels)))),
                "final_cluster_count": int(len(set(map(int, labels)))),
                "max_clusters_applied": False,
            })
        z_matrix = linkage(squareform(matrix, checks=False), method=self.config.song_linkage)
        threshold_labels = self._compact(fcluster(
            z_matrix,
            t=self.config.song_distance_threshold,
            criterion=self.config.criterion,
        ).tolist())
        threshold_count = int(len(set(map(int, threshold_labels))))
        if threshold_count <= self.config.song_max_clusters:
            return SongClusterResult(threshold_labels, {
                "mode": "distance_threshold",
                "threshold": self.config.song_distance_threshold,
                "threshold_cluster_count": threshold_count,
                "final_cluster_count": threshold_count,
                "max_clusters": self.config.song_max_clusters,
                "max_clusters_applied": False,
            })
        capped_labels = self._compact(fcluster(
            z_matrix,
            t=min(self.config.song_max_clusters, matrix.shape[0]),
            criterion="maxclust",
        ).tolist())
        return SongClusterResult(capped_labels, {
            "mode": "distance_threshold_with_maxclust_cap",
            "threshold": self.config.song_distance_threshold,
            "threshold_cluster_count": threshold_count,
            "final_cluster_count": int(len(set(map(int, capped_labels)))),
            "max_clusters": self.config.song_max_clusters,
            "max_clusters_applied": True,
        })

    def _compact(self, labels: Sequence[int]) -> List[int]:
        mapping: Dict[int, int] = {}
        compact: List[int] = []
        for label in labels:
            value = int(label)
            if value not in mapping:
                mapping[value] = len(mapping)
            compact.append(mapping[value])
        return compact


class LabelDistributionAnalyzer:
    """Summarize whether codebook assignment collapses into a few labels."""

    def __init__(
        self,
        density_analyzer: TokenDensityAnalyzer,
        distance_calculator: EditDistanceCalculator,
    ) -> None:
        self.density_analyzer = density_analyzer
        self.distance_calculator = distance_calculator

    def analyze(self, counts: Counter, codebook: Sequence[BarRecord]) -> Dict[str, Any]:
        values = [int(value) for value in counts.values() if int(value) > 0]
        total = int(sum(values))
        if total <= 0:
            return {
                "total_assignments": 0,
                "used_label_count": 0,
                "max_label": None,
                "max_label_count": 0,
                "max_label_ratio": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "effective_label_count": 0.0,
                "gini": 0.0,
            }
        probabilities = np.asarray(values, dtype=np.float64) / float(total)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        used_count = len(values)
        normalized_entropy = entropy / np.log(used_count) if used_count > 1 else 0.0
        max_label, max_count = max(counts.items(), key=lambda item: int(item[1]))
        return {
            "total_assignments": total,
            "used_label_count": used_count,
            "max_label": str(max_label),
            "max_label_count": int(max_count),
            "max_label_ratio": round(float(max_count) / float(total), 6),
            "entropy": round(entropy, 6),
            "normalized_entropy": round(float(normalized_entropy), 6),
            "effective_label_count": round(float(np.exp(entropy)), 6),
            "gini": round(self._gini(values), 6),
            "top_labels": [
                {"label": str(label), "count": int(count), "ratio": round(float(count) / float(total), 6)}
                for label, count in counts.most_common(20)
            ],
            "top_labels_detail": [
                self._label_detail(label, count, total, codebook)
                for label, count in counts.most_common(20)
            ],
        }

    def _label_detail(
        self,
        label: int,
        count: int,
        total: int,
        codebook: Sequence[BarRecord],
    ) -> Dict[str, Any]:
        index = int(label)
        if index < 0 or index >= len(codebook):
            return {
                "label": str(label),
                "count": int(count),
                "ratio": round(float(count) / float(total), 6),
                "error": "label is outside codebook range",
            }
        bar = codebook[index]
        edit_distance_tokens = self.distance_calculator.tokens_for_bar(bar)
        relative_tokens = bar.tokens_for_edit_distance("relative")
        absolute_tokens = list(bar.absolute_tokens)
        return {
            "label": str(label),
            "count": int(count),
            "ratio": round(float(count) / float(total), 6),
            "source_song": bar.song_id,
            "source_file": bar.file_path,
            "source_bar_index": int(bar.bar_index),
            "edit_distance_tokens": edit_distance_tokens,
            "token_strategy": self.distance_calculator.config.token_strategy,
            "relative_tokens": list(bar.relative_tokens),
            "absolute_tokens": absolute_tokens,
            "token_variance": round(float(bar.token_variance), 6),
            "sharing_score": round(float(bar.sharing_score), 6),
            "density": self.density_analyzer.analyze(relative_tokens).to_dict(),
        }

    def _gini(self, values: Sequence[int]) -> float:
        if not values:
            return 0.0
        sorted_values = np.sort(np.asarray(values, dtype=np.float64))
        n = len(sorted_values)
        total = float(np.sum(sorted_values))
        if total <= 0:
            return 0.0
        weighted_sum = float(np.sum((np.arange(1, n + 1) * sorted_values)))
        return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


class CodebookAssignmentDiagnosticsAnalyzer:
    """Summarize nearest-codebook assignment distances."""

    def analyze(self, assignments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        distances = np.asarray(
            [float(item["nearest_distance"]) for item in assignments],
            dtype=np.float64,
        )
        if not distances.size:
            return {
                "assignment_count": 0,
                "assignment_policy_counts": {},
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "quantiles": {},
                "worst_assignments": [],
            }
        return {
            "assignment_count": int(distances.size),
            "assignment_policy_counts": {
                str(key): int(value)
                for key, value in Counter(str(item.get("assignment_policy", "unknown")) for item in assignments).items()
            },
            "min": round(float(np.min(distances)), 6),
            "max": round(float(np.max(distances)), 6),
            "mean": round(float(np.mean(distances)), 6),
            "std": round(float(np.std(distances)), 6),
            "median": round(float(np.median(distances)), 6),
            "quantiles": {
                str(percentile): round(float(np.percentile(distances, percentile)), 6)
                for percentile in (50, 75, 90, 95, 99)
            },
            "worst_assignments": sorted(
                assignments,
                key=lambda item: float(item["nearest_distance"]),
                reverse=True,
            )[:20],
        }


class CodebookClusteringStrategy:
    """Build a global codebook from per-song medoid candidates."""

    def build(
        self,
        clusterer: "GlobalCodebookClusterer",
        candidates: Sequence[MedoidCandidate],
        songs: Sequence[SongRecord],
    ) -> CodebookBuildResult:
        raise NotImplementedError


class DefaultCodebookClusteringStrategy(CodebookClusteringStrategy):
    """Cluster all medoid candidates using the configured global codebook logic."""

    def build(
        self,
        clusterer: "GlobalCodebookClusterer",
        candidates: Sequence[MedoidCandidate],
        songs: Sequence[SongRecord],
    ) -> CodebookBuildResult:
        bars = [candidate.bar for candidate in candidates]
        codebook, diagnostics = clusterer._build_codebook_from_bars(bars)
        diagnostics["strategy"] = "default"
        diagnostics["candidate_count"] = len(candidates)
        return CodebookBuildResult(codebook, diagnostics)


class FilteredRareBarCodebookClusteringStrategy(CodebookClusteringStrategy):
    """Exclude rare medoids from main clustering and map them to a reserved cluster."""

    def build(
        self,
        clusterer: "GlobalCodebookClusterer",
        candidates: Sequence[MedoidCandidate],
        songs: Sequence[SongRecord],
    ) -> CodebookBuildResult:
        threshold = int(clusterer.config.codebook_filter_min_bar_count)
        all_bars = [bar for song in songs for bar in song.bars]
        physical_counts = Counter(self._bar_key(bar) for bar in all_bars)
        frequent_bars = [
            bar
            for bar in all_bars
            if int(physical_counts[self._bar_key(bar)]) > threshold
        ]
        rare_bars = [
            bar
            for bar in all_bars
            if int(physical_counts[self._bar_key(bar)]) <= threshold
        ]
        main_size = max(1, int(clusterer.config.codebook_size) - 1)
        codebook, diagnostics = clusterer._build_codebook_from_bars(
            frequent_bars,
            codebook_size=main_size,
            force_fixed_count=True,
        )
        rare_representative = self._rare_representative(clusterer, rare_bars)
        if rare_representative is not None:
            codebook.append(rare_representative)
        diagnostics.update({
            "strategy": "filtered_rare_bar",
            "filter_rule": "global_physical_bar_count_by_relative_tokens > threshold",
            "filter_threshold": threshold,
            "candidate_count": len(all_bars),
            "unique_bar_pattern_count": len(physical_counts),
            "frequent_candidate_count": len(frequent_bars),
            "rare_candidate_count": len(rare_bars),
            "main_codebook_size": len(codebook) - (1 if rare_representative is not None else 0),
            "rare_cluster_added": rare_representative is not None,
            "reserved_rare_cluster_id": len(codebook) - 1 if rare_representative is not None else None,
        })
        forced_rare_bar_ids = {
            id(bar)
            for bar in rare_bars
        }
        return CodebookBuildResult(
            codebook,
            diagnostics,
            forced_rare_bar_ids=forced_rare_bar_ids,
            reserved_rare_cluster_id=len(codebook) - 1 if rare_representative is not None else None,
        )

    def _rare_representative(
        self,
        clusterer: "GlobalCodebookClusterer",
        rare: Sequence[BarRecord],
    ) -> Optional[BarRecord]:
        if not rare:
            return None
        if len(rare) == 1:
            return rare[0]
        bars = list(rare)
        matrix = clusterer.distance_calculator.build_matrix(bars)
        medoid = clusterer._medoids(matrix, [0 for _ in bars])[0]
        return bars[int(medoid)]

    def _bar_key(self, bar: BarRecord) -> tuple[int, ...]:
        return tuple(int(token) for token in bar.tokens_for_edit_distance("relative"))


class CodebookClusteringStrategyFactory:
    """Create the configured global codebook clustering strategy."""

    def from_config(self, config: GlobalCodebookConfig) -> CodebookClusteringStrategy:
        if config.codebook_clustering_strategy == "default":
            return DefaultCodebookClusteringStrategy()
        if config.codebook_clustering_strategy == "filtered_rare_bar":
            return FilteredRareBarCodebookClusteringStrategy()
        raise ValueError(
            "global_agglomerative_clustering.codebook_clustering_strategy "
            "must be 'default' or 'filtered_rare_bar'."
        )


class GlobalCodebookClusterer:
    """Per-song clustering plus global medoid-codebook quantization."""

    def __init__(
        self,
        config: GlobalCodebookConfig,
        distance_calculator: EditDistanceCalculator,
        density_analyzer: TokenDensityAnalyzer,
    ) -> None:
        self.config = config
        self.distance_calculator = distance_calculator
        self.density_analyzer = density_analyzer
        self.diagnostics: Dict[str, Any] = {}
        self.codebook: List[BarRecord] = []

    @classmethod
    def from_style_config(
        cls,
        config: Dict[str, Any],
        distance_calculator: EditDistanceCalculator,
    ) -> "GlobalCodebookClusterer":
        section = ConfigView(config).section("global_agglomerative_clustering")
        return cls(GlobalCodebookConfig(
            song_distance_threshold=float(section.get("song_distance_threshold", 0.25)),
            song_max_clusters=int(section.get("song_max_clusters", 48)),
            codebook_size=int(section.get("codebook_size", 1024)),
            codebook_clustering_strategy=str(section.get("codebook_clustering_strategy", "default")),
            codebook_filter_min_bar_count=int(section.get("codebook_filter_min_bar_count", 1)),
            codebook_distance_threshold=cls._optional_float(section.get("codebook_distance_threshold")),
            assignment_distance_threshold=cls._optional_float(section.get("assignment_distance_threshold")),
            expand_codebook_on_assignment_miss=bool(section.get("expand_codebook_on_assignment_miss", False)),
            song_linkage=str(section.get("song_linkage", "average")),
            codebook_linkage=str(section.get("codebook_linkage", "average")),
            criterion=str(section.get("criterion", "distance")),
        ), distance_calculator, TokenDensityAnalyzer.from_style_config(config))

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        return None if value is None else float(value)

    def assign(self, songs: Sequence[SongRecord]) -> None:
        medoids: List[MedoidCandidate] = []
        song_diagnostics: List[Dict[str, Any]] = []
        matrix_analyzer = EditDistanceDiagnosticsAnalyzer(self.distance_calculator.config)
        song_clusterer = SongAgglomerativeClusterer(self.config)
        for song in songs:
            if not song.bars:
                continue
            matrix = self.distance_calculator.build_matrix(song.bars)
            log.info("Built song edit-distance matrix: song=%s shape=%s", song.song_id, matrix.shape)
            cluster_result = song_clusterer.cluster(matrix)
            labels = cluster_result.labels
            medoid_indices = self._medoids(matrix, labels)
            label_counts = Counter(map(int, labels))
            log.info(
                "Clustered song edit-distance matrix: song=%s bars=%d clusters=%d medoids=%d",
                song.song_id,
                len(song.bars),
                len(set(map(int, labels))),
                len(medoid_indices),
            )
            for local_label, local_index in medoid_indices.items():
                member_indices = [
                    index
                    for index, label in enumerate(labels)
                    if int(label) == int(local_label)
                ]
                medoids.append(MedoidCandidate(
                    bar=song.bars[local_index],
                    represented_bar_count=int(label_counts[int(local_label)]),
                    member_bar_ids=tuple(id(song.bars[index]) for index in member_indices),
                ))
            song_diagnostics.append({
                "song_id": song.song_id,
                "file_path": song.file_path,
                "n_bars": len(song.bars),
                "n_song_clusters": len(set(map(int, labels))),
                "distance_matrix": matrix_analyzer.summarize(
                    matrix,
                    thresholds=[
                        self.config.song_distance_threshold,
                        self.config.song_distance_threshold * 0.5,
                        self.config.song_distance_threshold * 2.0,
                    ],
                ),
                "cluster_reduction": cluster_result.diagnostics,
                "medoids": [
                    {
                        "song_cluster_label": int(label),
                        "bar_index": int(song.bars[index].bar_index),
                        "represented_bar_count": int(label_counts[int(label)]),
                    }
                    for label, index in sorted(medoid_indices.items())
                ],
            })
        build_result = CodebookClusteringStrategyFactory().from_config(self.config).build(self, medoids, songs)
        codebook = build_result.codebook
        codebook_matrix_diagnostics = build_result.diagnostics
        assignment_records = self._assign_nearest_edit_distance_id(
            songs,
            codebook,
            forced_rare_bar_ids=build_result.forced_rare_bar_ids or set(),
            reserved_rare_cluster_id=build_result.reserved_rare_cluster_id,
        )
        preserve_ids = (
            set(range(int(build_result.reserved_rare_cluster_id) + 1))
            if build_result.reserved_rare_cluster_id is not None
            else set()
        )
        deduplication_diagnostics, id_mapping = self._deduplicate_codebook_by_relative_tokens(
            songs,
            codebook,
            preserve_ids=preserve_ids,
        )
        self._remap_assignment_records(assignment_records, id_mapping)
        self.codebook = list(codebook)
        log.info(
            "Built global edit-distance codebook: medoids=%d codebook_size=%d assignments=%d forced_rare_assignments=%d duplicate_entries=%d",
            len(medoids),
            len(codebook),
            len(assignment_records),
            sum(1 for record in assignment_records if record.get("assignment_policy") == "forced_rare_cluster"),
            deduplication_diagnostics["duplicate_entry_count"],
        )
        counts = Counter(bar.edit_distance_id for song in songs for bar in song.bars)
        self.diagnostics = {
            "backend": "global_agglomerative",
            "config": asdict(self.config),
            "n_songs": len(songs),
            "n_song_medoids": len(medoids),
            "requested_codebook_size": self.config.codebook_size,
            "initial_codebook_size": codebook_matrix_diagnostics.get("selected_codebook_size", len(codebook)),
            "actual_codebook_size": len(codebook),
            "assignment_miss_expansion_enabled": self._can_expand_codebook_on_assignment_miss(),
            "codebook_deduplication": deduplication_diagnostics,
            "medoid_distance_matrix": codebook_matrix_diagnostics,
            "assignment_distance": CodebookAssignmentDiagnosticsAnalyzer().analyze(assignment_records),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
            "label_distribution": LabelDistributionAnalyzer(
                self.density_analyzer,
                self.distance_calculator,
            ).analyze(counts, codebook),
            "song_diagnostics": song_diagnostics,
            "codebook": [
                {
                    "edit_distance_id": int(index),
                    "source_song": bar.song_id,
                    "source_bar_index": int(bar.bar_index),
                    "token_strategy": self.distance_calculator.config.token_strategy,
                    "edit_distance_tokens": self.distance_calculator.tokens_for_bar(bar),
                    "relative_tokens": bar.tokens_for_edit_distance("relative"),
                    "token_variance": round(float(bar.token_variance), 6),
                    "sharing_score": round(float(bar.sharing_score), 6),
                }
                for index, bar in enumerate(codebook)
            ],
        }

    def _cluster_by_count(self, matrix: np.ndarray, count: int, method: str) -> List[int]:
        if matrix.shape[0] <= 1:
            return [0 for _ in range(matrix.shape[0])]
        target = min(max(1, count), matrix.shape[0])
        z_matrix = linkage(squareform(matrix, checks=False), method=method)
        return self._compact(self._labels_from_linkage_exact_count(z_matrix, matrix.shape[0], target))

    def _labels_from_linkage_exact_count(
        self,
        z_matrix: np.ndarray,
        item_count: int,
        target_count: int,
    ) -> List[int]:
        parent = list(range(item_count))
        members: Dict[int, List[int]] = {index: [index] for index in range(item_count)}
        next_cluster_id = item_count

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        merge_count = max(0, item_count - target_count)
        for left_raw, right_raw, _distance, _size in z_matrix[:merge_count]:
            left_id = int(left_raw)
            right_id = int(right_raw)
            left_members = members.pop(left_id)
            right_members = members.pop(right_id)
            merged_members = left_members + right_members
            parent.append(next_cluster_id)
            for member in merged_members:
                parent[find(member)] = next_cluster_id
            members[next_cluster_id] = merged_members
            next_cluster_id += 1

        root_to_label: Dict[int, int] = {}
        labels: List[int] = []
        for item in range(item_count):
            root = find(item)
            if root not in root_to_label:
                root_to_label[root] = len(root_to_label)
            labels.append(root_to_label[root])
        return labels

    def _cluster_codebook(self, matrix: np.ndarray, codebook_size: Optional[int] = None) -> tuple[List[int], Dict[str, Any]]:
        target_size = int(codebook_size if codebook_size is not None else self.config.codebook_size)
        if matrix.shape[0] <= 1:
            return [0 for _ in range(matrix.shape[0])], {
                "mode": "single_medoid",
                "threshold_cluster_count": int(matrix.shape[0]),
                "final_cluster_count": int(matrix.shape[0]),
                "max_clusters_applied": False,
            }
        z_matrix = linkage(squareform(matrix, checks=False), method=self.config.codebook_linkage)
        if self.config.codebook_distance_threshold is None:
            labels = self._compact(fcluster(
                z_matrix,
                t=min(max(1, target_size), matrix.shape[0]),
                criterion="maxclust",
            ).tolist())
            return labels, {
                "mode": "fixed_codebook_size",
                "threshold": None,
                "threshold_cluster_count": None,
                "final_cluster_count": len(set(map(int, labels))),
                "max_clusters": target_size,
                "max_clusters_applied": True,
            }
        threshold_labels = self._compact(fcluster(
            z_matrix,
            t=float(self.config.codebook_distance_threshold),
            criterion="distance",
        ).tolist())
        threshold_count = len(set(map(int, threshold_labels)))
        if threshold_count <= target_size:
            return threshold_labels, {
                "mode": "distance_threshold",
                "threshold": self.config.codebook_distance_threshold,
                "threshold_cluster_count": threshold_count,
                "final_cluster_count": threshold_count,
                "max_clusters": target_size,
                "max_clusters_applied": False,
            }
        capped_labels = self._compact(fcluster(
            z_matrix,
            t=min(max(1, target_size), matrix.shape[0]),
            criterion="maxclust",
        ).tolist())
        return capped_labels, {
            "mode": "distance_threshold_with_maxclust_cap",
            "threshold": self.config.codebook_distance_threshold,
            "threshold_cluster_count": threshold_count,
            "final_cluster_count": len(set(map(int, capped_labels))),
            "max_clusters": target_size,
            "max_clusters_applied": True,
        }

    def _build_codebook_from_bars(
        self,
        bars: Sequence[BarRecord],
        codebook_size: Optional[int] = None,
        force_fixed_count: bool = False,
    ) -> tuple[List[BarRecord], Dict[str, Any]]:
        if not bars:
            return [], {}
        matrix = self.distance_calculator.build_matrix(bars)
        log.info("Built global medoid edit-distance matrix: shape=%s", matrix.shape)
        if force_fixed_count:
            target_size = int(codebook_size if codebook_size is not None else self.config.codebook_size)
            labels = self._cluster_by_count(matrix, target_size, self.config.codebook_linkage)
            cluster_diagnostics = {
                "mode": "forced_fixed_count",
                "final_cluster_count": len(set(map(int, labels))),
                "max_clusters": min(max(1, target_size), matrix.shape[0]),
                "max_clusters_applied": True,
            }
        else:
            labels, cluster_diagnostics = self._cluster_codebook(matrix, codebook_size=codebook_size)
        medoid_indices = self._medoids(matrix, labels)
        diagnostics = EditDistanceDiagnosticsAnalyzer(self.distance_calculator.config).summarize(matrix)
        diagnostics["cluster_reduction"] = cluster_diagnostics
        diagnostics["selected_codebook_size"] = len(medoid_indices)
        return [bars[index] for _, index in sorted(medoid_indices.items())], diagnostics

    def _assign_nearest_edit_distance_id(
        self,
        songs: Sequence[SongRecord],
        codebook: List[BarRecord],
        forced_rare_bar_ids: set[int],
        reserved_rare_cluster_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        assignments: List[Dict[str, Any]] = []
        if not codebook:
            return assignments
        for song in songs:
            for bar in song.bars:
                assignment_policy = "nearest_edit_distance"
                distances = [self.distance_calculator.distance(bar, candidate) for candidate in codebook]
                nearest_id = int(np.argmin(distances))
                nearest_distance = float(distances[nearest_id])
                if id(bar) in forced_rare_bar_ids and reserved_rare_cluster_id is not None:
                    nearest_id = int(reserved_rare_cluster_id)
                    nearest_distance = float(distances[nearest_id])
                    assignment_policy = "forced_rare_cluster"
                elif (
                    self._can_expand_codebook_on_assignment_miss()
                    and self.config.assignment_distance_threshold is not None
                    and nearest_distance > float(self.config.assignment_distance_threshold)
                ):
                    codebook.append(bar)
                    nearest_id = len(codebook) - 1
                    nearest_distance = 0.0
                    assignment_policy = "assignment_miss_expansion"
                bar.edit_distance_id = nearest_id
                assignments.append({
                    "song_id": song.song_id,
                    "file_path": song.file_path,
                    "bar_index": int(bar.bar_index),
                    "edit_distance_id": nearest_id,
                    "nearest_distance": round(nearest_distance, 6),
                    "assignment_policy": assignment_policy,
                    "edit_distance_tokens": self.distance_calculator.tokens_for_bar(bar),
                    "relative_tokens": bar.tokens_for_edit_distance("relative"),
                    "absolute_tokens": list(bar.absolute_tokens),
                    "token_variance": round(float(bar.token_variance), 6),
                    "sharing_score": round(float(bar.sharing_score), 6),
                })
        return assignments

    def _can_expand_codebook_on_assignment_miss(self) -> bool:
        if self.config.codebook_clustering_strategy == "filtered_rare_bar":
            return False
        return bool(self.config.expand_codebook_on_assignment_miss)

    def _deduplicate_codebook_by_relative_tokens(
        self,
        songs: Sequence[SongRecord],
        codebook: List[BarRecord],
        preserve_ids: set[int],
    ) -> tuple[Dict[str, Any], Dict[int, int]]:
        canonical_by_tokens: Dict[tuple[int, ...], int] = {}
        canonical_entries: List[BarRecord] = []
        id_mapping: Dict[int, int] = {}
        duplicate_groups: Dict[int, List[int]] = defaultdict(list)
        for old_id, bar in enumerate(codebook):
            key = tuple(int(token) for token in bar.tokens_for_edit_distance("relative"))
            if old_id in preserve_ids:
                new_id = len(canonical_entries)
                canonical_entries.append(bar)
            elif key in canonical_by_tokens:
                new_id = canonical_by_tokens[key]
                duplicate_groups[new_id].append(old_id)
            else:
                new_id = len(canonical_entries)
                canonical_by_tokens[key] = new_id
                canonical_entries.append(bar)
            id_mapping[old_id] = new_id
        if len(canonical_entries) != len(codebook):
            codebook[:] = canonical_entries
            for song in songs:
                for bar in song.bars:
                    if bar.edit_distance_id is not None:
                        bar.edit_distance_id = id_mapping[int(bar.edit_distance_id)]
        return {
            "strategy": "relative_tokens",
            "preserved_edit_distance_ids": sorted(int(item) for item in preserve_ids),
            "before_size": len(id_mapping),
            "after_size": len(canonical_entries),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_entry_count": len(id_mapping) - len(canonical_entries),
            "largest_duplicate_groups": [
                {
                    "canonical_edit_distance_id": int(canonical_id),
                    "duplicate_edit_distance_ids": [int(item) for item in duplicates[:30]],
                    "duplicate_count": len(duplicates),
                    "relative_tokens": codebook[int(canonical_id)].tokens_for_edit_distance("relative"),
                }
                for canonical_id, duplicates in sorted(
                    duplicate_groups.items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )[:20]
            ],
        }, id_mapping

    def _remap_assignment_records(
        self,
        assignments: Sequence[Dict[str, Any]],
        id_mapping: Dict[int, int],
    ) -> None:
        for assignment in assignments:
            old_id = int(assignment["edit_distance_id"])
            assignment["edit_distance_id"] = int(id_mapping.get(old_id, old_id))

    def _medoids(self, matrix: np.ndarray, labels: Sequence[int]) -> Dict[int, int]:
        result: Dict[int, int] = {}
        label_array = np.asarray(labels, dtype=int)
        for label in sorted(set(map(int, labels))):
            members = np.where(label_array == label)[0]
            if len(members) == 1:
                result[int(label)] = int(members[0])
                continue
            sub_matrix = matrix[members[:, None], members]
            result[int(label)] = int(members[int(np.argmin(np.sum(sub_matrix, axis=1)))])
        return result

    def _compact(self, labels: Sequence[int]) -> List[int]:
        mapping: Dict[int, int] = {}
        compact: List[int] = []
        for label in labels:
            value = int(label)
            if value not in mapping:
                mapping[value] = len(mapping)
            compact.append(mapping[value])
        return compact


class KMeansFeatureClusterer:
    """Optional feature-vector clustering used as an extra observation facet."""

    def __init__(self, config: KMeansFeatureConfig) -> None:
        self.config = config
        self.diagnostics: Dict[str, Any] = {}

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "KMeansFeatureClusterer":
        section = ConfigView(config).section("kmeans_clustering")
        return cls(KMeansFeatureConfig(
            enabled=bool(section.get("enabled", False)),
            n_clusters=int(section.get("n_clusters", 8)),
            random_seed=int(section.get("random_seed", 42)),
        ))

    def assign(self, songs: Sequence[SongRecord]) -> None:
        bars = [bar for song in songs for bar in song.bars]
        if not self.config.enabled or not bars:
            for bar in bars:
                bar.kmeans_id = None
            self.diagnostics = {"enabled": False}
            return
        vectors = np.asarray([bar.feature_vector for bar in bars], dtype=np.float64)
        n_clusters = min(self.config.n_clusters, len(bars))
        labels = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed, n_init="auto").fit_predict(vectors)
        for bar, label in zip(bars, labels):
            bar.kmeans_id = int(label)
        self.diagnostics = {
            "enabled": True,
            "n_clusters": int(n_clusters),
            "label_counts": {str(k): int(v) for k, v in Counter(map(int, labels)).items()},
        }


class ObservationVocabBuilder:
    """Create contiguous HMM observation IDs from structured composite keys."""

    def __init__(self, config: ObservationVocabConfig | None = None) -> None:
        self.config = config or ObservationVocabConfig()

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "ObservationVocabBuilder":
        section = ConfigView(config).section("observation_vocab")
        return cls(ObservationVocabConfig(
            strategy=str(section.get("strategy", "composite")),
            position_strategy=str(section.get("position_strategy", "period_role")),
            position_modulo=int(section.get("position_modulo", 8)),
            position_source=str(section.get("position_source", "bar_index")),
            key_format=str(section.get("key_format", "structured")),
        ))

    def assign(self, songs: Sequence[SongRecord]) -> ObservationVocab:
        bars = [bar for song in songs for bar in song.bars]
        for bar in bars:
            bar.composite_key = self._observation_key(bar)
        unique_keys = sorted({str(bar.composite_key) for bar in bars})
        composite_to_observation = {key: index for index, key in enumerate(unique_keys)}
        observation_to_composite = {index: key for key, index in composite_to_observation.items()}
        composite_parts = {}
        for bar in bars:
            composite_parts[str(bar.composite_key)] = self._observation_parts(bar)
            bar.observation_id = composite_to_observation[str(bar.composite_key)]
        return ObservationVocab(composite_to_observation, observation_to_composite, composite_parts)

    def diagnostics(self, songs: Sequence[SongRecord], vocab: ObservationVocab) -> Dict[str, Any]:
        counts = Counter(bar.observation_id for song in songs for bar in song.bars)
        role_counts = Counter(
            self._position_context(bar)
            for song in songs
            for bar in song.bars
        )
        composite_counts = Counter(
            self._composite_key(bar)
            for song in songs
            for bar in song.bars
        )
        return {
            "config": asdict(self.config),
            "observation_count": len(vocab.composite_to_observation),
            "base_composite_count": len(composite_counts),
            "observation_expansion_ratio": round(
                float(len(vocab.composite_to_observation)) / max(1, len(composite_counts)),
                6,
            ),
            "observation_counts": {str(k): int(v) for k, v in counts.items()},
            "rare_observation_count": sum(1 for value in counts.values() if value == 1),
            "position_context_counts": {str(k): int(v) for k, v in sorted(role_counts.items())},
            "vocab": vocab.to_dict(),
        }

    def _observation_key(self, bar: BarRecord) -> str:
        composite = self._composite_key(bar)
        if self.config.strategy == "composite":
            return composite
        if self.config.strategy == "positioned_composite":
            return f"{self._position_key_prefix(bar)}_{composite}"
        raise ValueError("observation_vocab.strategy must be 'composite' or 'positioned_composite'.")

    def _observation_parts(self, bar: BarRecord) -> Dict[str, Any]:
        parts = bar.composite_parts()
        if self.config.strategy == "positioned_composite":
            parts = dict(parts)
            parts["phrase_position"] = self._phrase_position(bar)
            parts["period_role"] = self._period_role(self._phrase_position(bar))
            parts["position_context"] = self._position_context(bar)
            parts["position_strategy"] = self.config.position_strategy
            parts["position_modulo"] = int(self.config.position_modulo)
        return parts

    def _composite_key(self, bar: BarRecord) -> str:
        edit_id = bar.edit_distance_id if bar.edit_distance_id is not None else -1
        if bar.kmeans_id is None:
            return f"E{edit_id}"
        return f"E{edit_id}_K{bar.kmeans_id}"

    def _phrase_position(self, bar: BarRecord) -> int:
        modulo = max(1, int(self.config.position_modulo))
        if self.config.position_source != "bar_index":
            raise ValueError("observation_vocab.position_source currently supports only 'bar_index'.")
        return int(bar.bar_index) % modulo

    def _period_role(self, phrase_position: int) -> str:
        position = int(phrase_position) % max(1, int(self.config.position_modulo))
        if position in {0, 1}:
            return "begin"
        if position in {2, 3, 4, 5}:
            return "middle"
        return "end"

    def _position_context(self, bar: BarRecord) -> str:
        phrase_position = self._phrase_position(bar)
        if self.config.position_strategy == "exact_mod":
            return str(phrase_position)
        if self.config.position_strategy == "period_role":
            return self._period_role(phrase_position)
        raise ValueError("observation_vocab.position_strategy must be 'exact_mod' or 'period_role'.")

    def _position_key_prefix(self, bar: BarRecord) -> str:
        context = self._position_context(bar)
        if self.config.position_strategy == "exact_mod":
            return f"P{context}"
        return f"R{context}"


class BarClusteringPipeline:
    """Run edit-distance codebook, optional KMeans, and observation vocab."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.diagnostics: Dict[str, Any] = {}
        self.edit_distance_codebook: Dict[int, CodebookEntry] = {}
        self.distance = EditDistanceCalculator.from_style_config(config)

    def run(self, songs: Sequence[SongRecord]) -> ObservationVocab:
        self.distance.fit_corpus([bar for song in songs for bar in song.bars])
        codebook = GlobalCodebookClusterer.from_style_config(self.config, self.distance)
        codebook.assign(songs)
        density_analyzer = TokenDensityAnalyzer.from_style_config(self.config)
        kmeans = KMeansFeatureClusterer.from_style_config(self.config)
        kmeans.assign(songs)
        vocab_builder = ObservationVocabBuilder.from_style_config(self.config)
        vocab = vocab_builder.assign(songs)
        self.edit_distance_codebook = self._build_edit_distance_codebook(
            songs,
            codebook.codebook,
            density_analyzer,
        )
        self.diagnostics = {
            "edit_distance": codebook.diagnostics,
            "bar_autoencoder": self.distance.autoencoder.diagnostics,
            "codebook_density": self._codebook_density_diagnostics(),
            "kmeans": kmeans.diagnostics,
            "observation_vocab": vocab_builder.diagnostics(songs, vocab),
        }
        return vocab

    def _build_edit_distance_codebook(
        self,
        songs: Sequence[SongRecord],
        codebook: Sequence[BarRecord],
        density_analyzer: TokenDensityAnalyzer,
    ) -> Dict[int, CodebookEntry]:
        candidates_by_label: Dict[int, List[CodebookCandidate]] = defaultdict(list)
        for song in songs:
            for bar in song.bars:
                if bar.edit_distance_id is None:
                    continue
                candidates_by_label[int(bar.edit_distance_id)].append(
                    self._candidate_for_bar(bar, density_analyzer)
                )
        entries: Dict[int, CodebookEntry] = {}
        for index, bar in enumerate(codebook):
            relative_tokens = bar.tokens_for_edit_distance("relative")
            entries[int(index)] = CodebookEntry(
                edit_distance_id=int(index),
                source_song=bar.song_id,
                source_file=bar.file_path,
                source_bar_index=int(bar.bar_index),
                relative_tokens=relative_tokens,
                absolute_tokens=list(bar.absolute_tokens),
                density=density_analyzer.analyze(relative_tokens),
                token_variance=float(bar.token_variance),
                sharing_score=float(bar.sharing_score),
                candidates=candidates_by_label.get(int(index), []),
            )
        return entries

    def _candidate_for_bar(
        self,
        bar: BarRecord,
        density_analyzer: TokenDensityAnalyzer,
    ) -> CodebookCandidate:
        relative_tokens = bar.tokens_for_edit_distance("relative")
        return CodebookCandidate(
            source_song=bar.song_id,
            source_file=bar.file_path,
            source_bar_index=int(bar.bar_index),
            relative_tokens=relative_tokens,
            absolute_tokens=list(bar.absolute_tokens),
            density=density_analyzer.analyze(relative_tokens),
            token_variance=float(bar.token_variance),
            sharing_score=float(bar.sharing_score),
            kmeans_id=int(bar.kmeans_id) if bar.kmeans_id is not None else None,
            observation_id=int(bar.observation_id) if bar.observation_id is not None else None,
            position_ratio=self._position_ratio(bar),
        )

    def _position_ratio(self, bar: BarRecord) -> float:
        if bar.source_bar_count is None or int(bar.source_bar_count) <= 1:
            return 0.0
        return float(int(bar.bar_index) / max(1, int(bar.source_bar_count) - 1))

    def _codebook_density_diagnostics(self) -> Dict[str, Any]:
        entries = list(self.edit_distance_codebook.values())
        sparse = [entry for entry in entries if entry.density is not None and entry.density.is_sparse]
        semi_sparse = [
            entry
            for entry in entries
            if entry.density is not None and entry.density.is_semi_sparse
        ]
        return {
            "codebook_size": len(entries),
            "sparse_count": len(sparse),
            "semi_sparse_count": len(semi_sparse),
            "sparse_edit_distance_ids": [entry.edit_distance_id for entry in sparse],
            "semi_sparse_edit_distance_ids": [entry.edit_distance_id for entry in semi_sparse],
            "candidate_counts": {
                str(entry.edit_distance_id): len(entry.candidates)
                for entry in entries
            },
            "entries": [
                {
                    "edit_distance_id": entry.edit_distance_id,
                    "source_song": entry.source_song,
                    "source_file": entry.source_file,
                    "source_bar_index": entry.source_bar_index,
                    "density": entry.density.to_dict() if entry.density is not None else None,
                    "candidate_count": len(entry.candidates),
                    "candidate_density_summary": self._candidate_density_summary(entry),
                    "candidate_distribution": self._candidate_distribution(entry),
                    "candidate_examples": self._candidate_examples(entry),
                }
                for entry in entries
            ],
        }

    def _candidate_density_summary(self, entry: CodebookEntry) -> Dict[str, Any]:
        candidates = list(entry.candidates)
        if not candidates:
            return {
                "candidate_count": 0,
                "note_on_ratio": {},
                "sparse_count": 0,
                "semi_sparse_count": 0,
            }
        note_on_ratios = [
            float(candidate.density.note_on_ratio)
            for candidate in candidates
            if candidate.density is not None
        ]
        sparse_count = sum(
            1 for candidate in candidates
            if candidate.density is not None and candidate.density.is_sparse
        )
        semi_sparse_count = sum(
            1 for candidate in candidates
            if candidate.density is not None and candidate.density.is_semi_sparse
        )
        return {
            "candidate_count": len(candidates),
            "representative_note_on_ratio": (
                float(entry.density.note_on_ratio)
                if entry.density is not None
                else None
            ),
            "note_on_ratio": self._numeric_summary(note_on_ratios),
            "sparse_count": int(sparse_count),
            "sparse_ratio": round(float(sparse_count) / len(candidates), 6),
            "semi_sparse_count": int(semi_sparse_count),
            "semi_sparse_ratio": round(float(semi_sparse_count) / len(candidates), 6),
        }

    def _candidate_distribution(self, entry: CodebookEntry) -> Dict[str, Any]:
        candidates = list(entry.candidates)
        return {
            "top_kmeans_ids": self._top_counts(
                candidate.kmeans_id for candidate in candidates
                if candidate.kmeans_id is not None
            ),
            "top_observation_ids": self._top_counts(
                candidate.observation_id for candidate in candidates
                if candidate.observation_id is not None
            ),
            "top_source_songs": self._top_counts(
                candidate.source_song for candidate in candidates
                if candidate.source_song is not None
            ),
        }

    def _candidate_examples(self, entry: CodebookEntry, limit: int = 5) -> Dict[str, Any]:
        candidates = [
            candidate for candidate in entry.candidates
            if candidate.density is not None
        ]
        sparse_first = sorted(
            candidates,
            key=lambda candidate: (
                float(candidate.density.note_on_ratio),
                int(candidate.source_bar_index or 0),
            ),
        )[:limit]
        dense_first = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.density.note_on_ratio),
                int(candidate.source_bar_index or 0),
            ),
        )[:limit]
        return {
            "sparsest": [self._candidate_example(candidate) for candidate in sparse_first],
            "densest": [self._candidate_example(candidate) for candidate in dense_first],
        }

    def _candidate_example(self, candidate: CodebookCandidate) -> Dict[str, Any]:
        return {
            "source_song": candidate.source_song,
            "source_file": candidate.source_file,
            "source_bar_index": candidate.source_bar_index,
            "kmeans_id": candidate.kmeans_id,
            "observation_id": candidate.observation_id,
            "position_ratio": round(float(candidate.position_ratio), 6),
            "density": candidate.density.to_dict() if candidate.density is not None else None,
            "relative_tokens": list(candidate.relative_tokens),
        }

    def _numeric_summary(self, values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {}
        array = np.array(values, dtype=np.float64)
        return {
            "min": round(float(np.min(array)), 6),
            "max": round(float(np.max(array)), 6),
            "mean": round(float(np.mean(array)), 6),
            "median": round(float(np.median(array)), 6),
            "p10": round(float(np.quantile(array, 0.10)), 6),
            "p90": round(float(np.quantile(array, 0.90)), 6),
        }

    def _top_counts(self, values: Sequence[Any], limit: int = 10) -> List[Dict[str, Any]]:
        counts = Counter(values)
        return [
            {
                "value": str(value),
                "count": int(count),
            }
            for value, count in counts.most_common(limit)
        ]


class BarClusteringCLI:
    """Standalone CLI for assigning observation IDs to parsed songs."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Cluster parsed bars and build observation vocab.")
        parser.add_argument("--songs", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        payload = json.loads(args.songs.read_text(encoding="utf-8"))
        songs = [SongRecord.from_dict(item) for item in payload.get("songs", [])]
        pipeline = BarClusteringPipeline(config)
        vocab = pipeline.run(songs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"songs": [song.to_dict() for song in songs], "observation_vocab": vocab.to_dict()}, indent=2),
            encoding="utf-8",
        )
        if args.diagnostics_output:
            args.diagnostics_output.write_text(json.dumps(pipeline.diagnostics, indent=2), encoding="utf-8")
        print(f"Wrote clustered bars -> {args.output}")


def main() -> None:
    BarClusteringCLI().run()


if __name__ == "__main__":
    main()
