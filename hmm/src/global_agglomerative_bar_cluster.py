#!/usr/bin/env python3
"""Per-song agglomerative clustering aligned through a global bar codebook."""

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

from config_loader import ConfigLoader, ConfigView
from edit_distance_matrix import DistanceMatrixConfig, EditDistanceMatrixBuilder
from grid_tokenizer import BarGrid, GridTokenizer


log = logging.getLogger("global_agglomerative_bar_cluster")


@dataclass(frozen=True)
class GlobalAgglomerativeClusterConfig:
    song_distance_threshold: float = 0.25
    codebook_size: int = 1024
    codebook_linkage: str = "average"
    song_linkage: str = "average"
    criterion: str = "distance"
    min_clusters_warn: int = 3
    max_clusters_warn_fraction: float = 0.6


class GlobalAgglomerativeClusterer:
    """Cluster per song, collect medoids, then quantize bars to a global codebook."""

    def __init__(self, config: GlobalAgglomerativeClusterConfig) -> None:
        self.config = config
        self.diagnostics: Dict[str, Any] = {}

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "GlobalAgglomerativeClusterer":
        section = ConfigView(config).section("global_agglomerative_clustering")
        return cls(GlobalAgglomerativeClusterConfig(
            song_distance_threshold=float(section.get("song_distance_threshold", 0.25)),
            codebook_size=int(section.get("codebook_size", 1024)),
            codebook_linkage=str(section.get("codebook_linkage", "average")),
            song_linkage=str(section.get("song_linkage", "average")),
            criterion=str(section.get("criterion", "distance")),
            min_clusters_warn=int(section.get("min_clusters_warn", 3)),
            max_clusters_warn_fraction=float(section.get("max_clusters_warn_fraction", 0.6)),
        ))

    def fit_predict(self, bars: Sequence[BarGrid]) -> np.ndarray:
        if not bars:
            self.diagnostics = {"backend": "global_agglomerative", "n_bars": 0}
            return np.zeros(0, dtype=int)

        distance_builder = EditDistanceMatrixBuilder(DistanceMatrixConfig(normalize_distance=True))
        song_groups = self._group_bars_by_song(bars)
        song_diagnostics: List[Dict[str, Any]] = []
        medoid_bars: List[BarGrid] = []
        medoid_sources: List[Dict[str, Any]] = []

        for song_key, indices in song_groups.items():
            song_bars = [bars[index] for index in indices]
            song_distance = distance_builder.build_distance(song_bars)
            song_labels = self._cluster_distance(
                song_distance,
                threshold=self.config.song_distance_threshold,
                linkage_method=self.config.song_linkage,
            )
            medoid_local_indices = self._cluster_medoids(song_distance, song_labels)
            for local_label, local_index in medoid_local_indices.items():
                global_index = indices[local_index]
                medoid_bars.append(bars[global_index])
                medoid_sources.append({
                    "song": song_key,
                    "song_cluster_label": int(local_label),
                    "bar_global_index": int(global_index),
                    "bar_index": int(bars[global_index].bar_index),
                    "file_path": bars[global_index].file_path,
                })
            song_diagnostics.append({
                "song": song_key,
                "bar_indices": [int(index) for index in indices],
                "n_bars": len(indices),
                "n_song_clusters": len(set(map(int, song_labels))),
                "song_labels": [int(x) for x in song_labels],
                "medoids": medoid_sources[-len(medoid_local_indices):],
            })

        codebook_distance = distance_builder.build_distance(medoid_bars)
        codebook_labels = self._cluster_codebook(codebook_distance)
        codebook = self._codebook_medoids(codebook_distance, codebook_labels, medoid_bars)
        labels = self._quantize_to_codebook(bars, codebook, distance_builder)
        self._warn_if_degenerate(labels)

        self.diagnostics = {
            "backend": "global_agglomerative",
            "config": asdict(self.config),
            "n_bars": len(bars),
            "n_songs": len(song_groups),
            "n_song_medoids": len(medoid_bars),
            "requested_codebook_size": self.config.codebook_size,
            "actual_codebook_size": len(codebook),
            "song_diagnostics": song_diagnostics,
            "codebook": [
                {
                    "label": int(item["label"]),
                    "medoid_pool_index": int(item["medoid_pool_index"]),
                    "source": item["source"],
                    "tokens": item["tokens"],
                }
                for item in codebook
            ],
            "label_counts": {str(k): int(v) for k, v in Counter(map(int, labels)).items()},
            "labels": [int(x) for x in labels],
        }
        return labels

    def build_report(self, bars: Sequence[BarGrid], labels: Sequence[int]) -> Dict[str, Any]:
        by_label: Dict[int, List[BarGrid]] = defaultdict(list)
        for bar, label in zip(bars, labels):
            by_label[int(label)].append(bar)
        prototypes = {}
        for label, label_bars in by_label.items():
            prototype = Counter(tuple(bar.tokens) for bar in label_bars).most_common(1)[0][0]
            prototypes[str(label)] = list(prototype)
        return {
            "backend": "global_agglomerative",
            "config": asdict(self.config),
            "n_bars": len(bars),
            "n_clusters": len(by_label),
            "label_counts": {str(k): int(v) for k, v in Counter(map(int, labels)).items()},
            "labels": [int(x) for x in labels],
            "prototypes": prototypes,
            "diagnostics": self.diagnostics,
        }

    def save_labels(
        self,
        output_path: str | Path,
        bars: Sequence[BarGrid],
        labels: Sequence[int],
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build_report(bars, labels)
        report["bars"] = [
            {
                "source_index": bar.source_index,
                "file_path": bar.file_path,
                "bar_index": bar.bar_index,
                "label": int(label),
                "tokens": bar.tokens,
            }
            for bar, label in zip(bars, labels)
        ]
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def save_diagnostics(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.diagnostics, indent=2), encoding="utf-8")

    def _group_bars_by_song(self, bars: Sequence[BarGrid]) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = defaultdict(list)
        for index, bar in enumerate(bars):
            groups[bar.file_path].append(index)
        return dict(groups)

    def _cluster_distance(
        self,
        distance_matrix: np.ndarray,
        threshold: float,
        linkage_method: str,
    ) -> np.ndarray:
        n_items = int(distance_matrix.shape[0])
        if n_items == 0:
            return np.zeros(0, dtype=int)
        if n_items == 1:
            return np.zeros(1, dtype=int)
        condensed = squareform(distance_matrix, checks=False)
        z_matrix = linkage(condensed, method=linkage_method)
        raw_labels = fcluster(z_matrix, t=threshold, criterion=self.config.criterion)
        return self._compact_labels(raw_labels)

    def _cluster_codebook(self, distance_matrix: np.ndarray) -> np.ndarray:
        n_items = int(distance_matrix.shape[0])
        if n_items == 0:
            return np.zeros(0, dtype=int)
        if n_items == 1:
            return np.zeros(1, dtype=int)
        target = min(max(1, self.config.codebook_size), n_items)
        condensed = squareform(distance_matrix, checks=False)
        z_matrix = linkage(condensed, method=self.config.codebook_linkage)
        raw_labels = fcluster(z_matrix, t=target, criterion="maxclust")
        return self._compact_labels(raw_labels)

    def _cluster_medoids(
        self,
        distance_matrix: np.ndarray,
        labels: Sequence[int],
    ) -> Dict[int, int]:
        medoids: Dict[int, int] = {}
        label_array = np.asarray(labels, dtype=int)
        for label in sorted(set(map(int, labels))):
            member_indices = np.where(label_array == label)[0]
            if len(member_indices) == 1:
                medoids[label] = int(member_indices[0])
                continue
            sub_matrix = distance_matrix[member_indices[:, None], member_indices]
            distance_sums = np.sum(sub_matrix, axis=1)
            medoids[label] = int(member_indices[int(np.argmin(distance_sums))])
        return medoids

    def _codebook_medoids(
        self,
        distance_matrix: np.ndarray,
        labels: Sequence[int],
        medoid_bars: Sequence[BarGrid],
    ) -> List[Dict[str, Any]]:
        medoid_indices = self._cluster_medoids(distance_matrix, labels)
        codebook = []
        for label, medoid_pool_index in sorted(medoid_indices.items()):
            bar = medoid_bars[medoid_pool_index]
            codebook.append({
                "label": int(label),
                "medoid_pool_index": int(medoid_pool_index),
                "tokens": list(bar.tokens),
                "source": {
                    "file_path": bar.file_path,
                    "bar_index": int(bar.bar_index),
                    "source_index": int(bar.source_index),
                },
            })
        return codebook

    def _quantize_to_codebook(
        self,
        bars: Sequence[BarGrid],
        codebook: Sequence[Dict[str, Any]],
        distance_builder: EditDistanceMatrixBuilder,
    ) -> np.ndarray:
        if not codebook:
            return np.zeros(len(bars), dtype=int)
        codebook_bars = [
            BarGrid(
                tokens=list(item["tokens"]),
                file_path=item["source"]["file_path"],
                bar_index=int(item["source"]["bar_index"]),
                bar_offset_ql=0.0,
                bar_length_ql=bars[0].bar_length_ql if bars else 4.0,
            )
            for item in codebook
        ]
        labels: List[int] = []
        for bar in bars:
            distances = [
                distance_builder.build_distance([bar, codebook_bar])[0, 1]
                for codebook_bar in codebook_bars
            ]
            best = int(np.argmin(distances))
            labels.append(int(codebook[best]["label"]))
        return np.asarray(labels, dtype=int)

    def _compact_labels(self, labels: Sequence[int]) -> np.ndarray:
        mapping: Dict[int, int] = {}
        compact: List[int] = []
        for label in labels:
            label = int(label)
            if label not in mapping:
                mapping[label] = len(mapping)
            compact.append(mapping[label])
        return np.asarray(compact, dtype=int)

    def _warn_if_degenerate(self, labels: Sequence[int]) -> None:
        n_bars = len(labels)
        n_clusters = len(set(map(int, labels)))
        if n_clusters < self.config.min_clusters_warn:
            log.warning(
                "Global agglomerative clustering produced only %d clusters. "
                "codebook_size or song_distance_threshold may be too low/high.",
                n_clusters,
            )
        max_clusters = max(1, int(round(n_bars * self.config.max_clusters_warn_fraction)))
        if n_clusters > max_clusters:
            log.warning(
                "Global agglomerative clustering produced %d clusters for %d bars. "
                "codebook_size may be too high or song_distance_threshold too low.",
                n_clusters,
                n_bars,
            )


class GlobalAgglomerativeClusterCLI:
    """CLI for per-song clustering plus global codebook quantization."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Cluster bars with a global agglomerative codebook.")
        parser.add_argument("--bars", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--codebook-size", type=int, default=None)
        parser.add_argument("--song-distance-threshold", type=float, default=None)
        parser.add_argument("--song-linkage", default=None)
        parser.add_argument("--codebook-linkage", default=None)
        parser.add_argument("--verbose", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
        config = ConfigLoader().load(args.config)
        cluster_config = GlobalAgglomerativeClusterer.from_style_config(config).config
        cluster_config = GlobalAgglomerativeClusterConfig(
            song_distance_threshold=(
                args.song_distance_threshold
                if args.song_distance_threshold is not None
                else cluster_config.song_distance_threshold
            ),
            codebook_size=args.codebook_size if args.codebook_size is not None else cluster_config.codebook_size,
            codebook_linkage=args.codebook_linkage if args.codebook_linkage is not None else cluster_config.codebook_linkage,
            song_linkage=args.song_linkage if args.song_linkage is not None else cluster_config.song_linkage,
            criterion=cluster_config.criterion,
            min_clusters_warn=cluster_config.min_clusters_warn,
            max_clusters_warn_fraction=cluster_config.max_clusters_warn_fraction,
        )
        bars = GridTokenizer.load_bars_file(args.bars)
        clusterer = GlobalAgglomerativeClusterer(cluster_config)
        labels = clusterer.fit_predict(bars)
        clusterer.save_labels(args.output, bars, labels)
        if args.diagnostics_output:
            clusterer.save_diagnostics(args.diagnostics_output)
        print(f"Wrote {len(labels)} global agglomerative bar labels -> {args.output}")


def main() -> None:
    GlobalAgglomerativeClusterCLI().run()


if __name__ == "__main__":
    main()
