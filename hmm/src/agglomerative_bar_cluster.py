#!/usr/bin/env python3
"""Cluster bar grids by cutting an edit-distance dendrogram."""

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
from grid_tokenizer import BarGrid, GridTokenizer


log = logging.getLogger("agglomerative_bar_cluster")


@dataclass(frozen=True)
class AgglomerativeBarClusterConfig:
    distance_threshold: float = 4.0
    linkage: str = "average"
    criterion: str = "distance"
    min_clusters_warn: int = 3
    max_clusters_warn_fraction: float = 0.6


class AgglomerativeBarClusterer:
    """Assign bar labels by thresholding a hierarchical clustering tree."""

    def __init__(self, config: AgglomerativeBarClusterConfig) -> None:
        self.config = config
        self.linkage_matrix: Optional[np.ndarray] = None
        self.diagnostics: Dict[str, Any] = {}

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "AgglomerativeBarClusterer":
        section = ConfigView(config).section("agglomerative_clustering")
        return cls(AgglomerativeBarClusterConfig(
            distance_threshold=float(section.get("distance_threshold", 4.0)),
            linkage=str(section.get("linkage", "average")),
            criterion=str(section.get("criterion", "distance")),
            min_clusters_warn=int(section.get("min_clusters_warn", 3)),
            max_clusters_warn_fraction=float(section.get("max_clusters_warn_fraction", 0.6)),
        ))

    def fit_predict(self, distance_matrix: np.ndarray) -> np.ndarray:
        n_bars = int(distance_matrix.shape[0])
        if n_bars == 0:
            self.diagnostics = {
                "backend": "agglomerative",
                "config": asdict(self.config),
                "n_bars": 0,
            }
            return np.zeros(0, dtype=int)
        if n_bars == 1:
            self.diagnostics = {
                "backend": "agglomerative",
                "config": asdict(self.config),
                "n_bars": 1,
                "n_clusters": 1,
                "labels": [0],
            }
            return np.zeros(1, dtype=int)
        condensed = squareform(distance_matrix, checks=False)
        self.linkage_matrix = linkage(condensed, method=self.config.linkage)
        raw_labels = fcluster(
            self.linkage_matrix,
            t=self.config.distance_threshold,
            criterion=self.config.criterion,
        )
        labels = self._compact_labels(raw_labels)
        self._warn_if_degenerate(labels)
        self.diagnostics = {
            "backend": "agglomerative",
            "config": asdict(self.config),
            "n_bars": n_bars,
            "n_clusters": int(len(set(map(int, labels)))),
            "labels": [int(x) for x in labels],
            "distance_stats": {
                "min": float(np.min(distance_matrix)),
                "max": float(np.max(distance_matrix)),
                "mean": float(np.mean(distance_matrix)),
            },
            "linkage_matrix": self.linkage_matrix.tolist(),
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
        report: Dict[str, Any] = {
            "backend": "agglomerative",
            "config": asdict(self.config),
            "n_bars": len(bars),
            "n_clusters": len(by_label),
            "label_counts": {str(k): int(v) for k, v in Counter(map(int, labels)).items()},
            "labels": [int(x) for x in labels],
            "prototypes": prototypes,
            "diagnostics": self.diagnostics,
        }
        if self.linkage_matrix is not None:
            report["linkage_matrix"] = self.linkage_matrix.tolist()
        return report

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
                "Agglomerative clustering produced only %d clusters. "
                "distance_threshold may be too high.",
                n_clusters,
            )
        max_clusters = max(1, int(round(n_bars * self.config.max_clusters_warn_fraction)))
        if n_clusters > max_clusters:
            log.warning(
                "Agglomerative clustering produced %d clusters for %d bars. "
                "distance_threshold may be too low.",
                n_clusters,
                n_bars,
            )


class AgglomerativeBarClusterCLI:
    """CLI for dendrogram-threshold bar clustering."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Cluster bars by edit-distance dendrogram threshold.")
        parser.add_argument("--bars", type=Path, required=True)
        parser.add_argument("--matrix", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--distance-threshold", type=float, default=None)
        parser.add_argument("--linkage", default=None)
        parser.add_argument("--verbose", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
        config = ConfigLoader().load(args.config)
        cluster_config = AgglomerativeBarClusterer.from_style_config(config).config
        cluster_config = AgglomerativeBarClusterConfig(
            distance_threshold=(
                args.distance_threshold
                if args.distance_threshold is not None
                else cluster_config.distance_threshold
            ),
            linkage=args.linkage if args.linkage is not None else cluster_config.linkage,
            criterion=cluster_config.criterion,
            min_clusters_warn=cluster_config.min_clusters_warn,
            max_clusters_warn_fraction=cluster_config.max_clusters_warn_fraction,
        )
        bars = GridTokenizer.load_bars_file(args.bars)
        matrix_data = np.load(args.matrix, allow_pickle=False)
        clusterer = AgglomerativeBarClusterer(cluster_config)
        labels = clusterer.fit_predict(matrix_data["distance"])
        clusterer.save_labels(args.output, bars, labels)
        if args.diagnostics_output:
            clusterer.save_diagnostics(args.diagnostics_output)
        print(f"Wrote {len(labels)} agglomerative bar labels -> {args.output}")


def main() -> None:
    AgglomerativeBarClusterCLI().run()


if __name__ == "__main__":
    main()
