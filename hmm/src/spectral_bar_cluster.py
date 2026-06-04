#!/usr/bin/env python3
"""Cluster bar grids with spectral clustering over edit-distance affinity."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import SpectralClustering

from config_loader import ConfigLoader, ConfigView
from grid_tokenizer import BarGrid, GridTokenizer


@dataclass(frozen=True)
class SpectralBarClusterConfig:
    n_clusters: int = 8
    assign_labels: str = "kmeans"
    random_seed: int = 42


class SpectralBarClusterer:
    """Assign micro bar-type labels from a precomputed affinity matrix."""

    def __init__(self, config: SpectralBarClusterConfig) -> None:
        self.config = config
        self.diagnostics: Dict[str, Any] = {}

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "SpectralBarClusterer":
        section = ConfigView(config).section("spectral_clustering")
        return cls(SpectralBarClusterConfig(
            n_clusters=int(section.get("n_clusters", 8)),
            assign_labels=str(section.get("assign_labels", "kmeans")),
            random_seed=int(section.get("random_seed", 42)),
        ))

    def fit_predict(self, affinity_matrix: np.ndarray) -> np.ndarray:
        if affinity_matrix.shape[0] == 0:
            self.diagnostics = {
                "backend": "spectral",
                "config": asdict(self.config),
                "n_bars": 0,
            }
            return np.zeros(0, dtype=int)
        n_clusters = min(self.config.n_clusters, affinity_matrix.shape[0])
        if n_clusters <= 1:
            self.diagnostics = {
                "backend": "spectral",
                "config": asdict(self.config),
                "n_bars": int(affinity_matrix.shape[0]),
                "requested_n_clusters": int(self.config.n_clusters),
                "actual_n_clusters": 1,
                "labels": [0] * int(affinity_matrix.shape[0]),
            }
            return np.zeros(affinity_matrix.shape[0], dtype=int)
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels=self.config.assign_labels,
            random_state=self.config.random_seed,
        )
        labels = model.fit_predict(affinity_matrix).astype(int)
        self.diagnostics = {
            "backend": "spectral",
            "config": asdict(self.config),
            "n_bars": int(affinity_matrix.shape[0]),
            "requested_n_clusters": int(self.config.n_clusters),
            "actual_n_clusters": int(len(set(map(int, labels)))),
            "labels": [int(x) for x in labels],
            "affinity_stats": {
                "min": float(np.min(affinity_matrix)),
                "max": float(np.max(affinity_matrix)),
                "mean": float(np.mean(affinity_matrix)),
            },
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
            "config": asdict(self.config),
            "n_bars": len(bars),
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


class SpectralBarClusterCLI:
    """CLI for spectral bar clustering."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Cluster bars from a precomputed affinity matrix.")
        parser.add_argument("--bars", type=Path, required=True)
        parser.add_argument("--matrix", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--n-clusters", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        cluster_config = SpectralBarClusterer.from_style_config(config).config
        cluster_config = SpectralBarClusterConfig(
            n_clusters=args.n_clusters if args.n_clusters is not None else cluster_config.n_clusters,
            assign_labels=cluster_config.assign_labels,
            random_seed=args.seed if args.seed is not None else cluster_config.random_seed,
        )
        bars = GridTokenizer.load_bars_file(args.bars)
        matrix_data = np.load(args.matrix, allow_pickle=False)
        clusterer = SpectralBarClusterer(cluster_config)
        labels = clusterer.fit_predict(matrix_data["affinity"])
        clusterer.save_labels(args.output, bars, labels)
        if args.diagnostics_output:
            clusterer.save_diagnostics(args.diagnostics_output)
        print(f"Wrote {len(labels)} bar labels -> {args.output}")


def main() -> None:
    SpectralBarClusterCLI().run()


if __name__ == "__main__":
    main()
