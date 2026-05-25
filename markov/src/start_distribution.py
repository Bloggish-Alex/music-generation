#!/usr/bin/env python3
"""
Start Distribution — per-cluster probability of appearing as the first measure.

Counts how often each cluster appears as the starting state across files.
Counts are normalised to a probability distribution suitable for seeding
a Markov chain.

Usage (library)::

    from measure_clustering import classify_files, MeasureExtractor, MeasureClusterer
    from start_distribution import StartDistributionBuilder

    extractor = MeasureExtractor()
    clusterer = MeasureClusterer()
    clusterer.fit(extractor.extract_all("path/to/music/dir"), n_clusters=8)
    file_labels = classify_files("path/to/music/dir", clusterer, extractor)
    start_dist = StartDistributionBuilder.build(file_labels)
    print(start_dist.summary())
    print("sampled start state:", start_dist.sample())

Usage (CLI)::

    python start_distribution.py --music-dir ../../datasets/corelli --n-clusters 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from measure_clustering import (
    MeasureClusterer,
    MeasureExtractor,
    classify_files,
)

log = logging.getLogger("start_distribution")

# ---------------------------------------------------------------------------
# StartDistribution dataclass
# ---------------------------------------------------------------------------


class StartDistribution:
    """Per-cluster probability of being the first measure of a piece.

    Attributes:
        start_counts: Raw count per cluster, shape (n_clusters,).
        start_probs: Normalised probabilities, shape (n_clusters,).
        n_clusters: Number of cluster states.
        total_files: Number of files that contributed.
        states: Cluster label list [0 .. n_clusters-1].
    """

    def __init__(
        self,
        start_counts: np.ndarray,
        n_clusters: int,
        total_files: int,
    ) -> None:
        self.start_counts = start_counts
        self.n_clusters = n_clusters
        self.total_files = total_files
        self.states = list(range(n_clusters))
        total = float(start_counts.sum())
        self.start_probs = (
            start_counts.astype(np.float64) / total if total > 0
            else np.zeros(n_clusters, dtype=np.float64)
        )

    def sample(self, seed: Optional[int] = None) -> int:
        """Randomly select a start cluster weighted by start_probs.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            A cluster label.
        """
        rng = np.random.RandomState(seed) if seed is not None else np.random
        return int(rng.choice(self.states, p=self.start_probs))

    def summary(self) -> str:
        """Human-readable summary of the start distribution."""
        lines = [
            "=" * 55,
            "START DISTRIBUTION",
            "=" * 55,
            f"  States (clusters):  {self.n_clusters}",
            f"  Total files:        {self.total_files}",
            "",
            f"  {'Cluster':>8}  {'Count':>7}  {'Prob':>8}",
            "  " + "-" * 30,
        ]
        for c in self.states:
            lines.append(
                f"  {c:>8}  {self.start_counts[c]:>7}  "
                f"{self.start_probs[c]:>8.4f}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StartDistributionBuilder
# ---------------------------------------------------------------------------


class StartDistributionBuilder:
    """Builds a StartDistribution from per-file cluster label sequences.

    Pure computation — no internal state.  Extraction and clustering
    happen externally via :func:`classify_files`.

    Usage::

        from measure_clustering import classify_files, MeasureExtractor, MeasureClusterer
        from start_distribution import StartDistributionBuilder

        extractor = MeasureExtractor()
        clusterer = MeasureClusterer()
        clusterer.fit(extractor.extract_all(music_dir), n_clusters=8)
        file_labels = classify_files(music_dir, clusterer, extractor)
        start_dist = StartDistributionBuilder.build(file_labels)
        print(start_dist.summary())
    """

    @staticmethod
    def build(
        file_labels: List[List[int]],
    ) -> StartDistribution:
        """Build the start distribution from per-file labels.

        Only the first label of each file is counted.

        Args:
            file_labels: One ordered label list per file.

        Returns:
            StartDistribution with per-cluster counts and probabilities.
        """
        if not file_labels:
            raise ValueError("file_labels is empty")

        n = max(label for seq in file_labels for label in seq) + 1
        start_counts = np.zeros(n, dtype=np.int64)

        for labels in file_labels:
            if labels:
                start_counts[labels[0]] += 1

        total = int(start_counts.sum())
        log.info(
            "Start distribution: %d states across %d files.",
            n, total,
        )

        return StartDistribution(
            start_counts=start_counts,
            n_clusters=n,
            total_files=total,
        )


# ---------------------------------------------------------------------------
# StartDistributionVisualizer
# ---------------------------------------------------------------------------


class StartDistributionVisualizer:
    """Generate diagnostic plots for start distribution."""

    def __init__(self, start_dist: StartDistribution) -> None:
        self._start = start_dist

    def plot_bars(
        self,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """Save a bar chart of start probabilities per cluster.

        Args:
            save_path: Output image path (.png).
            title: Plot title. Auto-generated if None.
            figsize: Figure size in inches.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clusters = self._start.states
        probs = self._start.start_probs
        counts = self._start.start_counts

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            clusters, probs,
            color=plt.get_cmap("tab10")(clusters),
            edgecolor="white", alpha=0.85,
        )
        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Start Probability", fontsize=12)
        ax.set_title(
            title or f"Start Distribution by Cluster "
            f"(n={self._start.n_clusters}, files={self._start.total_files})",
            fontsize=14,
        )
        ax.set_xticks(clusters)
        ax.set_xticklabels([f"Cluster {c}" for c in clusters])

        for bar, prob, cnt in zip(bars, probs, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{prob:.3f}\n(n={cnt})",
                ha="center", fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved start distribution bar chart to %s", save_path)

    def plot_pie(
        self,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
    ) -> None:
        """Save a pie chart of start probabilities.

        Args:
            save_path: Output image path (.png).
            title: Plot title. Auto-generated if None.
            figsize: Figure size in inches.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        probs = self._start.start_probs
        labels = [f"Cluster {c}" for c in self._start.states]
        colors = plt.get_cmap("tab10")(self._start.states)

        # Only label slices with meaningful probability
        def _pct(pctval: float) -> str:
            return f"{pctval:.1f}%" if pctval > 2 else ""

        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(
            probs, labels=labels, autopct=_pct,
            colors=colors, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        ax.set_title(
            title or f"Start Distribution (n={self._start.n_clusters}, "
            f"files={self._start.total_files})",
            fontsize=14,
        )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved start distribution pie chart to %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse
    from _argparse_utils import (
        add_clustering_args,
        add_model_io_args,
        add_music_source_args,
        add_verbose_arg,
    )

    parser = argparse.ArgumentParser(
        description="Start Distribution — per-cluster first-measure "
        "probability distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_music_source_args(parser)
    add_clustering_args(parser)
    add_model_io_args(parser)
    add_verbose_arg(parser)
    parser.add_argument(
        "--barplot",
        default="start_distribution_bars.png",
        help="Path to save the bar chart image.",
    )
    parser.add_argument(
        "--piechart",
        default=None,
        help="Path to save the pie chart image (optional).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    patterns = [p.strip() for p in args.file_patterns.split(",") if p.strip()]

    extractor = MeasureExtractor()
    clusterer = MeasureClusterer()

    if args.load_model:
        log.info("Loading clusterer from %s ...", args.load_model)
        clusterer = MeasureClusterer.load(args.load_model)
    else:
        log.info("Fitting clusterer on %s ...", args.music_dir)
        vectors = extractor.extract_all(args.music_dir, file_patterns=patterns)
        clusterer.fit(vectors, n_clusters=args.n_clusters, random_seed=args.seed)
        log.info(
            "Clusterer fitted: k=%d, inertia=%.3f",
            args.n_clusters,
            clusterer.inertia,
        )
        if args.save_model:
            clusterer.save(args.save_model)

    log.info("Classifying files ...")
    file_labels = classify_files(args.music_dir, clusterer, extractor, patterns)

    log.info("Building start distribution ...")
    start_dist = StartDistributionBuilder.build(file_labels)

    # Print summary
    print()
    print(start_dist.summary())

    # Demonstrate sampling
    samples = [start_dist.sample() for _ in range(10)]
    print(f"\n  Sampled start states (n=10): {samples}")

    # Plots
    if not args.no_plots:
        viz = StartDistributionVisualizer(start_dist)
        viz.plot_bars(args.barplot)
        if args.piechart:
            viz.plot_pie(args.piechart)

    print("\nDone.")


if __name__ == "__main__":
    main()
