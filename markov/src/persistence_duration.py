#!/usr/bin/env python3
"""
Persistence Duration Distribution — run-length analysis of cluster labels.

Computes consecutive run-lengths (how many measures a cluster persists)
within each file.  No cross-file leakage.  Outputs per-cluster run-length
lists, mean, standard deviation, and weighted sample.

Usage (library)::

    from measure_clustering import classify_files, MeasureExtractor, MeasureClusterer
    from persistence_duration import PersistenceDurationBuilder

    extractor = MeasureExtractor()
    clusterer = MeasureClusterer()
    clusterer.fit(extractor.extract_all("path/to/music/dir"), n_clusters=8)
    file_labels = classify_files("path/to/music/dir", clusterer, extractor)
    persist = PersistenceDurationBuilder.build(file_labels)
    print(persist.summary())

Usage (CLI)::

    python persistence_duration.py --music-dir ../../datasets/corelli --n-clusters 5
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from measure_clustering import (
    MeasureClusterer,
    MeasureExtractor,
    classify_files,
)

log = logging.getLogger("persistence_duration")

# ---------------------------------------------------------------------------
# PersistenceDuration dataclass
# ---------------------------------------------------------------------------


class PersistenceDuration:
    """Holds per-cluster run-length distributions and statistics.

    Attributes:
        run_lengths: cluster → list of consecutive run-lengths (in measures).
        stats: Per-cluster (mean, std), index = cluster label.
        n_clusters: Number of cluster states.
        total_runs: Total number of runs across all clusters.
    """

    def __init__(
        self,
        run_lengths: Dict[int, List[int]],
        n_clusters: int,
    ) -> None:
        self.run_lengths = run_lengths
        self.n_clusters = n_clusters
        self.total_runs = sum(len(v) for v in run_lengths.values())
        self.stats = self._compute_stats()

    @staticmethod
    def _weighted_average(values: List[int]) -> int:
        """Compute the expected value (rounded mean) from the run-length histogram."""
        if not values:
            return 0
        from collections import Counter

        counter = Counter(values)
        weighted_sum = sum(duration * count for duration, count in counter.items())
        return int(round(weighted_sum / len(values)))

    def _compute_stats(self) -> List[Tuple[float, float]]:
        """Compute (mean, std) for each cluster's run-lengths."""
        result: List[Tuple[float, float]] = []
        for c in range(self.n_clusters):
            rl = self.run_lengths.get(c, [])
            if rl:
                mean = float(np.mean(rl))
                std = float(np.std(rl))
            else:
                mean = 0.0
                std = 0.0
            result.append((mean, std))
        return result

    def sample_duration(self, cluster: int) -> int:
        """Return the expected run-length (rounded mean) for *cluster*.

        Uses the weighted average of the run-length histogram — no random
        sampling, so the result is deterministic for each cluster.

        Returns:
            Rounded mean run-length, or 1 if the cluster has no data.
        """
        rl = self.run_lengths.get(cluster, [])
        if not rl:
            return 1
        return self._weighted_average(rl)

    def summary(self) -> str:
        """Human-readable summary of persistence duration distribution."""
        from collections import Counter

        lines = [
            "=" * 75,
            "PERSISTENCE DURATION DISTRIBUTION",
            "=" * 75,
            f"  States (clusters):  {self.n_clusters}",
            f"  Total runs:         {self.total_runs}",
            "",
            f"  {'Cluster':>8}  {'Runs':>6}  {'Mean':>8}  {'Std':>8}  "
            f"{'W.Avg':>6}  Run-length histogram (length: count)",
            "  " + "-" * 78,
        ]
        for c in range(self.n_clusters):
            rl = self.run_lengths.get(c, [])
            n_runs = len(rl)
            mean, std = self.stats[c]
            wavg = self._weighted_average(rl) if rl else 0
            if rl:
                counter = Counter(rl)
                top = counter.most_common(10)
                hist = " ".join(f"{length}:{cnt}" for length, cnt in top)
                remaining = len(counter) - len(top)
                if remaining > 0:
                    hist += f" (+{remaining} more)"
            else:
                hist = "(no data)"
            lines.append(
                f"  {c:>8}  {n_runs:>6}  {mean:>8.2f}  {std:>8.2f}  "
                f"{wavg:>6}  {hist}"
            )
        lines.append("=" * 75)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Label preprocessors — chainable label→label transforms
# ---------------------------------------------------------------------------


class LabelPreprocessor(ABC):
    """Base class for label sequence preprocessing. In: labels, Out: labels."""

    @abstractmethod
    def process(self, labels: List[int]) -> List[int]:
        ...

    @staticmethod
    def _build_runs(labels: List[int]) -> List[Tuple[int, int, int]]:
        """Build run list: [(label, length, start_index), ...]."""
        runs: List[Tuple[int, int, int]] = []
        current = labels[0]
        start = 0
        for i, s in enumerate(labels):
            if s != current:
                runs.append((current, i - start, start))
                current = s
                start = i
        runs.append((current, len(labels) - start, start))
        return runs


class ABASmoother(LabelPreprocessor):
    """Smooth isolated ABA patterns to AAA.

    When a single measure of cluster B is sandwiched between two runs of
    the same cluster A, the B measure is treated as noise and absorbed.
    """

    def process(self, labels: List[int]) -> List[int]:
        if len(labels) < 3:
            return list(labels)

        runs = self._build_runs(labels)
        smoothed = list(labels)
        for idx, (label, length, start_idx) in enumerate(runs):
            if length != 1:
                continue
            if idx == 0 or idx == len(runs) - 1:
                continue
            left_label, _left_len, _left_start = runs[idx - 1]
            right_label, _right_len, _right_start = runs[idx + 1]
            if left_label == right_label and left_label != label:
                for j in range(start_idx, start_idx + 1):
                    smoothed[j] = left_label
        return smoothed


class ShortRunMerger(LabelPreprocessor):
    """Merge runs shorter than *min_length* into adjacent longer runs.

    Short runs are absorbed by whichever neighbouring run is longer
    (right-tiebreak).
    """

    def __init__(self, min_length: int = 2) -> None:
        self.min_length = min_length

    def process(self, labels: List[int]) -> List[int]:
        if self.min_length <= 1 or len(labels) < 2:
            return list(labels)

        runs = self._build_runs(labels)
        merged = list(labels)
        for idx, (label, length, start_idx) in enumerate(runs):
            if length >= self.min_length:
                continue
            left_len = runs[idx - 1][1] if idx > 0 else 0
            right_len = runs[idx + 1][1] if idx + 1 < len(runs) else 0
            if left_len >= right_len and idx > 0:
                merge_to = runs[idx - 1][0]
            elif idx + 1 < len(runs):
                merge_to = runs[idx + 1][0]
            else:
                continue
            for j in range(start_idx, start_idx + length):
                merged[j] = merge_to
        return merged


# ---------------------------------------------------------------------------
# PersistenceDurationBuilder
# ---------------------------------------------------------------------------


class PersistenceDurationBuilder:
    """Builds a PersistenceDuration from per-file cluster label sequences.

    Pure computation — no internal state.  Extraction and clustering
    happen externally via :func:`classify_files`.

    Usage::

        from measure_clustering import classify_files, MeasureExtractor, MeasureClusterer
        from persistence_duration import PersistenceDurationBuilder

        extractor = MeasureExtractor()
        clusterer = MeasureClusterer()
        clusterer.fit(extractor.extract_all(music_dir), n_clusters=8)
        file_labels = classify_files(music_dir, clusterer, extractor)
        persist = PersistenceDurationBuilder.build(file_labels)
    """

    @staticmethod
    def build(
        file_labels: List[List[int]],
        min_run_length: int = 0,
    ) -> PersistenceDuration:
        """Build the persistence duration distribution from per-file labels.

        Args:
            file_labels: One ordered label list per file.
            min_run_length: If > 1, append a ShortRunMerger to the
                preprocessor chain (0 = ABA smoother only).

        Returns:
            PersistenceDuration with run-lengths and per-cluster stats.
        """
        if not file_labels:
            raise ValueError("file_labels is empty")

        n = max(label for seq in file_labels for label in seq) + 1

        # Build preprocessor chain
        preprocessors: List[LabelPreprocessor] = [ABASmoother()]
        if min_run_length > 1:
            preprocessors.append(ShortRunMerger(min_length=min_run_length))

        run_lengths: Dict[int, List[int]] = defaultdict(list)
        skipped = 0
        for labels in file_labels:
            # Apply preprocessors
            processed = labels
            for pp in preprocessors:
                processed = pp.process(processed)

            # Run-length computation
            file_runs: Dict[int, List[int]] = defaultdict(list)
            current = processed[0]
            count = 1
            for s in processed[1:]:
                if s == current:
                    count += 1
                else:
                    file_runs[current].append(count)
                    current = s
                    count = 1
            file_runs[current].append(count)

            total = sum(len(v) for v in file_runs.values())
            if total == 0:
                skipped += 1
                continue
            for cluster, lengths in file_runs.items():
                run_lengths[cluster].extend(lengths)

        log.info(
            "Processed %d/%d files (%d skipped)%s.",
            len(file_labels) - skipped, len(file_labels), skipped,
            f", min_run_length={min_run_length}" if min_run_length > 1 else "",
        )

        result = PersistenceDuration(
            run_lengths=dict(run_lengths),
            n_clusters=n,
        )

        log.info(
            "Persistence duration: %d states, %d total runs.",
            n,
            result.total_runs,
        )
        return result


# ---------------------------------------------------------------------------
# PersistenceDurationVisualizer
# ---------------------------------------------------------------------------


class PersistenceDurationVisualizer:
    """Generate diagnostic plots for persistence duration distribution."""

    def __init__(self, persist: PersistenceDuration) -> None:
        self._persist = persist

    def plot_histogram(
        self,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> None:
        """Save a grid of per-cluster histograms of run-lengths.

        Args:
            save_path: Output image path (.png).
            title: Plot title. Auto-generated if None.
            figsize: Figure size in inches.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = self._persist.n_clusters
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        # Ensure axes is always a flat array
        if n == 1:
            axes = np.array([axes])
        axes_flat = np.atleast_1d(axes).flatten()

        for c in range(n):
            ax = axes_flat[c]
            rl = self._persist.run_lengths.get(c, [])
            if rl:
                ax.hist(rl, bins=max(5, min(30, len(set(rl)))),
                        color=plt.get_cmap("tab10")(c), edgecolor="white",
                        alpha=0.8)
                mean, std = self._persist.stats[c]
                ax.axvline(mean, color="red", linestyle="--", linewidth=1.5,
                           label=f"mean={mean:.1f}")
                ax.legend(fontsize=8)
            ax.set_title(f"Cluster {c} ({len(rl)} runs)", fontsize=10)
            ax.set_xlabel("Run-length (measures)")
            ax.set_ylabel("Frequency")

        # Hide unused subplots
        for c in range(n, len(axes_flat)):
            axes_flat[c].set_visible(False)

        fig.suptitle(
            title or f"Persistence Duration Histograms by Cluster "
            f"(n={n})",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved persistence histogram to %s", save_path)

    def plot_mean_std(
        self,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """Save a bar chart of mean run-length ± std per cluster.

        Args:
            save_path: Output image path (.png).
            title: Plot title. Auto-generated if None.
            figsize: Figure size in inches.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = self._persist.n_clusters
        clusters = list(range(n))
        means = [self._persist.stats[c][0] for c in clusters]
        stds = [self._persist.stats[c][1] for c in clusters]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            clusters, means, yerr=stds,
            capsize=5, color=plt.get_cmap("tab10")(clusters),
            edgecolor="white", alpha=0.85,
        )
        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Mean Run-length (measures)", fontsize=12)
        ax.set_title(
            title or f"Mean Persistence Duration by Cluster "
            f"(±1 std, n={n})",
            fontsize=14,
        )
        ax.set_xticks(clusters)
        ax.set_xticklabels([f"Cluster {c}" for c in clusters])

        # Annotate bars with mean value
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{mean:.1f}",
                ha="center", fontsize=10,
            )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved mean/std bar chart to %s", save_path)


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
        description="Persistence Duration Distribution — run-length "
        "analysis of cluster labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_music_source_args(parser)
    add_clustering_args(parser)
    add_model_io_args(parser)
    add_verbose_arg(parser)
    parser.add_argument(
        "--min-run-length",
        type=int,
        default=0,
        help="Merge runs shorter than this into adjacent longer runs "
        "(0 = no filter).",
    )
    parser.add_argument(
        "--histogram",
        default="persistence_histogram.png",
        help="Path to save the run-length histogram image.",
    )
    parser.add_argument(
        "--barplot",
        default="persistence_barplot.png",
        help="Path to save the mean/std bar chart image.",
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

    log.info("Building persistence duration distribution ...")
    persist = PersistenceDurationBuilder.build(
        file_labels,
        min_run_length=args.min_run_length,
    )

    # Print summary
    print()
    print(persist.summary())

    # Plots
    if not args.no_plots:
        viz = PersistenceDurationVisualizer(persist)
        viz.plot_histogram(args.histogram)
        viz.plot_mean_std(args.barplot)

    print("\nDone.")


if __name__ == "__main__":
    main()
