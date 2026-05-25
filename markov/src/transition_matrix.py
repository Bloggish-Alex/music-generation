#!/usr/bin/env python3
"""
Transition Matrix Builder — cluster-based measure transition analysis.

Counts transitions between cluster labels within each file (no cross-file
leakage) and normalizes to a row-stochastic transition probability matrix.

Usage (library)::

    from measure_clustering import classify_files, MeasureExtractor, MeasureClusterer
    from transition_matrix import TransitionMatrixBuilder

    extractor = MeasureExtractor()
    clusterer = MeasureClusterer()
    clusterer.fit(extractor.extract_all("path/to/music/dir"), n_clusters=8)
    file_labels = classify_files("path/to/music/dir", clusterer, extractor)
    tmatrix = TransitionMatrixBuilder.build(file_labels)
    print(tmatrix.summary())

Usage (CLI)::

    python transition_matrix.py --music-dir ../../datasets/corelli --n-clusters 5
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

log = logging.getLogger("transition_matrix")

# ---------------------------------------------------------------------------
# TransitionMatrix dataclass
# ---------------------------------------------------------------------------


class TransitionMatrix:
    """Holds the transition count matrix, probability matrix, and metadata.

    Attributes:
        count_matrix: Raw integer transition counts, shape (n, n).
        prob_matrix: Row-normalized transition probabilities, shape (n, n).
        n_clusters: Number of cluster states.
        total_transitions: Sum of all transition counts.
    """

    def __init__(
        self,
        count_matrix: np.ndarray,
        prob_matrix: np.ndarray,
        n_clusters: int,
        total_transitions: int,
    ) -> None:
        self.count_matrix = count_matrix
        self.prob_matrix = prob_matrix
        self.n_clusters = n_clusters
        self.total_transitions = total_transitions

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_clusters, self.n_clusters)

    def as_dataframe(self) -> "pd.DataFrame":
        """Return the probability matrix as a labeled pandas DataFrame."""
        import pandas as pd

        labels = [f"Cluster {i}" for i in range(self.n_clusters)]
        return pd.DataFrame(
            self.prob_matrix,
            index=labels,
            columns=labels,
        )

    def as_count_dataframe(self) -> "pd.DataFrame":
        """Return the count matrix as a labeled pandas DataFrame."""
        import pandas as pd

        labels = [f"Cluster {i}" for i in range(self.n_clusters)]
        return pd.DataFrame(
            self.count_matrix,
            index=labels,
            columns=labels,
        )

    def summary(self) -> str:
        """Human-readable summary of the transition matrix."""
        lines = [
            "=" * 60,
            "TRANSITION MATRIX SUMMARY",
            "=" * 60,
            f"  States (clusters):  {self.n_clusters}",
            f"  Total transitions:  {self.total_transitions}",
            f"  Matrix shape:       {self.n_clusters} x {self.n_clusters}",
        ]

        # Per-row stats
        row_sums = self.count_matrix.sum(axis=1)
        zero_rows = int((row_sums == 0).sum())
        if zero_rows > 0:
            zero_indices = [i for i, s in enumerate(row_sums) if s == 0]
            lines.append(f"  Zero-outgoing states: {zero_rows} {zero_indices}")

        # Top transitions
        if self.total_transitions > 0:
            top_indices = np.argsort(self.count_matrix.ravel())[-5:][::-1]
            lines.append("  Top transitions (from -> to : count):")
            for flat_idx in top_indices:
                src, dst = divmod(int(flat_idx), self.n_clusters)
                cnt = self.count_matrix[src, dst]
                if cnt > 0:
                    lines.append(
                        f"    Cluster {src} -> Cluster {dst} : {cnt} "
                        f"({self.prob_matrix[src, dst]:.3f})"
                    )

        lines.append("=" * 60)
        return "\n".join(lines)

    def sample_next(self, current: int, seed: Optional[int] = None) -> int:
        """Sample the next state from *current*'s transition row.

        Args:
            current: Current cluster label.
            seed: Optional seed for reproducibility.

        Returns:
            A cluster label drawn from the row distribution, or a
            uniform random state if the row has no outgoing transitions.
        """
        row = self.prob_matrix[current]
        rng = np.random.RandomState(seed) if seed is not None else np.random
        if row.sum() == 0:
            return int(rng.choice(self.n_clusters))
        return int(rng.choice(self.n_clusters, p=row))


# ---------------------------------------------------------------------------
# TransitionMatrixBuilder
# ---------------------------------------------------------------------------


class TransitionMatrixBuilder:
    """Builds a TransitionMatrix from per-file cluster label sequences.

    Pure computation — no internal state.  Extraction and clustering
    happen externally via :func:`classify_files`.

    Usage::

        from measure_clustering import classify_files, MeasureExtractor, MeasureClusterer

        extractor = MeasureExtractor()
        clusterer = MeasureClusterer()
        clusterer.fit(extractor.extract_all(music_dir), n_clusters=8)
        file_labels = classify_files(music_dir, clusterer, extractor)
        tmatrix = TransitionMatrixBuilder.build(file_labels)
    """

    @staticmethod
    def build(
        file_labels: List[List[int]],
        skip_self_transitions: bool = True,
    ) -> TransitionMatrix:
        """Build the transition matrix from per-file label sequences.

        Args:
            file_labels: One ordered label list per file.
            skip_self_transitions: Exclude A→A transitions (default True).

        Returns:
            TransitionMatrix with count and probability matrices.
        """
        if not file_labels:
            raise ValueError("file_labels is empty")

        n = max(label for seq in file_labels for label in seq) + 1
        count_matrix = np.zeros((n, n), dtype=np.int64)

        skipped = 0
        for labels in file_labels:
            file_matrix = np.zeros((n, n), dtype=np.int64)
            for i in range(len(labels) - 1):
                src = labels[i]
                dst = labels[i + 1]
                if skip_self_transitions and src == dst:
                    continue
                file_matrix[src, dst] += 1
            if file_matrix.sum() == 0:
                skipped += 1
                continue
            count_matrix += file_matrix

        log.info(
            "Processed %d/%d files (%d skipped all-self).",
            len(file_labels) - skipped, len(file_labels), skipped,
        )

        prob_matrix = TransitionMatrixBuilder._normalize(count_matrix)
        total = int(count_matrix.sum())

        log.info(
            "Transition matrix: %d states, %d total transitions.",
            n, total,
        )

        return TransitionMatrix(
            count_matrix=count_matrix,
            prob_matrix=prob_matrix,
            n_clusters=n,
            total_transitions=total,
        )

    @staticmethod
    def _normalize(count_matrix: np.ndarray) -> np.ndarray:
        """Row-normalize a count matrix to transition probabilities.

        Each row sums to 1.0. Rows that sum to zero remain all zeros.
        """
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            prob = count_matrix.astype(np.float64) / row_sums
            prob[np.isnan(prob)] = 0.0
        zero_rows = int((row_sums.flatten() == 0).sum())
        if zero_rows > 0:
            zero_indices = [
                i for i, s in enumerate(row_sums.flatten()) if s == 0
            ]
            log.warning(
                "States with zero outgoing transitions: %s", zero_indices
            )
        return prob


# ---------------------------------------------------------------------------
# TransitionMatrixVisualizer
# ---------------------------------------------------------------------------


class TransitionMatrixVisualizer:
    """Generate heatmap visualizations for a TransitionMatrix."""

    def __init__(self, tmatrix: TransitionMatrix) -> None:
        self._tmatrix = tmatrix

    def plot_heatmap(
        self,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "YlOrRd",
        fmt: str = ".2f",
        annot: bool = True,
    ) -> None:
        """Save a heatmap of the transition probability matrix.

        Args:
            save_path: Output image path (.png).
            title: Plot title. Auto-generated if None.
            figsize: Figure size in inches.
            cmap: Matplotlib colormap name.
            fmt: Annotation format string.
            annot: Whether to show cell values.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self._tmatrix.as_dataframe()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            square=True,
            linewidths=0.5,
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"shrink": 0.8, "label": "Transition Probability"},
            ax=ax,
        )
        ax.set_title(
            title or f"Cluster Transition Probability Matrix "
            f"(n={self._tmatrix.n_clusters})",
            fontsize=16,
            pad=20,
        )
        ax.set_xlabel("To Cluster", fontsize=12)
        ax.set_ylabel("From Cluster", fontsize=12)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=9
        )
        ax.set_yticklabels(
            ax.get_yticklabels(), rotation=0, fontsize=9
        )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved probability heatmap to %s", save_path)

    def plot_count_heatmap(
        self,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "YlOrRd",
        annot: bool = True,
    ) -> None:
        """Save a heatmap of the raw transition counts.

        Args:
            save_path: Output image path (.png).
            title: Plot title. Auto-generated if None.
            figsize: Figure size in inches.
            cmap: Matplotlib colormap name.
            annot: Whether to show cell values.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = self._tmatrix.as_count_dataframe()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df,
            annot=annot,
            fmt="d",
            cmap=cmap,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Transition Count"},
            ax=ax,
        )
        ax.set_title(
            title or f"Cluster Transition Count Matrix "
            f"(n={self._tmatrix.n_clusters})",
            fontsize=16,
            pad=20,
        )
        ax.set_xlabel("To Cluster", fontsize=12)
        ax.set_ylabel("From Cluster", fontsize=12)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=9
        )
        ax.set_yticklabels(
            ax.get_yticklabels(), rotation=0, fontsize=9
        )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved count heatmap to %s", save_path)


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
        description="Transition Matrix Builder — cluster-based measure "
        "transition analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_music_source_args(parser)
    add_clustering_args(parser)
    add_model_io_args(parser)
    add_verbose_arg(parser)
    parser.add_argument(
        "--output",
        default="transition_matrix.npz",
        help="Path to save the transition matrix (.npz format).",
    )
    parser.add_argument(
        "--heatmap",
        default="transition_heatmap.png",
        help="Path to save the probability heatmap image.",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip heatmap generation.",
    )
    parser.add_argument(
        "--count-heatmap",
        default=None,
        help="Path to save the raw count heatmap (optional).",
    )
    parser.add_argument(
        "--include-self-transitions",
        action="store_true",
        help="Include A->A self-transitions in counts and probabilities.",
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

    log.info("Building transition matrix ...")
    tmatrix = TransitionMatrixBuilder.build(
        file_labels,
        skip_self_transitions=not args.include_self_transitions,
    )

    # Print summary and matrix
    print()
    print(tmatrix.summary())
    print("\nTransition Probability Matrix:")
    print(tmatrix.as_dataframe().to_string(float_format=lambda x: f"{x:.4f}"))

    # Save matrix
    np.savez(
        args.output,
        count=tmatrix.count_matrix,
        prob=tmatrix.prob_matrix,
    )
    log.info("Saved transition matrix to %s", args.output)

    # Heatmaps
    if not args.no_heatmap:
        viz = TransitionMatrixVisualizer(tmatrix)
        viz.plot_heatmap(args.heatmap)
        if args.count_heatmap:
            viz.plot_count_heatmap(args.count_heatmap)

    print("\nDone.")


if __name__ == "__main__":
    main()
