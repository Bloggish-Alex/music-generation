#!/usr/bin/env python3
"""
Phrase Generator — generates cluster label timelines from a MusicModel.

Uses start distribution → persistence duration → transition matrix
to produce a sequence of cluster labels of a target length.

Usage::

    from music_model import MusicModel
    from phrase_generator import PhraseGenerator, plot_timeline

    model = MusicModel.load("./models/my_model")
    gen = PhraseGenerator(model)
    labels = gen.generate(100, seed=42)
    plot_timeline(labels, "timeline.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from music_model import TsModel

log = logging.getLogger("phrase_generator")


class PhraseGenerator:
    """Generate measure label sequences from a trained time-signature sub-model.

    Algorithm::

        1. Sample start state from start_distribution
        2. Sample run-length from persistence_duration for that state
        3. Repeat state for run_length measures
        4. Sample next state from transition_matrix (skip row of zeros)
        5. Repeat from step 2 until target length reached
    """

    def __init__(self, ts_model: TsModel) -> None:
        self._ts_model = ts_model

    @property
    def ts_model(self) -> TsModel:
        return self._ts_model

    def generate(
        self, num_measures: int, seed: Optional[int] = None
    ) -> List[int]:
        """Generate a sequence of cluster labels.

        Args:
            num_measures: Target number of measures.
            seed: Optional random seed for reproducibility.

        Returns:
            List of cluster labels, length *num_measures*.
        """
        if num_measures <= 0:
            return []

        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)

        model = self._ts_model

        labels: List[int] = []
        n = seed
        state = model.start_distribution.sample(seed=n)
        n += 1

        while len(labels) < num_measures:
            run_len = model.persistence_duration.sample_duration(state)
            run_len = min(run_len, num_measures - len(labels))
            labels.extend([state] * run_len)
            if len(labels) >= num_measures:
                break
            state = model.transition_matrix.sample_next(state, seed=n)
            n += 1

        log.info(
            "Generated %d measures, %d state transitions.",
            len(labels),
            len(labels) - 1,
        )
        return labels

    def generate_phrases(
        self,
        phrase_lengths: List[int],
        seed: Optional[int] = None,
    ) -> List[List[int]]:
        """Generate multiple phrases with file boundaries honoured.

        Each phrase restarts from a fresh start state, producing
        independent label sequences with no cross-phrase transitions.

        Args:
            phrase_lengths: Target lengths for each phrase.
            seed: Optional base seed (each phrase uses seed + i).

        Returns:
            List of label lists, one per phrase.
        """
        result: List[List[int]] = []
        for i, length in enumerate(phrase_lengths):
            phrase_seed = None if seed is None else seed + i
            result.append(self.generate(length, seed=phrase_seed))
        return result

    # -- internal -------------------------------------------------------------


# ---------------------------------------------------------------------------
# Timeline visualization
# ---------------------------------------------------------------------------


def plot_timeline(
    labels: List[int],
    save_path: Union[str, Path],
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    square_size: float = 0.22,
    gap: float = 0.04,
    row_gap: float = 0.18,
) -> None:
    """Render a label sequence as a grid of coloured squares.

    Each measure is one square.  Consecutive identical labels are laid
    out left-to-right in one row; when the cluster changes, the next run
    starts a new row.  Squares have a small gap between them.

    Example: ``2222 / 000 / 222222 / 77 / 222 / 1`` produces six rows.

    Args:
        labels: Ordered list of cluster labels.
        save_path: Output image path (.png).
        title: Plot title. Auto-generated if None.
        figsize: Figure size in inches. Auto-computed if None.
        dpi: Output resolution.
        square_size: Width/height of each square in inches.
        gap: Horizontal gap between adjacent squares (inches).
        row_gap: Vertical gap between rows (inches).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not labels:
        log.warning("Empty label list, skipping timeline plot.")
        return

    # Build runs: (label, length)
    runs: List[Tuple[int, int]] = []
    current = labels[0]
    count = 1
    for s in labels[1:]:
        if s == current:
            count += 1
        else:
            runs.append((current, count))
            current = s
            count = 1
    runs.append((current, count))

    n_clusters = max(labels) + 1
    cmap = plt.get_cmap("tab10")
    cluster_colors = [cmap(i % 10) for i in range(n_clusters)]

    # Figure size — auto-compute if not provided
    if figsize is None:
        max_run = max(length for _, length in runs)
        fig_w = max_run * (square_size + gap) - gap + 1.2
        fig_h = len(runs) * (square_size + row_gap) - row_gap + 1.0
        figsize = (max(fig_w, 4.0), max(fig_h, 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # Draw squares row by row
    y = 0.0
    for label, length in runs:
        color = cluster_colors[label]
        for i in range(length):
            x = i * (square_size + gap)
            rect = Rectangle(
                (x, y), square_size, square_size,
                facecolor=color, edgecolor="white",
                linewidth=0.3,
            )
            ax.add_patch(rect)
        # Label at the end of each row
        ax.text(
            length * (square_size + gap) - gap + 0.08,
            y + square_size / 2,
            f"×{length}",
            ha="left", va="center", fontsize=7, color="#555555",
        )
        y -= (square_size + row_gap)

    # Axes limits
    max_run = max(length for _, length in runs)
    ax.set_xlim(
        -0.15,
        max_run * (square_size + gap) - gap + 0.6,
    )
    ax.set_ylim(
        y + row_gap - 0.15,
        square_size + 0.15,
    )
    ax.axis("off")

    # Title
    ax.set_title(
        title or f"Phrase Timeline ({len(labels)} measures, "
        f"{len(runs)} runs, {n_clusters} clusters)",
        fontsize=12, pad=8,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=cluster_colors[c], edgecolor="white",
              label=f"Cluster {c}")
        for c in sorted(set(labels))
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        ncol=min(8, len(legend_handles)), fontsize=7,
        markerscale=0.8,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved phrase timeline to %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse
    from _argparse_utils import add_verbose_arg

    parser = argparse.ArgumentParser(
        description="Phrase Generator — generate cluster label timelines "
        "from a trained MusicModel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_verbose_arg(parser)
    parser.add_argument(
        "--model",
        default="../../models/test",
        required=False,
        help="Path to a trained MusicModel directory.",
    )
    parser.add_argument(
        "--num-measures",
        type=int,
        default=100,
        help="Number of measures to generate (single phrase mode).",
    )
    parser.add_argument(
        "--phrase-lengths",
        default=None,
        help="Comma-separated phrase lengths for multi-phrase mode "
        "(overrides --num-measures).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        default="phrase_timeline.png",
        help="Path to save the timeline plot (.png).",
    )
    parser.add_argument(
        "--print-labels",
        action="store_true",
        help="Print generated labels to stdout.",
    )
    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="Time signature to generate for (default: 4/4).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading model from %s ...", args.model)
    model = MusicModel.load(args.model)
    ts_model = model.model_for(args.time_signature)
    print()
    print(model.summary())

    gen = PhraseGenerator(ts_model)

    if args.phrase_lengths:
        lengths = [int(x.strip()) for x in args.phrase_lengths.split(",") if x.strip()]
        log.info("Generating %d phrases: %s ...", len(lengths), lengths)
        phrases = gen.generate_phrases(lengths, seed=args.seed)
        labels = []
        for i, phrase in enumerate(phrases):
            labels.extend(phrase)
            if args.print_labels:
                print(f"\nPhrase {i} ({len(phrase)} measures):")
                print(phrase)
        print(f"\nGenerated {len(phrases)} phrases, {len(labels)} total measures.")
    else:
        log.info("Generating %d measures ...", args.num_measures)
        labels = gen.generate(args.num_measures, seed=args.seed)
        if args.print_labels:
            print(f"\nLabels ({len(labels)} measures):")
            print(labels)
        print(f"\nGenerated {len(labels)} measures.")

    print(f"Saving timeline to {args.output} ...")
    plot_timeline(labels, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
