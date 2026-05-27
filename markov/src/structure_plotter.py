#!/usr/bin/env python3
"""
Structure Plotter — render a generated timeline as a color-coded form diagram.

Usage::

    from structure_plotter import StructurePlotter

    StructurePlotter.plot(labels, event_log, n_clusters, "timeline.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class StructurePlotter:
    """Render a cluster-label timeline as a color-coded form diagram.

    Each measure is a colored square; section blocks are grouped and
    labeled (A, B, FREE).  Cadence boundaries are marked with vertical
    lines.  The output is a PNG image.
    """

    @staticmethod
    def plot(
        labels: List[int],
        event_log: List[Dict[str, Any]],
        n_clusters: int,
        save_path: Union[str, Path],
    ) -> None:
        """Save a structure visualization to *save_path* (.png)."""
        n_measures = len(labels)
        cmap = plt.get_cmap("tab10")
        cluster_colors = [cmap(i % 10) for i in range(n_clusters)]

        measures_per_row = 16
        n_rows = (n_measures + measures_per_row - 1) // measures_per_row
        square_w = 0.35
        square_h = 0.30
        gap = 0.03
        row_gap = 0.08
        label_offset = 0.12

        fig_w = measures_per_row * (square_w + gap) + 3.0
        fig_h = n_rows * (square_h + row_gap + label_offset) + 1.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(0, measures_per_row * (square_w + gap) + 2.5)
        ax.set_ylim(-n_rows * (square_h + row_gap + label_offset) - 0.5, 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        measure_section, measure_role = _build_section_map(
            event_log, n_measures,
        )

        for row in range(n_rows):
            y_base = -row * (square_h + row_gap + label_offset)
            for col in range(measures_per_row):
                idx = row * measures_per_row + col
                if idx >= n_measures:
                    break
                x = col * (square_w + gap)
                y = y_base
                color = cluster_colors[labels[idx]]
                ax.add_patch(Rectangle(
                    (x, y), square_w, square_h,
                    facecolor=color, edgecolor="white", linewidth=0.3,
                ))
                if idx == 0 or (idx > 0 and measure_section[idx] != measure_section[idx - 1]):
                    ax.plot(
                        [x - gap, x - gap],
                        [y - 0.05, y + square_h + 0.05],
                        color="#333333", linewidth=1.0,
                    )

            prev_sec = None
            for col in range(measures_per_row):
                idx = row * measures_per_row + col
                if idx >= n_measures:
                    break
                sec = measure_section[idx]
                if sec is not None and sec != prev_sec:
                    x_start = col * (square_w + gap)
                    end_col = col
                    while (end_col + 1 < measures_per_row and
                           row * measures_per_row + end_col + 1 < n_measures and
                           measure_section[row * measures_per_row + end_col + 1] == sec):
                        end_col += 1
                    x_mid = (x_start + end_col * (square_w + gap) + square_w) / 2
                    label_text = f"{sec}"
                    if measure_role[idx] and measure_role[idx] not in ("NEW",):
                        label_text = f"{sec}'" if measure_role[idx] == "RETURN" else sec
                    ax.text(
                        x_mid, y_base + square_h + 0.06,
                        label_text, ha="center", va="bottom",
                        fontsize=7, color="#333333", fontweight="bold",
                    )
                    prev_sec = sec

        for col in [4, 8, 12]:
            x = col * (square_w + gap) - gap
            ax.plot([x, x], [0.2, -n_rows * (square_h + row_gap + label_offset)],
                    color="#aaaaaa", linewidth=0.5, linestyle="--", alpha=0.5)
        x = 8 * (square_w + gap) - gap
        ax.plot([x, x], [0.2, -n_rows * (square_h + row_gap + label_offset)],
                color="#666666", linewidth=0.8, linestyle="--")

        for row in range(n_rows):
            for col in range(0, measures_per_row, 4):
                idx = row * measures_per_row + col
                if idx < n_measures:
                    ax.text(
                        col * (square_w + gap) + square_w / 2,
                        -row * (square_h + row_gap + label_offset) - 0.06,
                        str(idx + 1),
                        ha="center", va="top", fontsize=5, color="#999999",
                    )

        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=cluster_colors[c], edgecolor="white")
            for c in range(n_clusters)
        ]
        ax.legend(
            legend_handles, [f"Cluster {c}" for c in range(n_clusters)],
            loc="upper right", fontsize=6, ncol=min(4, n_clusters),
            markerscale=0.6, framealpha=0.8,
        )

        n_sections = sum(1 for e in event_log if e["kind"] == "SECTION")
        n_free = sum(1 for e in event_log if e["kind"] == "FREE")
        ax.set_title(
            f"Generated Structure — {n_measures} measures, "
            f"{n_sections} sections, {n_free} FREE blocks",
            fontsize=11, pad=6,
        )

        fig.tight_layout(pad=0.5)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _build_section_map(
    event_log: List[Dict[str, Any]],
    n_measures: int,
) -> tuple:
    """Build per-measure section label and role arrays."""
    measure_section: List[Optional[str]] = [None] * n_measures
    measure_role: List[Optional[str]] = [None] * n_measures
    pos = 0
    for event in event_log:
        length = event["length"]
        if event["kind"] == "SECTION":
            for k in range(length):
                if pos + k < n_measures:
                    measure_section[pos + k] = event["label"]
                    measure_role[pos + k] = event.get("role", "")
        elif event["kind"] == "FREE":
            for k in range(length):
                if pos + k < n_measures:
                    measure_section[pos + k] = "FREE"
        pos += length
    return measure_section, measure_role
