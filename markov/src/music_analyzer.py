"""
Music Analyzer: Standalone MIDI/ABC melody analysis and visualization.

Provides 5 advanced melody analysis panels composable into a 2×3 dashboard:
  8. Interval Distribution Histogram + step/leap ratios
  9. Chroma Rose (polar) + Krumhansl-Kessler key detection
 10. N-gram Frequent Patterns (horizontal bar chart)
 11. Self-Similarity Matrix (heatmap)
 12. Melody Statistics Summary (text panel)

Can be used independently without Markov chain training.

Usage:
    # As a library
    from music_analyzer import MelodyAnalyzer
    MelodyAnalyzer.plot_melody_dashboard(events, save_path="analysis.png")

    # Standalone file analysis
    stats = MelodyAnalyzer.analyze_file("song.mid", save_path="dashboard.png")

    # CLI
    python music_analyzer.py --input song.mid --save dashboard.png
    python music_analyzer.py --input tune.abc --n 5 --top-k 20
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Internal imports from sibling module
# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path when run directly
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from markov_highorder import (  # noqa: E402
    BEAT_DIVISIONS,
    DURATION_CATEGORIES,
    HigherOrderMarkovChain,
    MusicEvent,
    MidiMetadata,
    MidiParser,
    VELOCITY_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Krumhansl-Kessler tonal profiles (Cognitive Foundations of Musical Pitch, 1990)
# Normalized so each profile sums to 1 for cosine-similarity compatibility.
KK_MAJOR_PROFILE: np.ndarray = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
KK_MINOR_PROFILE: np.ndarray = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

_NOTE_NAMES: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Diatonic pitch-class sets (C-rooted; rotated per detected tonic).
_MAJOR_DIATONIC: Set[int] = {0, 2, 4, 5, 7, 9, 11}
_MINOR_DIATONIC: Set[int] = {0, 2, 3, 5, 7, 8, 10}

# Default parameters — all overridable via function arguments.
_DEFAULT_NGRAM_N: int = 4
_DEFAULT_NGRAM_TOP_K: int = 15
_DEFAULT_INTERVAL_RANGE: int = 24
_DEFAULT_SSM_WINDOW_FACTOR: int = 30
_DEFAULT_SSM_MIN_WINDOW: int = 8
_DEFAULT_SSM_CMAP: str = "magma"
_DEFAULT_DASH_FIGSIZE: Tuple[int, int] = (26, 17)
_DEFAULT_DASH_HSPACE: float = 0.45
_DEFAULT_DASH_WSPACE: float = 0.35
_DEFAULT_DPI: int = 150

# Interval classification boundaries (semitones, absolute value).
_STEP_MAX: int = 2
_SKIP_MAX: int = 5


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_note_name(pitch: int) -> str:
    """Convert MIDI pitch number to note name with octave.

    Examples: 60 → 'C4', 71 → 'B4', 48 → 'C3'.
    """
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _classify_interval(interval: int) -> str:
    """Classify a pitch interval into step / skip / leap / repeat."""
    if interval == 0:
        return "repeat"
    abs_int = abs(interval)
    if abs_int <= _STEP_MAX:
        return "step"
    if abs_int <= _SKIP_MAX:
        return "skip"
    return "leap"


def _classify_ngram(ngram: Tuple[int, ...]) -> str:
    """Classify a pitch n-gram into arpeggio / stepwise / repeat / other."""
    if len(ngram) < 2:
        return "other"
    if all(p == ngram[0] for p in ngram):
        return "repeat"
    intervals = [ngram[i + 1] - ngram[i] for i in range(len(ngram) - 1)]
    if all(1 <= abs(x) <= 2 for x in intervals):
        return "stepwise"
    if all(3 <= abs(x) <= 5 for x in intervals):
        return "arpeggio"
    return "other"


def _extract_pitches(events: Sequence[MusicEvent]) -> List[int]:
    """Return pitches from non-rest events."""
    return [ev.pitch for ev in events if ev.pitch >= 0]


def _ensure_ax(ax: Optional[plt.Axes], figsize: Tuple[float, float] = (8, 6),
               subplot_kw: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Figure], plt.Axes]:
    """Return (fig, ax).  If ax is given, fig is None (embedded mode)."""
    if ax is not None:
        return None, ax
    fig = plt.figure(figsize=figsize)
    kw = subplot_kw or {}
    ax_obj = fig.add_subplot(111, **kw)
    return fig, ax_obj


def _resolve_save_or_show(fig: Optional[Figure], save_path: Optional[str] = None,
                          dpi: int = _DEFAULT_DPI) -> Optional[Figure]:
    """Save the figure if save_path given, else show it. Return fig (or None if saved)."""
    if fig is None:
        return None
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# MusicVisualizer — migrated from music_markov_model
# ---------------------------------------------------------------------------


class MusicVisualizer:
    """Visualize musical data using seaborn and matplotlib.

    All methods are static.  This class is the original visualizer from
    ``music_markov_model``, moved here so all plotting code lives in one module.
    """

    @staticmethod
    def plot_pitch_distribution(
        events: Sequence[MusicEvent],
        title: str = "Pitch Distribution",
        save_path: Optional[str] = None,
    ):
        """Pitch histogram + piano-roll scatter (two subplots)."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        pitches = [ev.pitch for ev in events if ev.pitch >= 0]
        if not pitches:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        sns.histplot(pitches, bins=min(88, len(set(pitches))), kde=True, ax=axes[0])
        axes[0].set_title(f"{title} — Histogram")
        axes[0].set_xlabel("MIDI Pitch")
        axes[0].set_ylabel("Count")

        times = list(range(len(pitches)))
        axes[1].scatter(times, pitches, s=2, alpha=0.6, c=pitches, cmap="plasma")
        axes[1].set_title(f"{title} — Piano Roll")
        axes[1].set_xlabel("Event index")
        axes[1].set_ylabel("MIDI Pitch")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_duration_distribution(
        events: Sequence[MusicEvent],
        title: str = "Duration Distribution",
        save_path: Optional[str] = None,
    ):
        """Horizontal bar chart of duration categories."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        dur_names = list(DURATION_CATEGORIES.keys())
        dur_indices = [ev.duration_idx for ev in events]
        dur_labels = [dur_names[i] if i < len(dur_names) else f"unk_{i}"
                      for i in dur_indices]

        fig, ax = plt.subplots(figsize=(12, 5))
        order = sorted(set(dur_labels), key=lambda x: DURATION_CATEGORIES.get(x, 99))
        sns.countplot(y=dur_labels, order=order, ax=ax, hue=dur_labels,
                      palette="viridis", legend=False)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Duration Category")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_velocity_heatmap(
        events: Sequence[MusicEvent],
        title: str = "Velocity × Pitch Heatmap",
        save_path: Optional[str] = None,
    ):
        """Velocity × Pitch heatmap (12 pitch bins × 8 velocity categories)."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        vel_names = list(VELOCITY_CATEGORIES.keys())
        data = pd.DataFrame([
            {"pitch": ev.pitch,
             "velocity": vel_names[ev.velocity_idx] if ev.velocity_idx < len(vel_names) else "?"}
            for ev in events if ev.pitch >= 0
        ])

        if data.empty:
            return

        data["pitch_bin"] = pd.cut(data["pitch"], bins=12, labels=[
            f"{int(b.left)}-{int(b.right)}" for b in
            pd.cut(data["pitch"], bins=12).cat.categories
        ])

        pivot = data.pivot_table(
            index="pitch_bin", columns="velocity",
            aggfunc="size", fill_value=0, observed=False,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Velocity Category")
        ax.set_ylabel("Pitch Bin")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_markov_transition_heatmap(
        chain: HigherOrderMarkovChain,
        top_n: int = 30,
        title: str = "Markov Transition Matrix",
        save_path: Optional[str] = None,
    ):
        """First-order Markov transition matrix heatmap (top-N tokens)."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if 1 not in chain._chains:
            return

        trans1 = chain._chains[1]
        token_freq: Dict[str, float] = defaultdict(float)
        for (tok,), next_probs in trans1.items():
            token_freq[tok] += sum(next_probs.values())
        top_tokens = sorted(token_freq, key=token_freq.get, reverse=True)[:top_n]

        n = len(top_tokens)
        matrix = np.zeros((n, n))
        for i, t1 in enumerate(top_tokens):
            if (t1,) in trans1:
                for j, t2 in enumerate(top_tokens):
                    matrix[i, j] = trans1[(t1,)].get(t2, 0.0)

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(matrix, xticklabels=top_tokens, yticklabels=top_tokens,
                    cmap="YlOrRd", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Next Token")
        ax.set_ylabel("Current Token")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_summary_dashboard(
        events: Sequence[MusicEvent],
        transition_matrix: Optional[Mapping[Tuple[str, ...], Mapping[str, float]]] = None,
        title: str = "Music Analysis Dashboard",
        save_path: Optional[str] = None,
    ):
        """3×3 summary dashboard with 7 sub-panels."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        dur_names = list(DURATION_CATEGORIES.keys())
        vel_names = list(VELOCITY_CATEGORIES.keys())
        pitches = [ev.pitch for ev in events if ev.pitch >= 0]
        dur_labels = [dur_names[ev.duration_idx] if ev.duration_idx < len(dur_names) else "?"
                      for ev in events]
        vel_labels = [vel_names[ev.velocity_idx] if ev.velocity_idx < len(vel_names) else "?"
                      for ev in events]

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # (0,0) Pitch histogram
        ax1 = fig.add_subplot(gs[0, 0])
        if pitches:
            sns.histplot(pitches, bins=40, kde=True, ax=ax1)
            ax1.set_title("Pitch Distribution")
            ax1.set_xlabel("MIDI Pitch")

        # (0,1) Duration counts
        ax2 = fig.add_subplot(gs[0, 1])
        dur_order = sorted(set(dur_labels), key=lambda x: DURATION_CATEGORIES.get(x, 99))
        sns.countplot(y=dur_labels, order=dur_order, ax=ax2, hue=dur_labels,
                      palette="viridis", legend=False)
        ax2.set_title("Duration Categories")

        # (0,2) Velocity counts
        ax3 = fig.add_subplot(gs[0, 2])
        vel_order = sorted(set(vel_labels), key=lambda x: VELOCITY_CATEGORIES.get(x, 0))
        sns.countplot(y=vel_labels, order=vel_order, ax=ax3, hue=vel_labels,
                      palette="magma", legend=False)
        ax3.set_title("Velocity Categories")

        # (1,0) Beat position
        ax4 = fig.add_subplot(gs[1, 0])
        beat_positions = [ev.beat_position for ev in events]
        sns.histplot(beat_positions, bins=BEAT_DIVISIONS, ax=ax4)
        ax4.set_title("Beat Position Distribution")
        ax4.set_xlabel("Position (0–15)")

        # (1,1) Pitch × Event Index
        ax5 = fig.add_subplot(gs[1, 1])
        if pitches:
            ax5.scatter(range(len(pitches)), pitches, s=1, alpha=0.5,
                        c=pitches, cmap="plasma")
            ax5.set_title("Pitch × Event Index")
            ax5.set_xlabel("Event Index")
            ax5.set_ylabel("MIDI Pitch")

        # (1,2) Pitch vs Duration
        ax6 = fig.add_subplot(gs[1, 2])
        if pitches:
            dur_values = [DURATION_CATEGORIES.get(d, 1.0)
                          for ev, d in zip(events, dur_labels) if ev.pitch >= 0]
            ax6.scatter(pitches, dur_values, s=3, alpha=0.4, c=pitches, cmap="plasma")
            ax6.set_title("Pitch vs Duration")
            ax6.set_xlabel("MIDI Pitch")
            ax6.set_ylabel("Quarter Length")

        # (2, 0:3) Transition matrix
        ax7 = fig.add_subplot(gs[2, :])
        if transition_matrix:
            trans1 = transition_matrix
            token_freq: Dict[str, float] = defaultdict(float)
            for history, next_probs in trans1.items():
                if len(history) != 1:
                    continue
                tok = history[0]
                token_freq[tok] += sum(next_probs.values())
            top_tokens = sorted(token_freq, key=token_freq.get, reverse=True)[:20]

            n = len(top_tokens)
            matrix = np.zeros((n, n))
            short_labels = [t[:20] for t in top_tokens]
            for i, t1 in enumerate(top_tokens):
                if (t1,) in trans1:
                    for j, t2 in enumerate(top_tokens):
                        matrix[i, j] = trans1[(t1,)].get(t2, 0.0)
            if n > 0:
                sns.heatmap(matrix, xticklabels=short_labels, yticklabels=short_labels,
                            cmap="YlOrRd", ax=ax7, cbar_kws={"label": "Probability"})
                ax7.set_title("First-Order Transition Matrix (Top 20 Tokens)")
            else:
                ax7.text(0.5, 0.5, "No first-order transition rows",
                         ha="center", va="center", transform=ax7.transAxes)
                ax7.set_axis_off()
        else:
            ax7.text(0.5, 0.5, "No transition matrix provided",
                     ha="center", va="center", transform=ax7.transAxes)
            ax7.set_axis_off()

        fig.suptitle(title, fontsize=16, fontweight="bold")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# MelodyAnalyzer
# ---------------------------------------------------------------------------


class MelodyAnalyzer:
    """Collection of melody analysis and visualization methods.

    All methods are static.  Every parameter that could affect results is
    exposed as a function argument with a documented default.
    """

    # ==================================================================
    # Computation methods (pure data, no matplotlib)
    # ==================================================================

    @staticmethod
    def compute_intervals(
        events: Sequence[MusicEvent],
    ) -> Dict[str, Any]:
        """Compute adjacent pitch-interval statistics.

        Returns a dict with keys:
            intervals, steps, skips, leaps, repeats,
            ascending, descending, mean_interval, std_interval, total.
        All counts are 0 and 'intervals' is an empty list when fewer than
        2 pitched notes exist.
        """
        pitches = _extract_pitches(events)
        if len(pitches) < 2:
            return {
                "intervals": [],
                "steps": 0, "skips": 0, "leaps": 0, "repeats": 0,
                "ascending": 0, "descending": 0,
                "mean_interval": 0.0, "std_interval": 0.0,
                "total": 0,
            }
        intervals = [int(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
        arr = np.array(intervals, dtype=np.float64)
        return {
            "intervals": intervals,
            "steps": sum(1 for x in intervals if 1 <= abs(x) <= _STEP_MAX),
            "skips": sum(1 for x in intervals if _STEP_MAX < abs(x) <= _SKIP_MAX),
            "leaps": sum(1 for x in intervals if abs(x) > _SKIP_MAX),
            "repeats": sum(1 for x in intervals if x == 0),
            "ascending": sum(1 for x in intervals if x > 0),
            "descending": sum(1 for x in intervals if x < 0),
            "mean_interval": float(arr.mean()) if len(arr) > 0 else 0.0,
            "std_interval": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "total": len(intervals),
        }

    @staticmethod
    def compute_chroma(
        events: Sequence[MusicEvent],
    ) -> Dict[str, Any]:
        """Compute pitch-class distribution and detect key via KK profiles.

        Returns:
            chroma_counts: np.ndarray of shape (12,) — raw counts per pitch class
            total_pitched: int
            best_key: str like 'C major' / 'A minor' / 'N/A'
            best_r: float — Pearson r of best-matching profile
            all_scores: List[Tuple[str, float]] — all 24 keys sorted by r desc
        """
        pitches = _extract_pitches(events)
        chroma = np.zeros(12, dtype=np.float64)
        for p in pitches:
            chroma[p % 12] += 1.0
        total = len(pitches)

        if total == 0:
            return {
                "chroma_counts": chroma,
                "total_pitched": 0,
                "best_key": "N/A",
                "best_r": 0.0,
                "all_scores": [],
            }

        # Normalise chroma for correlation
        chroma_norm = chroma / chroma.sum()

        key_names_major = [f"{_NOTE_NAMES[r]} major" for r in range(12)]
        key_names_minor = [f"{_NOTE_NAMES[r]} minor" for r in range(12)]

        all_scores: List[Tuple[str, float]] = []
        for root in range(12):
            # Rotate profile so index 0 aligns with the candidate tonic
            rotated_major = np.roll(KK_MAJOR_PROFILE, root)
            rotated_minor = np.roll(KK_MINOR_PROFILE, root)
            r_maj = float(np.corrcoef(chroma_norm, rotated_major)[0, 1])
            r_min = float(np.corrcoef(chroma_norm, rotated_minor)[0, 1])
            all_scores.append((key_names_major[root], r_maj))
            all_scores.append((key_names_minor[root], r_min))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        best_key, best_r = all_scores[0]
        if np.isnan(best_r):
            best_r = 0.0

        return {
            "chroma_counts": chroma,
            "total_pitched": total,
            "best_key": best_key,
            "best_r": best_r,
            "all_scores": all_scores,
        }

    @staticmethod
    def compute_ngrams(
        events: Sequence[MusicEvent],
        n: int = _DEFAULT_NGRAM_N,
        top_k: int = _DEFAULT_NGRAM_TOP_K,
    ) -> Dict[str, Any]:
        """Extract the most frequent pitch n-grams.

        Returns:
            top_patterns: List[Tuple[Tuple[int,...], int]] sorted by frequency
            total_ngrams: int — total number of n-gram occurrences
            unique_ngrams: int — count of distinct n-grams
            coverage: float — fraction of total occurrences covered by top_k
        """
        pitches = _extract_pitches(events)
        if len(pitches) < n:
            return {
                "top_patterns": [],
                "total_ngrams": 0,
                "unique_ngrams": 0,
                "coverage": 0.0,
            }

        counts: Dict[Tuple[int, ...], int] = {}
        for i in range(len(pitches) - n + 1):
            pat = tuple(pitches[i : i + n])
            counts[pat] = counts.get(pat, 0) + 1

        total = sum(counts.values())
        unique = len(counts)
        sorted_pats = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top = sorted_pats[:top_k]
        coverage = sum(c for _, c in top) / total if total > 0 else 0.0

        return {
            "top_patterns": top,
            "total_ngrams": total,
            "unique_ngrams": unique,
            "coverage": coverage,
        }

    @staticmethod
    def compute_ssm(
        events: Sequence[MusicEvent],
        window: Optional[int] = None,
        hop: Optional[int] = None,
        window_factor: int = _DEFAULT_SSM_WINDOW_FACTOR,
        min_window: int = _DEFAULT_SSM_MIN_WINDOW,
    ) -> Dict[str, Any]:
        """Build a self-similarity matrix from the pitch sequence.

        Parameters:
            window: Segment size in notes.  Auto-computed if None:
                    ``max(min_window, len(pitches) // window_factor)``.
            hop: Step between segments.  Defaults to ``max(1, window // 2)``.
            window_factor: Divisor for auto window (default 30).
            min_window: Minimum segment size (default 8).

        Returns:
            matrix: np.ndarray (n_seg, n_seg), values in [0, 1]
            n_segments: int
            window: int
            hop: int
        """
        pitches = _extract_pitches(events)
        n_notes = len(pitches)

        if window is None:
            window = max(min_window, n_notes // window_factor)
        if hop is None:
            hop = max(1, window // 2)

        if n_notes < window or window < 1:
            return {
                "matrix": np.eye(1, dtype=np.float64),
                "n_segments": 0,
                "window": window,
                "hop": hop,
            }

        segments: List[np.ndarray] = []
        for i in range(0, n_notes - window + 1, hop):
            segments.append(np.array(pitches[i : i + window], dtype=np.float64))

        n_seg = len(segments)
        if n_seg == 0:
            return {
                "matrix": np.eye(1, dtype=np.float64),
                "n_segments": 0,
                "window": window,
                "hop": hop,
            }

        matrix = np.zeros((n_seg, n_seg), dtype=np.float64)
        max_dist = np.sqrt(window * 127.0 ** 2)  # theoretical max Euclidean distance

        for i in range(n_seg):
            matrix[i, i] = 1.0
            for j in range(i + 1, n_seg):
                dist = float(np.linalg.norm(segments[i] - segments[j]))
                sim = 1.0 - dist / max_dist
                matrix[i, j] = sim
                matrix[j, i] = sim

        return {
            "matrix": matrix,
            "n_segments": n_seg,
            "window": window,
            "hop": hop,
        }

    @staticmethod
    def compute_melody_stats(
        events: Sequence[MusicEvent],
        ngram_n: int = _DEFAULT_NGRAM_N,
        ngram_top_k: int = _DEFAULT_NGRAM_TOP_K,
        ssm_window: Optional[int] = None,
        ssm_hop: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Aggregate all melody metrics into a single dictionary.

        This is the primary high-level analysis entry point.
        Returns a flat dict suitable for display or programmatic use.
        """
        pitches = _extract_pitches(events)
        n_notes = len(pitches)
        n_rests = sum(1 for e in events if e.pitch < 0)
        total = len(events)

        # Pitch statistics
        pitch_arr = np.array(pitches, dtype=np.float64) if pitches else np.array([])
        pitch_min = int(pitch_arr.min()) if len(pitch_arr) > 0 else -1
        pitch_max = int(pitch_arr.max()) if len(pitch_arr) > 0 else -1
        pitch_range_semi = pitch_max - pitch_min if len(pitch_arr) > 0 else 0
        mean_pitch = float(pitch_arr.mean()) if len(pitch_arr) > 0 else 0.0
        median_pitch = float(np.median(pitch_arr)) if len(pitch_arr) > 0 else 0.0
        unique_pitches = len(set(pitches))

        # Sub-analyses
        interval_stats = MelodyAnalyzer.compute_intervals(events)
        chroma_stats = MelodyAnalyzer.compute_chroma(events)
        ngram_stats = MelodyAnalyzer.compute_ngrams(events, n=ngram_n, top_k=ngram_top_k)
        ssm_stats = MelodyAnalyzer.compute_ssm(events, window=ssm_window, hop=ssm_hop)

        it = interval_stats
        it_total = it["total"] or 1  # avoid division by zero

        return {
            # Event composition
            "total_events": total,
            "note_count": n_notes,
            "rest_count": n_rests,
            "note_ratio": n_notes / total if total > 0 else 0.0,
            "rest_ratio": n_rests / total if total > 0 else 0.0,
            # Pitch range
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "pitch_min_note": _to_note_name(pitch_min) if pitch_min >= 0 else "N/A",
            "pitch_max_note": _to_note_name(pitch_max) if pitch_max >= 0 else "N/A",
            "pitch_range_semitones": pitch_range_semi,
            "pitch_range_octaves": pitch_range_semi / 12.0,
            "mean_pitch": mean_pitch,
            "median_pitch": median_pitch,
            "mean_pitch_note": _to_note_name(int(round(mean_pitch))) if len(pitch_arr) > 0 else "N/A",
            "unique_pitches": unique_pitches,
            # Interval composition
            "step_count": it["steps"],
            "step_ratio": it["steps"] / it_total,
            "skip_count": it["skips"],
            "skip_ratio": it["skips"] / it_total,
            "leap_count": it["leaps"],
            "leap_ratio": it["leaps"] / it_total,
            "repeat_count": it["repeats"],
            "repeat_ratio": it["repeats"] / it_total,
            "ascending": it["ascending"],
            "asc_ratio": it["ascending"] / it_total,
            "descending": it["descending"],
            "desc_ratio": it["descending"] / it_total,
            "mean_interval": it["mean_interval"],
            "std_interval": it["std_interval"],
            # N-gram
            "unique_ngrams": ngram_stats["unique_ngrams"],
            "ngram_total": ngram_stats["total_ngrams"],
            "top_coverage": ngram_stats["coverage"],
            # Tonality
            "detected_key": chroma_stats["best_key"],
            "key_correlation": chroma_stats["best_r"],
            # SSM
            "ssm_n_segments": ssm_stats["n_segments"],
            "ssm_window": ssm_stats["window"],
        }

    # ==================================================================
    # Panel 8: Interval Distribution Histogram
    # ==================================================================

    @staticmethod
    def plot_interval_distribution(
        events: Sequence[MusicEvent],
        ax: Optional[plt.Axes] = None,
        title: str = "Interval Distribution",
        interval_range: int = _DEFAULT_INTERVAL_RANGE,
        show_kde: bool = True,
        show_step_annotation: bool = True,
        save_path: Optional[str] = None,
        color_asc: str = "#E74C3C",
        color_desc: str = "#3498DB",
        color_repeat: str = "#95A5A6",
        kde_alpha: float = 0.35,
        dpi: int = _DEFAULT_DPI,
    ) -> Optional[Figure]:
        """Plot adjacent-interval histogram with step/leap annotations.

        Ascending intervals are drawn in ``color_asc`` (default warm red),
        descending in ``color_desc`` (default steel blue), and zero-interval
        repeats in ``color_repeat``.
        """
        fig, ax_obj = _ensure_ax(ax, figsize=(10, 6))

        stats = MelodyAnalyzer.compute_intervals(events)
        intervals = stats["intervals"]

        if not intervals:
            ax_obj.text(0.5, 0.5, "Insufficient data\n(< 2 notes)",
                        transform=ax_obj.transAxes, ha="center", va="center",
                        fontsize=12, color="gray")
            ax_obj.set_title(title)
            return _resolve_save_or_show(fig, save_path, dpi)

        # Separate by direction
        asc_vals = [x for x in intervals if x > 0]
        desc_vals = [x for x in intervals if x < 0]
        repeat_vals = [x for x in intervals if x == 0]

        bins = np.arange(-interval_range - 0.5, interval_range + 1.5, 1)
        # Use ax.bar for precise colour control per bar
        counts, bin_edges = np.histogram(intervals, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for ctr, cnt in zip(bin_centers, counts):
            if cnt == 0:
                continue
            ival = int(round(ctr))
            if ival > 0:
                clr = color_asc
            elif ival < 0:
                clr = color_desc
            else:
                clr = color_repeat
            ax_obj.bar(ctr, cnt, width=0.9, color=clr, alpha=0.85, edgecolor="none")

        # KDE overlays
        if show_kde:
            import seaborn as sns
            if asc_vals:
                sns.kdeplot(asc_vals, ax=ax_obj, color=color_asc, alpha=kde_alpha,
                            linewidth=2, label="Ascending")
            if desc_vals:
                sns.kdeplot(desc_vals, ax=ax_obj, color=color_desc, alpha=kde_alpha,
                            linewidth=2, label="Descending")

        # Boundary lines
        for bound, ls, lbl in [(-(_SKIP_MAX + 0.5), "--", "leap"),
                                (-(_STEP_MAX + 0.5), ":", "skip"),
                                ((_STEP_MAX + 0.5), ":", "step"),
                                ((_SKIP_MAX + 0.5), "--", "leap")]:
            ax_obj.axvline(bound, color="gray", linestyle=ls, alpha=0.6, linewidth=1)

        ax_obj.set_xlim(-interval_range - 1, interval_range + 1)
        ax_obj.set_xlabel("Interval (semitones)")
        ax_obj.set_ylabel("Count")
        ax_obj.set_title(title)

        if show_step_annotation:
            it = stats
            total = it["total"] or 1
            text = (
                f"Step ($\\leq$2):  {it['steps']/total:5.1%}\n"
                f"Skip (3-5):      {it['skips']/total:5.1%}\n"
                f"Leap ($\\geq$6): {it['leaps']/total:5.1%}\n"
                f"Repeat:          {it['repeats']/total:5.1%}\n\n"
                f"Asc: {it['ascending']/total:.0%}  "
                f"Desc: {it['descending']/total:.0%}\n"
                f"Mean: {it['mean_interval']:+.1f}  "
                f"Std: {it['std_interval']:.1f}"
            )
            ax_obj.text(0.97, 0.97, text, transform=ax_obj.transAxes,
                        va="top", ha="right", fontsize=8, fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                  alpha=0.85, edgecolor="gray"))

        if show_kde and (asc_vals or desc_vals):
            ax_obj.legend(fontsize=8, loc="upper left")

        return _resolve_save_or_show(fig, save_path, dpi)

    # ==================================================================
    # Panel 9: Chroma Rose
    # ==================================================================

    @staticmethod
    def plot_chroma_rose(
        events: Sequence[MusicEvent],
        ax: Optional[plt.Axes] = None,
        title: str = "Chroma Rose",
        show_key_detection: bool = True,
        save_path: Optional[str] = None,
        color_diatonic: str = "#8E44AD",
        color_chromatic: str = "#BDC3C7",
        bar_alpha: float = 0.85,
        dpi: int = _DEFAULT_DPI,
    ) -> Optional[Figure]:
        """Plot a polar rose diagram of pitch-class distribution.

        Diatonic pitch classes of the detected key are coloured
        ``color_diatonic``; chromatic ones use ``color_chromatic``.
        A dashed reference circle marks the uniform-distribution baseline.
        """
        chroma = MelodyAnalyzer.compute_chroma(events)
        counts = chroma["chroma_counts"]
        total = chroma["total_pitched"]
        best_key = chroma["best_key"]
        best_r = chroma["best_r"]

        fig, ax_obj = _ensure_ax(ax, figsize=(8, 8),
                                 subplot_kw={"projection": "polar"})

        if total == 0:
            ax_obj.text(0.5, 0.5, "No pitched notes", transform=ax_obj.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
            ax_obj.set_title(title)
            return _resolve_save_or_show(fig, save_path, dpi)

        theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        ax_obj.set_theta_zero_location("N")
        ax_obj.set_theta_direction("clockwise")

        # Determine diatonic set for the detected key
        diatonic_set: Set[int] = set()
        if best_key != "N/A":
            tonic_str = best_key.split()[0]
            if tonic_str in _NOTE_NAMES:
                tonic_idx = _NOTE_NAMES.index(tonic_str)
                if "major" in best_key:
                    diatonic_set = {(p + tonic_idx) % 12 for p in _MAJOR_DIATONIC}
                else:
                    diatonic_set = {(p + tonic_idx) % 12 for p in _MINOR_DIATONIC}

        colors = [color_diatonic if i in diatonic_set else color_chromatic for i in range(12)]

        width = (2.0 * np.pi / 12) * 0.85
        bars = ax_obj.bar(theta, counts, width=width, color=colors,
                          alpha=bar_alpha, edgecolor="white", linewidth=0.5)

        # Reference circle
        uniform_val = total / 12.0
        ax_obj.axhline(y=uniform_val, color="gray", linestyle="--",
                       alpha=0.5, linewidth=1)

        ax_obj.set_xticks(theta)
        ax_obj.set_xticklabels(_NOTE_NAMES, fontsize=10)
        ax_obj.set_title(title, pad=20, fontsize=12)

        # Key detection annotation
        if show_key_detection:
            if best_key == "N/A" or best_r < 0.3:
                key_text = "Key: ambiguous"
            else:
                key_text = f"Detected: {best_key}\nr = {best_r:.3f}"
            ax_obj.text(0.5, -0.18, key_text, transform=ax_obj.transAxes,
                        ha="center", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.85, edgecolor="gray"))

        return _resolve_save_or_show(fig, save_path, dpi)

    # ==================================================================
    # Panel 10: N-gram Frequent Patterns
    # ==================================================================

    @staticmethod
    def plot_ngram_patterns(
        events: Sequence[MusicEvent],
        n: int = _DEFAULT_NGRAM_N,
        top_k: int = _DEFAULT_NGRAM_TOP_K,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        color_arpeggio: str = "#E67E22",
        color_stepwise: str = "#5DADE2",
        color_repeat: str = "#95A5A6",
        color_other: str = "#BDC3C7",
        bar_height: float = 0.72,
        dpi: int = _DEFAULT_DPI,
    ) -> Optional[Figure]:
        """Plot Top-K most frequent pitch n-grams as a horizontal bar chart.

        Patterns are labelled with note names (e.g. 'C4→E4→G4→C5') and
        colour-coded by type: arpeggio (orange), stepwise (blue),
        repeat (grey), other (light grey).
        """
        if title is None:
            title = f"Most Frequent Pitch Patterns ({n}-grams)"

        fig, ax_obj = _ensure_ax(ax, figsize=(10, 8))

        ngram_data = MelodyAnalyzer.compute_ngrams(events, n=n, top_k=top_k)
        patterns = ngram_data["top_patterns"]
        coverage = ngram_data["coverage"]
        unique = ngram_data["unique_ngrams"]

        if not patterns:
            ax_obj.text(0.5, 0.5, f"Insufficient data\n(need $\\geq$ {n} notes)",
                        transform=ax_obj.transAxes, ha="center", va="center",
                        fontsize=12, color="gray")
            ax_obj.set_title(title)
            return _resolve_save_or_show(fig, save_path, dpi)

        # Build labels and colours
        labels: List[str] = []
        colours: List[str] = []
        counts_list: List[int] = []
        for pat, cnt in patterns:
            labels.append("  →  ".join(_to_note_name(p) for p in pat))
            counts_list.append(cnt)
            cls = _classify_ngram(pat)
            if cls == "arpeggio":
                colours.append(color_arpeggio)
            elif cls == "stepwise":
                colours.append(color_stepwise)
            elif cls == "repeat":
                colours.append(color_repeat)
            else:
                colours.append(color_other)

        y_pos = list(range(len(labels) - 1, -1, -1))
        ax_obj.barh(y_pos, counts_list, height=bar_height, color=colours,
                    alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_obj.set_yticks(y_pos)
        ax_obj.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
        ax_obj.set_xlabel("Count")
        ax_obj.set_title(title, fontsize=11)

        # Count labels at bar ends
        max_cnt = max(counts_list) if counts_list else 1
        for y, cnt in zip(y_pos, counts_list):
            ax_obj.text(cnt + max_cnt * 0.01, y, str(cnt),
                        va="center", fontsize=8, color="dimgray")

        # Coverage annotation
        cov_text = (f"Top {len(patterns)} patterns cover {coverage:.0%} "
                    f"of {n}-grams\nUnique {n}-grams: {unique}")
        ax_obj.text(0.98, 0.02, cov_text, transform=ax_obj.transAxes,
                    va="bottom", ha="right", fontsize=8, color="dimgray",
                    fontfamily="monospace")

        return _resolve_save_or_show(fig, save_path, dpi)

    # ==================================================================
    # Panel 11: Self-Similarity Matrix
    # ==================================================================

    @staticmethod
    def plot_self_similarity(
        events: Sequence[MusicEvent],
        window: Optional[int] = None,
        hop: Optional[int] = None,
        window_factor: int = _DEFAULT_SSM_WINDOW_FACTOR,
        min_window: int = _DEFAULT_SSM_MIN_WINDOW,
        ax: Optional[plt.Axes] = None,
        title: str = "Self-Similarity Matrix",
        cmap: str = _DEFAULT_SSM_CMAP,
        show_guide: bool = True,
        save_path: Optional[str] = None,
        dpi: int = _DEFAULT_DPI,
    ) -> Optional[Figure]:
        """Plot a self-similarity matrix (SSM) heatmap.

        Diagonal blocks indicate repeated sections; off-diagonal bright
        spots indicate distant repetition (e.g. ABA form).
        """
        fig, ax_obj = _ensure_ax(ax, figsize=(10, 9))

        ssm = MelodyAnalyzer.compute_ssm(events, window=window, hop=hop,
                                         window_factor=window_factor,
                                         min_window=min_window)
        matrix = ssm["matrix"]
        n_seg = ssm["n_segments"]
        actual_window = ssm["window"]

        if n_seg < 2:
            n_notes = len(_extract_pitches(events))
            ax_obj.text(0.5, 0.5,
                        f"Sequence too short for SSM\n"
                        f"(need $\\geq$ {actual_window * 2} notes, have {n_notes})",
                        transform=ax_obj.transAxes, ha="center", va="center",
                        fontsize=12, color="gray")
            ax_obj.set_title(title)
            return _resolve_save_or_show(fig, save_path, dpi)

        im = ax_obj.imshow(matrix, cmap=cmap, aspect="equal",
                           vmin=0.0, vmax=1.0, origin="lower")
        cbar = fig.colorbar(im, ax=ax_obj, fraction=0.046, pad=0.04) if fig else None
        if cbar is not None:
            cbar.set_label("Similarity", fontsize=9)

        ax_obj.set_xlabel("Segment index")
        ax_obj.set_ylabel("Segment index")
        ax_obj.set_title(f"{title}\n(window={actual_window}, hop={ssm['hop']})",
                         fontsize=11)

        if show_guide:
            guide = (
                "Reading guide:\n"
                "▮ Diagonal blocks   = repeated sections\n"
                "▮ Off-diagonal spots = distant repetition (ABA)\n"
                "▮ Uniformly dark     = through-composed"
            )
            ax_obj.text(0.02, 0.98, guide, transform=ax_obj.transAxes,
                        va="top", ha="left", fontsize=7, color="white",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                                  alpha=0.55))

        return _resolve_save_or_show(fig, save_path, dpi)

    # ==================================================================
    # Panel 12: Melody Statistics Summary
    # ==================================================================

    @staticmethod
    def plot_melody_stats(
        events: Sequence[MusicEvent],
        ngram_n: int = _DEFAULT_NGRAM_N,
        ngram_top_k: int = _DEFAULT_NGRAM_TOP_K,
        ax: Optional[plt.Axes] = None,
        title: str = "Melody Statistics",
        save_path: Optional[str] = None,
        dpi: int = _DEFAULT_DPI,
    ) -> Optional[Figure]:
        """Plot a text-based melody statistics summary.

        Six sections: event composition, pitch range, interval composition
        (with Unicode mini-bars), direction, pattern diversity, and tonality.
        """
        fig, ax_obj = _ensure_ax(ax, figsize=(8, 10))

        s = MelodyAnalyzer.compute_melody_stats(events, ngram_n=ngram_n,
                                                ngram_top_k=ngram_top_k)

        if s["total_events"] == 0:
            ax_obj.text(0.5, 0.5, "No events to analyze",
                        transform=ax_obj.transAxes, ha="center", va="center",
                        fontsize=14, color="gray")
            ax_obj.set_title(title)
            return _resolve_save_or_show(fig, save_path, dpi)

        ax_obj.axis("off")

        # Mini-bar helper (20-char width)
        def _bar(ratio: float, width: int = 20) -> str:
            filled = int(round(ratio * width))
            return "█" * filled + "░" * (width - filled)

        lines: List[str] = []
        sep = "─" * 36

        # Section 1 — Event composition
        lines.append("●  Event Composition")
        lines.append(f"  Note count       {s['note_count']:>5}    ({s['note_ratio']:.0%})")
        lines.append(f"  Rest count       {s['rest_count']:>5}    ({s['rest_ratio']:.0%})")
        lines.append(sep)

        # Section 2 — Pitch range
        lines.append("●  Pitch Range")
        lines.append(f"  Range       {s['pitch_min_note']:>4s} – {s['pitch_max_note']:<4s} "
                     f"({s['pitch_range_octaves']:.1f} octaves)")
        lines.append(f"  Mean pitch  {s['mean_pitch_note']:>4s} ({s['mean_pitch']:.0f})")
        lines.append(f"  Unique       {s['unique_pitches']:>3d} / 128")
        lines.append(sep)

        # Section 3 — Interval composition
        lines.append("●  Interval Composition")
        lines.append(f"  Step (≤2)   {_bar(s['step_ratio'])} {s['step_ratio']:.0%}")
        lines.append(f"  Skip (3-5)  {_bar(s['skip_ratio'])} {s['skip_ratio']:.0%}")
        lines.append(f"  Leap (≥6)  {_bar(s['leap_ratio'])} {s['leap_ratio']:.0%}")
        lines.append(f"  Repeat      {_bar(s['repeat_ratio'])} {s['repeat_ratio']:.0%}")
        lines.append(sep)

        # Section 4 — Direction
        lines.append("●  Direction")
        lines.append(f"  Ascending     {s['asc_ratio']:.0%}")
        lines.append(f"  Descending    {s['desc_ratio']:.0%}")
        lines.append(sep)

        # Section 5 — Pattern diversity
        lines.append("●  Pattern Diversity")
        lines.append(f"  Unique {ngram_n}-grams   {s['unique_ngrams']}")
        lines.append(f"  Top {ngram_top_k} coverage  {s['top_coverage']:.0%}")
        lines.append(sep)

        # Section 6 — Tonality
        lines.append("●  Tonality")
        if s["detected_key"] == "N/A":
            lines.append("  No pitched notes")
        else:
            r = s["key_correlation"]
            confidence = "strong" if r > 0.85 else ("moderate" if r > 0.6 else "weak")
            lines.append(f"  Likely key   {s['detected_key']}  (r={r:.3f}, {confidence})")

        stats_text = "\n".join(lines)
        ax_obj.text(0.05, 0.98, stats_text, transform=ax_obj.transAxes,
                    va="top", ha="left", fontsize=10, fontfamily="monospace",
                    linespacing=1.35)
        ax_obj.set_title(title, fontsize=13, fontweight="bold", pad=12)

        return _resolve_save_or_show(fig, save_path, dpi)

    # ==================================================================
    # Dashboard Composer
    # ==================================================================

    @staticmethod
    def plot_melody_dashboard(
        events: Sequence[MusicEvent],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = _DEFAULT_DASH_FIGSIZE,
        hspace: float = _DEFAULT_DASH_HSPACE,
        wspace: float = _DEFAULT_DASH_WSPACE,
        dpi: int = _DEFAULT_DPI,
        **kwargs: Any,
    ) -> Optional[Figure]:
        """Compose the 5-panel melody analysis dashboard.

        Layout (2 rows × 3 columns):
          (0,0) Interval Distribution   (0,1) Chroma Rose   (0,2) N-gram Patterns
          (1, 0:2) Self-Similarity      (1,2) Melody Stats

        Extra ``**kwargs`` are forwarded to individual panel methods
        (e.g. ``n=5``, ``top_k=20``, ``interval_range=36``).
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=hspace, wspace=wspace,
                              height_ratios=[1.0, 1.2])

        # Filter kwargs per panel to avoid unknown-argument errors.
        panel_kwargs = {
            "n": kwargs.pop("n", _DEFAULT_NGRAM_N),
            "top_k": kwargs.pop("top_k", _DEFAULT_NGRAM_TOP_K),
            "window": kwargs.pop("window", None),
            "hop": kwargs.pop("hop", None),
            "window_factor": kwargs.pop("window_factor", _DEFAULT_SSM_WINDOW_FACTOR),
            "min_window": kwargs.pop("min_window", _DEFAULT_SSM_MIN_WINDOW),
            "interval_range": kwargs.pop("interval_range", _DEFAULT_INTERVAL_RANGE),
        }

        # Panel 8 — (0,0)
        ax8 = fig.add_subplot(gs[0, 0])
        try:
            MelodyAnalyzer.plot_interval_distribution(
                events, ax=ax8, interval_range=panel_kwargs["interval_range"])
        except Exception as exc:
            ax8.text(0.5, 0.5, f"Panel error:\n{exc}", transform=ax8.transAxes,
                     ha="center", va="center", fontsize=9, color="red")

        # Panel 9 — (0,1)
        ax9 = fig.add_subplot(gs[0, 1])
        try:
            MelodyAnalyzer.plot_chroma_rose(events, ax=ax9)
        except Exception as exc:
            ax9.text(0.5, 0.5, f"Panel error:\n{exc}", transform=ax9.transAxes,
                     ha="center", va="center", fontsize=9, color="red")

        # Panel 10 — (0,2)
        ax10 = fig.add_subplot(gs[0, 2])
        try:
            MelodyAnalyzer.plot_ngram_patterns(
                events, ax=ax10, n=panel_kwargs["n"], top_k=panel_kwargs["top_k"])
        except Exception as exc:
            ax10.text(0.5, 0.5, f"Panel error:\n{exc}", transform=ax10.transAxes,
                      ha="center", va="center", fontsize=9, color="red")

        # Panel 11 — (1, 0:2)
        ax11 = fig.add_subplot(gs[1, :2])
        try:
            MelodyAnalyzer.plot_self_similarity(
                events, ax=ax11, window=panel_kwargs["window"],
                hop=panel_kwargs["hop"],
                window_factor=panel_kwargs["window_factor"],
                min_window=panel_kwargs["min_window"])
        except Exception as exc:
            ax11.text(0.5, 0.5, f"Panel error:\n{exc}", transform=ax11.transAxes,
                      ha="center", va="center", fontsize=9, color="red")

        # Panel 12 — (1,2)
        ax12 = fig.add_subplot(gs[1, 2])
        try:
            MelodyAnalyzer.plot_melody_stats(
                events, ax=ax12, ngram_n=panel_kwargs["n"],
                ngram_top_k=panel_kwargs["top_k"])
        except Exception as exc:
            ax12.text(0.5, 0.5, f"Panel error:\n{exc}", transform=ax12.transAxes,
                      ha="center", va="center", fontsize=9, color="red")

        fig.suptitle("Melody Analysis Dashboard", fontsize=16, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        plt.show()
        return fig

    # ==================================================================
    # Standalone File Analysis
    # ==================================================================

    @staticmethod
    def analyze_file(
        file_path: Union[str, Path],
        save_path: Optional[str] = None,
        print_stats: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Parse a MIDI or ABC file and produce the melody dashboard + stats.

        Parameters:
            file_path: Path to a .mid / .midi / .abc file.
            save_path: If given, save the dashboard image to this path.
            print_stats: If True, print a summary to stdout.
            **kwargs: Forwarded to ``plot_melody_dashboard``.

        Returns:
            The ``compute_melody_stats`` dictionary.
        """
        parser = MidiParser()
        file_path = Path(file_path)
        try:
            events, _metadata = parser.parse(str(file_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse '{file_path}': {exc}"
            ) from exc

        if not events:
            print(f"Warning: No events extracted from '{file_path}'.")
            return MelodyAnalyzer.compute_melody_stats([])

        # Pass ngram-related kwargs through to compute_melody_stats.
        ngram_n = kwargs.get("n", _DEFAULT_NGRAM_N)
        ngram_top_k = kwargs.get("top_k", _DEFAULT_NGRAM_TOP_K)
        stats = MelodyAnalyzer.compute_melody_stats(
            events, ngram_n=ngram_n, ngram_top_k=ngram_top_k,
        )
        MelodyAnalyzer.plot_melody_dashboard(events, save_path=save_path, **kwargs)

        if print_stats:
            s = stats
            print(f"File:            {file_path}")
            print(f"Notes / Rests:   {s['note_count']} / {s['rest_count']} "
                  f"({s['note_ratio']:.0%} / {s['rest_ratio']:.0%})")
            print(f"Range:           {s['pitch_min_note']} – {s['pitch_max_note']} "
                  f"({s['pitch_range_octaves']:.1f} octaves)")
            print(f"Step/Skip/Leap:  {s['step_ratio']:.1%} / {s['skip_ratio']:.1%} / {s['leap_ratio']:.1%}")
            print(f"Detected Key:    {s['detected_key']} (r={s['key_correlation']:.3f})")
            print(f"Unique {ngram_n}-grams:   {s['unique_ngrams']}, "
                  f"Top-{ngram_top_k} coverage: {s['top_coverage']:.1%}")

        return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="Music Analyzer — standalone melody analysis for MIDI and ABC files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python music_analyzer.py --input song.mid
  python music_analyzer.py --input tune.abc --save melody_dashboard.png
  python music_analyzer.py --input song.mid --n 5 --top-k 20 --no-show
        """,
    )
    _parser.add_argument(
        "-i", "--input", required=True,
        help="Path to MIDI (.mid, .midi) or ABC (.abc) file",
    )
    _parser.add_argument(
        "-s", "--save", default=None,
        help="Save dashboard image to path",
    )
    _parser.add_argument(
        "--n", type=int, default=_DEFAULT_NGRAM_N,
        help=f"N-gram length (default: {_DEFAULT_NGRAM_N})",
    )
    _parser.add_argument(
        "--top-k", type=int, default=_DEFAULT_NGRAM_TOP_K,
        help=f"Number of top N-grams to show (default: {_DEFAULT_NGRAM_TOP_K})",
    )
    _parser.add_argument(
        "--window", type=int, default=None,
        help="SSM segment window size (default: auto)",
    )
    _parser.add_argument(
        "--interval-range", type=int, default=_DEFAULT_INTERVAL_RANGE,
        help=f"Max semitone range for interval histogram (default: {_DEFAULT_INTERVAL_RANGE})",
    )
    _parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the plot interactively (useful for headless environments)",
    )
    _args = _parser.parse_args()

    # If --no-show, switch to a non-interactive backend.
    if _args.no_show:
        matplotlib.use("Agg")

    _stats = MelodyAnalyzer.analyze_file(
        _args.input,
        save_path=_args.save,
        n=_args.n,
        top_k=_args.top_k,
        window=_args.window,
        interval_range=_args.interval_range,
    )
