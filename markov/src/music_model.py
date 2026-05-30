#!/usr/bin/env python3
"""
Music Model — unified model for phrase generation.

Trains a complete model from music files in a single pass: clusters measure
vectors, classifies all files, and builds the transition matrix, persistence
duration distribution, and start distribution.

Usage (library)::

    from music_model import MusicModel

    model = MusicModel.fit("path/to/music/dir", n_clusters=8)
    model.save("./my_model")
    ...
    model = MusicModel.load("./my_model")
    print(model.summary())

Usage (CLI — production training)::

    python music_model.py --music-dir ../../datasets/corelli --n-clusters 5 \
        --save-model ./models/corelli_k5
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from measure_clustering import (
    ClusterVisualizer,
    MeasureClusterer,
    MeasureExtractor,
    MeasureInfo,
    MeasureVector,
)
from transition_matrix import TransitionMatrix, TransitionMatrixBuilder
from persistence_duration import PersistenceDuration, PersistenceDurationBuilder
from start_distribution import StartDistribution, StartDistributionBuilder

log = logging.getLogger("music_model")

# ---------------------------------------------------------------------------
# MusicModel
# ---------------------------------------------------------------------------


@dataclass
class TsModel:
    """Sub-model for a single time signature."""

    time_signature: str
    clusterer: MeasureClusterer
    transition_matrix: TransitionMatrix
    persistence_duration: PersistenceDuration
    start_distribution: StartDistribution
    n_clusters: int


class MusicModel:
    """Complete music analysis model for phrase generation.

    Contains one :class:`TsModel` per time signature (e.g. "4/4", "3/4").
    Generation picks the right sub-model via :meth:`model_for`.
    """

    def __init__(self, ts_models: Dict[str, TsModel]) -> None:
        self._ts_models = ts_models

    @property
    def time_signatures(self) -> List[str]:
        return sorted(self._ts_models.keys())

    def model_for(self, time_signature: str) -> TsModel:
        """Return the sub-model for *time_signature*.

        Raises KeyError if the time signature was not seen during training.
        """
        if time_signature not in self._ts_models:
            raise KeyError(
                f"Time signature {time_signature!r} not in model. "
                f"Available: {self.time_signatures}"
            )
        return self._ts_models[time_signature]

    # -- factory ---------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        music_dir: Union[str, Path],
        n_clusters: int = 8,
        seed: int = 42,
        min_run_length: int = 0,
        skip_self_transitions: bool = True,
        file_patterns: Optional[Union[str, Sequence[str]]] = None,
        feature_weights: Optional[Sequence[float]] = None,
        min_measures: int = 20,
    ) -> "MusicModel":
        """Train a complete model from a directory of music files.

        Measures are grouped by time signature; each group gets its own
        clusterer, transition matrix, persistence, and start distribution.

        Args:
            music_dir: Root directory containing music files.
            n_clusters: Number of KMeans clusters (per time signature).
            seed: Random seed for reproducibility.
            min_run_length: Persistence noise-filter threshold.
            skip_self_transitions: Exclude self-transitions from matrix.
            file_patterns: Glob patterns for music files.
            feature_weights: Per-feature multiplier for KMeans.
            min_measures: Skip time signatures with fewer measures.

        Returns:
            A trained MusicModel.
        """
        music_dir = Path(music_dir)
        extractor = MeasureExtractor()

        if file_patterns is None:
            patterns = ["*.mid", "*.midi", "*.abc", "*.krn"]
        elif isinstance(file_patterns, str):
            patterns = [file_patterns]
        else:
            patterns = list(file_patterns)

        file_paths: List[Path] = []
        for pat in patterns:
            file_paths.extend(sorted(music_dir.rglob(pat)))
        file_paths = sorted(set(file_paths))

        # 1. Extract vectors + MeasureInfo in a single pass, grouped by ts
        log.info("Extracting measures from %s ...", music_dir)
        ts_vectors: Dict[str, List[MeasureVector]] = {}
        ts_measures: Dict[str, List[MeasureInfo]] = {}
        total = 0
        for fp in file_paths:
            try:
                measures = extractor.extract(fp)
                for m in measures:
                    v = extractor.vectorize(m)
                    ts_vectors.setdefault(v.time_signature, []).append(v)
                    ts_measures.setdefault(v.time_signature, []).append(m)
                    total += 1
            except Exception as exc:
                log.warning("Skipping %s: %s", fp, exc)
        log.info("Extracted %d measure vectors across %d time signatures.",
                 total, len(ts_vectors))
        log.info("Time signatures: %s",
                 ", ".join(f"{ts}({len(g)})" for ts, g in sorted(ts_vectors.items())))

        # 2. Train one TsModel per time signature
        ts_models: Dict[str, TsModel] = {}
        for ts in sorted(ts_vectors):
            vectors = ts_vectors[ts]
            measures = ts_measures[ts]
            if len(vectors) < min_measures:
                log.warning(
                    "Skipping %s: only %d measures (< %d).",
                    ts, len(vectors), min_measures,
                )
                continue

            log.info("--- Training sub-model for %s (%d measures) ---", ts, len(vectors))

            clusterer = MeasureClusterer()
            clusterer.fit(
                vectors, n_clusters=n_clusters, random_seed=seed,
                feature_weights=None,
            )
            log.info("Clusterer fitted: k=%d, inertia=%.3f", n_clusters, clusterer.inertia)

            labels = clusterer.predict_many(vectors)
            clusterer.store_measures(measures, labels)

            # Flat sequence: treat all same-ts measures as one virtual file
            # so transitions can form across source files.
            file_labels = [labels.tolist()]
            transition_matrix = TransitionMatrixBuilder.build(
                file_labels, skip_self_transitions=skip_self_transitions,
            )
            persistence_duration = PersistenceDurationBuilder.build(
                file_labels, min_run_length=min_run_length,
            )
            start_distribution = StartDistributionBuilder.build(file_labels)

            ts_models[ts] = TsModel(
                time_signature=ts,
                clusterer=clusterer,
                transition_matrix=transition_matrix,
                persistence_duration=persistence_duration,
                start_distribution=start_distribution,
                n_clusters=n_clusters,
            )

        if not ts_models:
            raise RuntimeError(
                f"No time signature had >= {min_measures} measures. "
                f"Available: { {ts: len(g) for ts, g in ts_vectors.items()} }"
            )

        log.info("MusicModel trained: %d time signature(s).", len(ts_models))
        return cls(ts_models=ts_models)

    # -- persistence -----------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.

        Creates::

            path/
              ts_models.json
              4_4/
                clusterer.pkl  transition.npz  persistence.json  start_dist.json  metadata.json
              3_4/
                ...
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        ts_list: List[str] = []
        for ts, tm in self._ts_models.items():
            ts_dir = path / ts.replace("/", "_")
            ts_dir.mkdir(parents=True, exist_ok=True)

            tm.clusterer.save(ts_dir / "clusterer.pkl")

            np.savez(
                ts_dir / "transition.npz",
                count=tm.transition_matrix.count_matrix,
                prob=tm.transition_matrix.prob_matrix,
            )

            rl_serialisable = {
                str(k): v for k, v in tm.persistence_duration.run_lengths.items()
            }
            with open(ts_dir / "persistence.json", "w") as f:
                json.dump(
                    {
                        "run_lengths": rl_serialisable,
                        "file_run_sequences": tm.persistence_duration.file_run_sequences,
                    },
                    f,
                )

            with open(ts_dir / "start_dist.json", "w") as f:
                json.dump(
                    {
                        "start_counts": tm.start_distribution.start_counts.tolist(),
                        "total_files": tm.start_distribution.total_files,
                    },
                    f,
                )

            with open(ts_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "n_clusters": tm.n_clusters,
                        "total_transitions": tm.transition_matrix.total_transitions,
                        "total_runs": tm.persistence_duration.total_runs,
                    },
                    f,
                )

            ts_list.append(ts)
            log.info("Saved sub-model for %s → %s", ts, ts_dir)

        with open(path / "ts_models.json", "w") as f:
            json.dump({"time_signatures": ts_list}, f)

        log.info("Saved model to %s (%d time signatures)", path, len(ts_list))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MusicModel":
        """Load a model from a directory created by :meth:`save`."""
        path = Path(path)

        with open(path / "ts_models.json") as f:
            ts_data = json.load(f)

        ts_models: Dict[str, TsModel] = {}
        for ts in ts_data["time_signatures"]:
            ts_dir = path / ts.replace("/", "_")
            clusterer = MeasureClusterer.load(ts_dir / "clusterer.pkl")
            n = clusterer.centroids.shape[0]  # type: ignore[union-attr]

            tdata = np.load(ts_dir / "transition.npz")
            transition_matrix = TransitionMatrix(
                count_matrix=tdata["count"],
                prob_matrix=tdata["prob"],
                n_clusters=n,
                total_transitions=int(tdata["count"].sum()),
            )

            with open(ts_dir / "persistence.json") as f:
                rl_data = json.load(f)
            if isinstance(rl_data, dict) and "run_lengths" in rl_data:
                run_lengths = {int(k): v for k, v in rl_data["run_lengths"].items()}
                file_run_sequences = [
                    [tuple(pair) for pair in seq]
                    for seq in rl_data["file_run_sequences"]
                ]
            else:
                run_lengths = {int(k): v for k, v in rl_data.items()}
                file_run_sequences = []
            persistence_duration = PersistenceDuration(
                run_lengths=run_lengths, n_clusters=n,
                file_run_sequences=file_run_sequences,
            )

            with open(ts_dir / "start_dist.json") as f:
                sd_data = json.load(f)
            start_distribution = StartDistribution(
                start_counts=np.array(sd_data["start_counts"], dtype=np.int64),
                n_clusters=n,
                total_files=sd_data["total_files"],
            )

            ts_models[ts] = TsModel(
                time_signature=ts,
                clusterer=clusterer,
                transition_matrix=transition_matrix,
                persistence_duration=persistence_duration,
                start_distribution=start_distribution,
                n_clusters=n,
            )
            log.info("Loaded sub-model for %s (k=%d)", ts, n)

        log.info("Loaded model from %s (%d time signatures)", path, len(ts_models))
        return cls(ts_models=ts_models)

    # -- summary ---------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "MUSIC MODEL SUMMARY",
            "=" * 65,
            f"  Time signatures: {', '.join(self.time_signatures)}",
            "",
        ]
        for ts in self.time_signatures:
            tm = self._ts_models[ts]
            lines.append(f"--- {ts} ({tm.n_clusters} clusters) ---")
            lines.append(str(tm.transition_matrix.summary()))
            lines.append("")
            lines.append(str(tm.persistence_duration.summary()))
            lines.append("")
            lines.append(str(tm.start_distribution.summary()))
            lines.append("")
            lines.append("-" * 65)
            tm.clusterer.print_sample_measures()
        lines.append("=" * 65)
        return "\n".join(lines)

    # -- visualization ----------------------------------------------------------

    def visualize(self, output_dir: Union[str, Path]) -> None:
        """Save cluster diagnostic plots for each time signature.

        Creates ``{output_dir}/{ts}_summary.png``, ``{ts}_silhouette.png``, etc.
        """
        output_dir = Path(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        for ts, tm in self._ts_models.items():
            # Gather vectors from stored measures
            extractor = MeasureExtractor()
            all_measures = [
                m for measures in tm.clusterer._cluster_measures.values()
                for m in measures
            ]
            if not all_measures:
                log.warning("No stored measures for %s, skipping viz.", ts)
                continue
            vectors = [extractor.vectorize(m) for m in all_measures]

            viz = ClusterVisualizer(vectors, tm.clusterer)
            safe_ts = ts.replace("/", "_")
            viz.plot_all(str(output_dir / safe_ts))
            log.info("Visualizations for %s saved to %s", ts, output_dir / safe_ts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse
    from _argparse_utils import (
        add_clustering_args,
        add_music_source_args,
        add_verbose_arg,
    )

    parser = argparse.ArgumentParser(
        description="Music Model — train and save a complete music model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_music_source_args(parser)
    add_clustering_args(parser)
    add_verbose_arg(parser)
    parser.add_argument(
        "--min-run-length",
        type=int,
        default=0,
        help="Persistence noise-filter threshold.",
    )
    parser.add_argument(
        "--include-self-transitions",
        action="store_true",
        help="Include A->A self-transitions (default: skip).",
    )
    parser.add_argument(
        "--feature-weights",
        default="1,1,1,1,1,1,1,1,0.5,10,1,1,1,1,1,1,1,1,1,1",
        help="Comma-separated per-feature weights for KMeans, e.g. "
        "'1,1,1,1,1,1,1,1,0.15' to dampen relative_position. "
        "Default: equal weight for all features.",
    )
    parser.add_argument(
        "--min-measures",
        type=int,
        default=20,
        help="Skip time signatures with fewer measures than this threshold.",
    )
    parser.add_argument(
        "--viz",
        default=True,
        metavar="DIR",
        help="Save cluster diagnostic plots to DIR (one subdir per time signature).",
    )
    parser.add_argument(
        "--save-model",
        default="../../models/test",
        help="Path to save the trained model directory.",
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

    feature_weights = None
    if args.feature_weights:
        feature_weights = [
            float(x.strip()) for x in args.feature_weights.split(",")
        ]

    log.info("Training MusicModel ...")
    model = MusicModel.fit(
        music_dir=args.music_dir,
        n_clusters=args.n_clusters,
        seed=args.seed,
        min_run_length=args.min_run_length,
        skip_self_transitions=not args.include_self_transitions,
        file_patterns=patterns,
        feature_weights=feature_weights,
        min_measures=args.min_measures,
    )

    print()
    print(model.summary())

    if args.viz:
        log.info("Generating cluster visualizations in %s ...", args.viz)
        model.visualize(args.viz)

    log.info("Saving model to %s ...", args.save_model)
    model.save(args.save_model)

    print(f"\nModel saved to {args.save_model}")
    print("Done.")


if __name__ == "__main__":
    main()
