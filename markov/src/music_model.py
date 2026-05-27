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
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from section_grammar import SectionGrammar

from measure_clustering import (
    MeasureClusterer,
    MeasureExtractor,
    classify_files,
)
from transition_matrix import TransitionMatrix, TransitionMatrixBuilder
from persistence_duration import PersistenceDuration, PersistenceDurationBuilder
from start_distribution import StartDistribution, StartDistributionBuilder

log = logging.getLogger("music_model")

# ---------------------------------------------------------------------------
# MusicModel
# ---------------------------------------------------------------------------


class MusicModel:
    """Complete music analysis model for phrase generation.

    Attributes:
        clusterer: Fitted MeasureClusterer.
        transition_matrix: Row-normalised transition probabilities.
        persistence_duration: Per-cluster run-length distributions.
        start_distribution: Per-cluster start probabilities.
        section_results: Per-file section structure dicts (filename → result dict),
            populated when ``analyze_sections=True`` is passed to :meth:`fit`.
    """

    def __init__(
        self,
        clusterer: MeasureClusterer,
        transition_matrix: TransitionMatrix,
        persistence_duration: PersistenceDuration,
        start_distribution: StartDistribution,
        section_results: Optional[Dict[str, Dict[str, Any]]] = None,
        grammar: Optional["SectionGrammar"] = None,
    ) -> None:
        self.clusterer = clusterer
        self.transition_matrix = transition_matrix
        self.persistence_duration = persistence_duration
        self.start_distribution = start_distribution
        self.section_results = section_results
        self.grammar = grammar

    @property
    def n_clusters(self) -> int:
        return self.transition_matrix.n_clusters

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
        analyze_sections: bool = False,
        section_config: Optional[Dict[str, Any]] = None,
    ) -> "MusicModel":
        """Train a complete model from a directory of music files.

        Extraction, clustering, and classification happen once, then all
        sub-models are built from the same label sequences.

        Args:
            music_dir: Root directory containing music files.
            n_clusters: Number of KMeans clusters.
            seed: Random seed for reproducibility.
            min_run_length: Persistence noise-filter threshold.
            skip_self_transitions: Exclude self-transitions from matrix.
            file_patterns: Glob patterns for music files.
            analyze_sections: If True, run the section miner on each file's
                bar vectors after clustering.  Results are stored in
                ``self.section_results``.
            section_config: Optional dict of SectionMinerConfig overrides
                (e.g. ``{"min_len": 4, "transposition_invariant": True}``).

        Returns:
            A trained MusicModel.
        """
        music_dir = Path(music_dir)
        extractor = MeasureExtractor()

        # 1. Extract per-file vectors (kept for optional section analysis)
        log.info("Extracting measures from %s ...", music_dir)
        file_map = extractor.extract_file_map(music_dir, file_patterns=file_patterns)
        vectors = [v for vecs in file_map.values() for v in vecs]
        log.info("Extracted %d measure vectors from %d files.", len(vectors), len(file_map))

        # 2. Fit KMeans on flat vector list (8-D texture features only)
        clusterer = MeasureClusterer()
        clusterer.fit(vectors, n_clusters=n_clusters, random_seed=seed)
        log.info("Clusterer fitted: k=%d, inertia=%.3f", n_clusters, clusterer.inertia)

        # 3. Classify all files → file_labels
        log.info("Classifying files ...")
        file_labels = classify_files(
            music_dir, clusterer, extractor, file_patterns
        )

        # 3b. Learn per-cluster statistics from note-level data
        log.info("Computing per-cluster pitch histograms ...")
        clusterer.compute_pitch_histograms(file_map, file_labels)
        log.info("Computing per-cluster note statistics ...")
        clusterer.compute_note_statistics(file_map, file_labels)
        log.info("Computing per-cluster bass histograms ...")
        clusterer.compute_bass_histograms(file_map, file_labels)

        # 4. Build sub-models
        log.info("Building transition matrix ...")
        transition_matrix = TransitionMatrixBuilder.build(
            file_labels,
            skip_self_transitions=skip_self_transitions,
        )

        log.info("Building persistence duration ...")
        persistence_duration = PersistenceDurationBuilder.build(
            file_labels,
            min_run_length=min_run_length,
        )

        log.info("Building start distribution ...")
        start_distribution = StartDistributionBuilder.build(file_labels)

        # 5. Optional section structure analysis
        section_results: Optional[Dict[str, Dict[str, Any]]] = None
        if analyze_sections:
            from section_miner import SectionMiner, SectionMinerConfig

            miner_config = SectionMinerConfig(
                **(section_config or {})
            )
            miner = SectionMiner(miner_config)
            section_results = {}
            for filepath, measure_vecs in file_map.items():
                if len(measure_vecs) < miner_config.min_len * 2:
                    continue
                # Feed 20-D full arrays (texture + pitch-class histogram)
                full_arrays = [v.as_full_array().tolist() for v in measure_vecs]
                try:
                    result = miner.analyze(full_arrays)
                    if result.families:
                        section_results[filepath] = result.to_dict()
                except Exception as exc:
                    log.warning("Section miner failed for %s: %s", filepath, exc)
            log.info(
                "Section analysis: %d/%d files have detected sections.",
                len(section_results), len(file_map),
            )

        # 6. Train section grammar from discovered section structure
        grammar: Optional["SectionGrammar"] = None
        if section_results:
            from section_grammar import SectionGrammar

            grammar = SectionGrammar.fit_from_results(
                section_results, clusterer, n_clusters,
            )

        log.info("MusicModel trained: k=%d.", n_clusters)
        return cls(
            clusterer=clusterer,
            transition_matrix=transition_matrix,
            persistence_duration=persistence_duration,
            start_distribution=start_distribution,
            section_results=section_results,
            grammar=grammar,
        )

    # -- persistence -----------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.

        Creates::

            path/
              clusterer.pkl
              transition.npz
              persistence.json
              start_dist.json
              metadata.json
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Clusterer
        self.clusterer.save(path / "clusterer.pkl")

        # Transition matrix
        np.savez(
            path / "transition.npz",
            count=self.transition_matrix.count_matrix,
            prob=self.transition_matrix.prob_matrix,
        )

        # Persistence duration
        rl_serialisable = {
            str(k): v for k, v in self.persistence_duration.run_lengths.items()
        }
        with open(path / "persistence.json", "w") as f:
            json.dump(
                {
                    "run_lengths": rl_serialisable,
                    "file_run_sequences": self.persistence_duration.file_run_sequences,
                },
                f,
            )

        # Start distribution
        with open(path / "start_dist.json", "w") as f:
            json.dump(
                {
                    "start_counts": self.start_distribution.start_counts.tolist(),
                    "total_files": self.start_distribution.total_files,
                },
                f,
            )

        # Metadata
        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "n_clusters": self.n_clusters,
                    "total_transitions": self.transition_matrix.total_transitions,
                    "total_runs": self.persistence_duration.total_runs,
                },
                f,
            )

        # Section structure (optional)
        if self.section_results:
            with open(path / "section_structure.json", "w") as f:
                json.dump(self.section_results, f, indent=2)

        # Section grammar (optional)
        if self.grammar is not None:
            self.grammar.save(path / "grammar")

        log.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MusicModel":
        """Load a model from a directory created by :meth:`save`."""
        path = Path(path)

        # Clusterer
        clusterer = MeasureClusterer.load(path / "clusterer.pkl")
        n = clusterer.centroids.shape[0]  # type: ignore[union-attr]

        # Transition matrix
        tdata = np.load(path / "transition.npz")
        count_matrix = tdata["count"]
        prob_matrix = tdata["prob"]
        total_transitions = int(count_matrix.sum())
        transition_matrix = TransitionMatrix(
            count_matrix=count_matrix,
            prob_matrix=prob_matrix,
            n_clusters=n,
            total_transitions=total_transitions,
        )

        # Persistence duration
        with open(path / "persistence.json") as f:
            rl_data = json.load(f)
        if isinstance(rl_data, dict) and "run_lengths" in rl_data:
            # New format with file_run_sequences
            run_lengths: Dict[int, List[int]] = {
                int(k): v for k, v in rl_data["run_lengths"].items()
            }
            file_run_sequences: List[List[Tuple[int, int]]] = [
                [tuple(pair) for pair in seq]  # type: ignore[misc]
                for seq in rl_data["file_run_sequences"]
            ]
        else:
            # Legacy format (plain run_lengths dict)
            run_lengths = {int(k): v for k, v in rl_data.items()}
            file_run_sequences = []
        persistence_duration = PersistenceDuration(
            run_lengths=run_lengths,
            n_clusters=n,
            file_run_sequences=file_run_sequences,
        )

        # Start distribution
        with open(path / "start_dist.json") as f:
            sd_data = json.load(f)
        start_distribution = StartDistribution(
            start_counts=np.array(sd_data["start_counts"], dtype=np.int64),
            n_clusters=n,
            total_files=sd_data["total_files"],
        )

        # Section structure (optional)
        section_results: Optional[Dict[str, Dict[str, Any]]] = None
        section_path = path / "section_structure.json"
        if section_path.exists():
            with open(section_path) as f:
                section_results = json.load(f)

        # Section grammar (optional)
        grammar: Optional["SectionGrammar"] = None
        grammar_path = path / "grammar" / "grammar.json"
        if grammar_path.exists():
            from section_grammar import SectionGrammar

            grammar = SectionGrammar.load(path / "grammar")

        log.info("Loaded model from %s (k=%d)", path, n)
        return cls(
            clusterer=clusterer,
            transition_matrix=transition_matrix,
            persistence_duration=persistence_duration,
            start_distribution=start_distribution,
            section_results=section_results,
            grammar=grammar,
        )

    # -- summary ---------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "MUSIC MODEL SUMMARY",
            "=" * 65,
            f"  States (clusters):  {self.n_clusters}",
            "",
            "--- Transition Matrix ---",
            str(self.transition_matrix.summary()),
            "",
            "--- Persistence Duration ---",
            str(self.persistence_duration.summary()),
            "",
            "--- Start Distribution ---",
            str(self.start_distribution.summary()),
        ]
        if self.section_results:
            total_families = sum(
                len(result.get("families", []))
                for result in self.section_results.values()
            )
            files_with_sections = len(self.section_results)
            lines.extend([
                "",
                "--- Section Structure ---",
                f"  Files with sections: {files_with_sections}",
                f"  Total section families: {total_families}",
            ])
        if self.grammar is not None and self.grammar.files:
            lines.extend([
                "",
                "--- Section Grammar ---",
                f"  Template files: {len(self.grammar.files)}",
                f"  FREE length range: {min(self.grammar.global_free_lengths)}"
                f" – {max(self.grammar.global_free_lengths)} bars",
            ])
        lines.append("=" * 65)
        return "\n".join(lines)


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
        "--save-model",
        required=True,
        help="Path to save the trained model directory.",
    )
    parser.add_argument(
        "--analyze-sections",
        action="store_true",
        help="Run section structure miner on each file after clustering.",
    )
    parser.add_argument(
        "--section-min-len",
        type=int,
        default=3,
        help="Minimum section length in bars (for --analyze-sections).",
    )
    parser.add_argument(
        "--section-transposition-invariant", "--section-ti",
        action="store_true",
        dest="section_ti",
        help="Enable transposition-invariant section matching.",
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

    section_config = None
    if args.analyze_sections:
        section_config = {
            "min_len": args.section_min_len,
            "transposition_invariant": args.section_ti,
        }

    log.info("Training MusicModel ...")
    model = MusicModel.fit(
        music_dir=args.music_dir,
        n_clusters=args.n_clusters,
        seed=args.seed,
        min_run_length=args.min_run_length,
        skip_self_transitions=not args.include_self_transitions,
        file_patterns=patterns,
        analyze_sections=args.analyze_sections,
        section_config=section_config,
    )

    print()
    print(model.summary())

    log.info("Saving model to %s ...", args.save_model)
    model.save(args.save_model)

    print(f"\nModel saved to {args.save_model}")
    print("Done.")


if __name__ == "__main__":
    main()
