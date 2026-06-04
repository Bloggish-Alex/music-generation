#!/usr/bin/env python3
"""Train the grid/edit-distance/spectral/HMM model as one pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from agglomerative_bar_cluster import AgglomerativeBarClusterConfig, AgglomerativeBarClusterer
from config_loader import ConfigLoader, ConfigView
from edit_distance_matrix import EditDistanceMatrixBuilder
from global_agglomerative_bar_cluster import (
    GlobalAgglomerativeClusterConfig,
    GlobalAgglomerativeClusterer,
)
from grid_tokenizer import GridTokenizer
from hmm_model import HMMMusicModel
from hmmlearn_section_hmm import HmmlearnSectionHMM, HmmlearnSectionHMMConfig
from section_hmm import DiscreteSectionHMM, SectionHMMConfig
from spectral_bar_cluster import SpectralBarClusterConfig, SpectralBarClusterer


class BarClusteringBackend:
    """Base adapter for train-time bar clustering implementations."""

    name = ""

    def fit(
        self,
        config: dict,
        bars: Any,
        distance: Any,
        affinity: Any,
        args: argparse.Namespace,
    ):
        raise NotImplementedError


class SpectralBarClusteringBackend(BarClusteringBackend):
    """Train adapter for spectral clustering."""

    name = "spectral"

    def fit(
        self,
        config: dict,
        bars: Any,
        distance: Any,
        affinity: Any,
        args: argparse.Namespace,
    ):
        cluster_config = SpectralBarClusterer.from_style_config(config).config
        cluster_config = SpectralBarClusterConfig(
            n_clusters=args.n_bar_clusters if args.n_bar_clusters is not None else cluster_config.n_clusters,
            assign_labels=cluster_config.assign_labels,
            random_seed=args.seed if args.seed is not None else cluster_config.random_seed,
        )
        clusterer = SpectralBarClusterer(cluster_config)
        return clusterer, clusterer.fit_predict(affinity)


class AgglomerativeBarClusteringBackend(BarClusteringBackend):
    """Train adapter for dendrogram-threshold clustering."""

    name = "agglomerative"

    def fit(
        self,
        config: dict,
        bars: Any,
        distance: Any,
        affinity: Any,
        args: argparse.Namespace,
    ):
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
        clusterer = AgglomerativeBarClusterer(cluster_config)
        return clusterer, clusterer.fit_predict(distance)


class GlobalAgglomerativeBarClusteringBackend(BarClusteringBackend):
    """Train adapter for per-song clustering aligned by a global codebook."""

    name = "global_agglomerative"

    def fit(
        self,
        config: dict,
        bars: Any,
        distance: Any,
        affinity: Any,
        args: argparse.Namespace,
    ):
        cluster_config = GlobalAgglomerativeClusterer.from_style_config(config).config
        cluster_config = GlobalAgglomerativeClusterConfig(
            song_distance_threshold=(
                args.song_distance_threshold
                if args.song_distance_threshold is not None
                else cluster_config.song_distance_threshold
            ),
            codebook_size=(
                args.codebook_size
                if args.codebook_size is not None
                else cluster_config.codebook_size
            ),
            codebook_linkage=(
                args.codebook_linkage
                if args.codebook_linkage is not None
                else cluster_config.codebook_linkage
            ),
            song_linkage=(
                args.song_linkage
                if args.song_linkage is not None
                else cluster_config.song_linkage
            ),
            criterion=cluster_config.criterion,
            min_clusters_warn=cluster_config.min_clusters_warn,
            max_clusters_warn_fraction=cluster_config.max_clusters_warn_fraction,
        )
        clusterer = GlobalAgglomerativeClusterer(cluster_config)
        return clusterer, clusterer.fit_predict(bars)


class BarClusteringBackendRegistry:
    """Registry/factory so the training flow does not branch by backend."""

    def __init__(self, backends: Sequence[BarClusteringBackend]) -> None:
        self._backends: Dict[str, BarClusteringBackend] = {
            backend.name: backend for backend in backends
        }

    @property
    def choices(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def require(self, name: str) -> BarClusteringBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            choices = ", ".join(self.choices)
            raise ValueError(f"Unknown bar clustering backend '{name}'. Available: {choices}") from exc


BAR_CLUSTERING_BACKENDS = BarClusteringBackendRegistry([
    AgglomerativeBarClusteringBackend(),
    GlobalAgglomerativeBarClusteringBackend(),
    SpectralBarClusteringBackend(),
])


class HMMTrainingCLI:
    """Train all stages and persist an HMM music model."""
    def __init__(self,
                 default_music_dir: str = "",
                 default_model_dir: str = ""):
        self.default_music_dir = default_music_dir
        self.default__model_dir = default_model_dir

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train HMM music model from a MIDI corpus.")
        parser.add_argument("--music-dir", type=Path, default=self.default_music_dir)
        parser.add_argument("--model-dir", type=Path, default=self.default__model_dir)
        parser.add_argument("--work-dir", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--limit-files", type=int, default=None)
        parser.add_argument("--steps-per-bar", type=int, default=None)
        parser.add_argument("--bar-length-ql", type=float, default=None)
        parser.add_argument("--no-input-quantize", action="store_true")
        parser.add_argument(
            "--quantize-divisors",
            default=None,
            help="Comma-separated music21 quarterLengthDivisors, e.g. 4,3 or 8,4,3.",
        )
        parser.add_argument("--n-bar-clusters", type=int, default=None)
        parser.add_argument(
            "--bar-cluster-backend",
            choices=BAR_CLUSTERING_BACKENDS.choices,
            default=None,
            help="Bar clustering implementation. Defaults to bar_clustering.backend in config.",
        )
        parser.add_argument("--distance-threshold", type=float, default=None)
        parser.add_argument("--linkage", default=None)
        parser.add_argument("--codebook-size", type=int, default=None)
        parser.add_argument("--song-distance-threshold", type=float, default=None)
        parser.add_argument("--song-linkage", default=None)
        parser.add_argument("--codebook-linkage", default=None)
        parser.add_argument("--n-sections", type=int, default=None)
        parser.add_argument("--max-iter", type=int, default=None)
        parser.add_argument(
            "--hmm-backend",
            choices=("numpy", "hmmlearn"),
            default=None,
            help="HMM implementation. Defaults to section_hmm.backend in config.",
        )
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--verbose", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        config = ConfigLoader().load(args.config)
        work_dir = args.work_dir or args.model_dir / "training_artifacts"
        work_dir.mkdir(parents=True, exist_ok=True)
        args.model_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = GridTokenizer.from_style_config(config)
        if (
            args.steps_per_bar is not None
            or args.bar_length_ql is not None
            or args.no_input_quantize
            or args.quantize_divisors is not None
        ):
            current = tokenizer.config
            tokenizer = GridTokenizer(type(current)(
                steps_per_bar=args.steps_per_bar if args.steps_per_bar is not None else current.steps_per_bar,
                bar_length_ql=args.bar_length_ql if args.bar_length_ql is not None else current.bar_length_ql,
                rest_token=current.rest_token,
                sustain_token=current.sustain_token,
                polyphonic_strategy=current.polyphonic_strategy,
                quantize_input=False if args.no_input_quantize else current.quantize_input,
                quantize_divisors=(
                    tuple(int(x.strip()) for x in args.quantize_divisors.split(",") if x.strip())
                    if args.quantize_divisors is not None else current.quantize_divisors
                ),
                quantize_offsets=current.quantize_offsets,
                quantize_durations=current.quantize_durations,
                quantize_policy=current.quantize_policy,
                min_overlap_fraction=current.min_overlap_fraction,
            ))
        bars = tokenizer.tokenize_directory(args.music_dir, limit_files=args.limit_files)
        if not bars:
            raise ValueError(f"No bars extracted from {args.music_dir}")
        bars_path = work_dir / "bars.json"
        tokenizer.save_bars(bars, bars_path)
        logging.info("Extracted %d bars", len(bars))

        cluster_backend = (
            args.bar_cluster_backend
            or str(ConfigView(config).section("bar_clustering").get("backend", "agglomerative"))
        )
        distance = None
        affinity = None
        matrix_path = None
        if cluster_backend != "global_agglomerative":
            distance_builder = EditDistanceMatrixBuilder.from_style_config(config)
            distance = distance_builder.build_distance(bars)
            affinity = distance_builder.build_affinity(distance)
            matrix_path = work_dir / "distance_affinity.npz"
            distance_builder.save(matrix_path, distance, affinity)
        clusterer, labels = self._fit_bar_clusterer(
            cluster_backend,
            config,
            bars,
            distance,
            affinity,
            args,
        )
        labels_path = work_dir / "bar_labels.json"
        clusterer.save_labels(labels_path, bars, labels)
        diagnostics_path = work_dir / "bar_clustering_diagnostics.json"
        if hasattr(clusterer, "save_diagnostics"):
            clusterer.save_diagnostics(diagnostics_path)

        sequences = self._label_sequences(labels_path)
        backend = args.hmm_backend or str(ConfigView(config).section("section_hmm").get("backend", "numpy"))
        hmm = self._fit_hmm(backend, config, sequences, args)
        model = HMMMusicModel.from_training(config, bars, labels, hmm)
        model.save(args.model_dir)

        summary = {
            "music_dir": str(args.music_dir),
            "model_dir": str(args.model_dir),
            "work_dir": str(work_dir),
            "bars": str(bars_path),
            "matrix": str(matrix_path) if matrix_path is not None else None,
            "labels": str(labels_path),
            "bar_clustering_diagnostics": str(diagnostics_path),
            "n_bars": len(bars),
            "bar_cluster_backend": cluster_backend,
            "n_bar_clusters": int(max(labels) + 1),
            "n_sections": hmm.config.n_sections,
            "hmm_backend": backend,
            "hmm_training_log": hmm.training_log,
        }
        (args.model_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(f"Trained HMM model -> {args.model_dir}")
        print(f"Artifacts -> {work_dir}")

    def _label_sequences(self, labels_path: Path) -> list[list[int]]:
        from section_hmm import _sequences_from_label_payload

        return _sequences_from_label_payload(json.loads(labels_path.read_text(encoding="utf-8")))

    def _fit_bar_clusterer(
        self,
        backend: str,
        config: dict,
        bars: Any,
        distance: Any,
        affinity: Any,
        args: argparse.Namespace,
    ):
        return BAR_CLUSTERING_BACKENDS.require(backend).fit(
            config=config,
            bars=bars,
            distance=distance,
            affinity=affinity,
            args=args,
        )

    def _fit_hmm(
        self,
        backend: str,
        config: dict,
        sequences: list[list[int]],
        args: argparse.Namespace,
    ):
        if backend == "hmmlearn":
            hmm_config = HmmlearnSectionHMM.from_style_config(config).config
            hmm_config = HmmlearnSectionHMMConfig(
                n_sections=args.n_sections if args.n_sections is not None else hmm_config.n_sections,
                max_iter=args.max_iter if args.max_iter is not None else hmm_config.max_iter,
                tol=hmm_config.tol,
                self_transition_bias=hmm_config.self_transition_bias,
                random_seed=args.seed if args.seed is not None else hmm_config.random_seed,
                algorithm=hmm_config.algorithm,
                implementation=hmm_config.implementation,
            )
            return HmmlearnSectionHMM(hmm_config).fit(sequences)

        hmm_config = DiscreteSectionHMM.from_style_config(config).config
        hmm_config = SectionHMMConfig(
            n_sections=args.n_sections if args.n_sections is not None else hmm_config.n_sections,
            max_iter=args.max_iter if args.max_iter is not None else hmm_config.max_iter,
            tol=hmm_config.tol,
            self_transition_bias=hmm_config.self_transition_bias,
            emission_smoothing=hmm_config.emission_smoothing,
            transition_smoothing=hmm_config.transition_smoothing,
            random_seed=args.seed if args.seed is not None else hmm_config.random_seed,
        )
        return DiscreteSectionHMM(hmm_config).fit(sequences)


def main() -> None:
    HMMTrainingCLI("../../datasets/test", default_model_dir="../models/test", ).run()


if __name__ == "__main__":
    main()
