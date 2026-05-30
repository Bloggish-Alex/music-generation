#!/usr/bin/env python3
"""Shared argparse argument groups used across module CLIs."""

from __future__ import annotations


def add_music_source_args(parser: "argparse.ArgumentParser") -> None:
    """Add --music-dir and --file-patterns to *parser*."""
    parser.add_argument(
        "--music-dir",
        default="../../datasets/test",
        help="Directory containing music files (recursive).",
    )
    parser.add_argument(
        "--file-patterns",
        default="*.mid,*.midi,*.abc,*.krn",
        help="Comma-separated glob patterns for music files.",
    )


def add_clustering_args(parser: "argparse.ArgumentParser") -> None:
    """Add --n-clusters and --seed to *parser*."""
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=8,
        help="Number of KMeans clusters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )


def add_model_io_args(parser: "argparse.ArgumentParser") -> None:
    """Add --load-model and --save-model (both optional) to *parser*.

    These save/load a MeasureClusterer .pkl file, not the full MusicModel.
    """
    parser.add_argument(
        "--load-model",
        default=None,
        help="Load a pre-trained MeasureClusterer from a .pkl file "
        "(skips clustering).",
    )
    parser.add_argument(
        "--save-model",
        default="../../models/corelli",
        help="Save the fitted clusterer to a .pkl file.",
    )


def add_verbose_arg(parser: "argparse.ArgumentParser") -> None:
    """Add --verbose (store_true) to *parser*."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
