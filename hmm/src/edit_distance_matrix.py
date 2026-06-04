#!/usr/bin/env python3
"""Build edit-distance and RBF affinity matrices for bar token grids."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from Levenshtein import distance as levenshtein_distance

from config_loader import ConfigLoader, ConfigView
from grid_tokenizer import BarGrid, GridTokenizer


@dataclass(frozen=True)
class DistanceMatrixConfig:
    gamma: Optional[float] = None
    gamma_scale: float = 1.0
    normalize_distance: bool = True
    token_offset: int = 2


class EditDistanceMatrixBuilder:
    """Compute token-level Levenshtein distance and RBF affinity matrices."""

    def __init__(self, config: DistanceMatrixConfig) -> None:
        self.config = config

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "EditDistanceMatrixBuilder":
        section = ConfigView(config).section("distance_matrix")
        gamma = section.get("gamma")
        return cls(DistanceMatrixConfig(
            gamma=float(gamma) if gamma is not None else None,
            gamma_scale=float(section.get("gamma_scale", 1.0)),
            normalize_distance=bool(section.get("normalize_distance", True)),
            token_offset=int(section.get("token_offset", 2)),
        ))

    def build_distance(self, bars: Sequence[BarGrid]) -> np.ndarray:
        encoded = [self._encode_tokens(bar.tokens) for bar in bars]
        n_bars = len(encoded)
        matrix = np.zeros((n_bars, n_bars), dtype=np.float64)
        for i in range(n_bars):
            for j in range(i + 1, n_bars):
                value = levenshtein_distance(encoded[i], encoded[j])
                if self.config.normalize_distance:
                    value = value / max(1, max(len(encoded[i]), len(encoded[j])))
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix

    def build_affinity(self, distance_matrix: np.ndarray) -> np.ndarray:
        if distance_matrix.size == 0:
            return distance_matrix.copy()
        gamma = self.config.gamma
        non_zero = distance_matrix[distance_matrix > 0]
        if gamma is None:
            sigma = float(np.median(non_zero)) if non_zero.size else 1.0
            sigma = max(sigma, 1e-6)
            gamma = self.config.gamma_scale / (2.0 * sigma * sigma)
        affinity = np.exp(-float(gamma) * np.square(distance_matrix))
        np.fill_diagonal(affinity, 1.0)
        return affinity

    def save(
        self,
        output_path: str | Path,
        distance_matrix: np.ndarray,
        affinity_matrix: np.ndarray,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            distance=distance_matrix,
            affinity=affinity_matrix,
            config=json.dumps(asdict(self.config)),
        )

    def _encode_tokens(self, tokens: Sequence[int]) -> str:
        values = [int(token) + self.config.token_offset for token in tokens]
        return "".join(chr(value + 1) for value in values)


class EditDistanceMatrixCLI:
    """CLI for distance/affinity matrix generation."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Build edit distance matrix for bar grids.")
        parser.add_argument("--bars", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--gamma", type=float, default=None)
        parser.add_argument("--gamma-scale", type=float, default=None)
        parser.add_argument("--raw-distance", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        builder_config = EditDistanceMatrixBuilder.from_style_config(config).config
        builder_config = DistanceMatrixConfig(
            gamma=args.gamma if args.gamma is not None else builder_config.gamma,
            gamma_scale=args.gamma_scale if args.gamma_scale is not None else builder_config.gamma_scale,
            normalize_distance=False if args.raw_distance else builder_config.normalize_distance,
            token_offset=builder_config.token_offset,
        )
        builder = EditDistanceMatrixBuilder(builder_config)
        bars = GridTokenizer.load_bars_file(args.bars)
        distance = builder.build_distance(bars)
        affinity = builder.build_affinity(distance)
        builder.save(args.output, distance, affinity)
        print(f"Wrote distance and affinity matrices -> {args.output}")


def main() -> None:
    EditDistanceMatrixCLI().run()


if __name__ == "__main__":
    main()
