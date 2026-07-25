#!/usr/bin/env python3
"""CLI entrypoint for offline DVAE reconstruction analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.dvae_analysis import DVAEAnalysisConfig, DVAEAnalysisWriter, DVAEArtifactLoader, DVAEReconstructionAnalyzer


class DVAEAnalysisCLI:
    """Command-line adapter for DVAE reconstruction analysis."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Analyze a trained DVAE reconstruction quality.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--sample-count", type=int, default=32)
        parser.add_argument("--seed", type=int, default=42)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run analysis from CLI arguments."""
        args = self.build_parser().parse_args(argv)
        model_dir = Path(args.model_dir)
        output_dir = Path(args.output_dir) if args.output_dir else model_dir / "analysis"
        loader = DVAEArtifactLoader()
        model = loader.load_model(model_dir / "dvae.pt", args.device)
        tensors, keys = loader.load_tensor_array(model_dir / "encoded" / "bar_tensors.npz")
        index = loader.load_index(model_dir / "encoded" / "bar_tensor_index.json")
        action_map = loader.load_action_map(model_dir / "encoded" / "songs.json")
        config = DVAEAnalysisConfig(
            batch_size=int(args.batch_size),
            device=str(args.device),
            sample_count=int(args.sample_count),
            random_seed=int(args.seed),
        )
        analyzer = DVAEReconstructionAnalyzer(config)
        report = analyzer.analyze(model, tensors, keys, index, action_map)
        samples = analyzer.sample_reconstructions(model, tensors, keys)
        outputs = DVAEAnalysisWriter().write(output_dir, report, samples)
        for name, path in outputs.items():
            if path:
                print(f"{name}: {path}")


def main() -> None:
    """Run DVAE analysis CLI."""
    DVAEAnalysisCLI().run()


if __name__ == "__main__":
    main()
