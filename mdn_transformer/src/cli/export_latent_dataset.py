#!/usr/bin/env python3
"""CLI entrypoint for exporting DVAE latent datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.latent_dataset_exporter import LatentDatasetExporter, LatentExportConfig


class LatentExportCLI:
    """Command-line adapter for latent dataset export."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Export a trained DVAE latent dataset.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--save-z", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run latent export from CLI arguments."""
        args = self.build_parser().parse_args(argv)
        model_dir = Path(args.model_dir)
        output_dir = Path(args.output_dir) if args.output_dir else model_dir / "latent"
        config = LatentExportConfig(
            batch_size=int(args.batch_size),
            device=str(args.device),
            save_z=bool(args.save_z),
        )
        summary = LatentDatasetExporter(config).export_from_model_dir(model_dir, output_dir)
        print(f"Latent dataset -> {output_dir}")
        print(f"Samples: {summary['sample_count']}")
        print(f"Latent dim: {summary['latent_dim']}")


def main() -> None:
    """Run latent export CLI."""
    LatentExportCLI().run()


if __name__ == "__main__":
    main()
