#!/usr/bin/env python3
"""CLI entrypoint for rendering DVAE reconstruction samples to MIDI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.dvae_midi_render import DVAEMidiRenderConfig, DVAESampleMidiBatchRenderer


class DVAESampleRenderCLI:
    """Command-line adapter for DVAE sample MIDI rendering."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Render DVAE target/reconstruction sample tensors to MIDI.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--samples", type=Path, default=None)
        parser.add_argument("--index", type=Path, default=None)
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--tempo", type=int, default=120)
        parser.add_argument("--base-pitch", type=int, default=60)
        parser.add_argument("--pitch-scale", type=float, default=24.0)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Render samples according to CLI arguments."""
        args = self.build_parser().parse_args(argv)
        model_dir = Path(args.model_dir)
        samples = args.samples or model_dir / "analysis" / "dvae_reconstruction_samples.npz"
        index = args.index or model_dir / "encoded" / "bar_tensor_index.json"
        output_dir = args.output_dir or model_dir / "analysis" / "midi_samples"
        config = DVAEMidiRenderConfig(
            tempo_bpm=int(args.tempo),
            default_base_pitch=int(args.base_pitch),
            pitch_scale=float(args.pitch_scale),
        )
        report = DVAESampleMidiBatchRenderer(config).render(samples, index, output_dir)
        print(f"Rendered MIDI files: {report['rendered_count']}")
        print(f"Output -> {output_dir}")


def main() -> None:
    """Run sample MIDI render CLI."""
    DVAESampleRenderCLI().run()


if __name__ == "__main__":
    main()
