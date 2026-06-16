#!/usr/bin/env python3
"""Generation CLI entrypoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from common.config_loader import ConfigLoader, ConfigView
from common.model_store import ModelBundle
from decoder.form_generator import FormDrivenGenerator
from renderer.generation_output import GenerationOutputWriter
from renderer.harmonic_engine import HarmonicCliOverrides


class GenerationCLI:
    """CLI adapter for form-driven generation."""

    GENERATION_CONFIG_SECTIONS = (
        "decoder",
        "temporal_graph_templates",
        "harmonic_engine",
        "harmonic_render",
        "midi_render",
        "candidate_selector",
    )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Generate JSON/MIDI from a trained model.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--form", default="ternary")
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument(
            "--decoder-backend",
            choices=["lstm_token", "lstm_rerank", "temporal_lstm"],
            default=None,
        )
        HarmonicCliOverrides.add_arguments(parser)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        bundle = ModelBundle.load(args.model_dir)
        config = self._generation_config(bundle.config, ConfigLoader().load(args.config))
        config = self._decoder_cli_overrides(config, args)
        config = HarmonicCliOverrides.from_args(args).apply(config)

        generator = FormDrivenGenerator(bundle, config)
        generation = generator.generate(args.form, seed=args.seed)
        GenerationOutputWriter(bundle, config, generator.diagnostics).write(
            generation,
            args.output_json,
            args.output_midi,
        )

        diagnostics_path = args.diagnostics_output or args.output_json.with_suffix(".generation_diagnostics.json")
        generator.diagnostics.write(diagnostics_path)
        print(f"Wrote JSON -> {args.output_json}")
        print(f"Wrote MIDI -> {args.output_midi}")
        print(f"Diagnostics -> {diagnostics_path}")

    def _generation_config(self, model_config: Dict[str, Any], style_config: Dict[str, Any]) -> Dict[str, Any]:
        config = deepcopy(model_config)
        for section in self.GENERATION_CONFIG_SECTIONS:
            if section in style_config:
                config[section] = deepcopy(style_config[section])
        return config

    def _decoder_cli_overrides(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        result = deepcopy(config)
        decoder = dict(ConfigView(result).section("decoder"))
        if args.decoder_backend is not None:
            decoder["backend"] = str(args.decoder_backend)
        if str(decoder.get("backend", "temporal_lstm")) in {"lstm_token", "lstm_rerank", "temporal_lstm"}:
            decoder["model_dir"] = str(args.model_dir)
        result["decoder"] = decoder
        return result


def main() -> None:
    GenerationCLI().run()


if __name__ == "__main__":
    main()
