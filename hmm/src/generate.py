#!/usr/bin/env python3
"""Generate JSON and MIDI from a trained HMM music model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from config_loader import ConfigLoader, ConfigView
from hmm_model import HMMGenerator, HMMMusicModel, MidiRenderConfig


class HMMGenerationCLI:
    """Forward-sample an HMM model and render prototype bars."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Generate structure JSON and MIDI from HMM model.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--measures", type=int, default=32)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--tempo", type=int, default=None)
        parser.add_argument("--velocity", type=int, default=None)
        parser.add_argument("--time-signature", default=None)
        parser.add_argument("--transition-temperature", type=float, default=None)
        parser.add_argument("--emission-temperature", type=float, default=None)
        parser.add_argument("--max-same-label-run", type=int, default=None)
        parser.add_argument("--max-same-state-run", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        model = HMMMusicModel.load(args.model_dir)
        config = model.config
        if args.config:
            config = ConfigLoader().load(args.config)
        model.config = config
        self._apply_generation_overrides(model.config, args)
        render_config = self._render_config(config)
        render_config = MidiRenderConfig(
            tempo=args.tempo if args.tempo is not None else render_config.tempo,
            time_signature_num=(
                self._parse_time_signature(args.time_signature)[0]
                if args.time_signature else render_config.time_signature_num
            ),
            time_signature_den=(
                self._parse_time_signature(args.time_signature)[1]
                if args.time_signature else render_config.time_signature_den
            ),
            velocity=args.velocity if args.velocity is not None else render_config.velocity,
            channel=render_config.channel,
        )
        generator = HMMGenerator(model)
        generation = generator.generate(args.measures, seed=args.seed)
        generation["render_config"] = asdict(render_config)
        generator.write_json(generation, args.output_json)
        generator.write_midi(generation, args.output_midi, render_config)
        print(f"Wrote JSON -> {args.output_json}")
        print(f"Wrote MIDI -> {args.output_midi}")

    def _apply_generation_overrides(self, config: dict, args: argparse.Namespace) -> None:
        section = config.setdefault("hmm_generation", {})
        if args.transition_temperature is not None:
            section["transition_temperature"] = args.transition_temperature
        if args.emission_temperature is not None:
            section["emission_temperature"] = args.emission_temperature
        if args.max_same_label_run is not None:
            section["max_same_label_run"] = args.max_same_label_run
        if args.max_same_state_run is not None:
            section["max_same_state_run"] = args.max_same_state_run

    def _render_config(self, config: dict) -> MidiRenderConfig:
        section = ConfigView(config).section("midi_render")
        ts = self._parse_time_signature(str(section.get("time_signature", "4/4")))
        return MidiRenderConfig(
            tempo=int(section.get("tempo", 120)),
            time_signature_num=ts[0],
            time_signature_den=ts[1],
            velocity=int(section.get("velocity", 72)),
            channel=int(section.get("channel", 0)),
        )

    def _parse_time_signature(self, value: str) -> tuple[int, int]:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid time signature: {value}")
        return int(parts[0]), int(parts[1])


def main() -> None:
    HMMGenerationCLI().run()


if __name__ == "__main__":
    main()
