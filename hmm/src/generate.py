#!/usr/bin/env python3
"""DFA-driven generation CLI for trained form HMM models."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigLoader
from core_data import BarRecord
from diagnostics import GenerationDiagnostics
from generation_data import GenerationResult, SampledBar, SectionPlanItem
from harmonic_engine import HarmonicCliOverrides, HarmonicEngine, HarmonicMidiRenderer
from model_store import ModelBundle


class SectionPlanBuilder:
    """Build a deterministic section plan from a trained form template."""

    def build(self, bundle: ModelBundle, form_name: str) -> List[SectionPlanItem]:
        template = bundle.form_templates.get(form_name, {})
        sections = template.get("sections", [])
        if not sections:
            model = bundle.form_models[form_name]
            sections = [
                {
                    "name": model.state_role_map.get(index, f"State_{index}"),
                    "length": model.section_lengths[index] if index < len(model.section_lengths) else 1,
                    "source": None,
                }
                for index in range(model.n_states)
            ]
        plan = []
        for state_id, section in enumerate(sections):
            plan.append(SectionPlanItem(
                state_id=int(state_id),
                name=str(section.get("name", f"State_{state_id}")),
                bars=int(section.get("length", 1)),
                source=section.get("source"),
                pitch_offset=int(section.get("pitch_offset", 0) or 0),
                cadence=str(section.get("cadence", "none")),
                start_degree=section.get("start_degree"),
            ))
        return plan


class FormDrivenGenerator:
    """Generate bar IDs and concrete bars from a fixed DFA section plan."""

    def __init__(self, bundle: ModelBundle, config: Optional[dict] = None) -> None:
        self.bundle = bundle
        self.config = config or bundle.config
        self.diagnostics = GenerationDiagnostics()

    def generate(self, form_name: str, seed: Optional[int] = None) -> GenerationResult:
        if form_name not in self.bundle.form_models:
            raise ValueError(f"Model does not contain form '{form_name}'.")
        rng = np.random.default_rng(seed)
        model = self.bundle.form_models[form_name]
        plan = SectionPlanBuilder().build(self.bundle, form_name)
        self.diagnostics.record_section_plan([section.to_dict() for section in plan])
        sampled = []
        for section in plan:
            state_id = int(section.state_id)
            for local_index in range(int(section.bars)):
                observation_id, emission_prob = model.sample_from_state(state_id, rng)
                selected_bar = self._sample_bar(observation_id, rng)
                composite = self.bundle.observation_vocab.composite_for(observation_id)
                event = SampledBar(
                    output_bar_index=len(sampled),
                    section=section.name,
                    section_local_index=local_index,
                    hidden_state=state_id,
                    observation_id=int(observation_id),
                    composite_key=composite,
                    emission_probability=emission_prob,
                    source_file=selected_bar.file_path,
                    source_bar_index=selected_bar.bar_index,
                    edit_distance_id=int(selected_bar.edit_distance_id),
                    kmeans_id=selected_bar.kmeans_id,
                    absolute_tokens=list(selected_bar.absolute_tokens),
                    relative_tokens=list(selected_bar.relative_tokens),
                )
                sampled.append(event)
                self.diagnostics.record_sampled_bar(event.to_dict())
        return GenerationResult(
            form=form_name,
            seed=seed,
            section_plan=plan,
            sampled_bars=sampled,
        )

    def write_outputs(self, generation: GenerationResult, json_path: Path, midi_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        engine = HarmonicEngine(self.config, self.bundle.edit_distance_codebook)
        realized = engine.realize(generation)
        render_diag = HarmonicMidiRenderer.from_style_config(self.config).write(
            realized.harmonic_bars,
            midi_path,
        )
        engine.diagnostics["render"] = render_diag
        json_path.write_text(json.dumps(realized.to_dict(), indent=2), encoding="utf-8")
        self.diagnostics.record_stage("harmonic_engine", engine.diagnostics)
        for event in engine.diagnostics.get("rare_bar_selection", {}).get("events", []):
            self.diagnostics.record_rare_bar_selection(event)

    def _sample_bar(self, observation_id: int, rng: np.random.Generator) -> BarRecord:
        pool = self.bundle.observation_to_bars.get(int(observation_id), [])
        if not pool:
            raise ValueError(f"No bar pool for observation_id {observation_id}")
        return pool[int(rng.integers(0, len(pool)))]


class GenerationCLI:
    """CLI for form-driven generation."""

    GENERATION_CONFIG_SECTIONS = (
        "harmonic_engine",
        "harmonic_render",
        "midi_render",
        "hmm_generation",
    )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Generate JSON/MIDI from a trained DFA/HMM model.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--form", default="ternary")
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--seed", type=int, default=None)
        HarmonicCliOverrides.add_arguments(parser)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        bundle = ModelBundle.load(args.model_dir)
        config = self._generation_config(bundle.config, ConfigLoader().load(args.config))
        config = HarmonicCliOverrides.from_args(args).apply(config)
        generator = FormDrivenGenerator(bundle, config)
        generation = generator.generate(args.form, seed=args.seed)
        generator.write_outputs(generation, args.output_json, args.output_midi)
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


def main() -> None:
    GenerationCLI().run()


if __name__ == "__main__":
    main()
