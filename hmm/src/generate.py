#!/usr/bin/env python3
"""DFA-driven generation CLI for trained form HMM models."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigLoader, ConfigView
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
        sampled_by_section: Dict[str, List[SampledBar]] = {}
        for section in plan:
            state_id = int(section.state_id)
            section_events: List[SampledBar] = []
            for local_index in range(int(section.bars)):
                event = self._source_reuse_event(
                    model,
                    section,
                    local_index,
                    len(sampled),
                    sampled_by_section,
                    rng,
                )
                if event is None:
                    event = self._sample_event(model, state_id, section, local_index, len(sampled), rng)
                sampled.append(event)
                section_events.append(event)
                self.diagnostics.record_sampled_bar(event.to_dict())
            sampled_by_section[section.name] = section_events
        return GenerationResult(
            form=form_name,
            seed=seed,
            section_plan=plan,
            sampled_bars=sampled,
        )

    def _sample_event(
        self,
        model: Any,
        state_id: int,
        section: SectionPlanItem,
        local_index: int,
        output_bar_index: int,
        rng: np.random.Generator,
    ) -> SampledBar:
        observation_id, emission_prob = model.sample_from_state(state_id, rng)
        selected_bar = self._sample_bar(observation_id, rng)
        composite = self.bundle.observation_vocab.composite_for(observation_id)
        return SampledBar(
            output_bar_index=output_bar_index,
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

    def _source_reuse_event(
        self,
        model: Any,
        section: SectionPlanItem,
        local_index: int,
        output_bar_index: int,
        sampled_by_section: Dict[str, List[SampledBar]],
        rng: np.random.Generator,
    ) -> Optional[SampledBar]:
        mode = self._source_reuse_mode()
        if mode == "none" or not section.source:
            return None
        source_events = sampled_by_section.get(str(section.source), [])
        if local_index >= len(source_events):
            self.diagnostics.append_event("section_source_reuse", {
                "section": section.name,
                "source": section.source,
                "section_local_index": int(local_index),
                "used": False,
                "reason": "source_local_index_missing",
                "mode": mode,
            })
            return None
        source_event = source_events[local_index]
        if mode == "anchor_resample":
            return self._anchor_resample_event(model, section, local_index, output_bar_index, source_event, rng)
        reused = SampledBar(
            output_bar_index=output_bar_index,
            section=section.name,
            section_local_index=local_index,
            hidden_state=source_event.hidden_state,
            observation_id=source_event.observation_id,
            composite_key=source_event.composite_key,
            emission_probability=source_event.emission_probability,
            source_file=source_event.source_file,
            source_bar_index=source_event.source_bar_index,
            edit_distance_id=source_event.edit_distance_id,
            kmeans_id=source_event.kmeans_id,
            absolute_tokens=list(source_event.absolute_tokens),
            relative_tokens=list(source_event.relative_tokens),
        )
        self.diagnostics.append_event("section_source_reuse", {
            "section": section.name,
            "source": section.source,
            "section_local_index": int(local_index),
            "output_bar_index": int(output_bar_index),
            "source_output_bar_index": int(source_event.output_bar_index),
            "used": True,
            "mode": mode,
            "reused_hidden_state": int(source_event.hidden_state),
            "reused_observation_id": int(source_event.observation_id),
            "reused_edit_distance_id": int(source_event.edit_distance_id),
            "reused_kmeans_id": source_event.kmeans_id,
        })
        return reused

    def _anchor_resample_event(
        self,
        model: Any,
        section: SectionPlanItem,
        local_index: int,
        output_bar_index: int,
        source_event: SampledBar,
        rng: np.random.Generator,
    ) -> SampledBar:
        state_id = int(section.state_id)
        selected_bar, policy = self._anchor_resample_bar(source_event, rng)
        observation_id = int(selected_bar.observation_id)
        composite = self.bundle.observation_vocab.composite_for(observation_id)
        emission_prob = float(model.emissionprob[state_id, observation_id])
        event = SampledBar(
            output_bar_index=output_bar_index,
            section=section.name,
            section_local_index=local_index,
            hidden_state=state_id,
            observation_id=observation_id,
            composite_key=composite,
            emission_probability=emission_prob,
            source_file=selected_bar.file_path,
            source_bar_index=selected_bar.bar_index,
            edit_distance_id=int(selected_bar.edit_distance_id),
            kmeans_id=selected_bar.kmeans_id,
            absolute_tokens=list(selected_bar.absolute_tokens),
            relative_tokens=list(selected_bar.relative_tokens),
        )
        self.diagnostics.append_event("section_source_reuse", {
            "section": section.name,
            "source": source_event.section,
            "section_local_index": int(local_index),
            "output_bar_index": int(output_bar_index),
            "source_output_bar_index": int(source_event.output_bar_index),
            "used": True,
            "mode": "anchor_resample",
            "policy": policy,
            "anchor_hidden_state": int(source_event.hidden_state),
            "anchor_observation_id": int(source_event.observation_id),
            "anchor_edit_distance_id": int(source_event.edit_distance_id),
            "anchor_kmeans_id": source_event.kmeans_id,
            "selected_hidden_state": int(event.hidden_state),
            "selected_observation_id": int(event.observation_id),
            "selected_edit_distance_id": int(event.edit_distance_id),
            "selected_kmeans_id": event.kmeans_id,
            "selected_source_file": event.source_file,
            "selected_source_bar_index": int(event.source_bar_index),
        })
        return event

    def _anchor_resample_bar(self, source_event: SampledBar, rng: np.random.Generator) -> tuple[BarRecord, str]:
        same_observation = self._candidate_bars(
            lambda bar: int(bar.observation_id) == int(source_event.observation_id)
        )
        selected = self._choose_non_identical(same_observation, source_event, rng)
        if selected is not None:
            return selected, "same_observation"

        same_edit_kmeans = self._candidate_bars(
            lambda bar: (
                int(bar.edit_distance_id) == int(source_event.edit_distance_id)
                and bar.kmeans_id is not None
                and source_event.kmeans_id is not None
                and int(bar.kmeans_id) == int(source_event.kmeans_id)
            )
        )
        selected = self._choose_non_identical(same_edit_kmeans, source_event, rng)
        if selected is not None:
            return selected, "same_edit_distance_and_kmeans"

        same_edit = self._candidate_bars(
            lambda bar: int(bar.edit_distance_id) == int(source_event.edit_distance_id)
        )
        selected = self._choose_non_identical(same_edit, source_event, rng)
        if selected is not None:
            return selected, "same_edit_distance"

        fallback = self._sample_bar(source_event.observation_id, rng)
        return fallback, "hard_reuse_fallback"

    def _candidate_bars(self, predicate: Any) -> List[BarRecord]:
        candidates: List[BarRecord] = []
        for pool in self.bundle.observation_to_bars.values():
            for bar in pool:
                if (
                    bar.observation_id is not None
                    and bar.edit_distance_id is not None
                    and predicate(bar)
                ):
                    candidates.append(bar)
        return candidates

    def _choose_non_identical(
        self,
        candidates: Sequence[BarRecord],
        source_event: SampledBar,
        rng: np.random.Generator,
    ) -> Optional[BarRecord]:
        if not candidates:
            return None
        non_identical = [
            bar for bar in candidates
            if not (
                str(bar.file_path) == str(source_event.source_file)
                and int(bar.bar_index) == int(source_event.source_bar_index)
            )
        ]
        pool = non_identical or list(candidates)
        return pool[int(rng.integers(0, len(pool)))]

    def _source_reuse_mode(self) -> str:
        mode = str(ConfigView(self.config).section("hmm_generation").get("source_reuse_mode", "sampled_path"))
        if mode not in {"none", "sampled_path", "anchor_resample"}:
            raise ValueError("hmm_generation.source_reuse_mode must be 'none', 'sampled_path', or 'anchor_resample'.")
        return mode

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
