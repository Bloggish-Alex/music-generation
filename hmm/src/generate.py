#!/usr/bin/env python3
"""DFA-driven generation CLI for trained form HMM models."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from architecture import DecodeContext
from candidate_selector import CandidateSelectionContext, LearnedCandidateSelector
from config_loader import ConfigLoader, ConfigView
from core_data import BarRecord
from decoder import SectionPlanBuilder
from diagnostics import GenerationDiagnostics
from generation_data import CodebookCandidate, GenerationResult, HarmonyBarPlan, SampledBar, SectionPlanItem
from harmonic_engine import HarmonicCliOverrides, HarmonyProgressionPlanner
from model_store import ModelBundle
from renderer import HarmonicPhysicalRenderer


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
        harmony_plan = HarmonyProgressionPlanner.from_style_config(self.config).plan(plan, seed)
        self.diagnostics.record_stage("decoder_model", {
            "form": form_name,
            "model_type": str(getattr(model, "diagnostics", {}).get("model_type", model.__class__.__name__)),
            "state_count": int(getattr(model, "n_states", len(plan))),
            "observation_count": int(getattr(model, "n_observations", 0)),
            "uses_explicit_duration": hasattr(model, "durationprob"),
            "section_lengths": list(getattr(model, "section_lengths", [])),
            "duration_mode_by_state": list(
                getattr(model, "diagnostics", {}).get("duration_mode_by_state", [])
            ),
        })
        self.diagnostics.record_section_plan([section.to_dict() for section in plan])
        sampled = []
        sampled_by_section: Dict[str, List[SampledBar]] = {}
        previous_candidate: Optional[CodebookCandidate] = None
        for section in plan:
            state_id = int(section.state_id)
            section_events: List[SampledBar] = []
            for local_index in range(int(section.bars)):
                harmony = harmony_plan[len(sampled)]
                event = self._source_reuse_event(
                    model,
                    section,
                    local_index,
                    len(sampled),
                    sampled_by_section,
                    rng,
                )
                if event is not None:
                    sampled.append(event)
                    section_events.append(event)
                    previous_candidate = self._candidate_from_sampled_event(event)
                    self.diagnostics.record_sampled_bar(event.to_dict())
                    continue
                event, selected_candidate = self._sample_event(
                    model,
                    state_id,
                    section,
                    local_index,
                    len(sampled),
                    harmony,
                    int(section.bars),
                    previous_candidate,
                    rng,
                )
                sampled.append(event)
                section_events.append(event)
                previous_candidate = selected_candidate or self._candidate_from_sampled_event(event)
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
        harmony: HarmonyBarPlan,
        section_length: int,
        previous_candidate: Optional[CodebookCandidate],
        rng: np.random.Generator,
    ) -> tuple[SampledBar, Optional[CodebookCandidate]]:
        joint = self._joint_sample_event(
            model,
            state_id,
            section,
            local_index,
            output_bar_index,
            harmony,
            section_length,
            previous_candidate,
            rng,
        )
        if joint is not None:
            return joint
        observation_id, emission_prob, sampling_policy = self._sample_observation(
            model,
            state_id,
            local_index,
            rng,
        )
        selected_bar = self._sample_bar(observation_id, rng)
        composite = self.bundle.symbol_vocabulary.descriptor_key_for(observation_id)
        event = SampledBar(
            output_bar_index=output_bar_index,
            section=section.name,
            section_local_index=local_index,
            hidden_state=state_id,
            observation_id=int(observation_id),
            composite_key=composite,
            emission_probability=emission_prob,
            source_file=selected_bar.file_path,
            source_bar_index=selected_bar.bar_index,
            codebook_id=int(selected_bar.codebook_id),
            kmeans_id=selected_bar.kmeans_id,
            absolute_tokens=list(selected_bar.absolute_tokens),
            relative_tokens=list(selected_bar.relative_tokens),
            selection_mode="observation_sampled_bar",
        )
        self.diagnostics.append_event("observation_position_sampling", {
            "output_bar_index": int(output_bar_index),
            "section": section.name,
            "section_local_index": int(local_index),
            "state_id": int(state_id),
            "observation_id": int(observation_id),
            "composite_key": composite,
            **sampling_policy,
        })
        return event, self._candidate_from_sampled_event(event)

    def _joint_sample_event(
        self,
        model: Any,
        state_id: int,
        section: SectionPlanItem,
        local_index: int,
        output_bar_index: int,
        harmony: HarmonyBarPlan,
        section_length: int,
        previous_candidate: Optional[CodebookCandidate],
        rng: np.random.Generator,
    ) -> Optional[tuple[SampledBar, Optional[CodebookCandidate]]]:
        if self.bundle.candidate_selector_model is None or self.bundle.encoder_model is None:
            return None
        encoder_backend = self.bundle.encoder_model.metadata.get("backend")
        if encoder_backend != "vae_latent":
            return None
        selector = LearnedCandidateSelector(
            self.config,
            self.bundle.candidate_selector_model,
            mode=str(self.config.get("harmonic_engine", {}).get("mode", "major")),
        )
        if (
            not selector.config.enabled
            or selector.config.backend != "learned_ranker"
            or selector.config.selection_stage != "joint"
        ):
            return None
        probs = np.asarray(model.emissionprob[int(state_id)], dtype=np.float64)
        top_k = max(1, int(selector.config.observation_top_k))
        observation_indices = np.array(
            sorted(
                np.argpartition(probs, -min(top_k, len(probs)))[-min(top_k, len(probs)):],
                key=lambda index: float(probs[int(index)]),
                reverse=True,
            ),
            dtype=np.int64,
        )
        observation_entries = []
        for observation_id in observation_indices:
            obs = int(observation_id)
            try:
                codebook_id = self.bundle.symbol_vocabulary.codebook_id_for(obs)
                entry = self.bundle.encoder_model.codebook.entries[int(codebook_id)]
                composite = self.bundle.symbol_vocabulary.descriptor_key_for(obs)
            except KeyError:
                continue
            observation_entries.append((obs, float(probs[obs]), composite, entry))
        if not observation_entries:
            return None
        context_sampled = self._placeholder_sampled_bar(
            int(observation_entries[0][0]),
            str(observation_entries[0][2]),
            int(observation_entries[0][3].codebook_id),
            section,
            local_index,
            output_bar_index,
            state_id,
            float(observation_entries[0][1]),
            observation_entries[0][3],
        )
        observation_id, emission_prob, composite, selection = selector.select_from_observations(
            observation_entries,
            CandidateSelectionContext(
                sampled=context_sampled,
                harmony=harmony,
                section_length=section_length,
                previous_candidate=previous_candidate,
            ),
            rng,
        )
        if observation_id is None or emission_prob is None or composite is None:
            self.diagnostics.append_event("joint_observation_candidate_selection", {
                "output_bar_index": int(output_bar_index),
                "section": section.name,
                "section_local_index": int(local_index),
                "state_id": int(state_id),
                **selection.diagnostics,
            })
            return None
        selected_entry = selection.entry
        event = SampledBar(
            output_bar_index=output_bar_index,
            section=section.name,
            section_local_index=local_index,
            hidden_state=state_id,
            observation_id=int(observation_id),
            composite_key=str(composite),
            emission_probability=float(emission_prob),
            source_file=str(selected_entry.source_file or ""),
            source_bar_index=int(selected_entry.source_bar_index or 0),
            codebook_id=int(selected_entry.codebook_id),
            kmeans_id=None,
            absolute_tokens=list(selected_entry.absolute_tokens),
            relative_tokens=list(selected_entry.relative_tokens),
            selection_mode="joint_observation_candidate",
        )
        self.diagnostics.append_event("joint_observation_candidate_selection", {
            "output_bar_index": int(output_bar_index),
            "section": section.name,
            "section_local_index": int(local_index),
            "state_id": int(state_id),
            **selection.diagnostics,
        })
        self.diagnostics.append_event("observation_position_sampling", {
            "output_bar_index": int(output_bar_index),
            "section": section.name,
            "section_local_index": int(local_index),
            "state_id": int(state_id),
            "observation_id": int(observation_id),
            "composite_key": str(composite),
            "strategy": "joint_observation_candidate",
            "used_position_conditioning": False,
            "emission_probability": float(emission_prob),
            "observation_top_k": int(top_k),
        })
        return event, self._candidate_from_codebook_entry(selected_entry)

    def _placeholder_sampled_bar(
        self,
        observation_id: int,
        composite: str,
        codebook_id: int,
        section: SectionPlanItem,
        local_index: int,
        output_bar_index: int,
        state_id: int,
        emission_probability: float,
        entry: Any,
    ) -> SampledBar:
        return SampledBar(
            output_bar_index=output_bar_index,
            section=section.name,
            section_local_index=local_index,
            hidden_state=state_id,
            observation_id=int(observation_id),
            composite_key=str(composite),
            emission_probability=float(emission_probability),
            source_file=str(entry.source_file or ""),
            source_bar_index=int(entry.source_bar_index or 0),
            codebook_id=int(codebook_id),
            kmeans_id=None,
            absolute_tokens=list(entry.absolute_tokens),
            relative_tokens=list(entry.relative_tokens),
            selection_mode="placeholder",
        )

    def _sample_observation(
        self,
        model: Any,
        state_id: int,
        local_index: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, Dict[str, Any]]:
        encoder_backend = (
            self.bundle.encoder_model.metadata.get("backend")
            if self.bundle.encoder_model is not None
            else None
        )
        if encoder_backend == "vae_latent":
            observation_id, emission_prob = model.sample_from_state(state_id, rng)
            return int(observation_id), float(emission_prob), {
                "strategy": "vae_latent",
                "used_position_conditioning": False,
                "reason": "vae_latent_symbols_do_not_expand_by_position",
            }
        vocab_config = ConfigView(self.config).section("observation_vocab")
        strategy = str(vocab_config.get("strategy", "composite"))
        if strategy != "positioned_composite":
            observation_id, emission_prob = model.sample_from_state(state_id, rng)
            return int(observation_id), float(emission_prob), {
                "strategy": strategy,
                "used_position_conditioning": False,
            }
        modulo = max(1, int(vocab_config.get("position_modulo", 8)))
        target_position = int(local_index) % modulo
        position_strategy = str(vocab_config.get("position_strategy", "period_role"))
        target_context = self._position_context(target_position, position_strategy, modulo)
        probs = np.asarray(model.emissionprob[int(state_id)], dtype=np.float64)
        allowed = self._observations_for_position_context(target_context, position_strategy, target_position)
        if allowed:
            masked = np.zeros_like(probs)
            masked[allowed] = probs[allowed]
            total = float(masked.sum())
            if total > 0:
                conditioned = masked / total
                observation_id = int(rng.choice(len(conditioned), p=conditioned))
                return observation_id, float(probs[observation_id]), {
                    "strategy": strategy,
                    "used_position_conditioning": True,
                    "target_phrase_position": int(target_position),
                    "target_position_context": target_context,
                    "position_strategy": position_strategy,
                    "position_modulo": int(modulo),
                    "allowed_observation_count": int(len(allowed)),
                    "fallback": None,
                }
        observation_id, emission_prob = model.sample_from_state(state_id, rng)
        return int(observation_id), float(emission_prob), {
            "strategy": strategy,
            "used_position_conditioning": False,
            "target_phrase_position": int(target_position),
            "target_position_context": target_context,
            "position_strategy": position_strategy,
            "position_modulo": int(modulo),
            "allowed_observation_count": int(len(allowed)),
            "fallback": "no_positive_position_emission",
        }

    def _observations_for_position_context(
        self,
        context: str,
        position_strategy: str,
        phrase_position: Optional[int] = None,
    ) -> List[int]:
        decode_context = DecodeContext(
            phrase_position=phrase_position if position_strategy == "exact_mod" else None,
            position_context=str(context),
        )
        return self.bundle.symbol_vocabulary.symbols_for_context(
            decode_context,
            position_strategy=position_strategy,
        )

    def _position_context(self, phrase_position: int, position_strategy: str, modulo: int) -> str:
        position = int(phrase_position) % max(1, int(modulo))
        if position_strategy == "exact_mod":
            return str(position)
        if position_strategy == "period_role":
            if position in {0, 1}:
                return "begin"
            if position in {2, 3, 4, 5}:
                return "middle"
            return "end"
        raise ValueError("observation_vocab.position_strategy must be 'exact_mod' or 'period_role'.")

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
            codebook_id=source_event.codebook_id,
            kmeans_id=source_event.kmeans_id,
            absolute_tokens=list(source_event.absolute_tokens),
            relative_tokens=list(source_event.relative_tokens),
            selection_mode=source_event.selection_mode,
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
            "reused_codebook_id": int(source_event.codebook_id),
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
        composite = self.bundle.symbol_vocabulary.descriptor_key_for(observation_id)
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
            codebook_id=int(selected_bar.codebook_id),
            kmeans_id=selected_bar.kmeans_id,
            absolute_tokens=list(selected_bar.absolute_tokens),
            relative_tokens=list(selected_bar.relative_tokens),
            selection_mode="anchor_resample",
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
            "anchor_codebook_id": int(source_event.codebook_id),
            "anchor_kmeans_id": source_event.kmeans_id,
            "selected_hidden_state": int(event.hidden_state),
            "selected_observation_id": int(event.observation_id),
            "selected_codebook_id": int(event.codebook_id),
            "selected_kmeans_id": event.kmeans_id,
            "selected_source_file": event.source_file,
            "selected_source_bar_index": int(event.source_bar_index),
        })
        return event

    def _anchor_resample_bar(self, source_event: SampledBar, rng: np.random.Generator) -> tuple[BarRecord, str]:
        target_context = self._event_position_context(source_event)
        same_observation = self._candidate_bars(
            lambda bar: int(bar.observation_id) == int(source_event.observation_id)
        )
        selected = self._choose_non_identical(same_observation, source_event, rng)
        if selected is not None:
            return selected, "same_observation"

        same_edit_kmeans = self._candidate_bars(
            lambda bar: (
                int(bar.codebook_id) == int(source_event.codebook_id)
                and bar.kmeans_id is not None
                and source_event.kmeans_id is not None
                and int(bar.kmeans_id) == int(source_event.kmeans_id)
                and self._bar_position_context(bar) == target_context
            )
        )
        selected = self._choose_non_identical(same_edit_kmeans, source_event, rng)
        if selected is not None:
            return selected, "same_edit_distance_and_kmeans"

        same_edit = self._candidate_bars(
            lambda bar: (
                int(bar.codebook_id) == int(source_event.codebook_id)
                and self._bar_position_context(bar) == target_context
            )
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
                    and bar.codebook_id is not None
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

    def _event_position_context(self, event: SampledBar) -> Optional[str]:
        try:
            value = self.bundle.symbol_vocabulary.descriptor_for(int(event.observation_id)).position_context
        except KeyError:
            return None
        return str(value) if value is not None else None

    def _bar_position_context(self, bar: BarRecord) -> Optional[str]:
        if bar.observation_id is not None:
            try:
                value = self.bundle.symbol_vocabulary.descriptor_for(int(bar.observation_id)).position_context
                if value is not None:
                    return str(value)
            except KeyError:
                pass
        vocab_config = ConfigView(self.config).section("observation_vocab")
        if str(vocab_config.get("strategy", "composite")) != "positioned_composite":
            return None
        modulo = max(1, int(vocab_config.get("position_modulo", 8)))
        position_strategy = str(vocab_config.get("position_strategy", "period_role"))
        return self._position_context(int(bar.bar_index) % modulo, position_strategy, modulo)

    def _source_reuse_mode(self) -> str:
        mode = str(ConfigView(self.config).section("hmm_generation").get("source_reuse_mode", "sampled_path"))
        if mode not in {"none", "sampled_path", "anchor_resample"}:
            raise ValueError("hmm_generation.source_reuse_mode must be 'none', 'sampled_path', or 'anchor_resample'.")
        return mode

    def _candidate_from_sampled_event(self, event: SampledBar) -> CodebookCandidate:
        codebook = (
            self.bundle.encoder_model.codebook.entries
            if self.bundle.encoder_model is not None
            else self.bundle.global_codebook
        )
        entry = codebook.get(int(event.codebook_id))
        if entry is not None:
            for candidate in entry.candidates:
                if (
                    str(candidate.source_file) == str(event.source_file)
                    and candidate.source_bar_index is not None
                    and int(candidate.source_bar_index) == int(event.source_bar_index)
                ):
                    return candidate
            return self._candidate_from_codebook_entry(entry)
        return CodebookCandidate(
            source_song=None,
            source_file=event.source_file,
            source_bar_index=int(event.source_bar_index),
            relative_tokens=list(event.relative_tokens),
            absolute_tokens=list(event.absolute_tokens),
            kmeans_id=event.kmeans_id,
            observation_id=int(event.observation_id),
        )

    def _candidate_from_codebook_entry(self, entry: Any) -> CodebookCandidate:
        return CodebookCandidate(
            source_song=entry.source_song,
            source_file=entry.source_file,
            source_bar_index=entry.source_bar_index,
            relative_tokens=list(entry.relative_tokens),
            absolute_tokens=list(entry.absolute_tokens),
            density=entry.density,
            token_variance=float(entry.token_variance),
            sharing_score=float(entry.sharing_score),
            kmeans_id=None,
            observation_id=None,
            position_ratio=float(getattr(entry, "position_ratio", 0.0)),
            latent_vector=(
                [float(value) for value in entry.latent_vector]
                if getattr(entry, "latent_vector", None) is not None
                else None
            ),
        )

    def write_outputs(self, generation: GenerationResult, json_path: Path, midi_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        codebook = (
            self.bundle.encoder_model.codebook.entries
            if self.bundle.encoder_model is not None
            else self.bundle.global_codebook
        )
        renderer = HarmonicPhysicalRenderer(
            self.config,
            codebook,
            candidate_selector_model=self.bundle.candidate_selector_model,
        )
        render_result = renderer.realize(generation)
        realized = render_result.generation
        render_diag = renderer.write_midi(realized, midi_path)
        render_result.diagnostics["render"] = render_diag
        json_path.write_text(json.dumps(realized.to_dict(), indent=2), encoding="utf-8")
        self.diagnostics.record_stage("harmonic_engine", render_result.diagnostics)
        for event in render_result.diagnostics.get("rare_bar_selection", {}).get("events", []):
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
        "candidate_selector",
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
