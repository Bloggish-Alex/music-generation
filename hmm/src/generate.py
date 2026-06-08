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
        self._chunk_cache: Dict[int, List[List[BarRecord]]] = {}

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
            local_index = 0
            while local_index < int(section.bars):
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
                    self.diagnostics.record_sampled_bar(event.to_dict())
                    local_index += 1
                    continue
                chunk_events = self._sample_phrase_chunk(
                    model,
                    state_id,
                    section,
                    local_index,
                    int(section.bars) - local_index,
                    len(sampled),
                    rng,
                )
                if chunk_events:
                    for chunk_event in chunk_events:
                        sampled.append(chunk_event)
                        section_events.append(chunk_event)
                        self.diagnostics.record_sampled_bar(chunk_event.to_dict())
                    local_index += len(chunk_events)
                    continue
                event = self._sample_event(model, state_id, section, local_index, len(sampled), rng)
                sampled.append(event)
                section_events.append(event)
                self.diagnostics.record_sampled_bar(event.to_dict())
                local_index += 1
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
        observation_id, emission_prob, sampling_policy = self._sample_observation(
            model,
            state_id,
            local_index,
            rng,
        )
        selected_bar = self._sample_bar(observation_id, rng)
        composite = self.bundle.observation_vocab.composite_for(observation_id)
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
            edit_distance_id=int(selected_bar.edit_distance_id),
            kmeans_id=selected_bar.kmeans_id,
            absolute_tokens=list(selected_bar.absolute_tokens),
            relative_tokens=list(selected_bar.relative_tokens),
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
        return event

    def _sample_phrase_chunk(
        self,
        model: Any,
        state_id: int,
        section: SectionPlanItem,
        local_index: int,
        remaining_bars: int,
        output_bar_index: int,
        rng: np.random.Generator,
    ) -> Optional[List[SampledBar]]:
        generation = ConfigView(self.config).section("hmm_generation")
        if str(generation.get("sampling_unit", "bar")) != "phrase_chunk":
            return None
        chunk_size = int(generation.get("phrase_chunk_bars", 2))
        if chunk_size <= 1 or remaining_bars < chunk_size:
            return None
        candidates = self._phrase_chunk_candidates(model, state_id, local_index, chunk_size)
        if not candidates:
            self.diagnostics.append_event("phrase_chunk_sampling", {
                "section": section.name,
                "section_local_index": int(local_index),
                "output_bar_index": int(output_bar_index),
                "used": False,
                "reason": "no_matching_chunk",
                "chunk_bars": int(chunk_size),
            })
            return None
        top_k = int(generation.get("phrase_chunk_top_k", 32))
        temperature = float(generation.get("phrase_chunk_temperature", 0.7))
        candidate_indices = self._top_score_indices(
            np.array([candidate["score"] for candidate in candidates], dtype=np.float64),
            top_k,
        )
        probabilities = self._temperature_probabilities(
            np.array([candidates[index]["score"] for index in candidate_indices], dtype=np.float64),
            temperature,
        )
        selected_local = int(rng.choice(len(candidate_indices), p=probabilities))
        selected_index = int(candidate_indices[selected_local])
        selected = candidates[selected_index]
        full_probabilities = np.zeros(len(candidates), dtype=np.float64)
        for index, probability in zip(candidate_indices, probabilities):
            full_probabilities[int(index)] = float(probability)
        events = [
            self._event_from_bar(
                bar,
                section,
                local_index + offset,
                output_bar_index + offset,
                state_id,
                float(model.emissionprob[int(state_id), int(bar.observation_id)]),
            )
            for offset, bar in enumerate(selected["bars"])
        ]
        self.diagnostics.append_event("phrase_chunk_sampling", {
            "section": section.name,
            "section_local_index": int(local_index),
            "output_bar_index": int(output_bar_index),
            "used": True,
            "chunk_bars": int(chunk_size),
            "candidate_count": len(candidates),
            "sampling_candidate_count": int(len(candidate_indices)),
            "top_k": int(top_k),
            "temperature": float(temperature),
            "selected_index": int(selected_index),
            "selected_probability": round(float(full_probabilities[selected_index]), 6),
            "selected": self._chunk_diagnostics(selected),
            "top_candidates": [
                {
                    **self._chunk_diagnostics(candidates[index]),
                    "candidate_index": int(index),
                    "probability": round(float(full_probabilities[index]), 6),
                }
                for index in sorted(
                    range(len(candidates)),
                    key=lambda idx: float(candidates[idx]["score"]),
                    reverse=True,
                )[:5]
            ],
        })
        return events

    def _phrase_chunk_candidates(
        self,
        model: Any,
        state_id: int,
        local_index: int,
        chunk_size: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        probs = np.asarray(model.emissionprob[int(state_id)], dtype=np.float64)
        for bars in self._training_chunks(chunk_size):
            if not self._chunk_matches_position(bars, local_index):
                continue
            observation_ids = [int(bar.observation_id) for bar in bars if bar.observation_id is not None]
            if len(observation_ids) != chunk_size:
                continue
            emission_probs = [float(probs[observation_id]) for observation_id in observation_ids]
            if any(prob <= 0.0 for prob in emission_probs):
                continue
            score = float(np.prod(np.array(emission_probs, dtype=np.float64)))
            candidates.append({
                "bars": list(bars),
                "score": score,
                "emission_probabilities": emission_probs,
            })
        return candidates

    def _training_chunks(self, chunk_size: int) -> List[List[BarRecord]]:
        if int(chunk_size) in self._chunk_cache:
            return self._chunk_cache[int(chunk_size)]
        groups: Dict[str, List[BarRecord]] = {}
        for pool in self.bundle.observation_to_bars.values():
            for bar in pool:
                groups.setdefault(str(bar.file_path), []).append(bar)
        chunks: List[List[BarRecord]] = []
        for bars in groups.values():
            unique = {
                int(bar.bar_index): bar
                for bar in bars
            }
            ordered = [unique[index] for index in sorted(unique)]
            for start in range(0, max(0, len(ordered) - chunk_size + 1)):
                candidate = ordered[start:start + chunk_size]
                if all(
                    int(candidate[offset + 1].bar_index) == int(candidate[offset].bar_index) + 1
                    for offset in range(chunk_size - 1)
                ):
                    chunks.append(candidate)
        self._chunk_cache[int(chunk_size)] = chunks
        return chunks

    def _chunk_matches_position(self, bars: Sequence[BarRecord], local_index: int) -> bool:
        vocab_config = ConfigView(self.config).section("observation_vocab")
        if str(vocab_config.get("strategy", "composite")) != "positioned_composite":
            return True
        modulo = max(1, int(vocab_config.get("position_modulo", 8)))
        position_strategy = str(vocab_config.get("position_strategy", "period_role"))
        for offset, bar in enumerate(bars):
            expected = self._position_context((int(local_index) + offset) % modulo, position_strategy, modulo)
            if self._bar_position_context(bar) != expected:
                return False
        return True

    def _event_from_bar(
        self,
        bar: BarRecord,
        section: SectionPlanItem,
        local_index: int,
        output_bar_index: int,
        state_id: int,
        emission_probability: float,
    ) -> SampledBar:
        observation_id = int(bar.observation_id)
        return SampledBar(
            output_bar_index=int(output_bar_index),
            section=section.name,
            section_local_index=int(local_index),
            hidden_state=int(state_id),
            observation_id=observation_id,
            composite_key=self.bundle.observation_vocab.composite_for(observation_id),
            emission_probability=float(emission_probability),
            source_file=bar.file_path,
            source_bar_index=int(bar.bar_index),
            edit_distance_id=int(bar.edit_distance_id),
            kmeans_id=bar.kmeans_id,
            absolute_tokens=list(bar.absolute_tokens),
            relative_tokens=list(bar.relative_tokens),
        )

    def _chunk_diagnostics(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        bars = candidate["bars"]
        return {
            "score": round(float(candidate["score"]), 12),
            "emission_probabilities": [
                round(float(value), 12)
                for value in candidate["emission_probabilities"]
            ],
            "source_file": bars[0].file_path,
            "source_bar_indices": [int(bar.bar_index) for bar in bars],
            "observation_ids": [int(bar.observation_id) for bar in bars],
            "composite_keys": [
                self.bundle.observation_vocab.composite_for(int(bar.observation_id))
                for bar in bars
            ],
            "edit_distance_ids": [int(bar.edit_distance_id) for bar in bars],
            "kmeans_ids": [bar.kmeans_id for bar in bars],
            "position_contexts": [self._bar_position_context(bar) for bar in bars],
            "relative_tokens": [list(bar.relative_tokens) for bar in bars],
        }

    def _top_score_indices(self, scores: np.ndarray, top_k: int) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=np.int64)
        if top_k <= 0 or top_k >= len(scores):
            return np.arange(len(scores), dtype=np.int64)
        return np.array(
            sorted(
                np.argpartition(scores, -top_k)[-top_k:],
                key=lambda index: float(scores[int(index)]),
                reverse=True,
            ),
            dtype=np.int64,
        )

    def _temperature_probabilities(self, scores: np.ndarray, temperature: float) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=np.float64)
        adjusted = np.power(np.maximum(scores, 1.0e-12), 1.0 / max(float(temperature), 1.0e-6))
        total = float(adjusted.sum())
        if total <= 0.0:
            return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)
        return adjusted / total

    def _sample_observation(
        self,
        model: Any,
        state_id: int,
        local_index: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, Dict[str, Any]]:
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
        allowed = self._observations_for_position_context(target_context, position_strategy)
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

    def _observations_for_position_context(self, context: str, position_strategy: str) -> List[int]:
        result: List[int] = []
        for observation_id, composite in self.bundle.observation_vocab.observation_to_composite.items():
            parts = self.bundle.observation_vocab.composite_parts.get(str(composite), {})
            if str(parts.get("position_strategy", position_strategy)) != str(position_strategy):
                continue
            if str(parts.get("position_context")) == str(context):
                result.append(int(observation_id))
        return result

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
        target_context = self._event_position_context(source_event)
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
                and self._bar_position_context(bar) == target_context
            )
        )
        selected = self._choose_non_identical(same_edit_kmeans, source_event, rng)
        if selected is not None:
            return selected, "same_edit_distance_and_kmeans"

        same_edit = self._candidate_bars(
            lambda bar: (
                int(bar.edit_distance_id) == int(source_event.edit_distance_id)
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

    def _event_position_context(self, event: SampledBar) -> Optional[str]:
        composite = self.bundle.observation_vocab.observation_to_composite.get(int(event.observation_id))
        if composite is None:
            return None
        parts = self.bundle.observation_vocab.composite_parts.get(str(composite), {})
        value = parts.get("position_context")
        return str(value) if value is not None else None

    def _bar_position_context(self, bar: BarRecord) -> Optional[str]:
        if bar.observation_id is not None:
            composite = self.bundle.observation_vocab.observation_to_composite.get(int(bar.observation_id))
            if composite is not None:
                parts = self.bundle.observation_vocab.composite_parts.get(str(composite), {})
                value = parts.get("position_context")
                if value is not None:
                    return str(value)
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
