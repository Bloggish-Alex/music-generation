#!/usr/bin/env python3
"""Decoder-layer form-driven generator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.architecture import DecodeContext
from decoder.candidate_selector import CandidateSelectionContext, LearnedCandidateSelector
from common.config_loader import ConfigView
from data.core_data import BarRecord
from decoder.decoder import SectionPlanBuilder
from diagnostics.diagnostics import GenerationDiagnostics
from data.generation_data import CodebookCandidate, GenerationResult, HarmonyBarPlan, SampledBar, SectionPlanItem
from decoder.latent_token_model import LatentTokenContext
from decoder.lstm_token_model import LSTMTokenModel
from encoder.opening_seed import OpeningSeedSelector
from decoder.temporal_graph import (
    CandidateTokenMask,
    RuntimeThemeGraphConfig,
    RuntimeThemeGraphController,
    TemporalGraphTemplate,
)
from renderer.dvae_runtime import DVAEDecodeRequest, TrainedDVAEDecoderRuntime
from renderer.harmonic_engine import HarmonyProgressionPlanner
from common.model_store import ModelBundle


@dataclass(frozen=True)
class PhraseBeam:
    """One candidate path in short-horizon temporal graph planning."""

    token_ids: List[int]
    latent_sequence: List[List[float]]
    score: float
    first_token_id: Optional[int]
    steps: List[Dict[str, Any]]


class FormDrivenGenerator:
    """Generate bar IDs and concrete bars from a fixed DFA section plan."""

    def __init__(self, bundle: ModelBundle, config: Optional[dict] = None) -> None:
        self.bundle = bundle
        self.config = config or bundle.config
        self.diagnostics = GenerationDiagnostics()
        self.lstm_ranker = self._load_lstm_ranker()
        self.dvae_decoder = self._load_dvae_decoder()
        self.opening_seed_selector = self._build_opening_seed_selector()
        self.opening_theme_pipeline: Optional[Dict[str, Any]] = None
        self.theme_graph: Optional[RuntimeThemeGraphController] = None

    def generate(self, form_name: str, seed: Optional[int] = None) -> GenerationResult:
        form_name = self._resolve_form_name(form_name)
        rng = np.random.default_rng(seed)
        model = self.bundle.form_models[form_name]
        plan = SectionPlanBuilder().build(self.bundle, form_name)
        harmony_plan = HarmonyProgressionPlanner.from_style_config(self.config).plan(plan, seed)
        self.theme_graph = self._build_theme_graph(form_name)
        self.opening_theme_pipeline = self._select_opening_theme_pipeline(rng)
        self.diagnostics.record_stage("decoder_model", {
            "form": form_name,
            "model_type": str(getattr(model, "diagnostics", {}).get("model_type", model.__class__.__name__)),
            "state_count": int(getattr(model, "n_states", len(plan))),
            "observation_count": int(getattr(model, "n_observations", 0)),
            "section_lengths": list(getattr(model, "section_lengths", [])),
        })
        if self.opening_theme_pipeline is not None:
            self.diagnostics.record_stage("opening_theme_pipeline", self.opening_theme_pipeline)
        self.diagnostics.record_section_plan([section.to_dict() for section in plan])
        sampled = []
        sampled_by_section: Dict[str, List[SampledBar]] = {}
        previous_candidate: Optional[CodebookCandidate] = None
        for section in plan:
            state_id = int(section.state_id)
            section_events: List[SampledBar] = []
            for local_index in range(int(section.bars)):
                harmony = harmony_plan[len(sampled)]
                event = None
                if not self._uses_temporal_graph_decoder():
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
                    sampled,
                    harmony_plan[len(sampled) - 1] if sampled else None,
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

    def _resolve_form_name(self, form_name: str) -> str:
        requested = str(form_name)
        if requested in self.bundle.form_models:
            return requested
        lowered = requested.lower()
        matches = [
            name for name in self.bundle.form_models
            if str(name).lower() == lowered
        ]
        if matches:
            return str(matches[0])
        available = sorted(str(name) for name in self.bundle.form_models)
        templates = sorted(str(name) for name in getattr(self.bundle, "form_templates", {}).keys())
        raise ValueError(
            f"Model does not contain form '{requested}'. "
            f"Available trained forms: {available}. "
            f"Available templates: {templates}. "
            "Use --form with one of the trained forms, or retrain with form.json entries for that form."
        )

    def _build_theme_graph(self, form_name: str) -> Optional[RuntimeThemeGraphController]:
        decoder_config = ConfigView(self.config).section("decoder")
        backend = str(decoder_config.get("backend", "temporal_lstm"))
        if backend != "temporal_lstm":
            return None
        if self.lstm_ranker is None:
            raise ValueError("decoder.backend='temporal_lstm' requires LSTM decoder artifacts in model_dir.")
        graph_section = dict(decoder_config.get("temporal_graph", {}) or {})
        graph_section["enabled"] = True
        graph_config = RuntimeThemeGraphConfig.from_decoder_config({
            **decoder_config,
            "temporal_graph": graph_section,
        })
        template = TemporalGraphTemplate.from_config(form_name, self.config)
        if not template.action_script:
            raise ValueError(
                f"decoder.backend='temporal_lstm' requires temporal_graph_templates.{form_name}.action_script."
            )
        controller = RuntimeThemeGraphController(
            template=template,
            vocab_size=int(self.lstm_ranker.metadata.vocab_size),
            config=graph_config,
        )
        self.diagnostics.record_stage("temporal_graph", {
            "backend": "temporal_lstm",
            "form": str(form_name),
            "config": {
                "enabled": bool(graph_config.enabled),
                "candidate_top_k": int(graph_config.candidate_top_k),
                "min_candidates": int(graph_config.min_candidates),
                "memory_window": int(graph_config.memory_window),
                "memory_bias": float(graph_config.memory_bias),
                "anchor_span_bars": int(graph_config.anchor_span_bars),
                "aligned_anchor_bias": float(graph_config.aligned_anchor_bias),
                "same_theme_memory_bias": float(graph_config.same_theme_memory_bias),
                "planning_enabled": bool(graph_config.planning_enabled),
                "planning_window_bars": int(graph_config.planning_window_bars),
                "planning_beam_width": int(graph_config.planning_beam_width),
                "planning_branch_top_k": int(graph_config.planning_branch_top_k),
                "planning_path_temperature": float(graph_config.planning_path_temperature),
                "action_temperature": dict(graph_config.action_temperature),
                "diagnostics_top_k": int(graph_config.diagnostics_top_k),
            },
            "configured_sections": sorted(template.action_script.keys()),
            "theme_memory_config": template.theme_memory_config,
        })
        return controller

    def _uses_temporal_graph_decoder(self) -> bool:
        return str(ConfigView(self.config).section("decoder").get("backend", "temporal_lstm")) == "temporal_lstm"

    def _build_opening_seed_selector(self) -> OpeningSeedSelector:
        decoder_config = ConfigView(self.config).section("decoder")
        return OpeningSeedSelector.from_config_dict(decoder_config.get("opening_seed", {}))

    def _opening_seed_config(self) -> Dict[str, Any]:
        config = ConfigView(self.config).section("decoder").get("opening_seed", {})
        return config if isinstance(config, dict) else {}

    def _opening_seed_mode(self) -> str:
        config = self._opening_seed_config()
        if not bool(config.get("enabled", True)):
            return "disabled"
        return str(config.get("mode", "single_bar_pool"))

    def _select_opening_theme_pipeline(self, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
        if self._opening_seed_mode() != "theme_pipeline":
            return None
        if self.bundle.encoder_model is None:
            return None
        return self.opening_seed_selector.select_theme_pipeline(
            self.bundle.encoder_model.metadata,
            rng,
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
        sampled_history: Sequence[SampledBar],
        previous_harmony: Optional[HarmonyBarPlan],
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
            section_length,
            sampled_history,
            harmony,
            previous_harmony,
            output_bar_index,
            rng,
        )
        composite = self.bundle.symbol_vocabulary.descriptor_key_for(observation_id)
        selection_mode = (
            "lstm_token_logits"
            if str(sampling_policy.get("strategy")) in {"lstm_token_logits", "temporal_lstm_logits"}
            else "observation_sampled_bar"
        )
        if selection_mode == "lstm_token_logits":
            event = self._dvae_decoded_event(
                observation_id,
                composite,
                emission_prob,
                section,
                local_index,
                output_bar_index,
                state_id,
                sampling_policy,
                sampled_history,
            )
            self.diagnostics.append_event("dvae_token_decode", {
                "output_bar_index": int(output_bar_index),
                "section": section.name,
                "section_local_index": int(local_index),
                "hidden_state": int(state_id),
                "observation_id": int(observation_id),
                "codebook_id": int(event.codebook_id),
                "decoded_token_diagnostics": self._token_diagnostics(event.relative_tokens),
                **sampling_policy.get("dvae_decode", {}),
            })
            self.diagnostics.append_event("observation_position_sampling", {
                "output_bar_index": int(output_bar_index),
                "section": section.name,
                "section_local_index": int(local_index),
                "state_id": int(state_id),
                "observation_id": int(observation_id),
                "composite_key": composite,
                **{key: value for key, value in sampling_policy.items() if key != "dvae_decode"},
            })
            return event, self._candidate_from_sampled_event(event)
        selected_bar = self._sample_bar(observation_id, rng)
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
            selection_mode=selection_mode,
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

    def _dvae_decoded_event(
        self,
        observation_id: int,
        composite: str,
        emission_prob: float,
        section: SectionPlanItem,
        local_index: int,
        output_bar_index: int,
        state_id: int,
        sampling_policy: Dict[str, Any],
        sampled_history: Sequence[SampledBar],
    ) -> SampledBar:
        if self.dvae_decoder is None:
            raise RuntimeError("decoder.backend='lstm_token' requires a trained VAE decoder runtime.")
        if self.bundle.encoder_model is None:
            raise RuntimeError("VAE decode requires encoder_model in the model bundle.")
        entry = self.bundle.encoder_model.codebook_entry_for_symbol(int(observation_id))
        if entry.latent_vector is None:
            raise RuntimeError(f"Codebook entry for observation {observation_id} has no latent vector.")
        previous_last_pitch_scalar = self._previous_last_pitch_scalar(sampled_history)
        result = self.dvae_decoder.decode(DVAEDecodeRequest(
            token_id=int(observation_id),
            latent_vector=[float(value) for value in entry.latent_vector],
            previous_last_pitch_scalar=previous_last_pitch_scalar,
            source="codebook_medoid_mu",
            metadata={
                "output_bar_index": int(output_bar_index),
                "section": section.name,
                "section_local_index": int(local_index),
                "hidden_state": int(state_id),
                "codebook_id": int(entry.codebook_id),
                "previous_last_pitch_scalar": float(previous_last_pitch_scalar),
            },
        ))
        sampling_policy["dvae_decode"] = result.diagnostics
        return SampledBar(
            output_bar_index=output_bar_index,
            section=section.name,
            section_local_index=local_index,
            hidden_state=state_id,
            observation_id=int(observation_id),
            composite_key=str(composite),
            emission_probability=float(emission_prob),
            source_file="vae_decoder",
            source_bar_index=-1,
            codebook_id=int(entry.codebook_id),
            kmeans_id=None,
            absolute_tokens=list(result.relative_tokens),
            relative_tokens=list(result.relative_tokens),
            selection_mode="lstm_token_logits",
        )

    def _previous_last_pitch_scalar(self, sampled_history: Sequence[SampledBar]) -> float:
        previous_pitch: Optional[int] = None
        if sampled_history:
            previous_pitch = self._last_relative_note_pitch(sampled_history[-1].relative_tokens)
        if previous_pitch is None:
            return 0.0
        if self.bundle.encoder_model is not None:
            vae_config = self.bundle.encoder_model.metadata.get("vae_config", {})
            scale = float(vae_config.get("previous_last_pitch_scale", vae_config.get("pitch_scale", 24.0)))
        else:
            scale = 24.0
        return float(previous_pitch) / max(1.0e-6, scale)

    def _last_relative_note_pitch(self, tokens: Sequence[int]) -> Optional[int]:
        for token in reversed(tokens):
            if int(token) >= 0:
                return int(token)
        return None

    def _token_diagnostics(self, tokens: Sequence[int]) -> Dict[str, Any]:
        values = [int(token) for token in tokens]
        total = len(values)
        rest_count = sum(1 for token in values if token == -1)
        sustain_count = sum(1 for token in values if token == -2)
        note_on_values = [token for token in values if token >= 0]
        return {
            "token_count": int(total),
            "rest_count": int(rest_count),
            "sustain_count": int(sustain_count),
            "note_on_count": int(len(note_on_values)),
            "rest_ratio": float(rest_count / total) if total else 0.0,
            "sustain_ratio": float(sustain_count / total) if total else 0.0,
            "note_on_ratio": float(len(note_on_values) / total) if total else 0.0,
            "first_note_pitch": int(note_on_values[0]) if note_on_values else None,
            "last_note_pitch": int(note_on_values[-1]) if note_on_values else None,
            "leading_rest_slots": int(self._leading_run(values, -1)),
            "trailing_rest_slots": int(self._trailing_run(values, -1)),
            "max_rest_run": int(self._max_run(values, -1)),
            "max_sustain_run": int(self._max_run(values, -2)),
        }

    def _leading_run(self, values: Sequence[int], target: int) -> int:
        count = 0
        for value in values:
            if int(value) != int(target):
                break
            count += 1
        return count

    def _trailing_run(self, values: Sequence[int], target: int) -> int:
        count = 0
        for value in reversed(values):
            if int(value) != int(target):
                break
            count += 1
        return count

    def _max_run(self, values: Sequence[int], target: int) -> int:
        best = 0
        current = 0
        for value in values:
            if int(value) == int(target):
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

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
        section_length: int,
        sampled_history: Sequence[SampledBar],
        harmony: HarmonyBarPlan,
        previous_harmony: Optional[HarmonyBarPlan],
        output_bar_index: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, Dict[str, Any]]:
        reranked = self._sample_observation_with_lstm_ranker(
            model,
            state_id,
            local_index,
            section_length,
            sampled_history,
            harmony,
            previous_harmony,
            output_bar_index,
            rng,
        )
        if reranked is not None:
            return reranked
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

    def _sample_observation_with_lstm_ranker(
        self,
        model: Any,
        state_id: int,
        local_index: int,
        section_length: int,
        sampled_history: Sequence[SampledBar],
        harmony: HarmonyBarPlan,
        previous_harmony: Optional[HarmonyBarPlan],
        output_bar_index: int,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, float, Dict[str, Any]]]:
        decoder_config = ConfigView(self.config).section("decoder")
        backend = str(decoder_config.get("backend", "temporal_lstm"))
        if backend not in {"lstm_token", "lstm_rerank", "temporal_lstm"}:
            return None
        if self.lstm_ranker is None:
            raise ValueError("decoder.backend is LSTM-based but no LSTM decoder model was loaded.")
        probs = np.asarray(model.emissionprob[int(state_id)], dtype=np.float64)
        if probs.size == 0:
            return None
        context = self._lstm_context(sampled_history, state_id, local_index)
        context = self._opening_seeded_context(context, int(output_bar_index), rng)
        temporal_action_label: Optional[str] = None
        action_state = None
        if backend == "temporal_lstm":
            if self.theme_graph is None:
                raise ValueError("decoder.backend='temporal_lstm' requires temporal graph templates in config.")
            all_mask = CandidateTokenMask(
                allowed_token_ids=list(range(int(self.lstm_ranker.metadata.vocab_size))),
                reason="temporal_graph_raw_logits",
                diagnostics={"state_id": int(state_id), "local_index": int(local_index)},
            )
            raw_result = self.lstm_ranker.score_next(context, all_mask)
            action_state = self.theme_graph.action_state(
                section=str(harmony.section),
                section_local_index=int(local_index),
                section_length=int(section_length),
                harmony_degree=harmony.degree,
            )
            temporal_action_label = action_state.action.value
            context = LatentTokenContext(
                token_ids=context.token_ids,
                latent_sequence=context.latent_sequence,
                action_state=action_state,
                metadata={**context.metadata, "temporal_action": temporal_action_label},
            )
            plan_step = self.theme_graph.plan_from_logits(action_state, raw_result.logits)
            mask = plan_step.mask
            result = self.lstm_ranker.score_next(context, mask)
            candidates = np.asarray([int(obs) for obs in mask.allowed_token_ids], dtype=np.int64)
            temporal_diagnostics = plan_step.to_dict()
            candidate_source = "temporal_graph"
        else:
            top_k = max(1, int(decoder_config.get("lstm_candidate_top_k", 32)))
            finite_k = min(top_k, int(probs.size))
            candidates = np.array(
                sorted(
                    np.argpartition(probs, -finite_k)[-finite_k:],
                    key=lambda index: float(probs[int(index)]),
                    reverse=True,
                ),
                dtype=np.int64,
            )
            candidates = np.asarray([int(obs) for obs in candidates if float(probs[int(obs)]) > 0.0], dtype=np.int64)
            if candidates.size == 0:
                return None
            mask = CandidateTokenMask(
                allowed_token_ids=[int(obs) for obs in candidates.tolist()],
                reason="form_prior_top_k",
                diagnostics={
                    "state_id": int(state_id),
                    "local_index": int(local_index),
                    "candidate_top_k": int(top_k),
                },
            )
            result = self.lstm_ranker.score_next(context, mask)
            temporal_diagnostics = None
            candidate_source = "form_prior_top_k"
        if candidates.size == 0:
            return None
        masked_logits = np.asarray(result.masked_logits, dtype=np.float64)
        if backend == "temporal_lstm" and self.theme_graph is not None and temporal_diagnostics is not None:
            memory_biases = self.theme_graph.memory_biases_for_state(action_state)
            for token_id, bias in memory_biases.items():
                if 0 <= token_id < len(masked_logits) and np.isfinite(masked_logits[token_id]):
                    masked_logits[token_id] += float(bias)
        processed_logits, boundary_diagnostics = self._apply_boundary_continuity_processor(
            masked_logits,
            candidates,
            sampled_history,
            harmony,
            previous_harmony,
            decoder_config,
            temporal_action_label,
        )
        theme_pipeline_selection = self._theme_pipeline_selection(
            int(output_bar_index),
            candidates,
            processed_logits,
        )
        if theme_pipeline_selection is not None:
            observation_id = int(theme_pipeline_selection["selected_symbol_id"])
            phrase_planning = {
                "enabled": False,
                "reason": "opening_theme_pipeline_bootstrap",
            }
            rank_rows = self._lstm_rank_rows(
                candidates,
                processed_logits,
                probs,
                int(decoder_config.get("lstm_diagnostics_top_k", 8)),
            )
            selected_rank = next(
                (int(row["rank"]) for row in rank_rows if int(row["observation_id"]) == observation_id),
                None,
            )
            selected_lstm_probability = self._probability_from_logits(processed_logits, observation_id, candidates)
            if backend == "temporal_lstm" and self.theme_graph is not None:
                self.theme_graph.remember_selection(
                    action_state,
                    int(observation_id),
                )
            return observation_id, float(probs[observation_id]) if observation_id < len(probs) else 0.0, {
                "strategy": "temporal_lstm_logits" if backend == "temporal_lstm" else "lstm_token_logits",
                "used_position_conditioning": False,
                "hidden_state": int(state_id),
                "context_token_ids": [int(token) for token in context.token_ids],
                "candidate_source": candidate_source,
                "candidate_count": int(len(candidates)),
                "lstm_temperature": float(decoder_config.get("lstm_temperature", 0.9)),
                "lstm_diagnostics": result.diagnostics,
                "opening_seed_context": context.metadata.get("opening_seed"),
                "selected_lstm_rank": selected_rank,
                "selected_lstm_probability": selected_lstm_probability,
                "lstm_top_tokens": rank_rows,
                "boundary_continuity": boundary_diagnostics,
                "temporal_graph": temporal_diagnostics,
                "phrase_planning": phrase_planning,
                "opening_seed": theme_pipeline_selection,
            }
        candidate_values = processed_logits[candidates]
        finite = np.isfinite(candidate_values)
        if not np.any(finite):
            return None
        temperature = max(1e-6, float(decoder_config.get("lstm_temperature", 0.9)))
        phrase_planning = None
        observation_id: Optional[int] = None
        if backend == "temporal_lstm" and self.theme_graph is not None:
            planned = self._temporal_phrase_plan(
                context=context,
                current_logits=processed_logits,
                current_candidates=candidates,
                state_id=state_id,
                local_index=local_index,
                section_length=section_length,
                harmony=harmony,
                temperature=temperature,
                rng=rng,
            )
            if planned is not None:
                observation_id, phrase_planning = planned
        if observation_id is None:
            scaled = candidate_values.copy()
            scaled[finite] = scaled[finite] / temperature
            max_value = float(np.max(scaled[finite]))
            weights = np.zeros_like(scaled, dtype=np.float64)
            weights[finite] = np.exp(scaled[finite] - max_value)
            weights_sum = float(weights.sum())
            if weights_sum <= 0.0:
                return None
            sampled_index = int(rng.choice(len(candidates), p=weights / weights_sum))
            observation_id = int(candidates[sampled_index])
            phrase_planning = {
                "enabled": False,
                "reason": "single_step_sampling",
            }
        rank_rows = self._lstm_rank_rows(
            candidates,
            processed_logits,
            probs,
            int(decoder_config.get("lstm_diagnostics_top_k", 8)),
        )
        selected_rank = next(
            (int(row["rank"]) for row in rank_rows if int(row["observation_id"]) == observation_id),
            None,
        )
        selected_lstm_probability = self._probability_from_logits(processed_logits, observation_id, candidates)
        if backend == "temporal_lstm" and self.theme_graph is not None:
            self.theme_graph.remember_selection(
                action_state,
                int(observation_id),
            )
        return observation_id, float(probs[observation_id]), {
            "strategy": "temporal_lstm_logits" if backend == "temporal_lstm" else "lstm_token_logits",
            "used_position_conditioning": False,
            "hidden_state": int(state_id),
            "context_token_ids": [int(token) for token in context.token_ids],
            "candidate_source": candidate_source,
            "candidate_count": int(len(candidates)),
            "lstm_temperature": float(temperature),
            "lstm_diagnostics": result.diagnostics,
            "opening_seed_context": context.metadata.get("opening_seed"),
            "selected_lstm_rank": selected_rank,
            "selected_lstm_probability": selected_lstm_probability,
            "lstm_top_tokens": rank_rows,
            "boundary_continuity": boundary_diagnostics,
            "temporal_graph": temporal_diagnostics,
            "phrase_planning": phrase_planning,
        }

    def _apply_boundary_continuity_processor(
        self,
        logits: np.ndarray,
        candidates: np.ndarray,
        sampled_history: Sequence[SampledBar],
        harmony: HarmonyBarPlan,
        previous_harmony: Optional[HarmonyBarPlan],
        decoder_config: Dict[str, Any],
        temporal_action: Optional[str] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        config = decoder_config.get("boundary_continuity", {})
        if not isinstance(config, dict):
            config = {}
        enabled = bool(config.get("enabled", False))
        mode = str(config.get("mode", "soft_penalty"))
        max_jump = float(config.get("max_jump_semitones", 12.0))
        base_penalty_weight = float(config.get("penalty_weight", 1.0))
        action_penalty_weights = config.get("action_penalty_weight", {})
        if isinstance(action_penalty_weights, dict) and temporal_action in action_penalty_weights:
            penalty_weight = float(action_penalty_weights[str(temporal_action)])
        else:
            penalty_weight = base_penalty_weight
        min_allowed = max(1, int(config.get("min_allowed_candidates", 3)))
        diagnostics_top_k = max(0, int(config.get("diagnostics_top_k", 8)))
        processed = np.asarray(logits, dtype=np.float64).copy()
        diagnostics: Dict[str, Any] = {
            "enabled": enabled,
            "mode": mode,
            "max_jump_semitones": max_jump,
            "temporal_action": temporal_action,
            "base_penalty_weight": base_penalty_weight,
            "penalty_weight": penalty_weight,
            "action_penalty_weight": dict(action_penalty_weights) if isinstance(action_penalty_weights, dict) else {},
            "min_allowed_candidates": min_allowed,
            "applied": False,
        }
        if not enabled:
            diagnostics["reason"] = "disabled"
            return processed, diagnostics
        previous_last_relative = self._previous_last_relative_note_pitch(sampled_history)
        if previous_last_relative is None or previous_harmony is None:
            diagnostics["reason"] = "missing_previous_pitch_or_harmony"
            return processed, diagnostics
        previous_last_abs = int(previous_harmony.bar_base_pitch) + int(previous_last_relative)
        rows = []
        hard_masked: List[int] = []
        penalized: List[int] = []
        for token_id in [int(value) for value in candidates.tolist()]:
            first_relative = self._candidate_first_relative_pitch(token_id)
            if first_relative is None:
                rows.append({
                    "observation_id": int(token_id),
                    "first_relative_pitch": None,
                    "candidate_first_abs_pitch": None,
                    "boundary_jump": None,
                    "penalty": 0.0,
                    "masked": False,
                })
                continue
            candidate_first_abs = int(harmony.bar_base_pitch) + int(first_relative)
            jump = abs(candidate_first_abs - previous_last_abs)
            penalty = max(0.0, float(jump) - max_jump) * penalty_weight
            masked = False
            if penalty > 0.0:
                penalized.append(int(token_id))
                if mode == "hard_mask":
                    processed[token_id] = -np.inf
                    masked = True
                    hard_masked.append(int(token_id))
                else:
                    processed[token_id] = float(processed[token_id]) - penalty
            rows.append({
                "observation_id": int(token_id),
                "first_relative_pitch": int(first_relative),
                "candidate_first_abs_pitch": int(candidate_first_abs),
                "boundary_jump": int(jump),
                "penalty": float(penalty),
                "masked": bool(masked),
            })
        finite_after = int(np.isfinite(processed[candidates]).sum())
        fallback = None
        if mode == "hard_mask" and finite_after < min_allowed:
            fallback = "hard_mask_would_drop_below_min_allowed_candidates"
            processed = np.asarray(logits, dtype=np.float64).copy()
            for row in rows:
                token_id = int(row["observation_id"])
                penalty = float(row["penalty"])
                if penalty > 0.0:
                    processed[token_id] = float(processed[token_id]) - penalty
                    row["masked"] = False
            finite_after = int(np.isfinite(processed[candidates]).sum())
        valid_jumps = [int(row["boundary_jump"]) for row in rows if row.get("boundary_jump") is not None]
        diagnostics.update({
            "applied": True,
            "current_bar_base_pitch": int(harmony.bar_base_pitch),
            "previous_bar_base_pitch": int(previous_harmony.bar_base_pitch),
            "previous_last_relative_pitch": int(previous_last_relative),
            "previous_last_abs_pitch": int(previous_last_abs),
            "candidate_count": int(len(candidates)),
            "finite_after_count": int(finite_after),
            "penalized_count": int(len(penalized)),
            "hard_masked_count": int(len(hard_masked)),
            "max_boundary_jump": max(valid_jumps) if valid_jumps else None,
            "mean_boundary_jump": float(sum(valid_jumps) / len(valid_jumps)) if valid_jumps else 0.0,
            "fallback": fallback,
            "top_boundary_candidates": sorted(
                rows,
                key=lambda item: (
                    -1 if item.get("boundary_jump") is None else int(item["boundary_jump"]),
                    float(item.get("penalty", 0.0)),
                ),
                reverse=True,
            )[:diagnostics_top_k],
        })
        return processed, diagnostics

    def _temporal_phrase_plan(
        self,
        context: LatentTokenContext,
        current_logits: np.ndarray,
        current_candidates: np.ndarray,
        state_id: int,
        local_index: int,
        section_length: int,
        harmony: HarmonyBarPlan,
        temperature: float,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, Dict[str, Any]]]:
        if self.theme_graph is None or self.lstm_ranker is None:
            return None
        graph_config = self.theme_graph.config
        if not bool(graph_config.planning_enabled):
            return None
        remaining = max(1, int(section_length) - int(local_index))
        horizon = max(1, min(int(graph_config.planning_window_bars), remaining))
        beam_width = max(1, int(graph_config.planning_beam_width))
        branch_top_k = max(1, int(graph_config.planning_branch_top_k))
        beams = [PhraseBeam(token_ids=[], latent_sequence=[], score=0.0, first_token_id=None, steps=[])]
        for offset in range(horizon):
            next_beams: List[PhraseBeam] = []
            step_local_index = int(local_index) + int(offset)
            action_state = self.theme_graph.action_state(
                section=str(harmony.section),
                section_local_index=step_local_index,
                section_length=int(section_length),
                harmony_degree=harmony.degree,
            )
            step_temperature = graph_config.temperature_for_action(action_state.action, fallback=float(temperature))
            for beam in beams:
                beam_context = self._planned_context(context, beam, action_state, int(state_id), step_local_index)
                if offset == 0:
                    logits = np.asarray(current_logits, dtype=np.float64)
                    candidates = np.asarray(current_candidates, dtype=np.int64)
                    plan_reason = "current_step_mask"
                else:
                    all_mask = CandidateTokenMask(
                        allowed_token_ids=list(range(int(self.lstm_ranker.metadata.vocab_size))),
                        reason="temporal_phrase_planning_raw_logits",
                        diagnostics={"state_id": int(state_id), "local_index": int(step_local_index)},
                    )
                    raw_result = self.lstm_ranker.score_next(beam_context, all_mask)
                    plan_step = self.theme_graph.plan_from_logits(action_state, raw_result.logits)
                    result = self.lstm_ranker.score_next(beam_context, plan_step.mask)
                    logits = np.asarray(result.masked_logits, dtype=np.float64)
                    bias_map = self.theme_graph.memory_biases_for_state(action_state)
                    for token_id, bias in bias_map.items():
                        if 0 <= token_id < len(logits) and np.isfinite(logits[token_id]):
                            logits[token_id] += float(bias)
                    candidates = np.asarray([int(token) for token in plan_step.mask.allowed_token_ids], dtype=np.int64)
                    plan_reason = plan_step.mask.reason
                if candidates.size == 0:
                    continue
                probabilities = self._softmax_subset(logits / max(1.0e-6, float(step_temperature)), candidates)
                ordered = sorted(
                    [
                        (int(token_id), float(probability))
                        for token_id, probability in probabilities.items()
                        if float(probability) > 0.0
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )[:branch_top_k]
                for token_id, probability in ordered:
                    latent = self._latent_for_observation(int(token_id))
                    if latent is None:
                        continue
                    next_beams.append(PhraseBeam(
                        token_ids=[*beam.token_ids, int(token_id)],
                        latent_sequence=[*beam.latent_sequence, latent],
                        score=float(beam.score + math.log(max(float(probability), 1.0e-12))),
                        first_token_id=int(beam.first_token_id if beam.first_token_id is not None else token_id),
                        steps=[
                            *beam.steps,
                            {
                                "offset": int(offset),
                                "section_local_index": int(step_local_index),
                                "temporal_action": action_state.action.value,
                                "target_theme_id": action_state.target_theme_id,
                                "selected_token_id": int(token_id),
                                "step_probability": float(probability),
                                "step_temperature": float(step_temperature),
                                "plan_reason": plan_reason,
                            },
                        ],
                    ))
            if not next_beams:
                return None
            beams = sorted(next_beams, key=lambda item: item.score, reverse=True)[:beam_width]
        selected_beam_index = self._sample_beam_index(
            beams,
            temperature=max(1.0e-6, float(graph_config.planning_path_temperature)),
            rng=rng,
        )
        best = beams[selected_beam_index]
        if best.first_token_id is None:
            return None
        return int(best.first_token_id), {
            "enabled": True,
            "mode": "stochastic_phrase_beam_search",
            "score_source": "lstm_log_probability_only",
            "temperature_source": "temporal_action",
            "horizon_bars": int(horizon),
            "beam_width": int(beam_width),
            "branch_top_k": int(branch_top_k),
            "planning_path_temperature": float(graph_config.planning_path_temperature),
            "action_temperature": dict(graph_config.action_temperature),
            "selected_first_token_id": int(best.first_token_id),
            "selected_path_rank": int(selected_beam_index + 1),
            "best_path_score": float(best.score),
            "best_path_token_ids": [int(token_id) for token_id in best.token_ids],
            "best_path_steps": best.steps,
            "beam_count": int(len(beams)),
            "top_paths": [
                {
                    "rank": int(index + 1),
                    "score": float(beam.score),
                    "token_ids": [int(token_id) for token_id in beam.token_ids],
                }
                for index, beam in enumerate(beams[: max(0, int(graph_config.diagnostics_top_k))])
            ],
        }

    def _sample_beam_index(
        self,
        beams: Sequence[PhraseBeam],
        temperature: float,
        rng: np.random.Generator,
    ) -> int:
        if len(beams) <= 1:
            return 0
        scores = np.asarray([float(beam.score) for beam in beams], dtype=np.float64)
        scaled = scores / max(1.0e-6, float(temperature))
        scaled = scaled - float(np.max(scaled))
        weights = np.exp(scaled)
        total = float(weights.sum())
        if total <= 0.0:
            return 0
        return int(rng.choice(len(beams), p=weights / total))

    def _planned_context(
        self,
        context: LatentTokenContext,
        beam: PhraseBeam,
        action_state: Any,
        state_id: int,
        local_index: int,
    ) -> LatentTokenContext:
        context_size = int(self.lstm_ranker.metadata.config.context_size) if self.lstm_ranker else len(context.token_ids)
        token_ids = [*context.token_ids, *beam.token_ids][-context_size:]
        latents = [*context.latent_sequence, *beam.latent_sequence][-context_size:]
        return LatentTokenContext(
            token_ids=[int(token_id) for token_id in token_ids],
            latent_sequence=[[float(value) for value in latent] for latent in latents],
            action_state=action_state,
            metadata={
                **context.metadata,
                "hidden_state": int(state_id),
                "section_local_index": int(local_index),
                "temporal_action": action_state.action.value,
            },
        )

    def _opening_seeded_context(
        self,
        context: LatentTokenContext,
        output_bar_index: int,
        rng: np.random.Generator,
    ) -> LatentTokenContext:
        config = self._opening_seed_config()
        if self._opening_seed_mode() != "single_bar_pool":
            return context
        if int(output_bar_index) >= max(0, int(config.get("bootstrap_bars", 1))):
            return context
        if self.bundle.encoder_model is None:
            return context
        selected = self.opening_seed_selector.select_seed_context(
            self.bundle.encoder_model.metadata,
            rng,
        )
        if selected is None:
            return context
        symbol_id = int(selected.get("symbol_id", selected.get("token_id", -1)))
        latent = self._latent_for_observation(symbol_id)
        if latent is None:
            return context
        return LatentTokenContext(
            token_ids=list(context.token_ids),
            latent_sequence=[list(row) for row in context.latent_sequence],
            action_state=context.action_state,
            metadata={
                **context.metadata,
                "latent_context_seed": [float(value) for value in latent],
                "latent_context_seed_source": "opening_seed_single_bar_pool",
                "opening_seed": selected,
            },
        )

    def _theme_pipeline_selection(
        self,
        output_bar_index: int,
        candidates: np.ndarray,
        logits: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        if self._opening_seed_mode() != "theme_pipeline":
            return None
        pipeline = self.opening_theme_pipeline
        if not pipeline:
            return None
        symbol_ids = [int(value) for value in pipeline.get("symbol_ids", [])]
        if int(output_bar_index) < 0 or int(output_bar_index) >= len(symbol_ids):
            return None
        selected = int(symbol_ids[int(output_bar_index)])
        if selected < 0 or selected >= len(logits):
            return None
        allowed = {int(value) for value in candidates.tolist()}
        return {
            "enabled": True,
            "strategy": "opening_theme_pipeline",
            "mode": "theme_pipeline",
            "selected_symbol_id": int(selected),
            "pipeline_position": int(output_bar_index),
            "pipeline_length": int(len(symbol_ids)),
            "selected_symbol_allowed_by_graph": bool(selected in allowed),
            "selected_logit_finite_before_force": bool(np.isfinite(logits[selected])),
            "pipeline": {
                key: value
                for key, value in pipeline.items()
                if key != "bars"
            },
        }

    def _previous_last_relative_note_pitch(self, sampled_history: Sequence[SampledBar]) -> Optional[int]:
        if not sampled_history:
            return None
        return self._last_relative_note_pitch(sampled_history[-1].relative_tokens)

    def _candidate_first_relative_pitch(self, observation_id: int) -> Optional[int]:
        if self.bundle.encoder_model is None:
            return None
        try:
            entry = self.bundle.encoder_model.codebook_entry_for_symbol(int(observation_id))
        except Exception:
            return None
        for token in entry.relative_tokens:
            if int(token) >= 0:
                return int(token)
        return None

    def _lstm_context(
        self,
        sampled_history: Sequence[SampledBar],
        state_id: int,
        local_index: int,
    ) -> LatentTokenContext:
        token_ids: List[int] = []
        latents: List[List[float]] = []
        for event in sampled_history[-int(self.lstm_ranker.metadata.config.context_size):] if self.lstm_ranker else []:
            latent = self._latent_for_observation(int(event.observation_id))
            if latent is None:
                continue
            token_ids.append(int(event.observation_id))
            latents.append(latent)
        return LatentTokenContext(
            token_ids=token_ids,
            latent_sequence=latents,
            metadata={
                "hidden_state": int(state_id),
                "section_local_index": int(local_index),
            },
        )

    def _latent_for_observation(self, observation_id: int) -> Optional[List[float]]:
        if self.bundle.encoder_model is None:
            return None
        try:
            entry = self.bundle.encoder_model.codebook_entry_for_symbol(int(observation_id))
        except KeyError:
            return None
        if entry.latent_vector is None:
            return None
        return [float(value) for value in entry.latent_vector]

    def _lstm_rank_rows(
        self,
        candidates: np.ndarray,
        logits: np.ndarray,
        form_prior_probs: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        probs = self._softmax_subset(logits, candidates)
        ordered = sorted(
            [
                (
                    int(obs),
                    float(logits[int(obs)]),
                    float(probs.get(int(obs), 0.0)),
                    float(form_prior_probs[int(obs)]),
                )
                for obs in candidates
                if math.isfinite(float(logits[int(obs)]))
            ],
            key=lambda item: item[2],
            reverse=True,
        )
        return [
            {
                "rank": index + 1,
                "observation_id": obs,
                "lstm_logit": logit,
                "lstm_probability": probability,
                "form_prior_probability": form_prior_probability,
            }
            for index, (obs, logit, probability, form_prior_probability) in enumerate(ordered[: max(1, int(top_k))])
        ]

    def _probability_from_logits(self, logits: np.ndarray, observation_id: int, candidates: np.ndarray) -> float:
        return float(self._softmax_subset(logits, candidates).get(int(observation_id), 0.0))

    def _softmax_subset(self, logits: np.ndarray, candidates: np.ndarray) -> Dict[int, float]:
        values = np.asarray([float(logits[int(obs)]) for obs in candidates], dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return {}
        max_value = float(np.max(values[finite]))
        weights = np.zeros_like(values)
        weights[finite] = np.exp(values[finite] - max_value)
        total = float(weights.sum())
        if total <= 0:
            return {}
        return {
            int(obs): float(weights[index] / total)
            for index, obs in enumerate(candidates)
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
            return selected, "same_codebook_and_kmeans"

        same_edit = self._candidate_bars(
            lambda bar: (
                int(bar.codebook_id) == int(source_event.codebook_id)
                and self._bar_position_context(bar) == target_context
            )
        )
        selected = self._choose_non_identical(same_edit, source_event, rng)
        if selected is not None:
            return selected, "same_codebook"

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
        mode = str(ConfigView(self.config).section("decoder").get("source_reuse_mode", "sampled_path"))
        if mode not in {"none", "sampled_path", "anchor_resample"}:
            raise ValueError("decoder.source_reuse_mode must be 'none', 'sampled_path', or 'anchor_resample'.")
        decoder_backend = str(ConfigView(self.config).section("decoder").get("backend", "temporal_lstm"))
        if decoder_backend in {"lstm_token", "lstm_rerank", "temporal_lstm"} and mode == "anchor_resample":
            return "sampled_path"
        return mode

    def _load_lstm_ranker(self) -> Optional[LSTMTokenModel]:
        decoder_config = ConfigView(self.config).section("decoder")
        if str(decoder_config.get("backend", "temporal_lstm")) not in {"lstm_token", "lstm_rerank", "temporal_lstm"}:
            return None
        model_dir = decoder_config.get("model_dir")
        if not model_dir:
            raise ValueError("decoder.backend is LSTM-based but the generation model_dir was not attached to decoder config.")
        device = decoder_config.get("lstm_device")
        return LSTMTokenModel.load(Path(str(model_dir)), device=str(device) if device else None)

    def _load_dvae_decoder(self) -> Optional[TrainedDVAEDecoderRuntime]:
        decoder_config = ConfigView(self.config).section("decoder")
        if str(decoder_config.get("backend", "temporal_lstm")) not in {"lstm_token", "lstm_rerank", "temporal_lstm"}:
            return None
        path = decoder_config.get("vae_decoder_path")
        if not path and decoder_config.get("model_dir"):
            path = str(Path(str(decoder_config["model_dir"])) / "vae_decoder.pt")
        if not path:
            raise ValueError(
                "LSTM-based generation requires <model-dir>/vae_decoder.pt. "
                "Retrain the main model if that file is missing."
            )
        device = decoder_config.get("vae_decoder_device") or decoder_config.get("lstm_device")
        return TrainedDVAEDecoderRuntime(Path(str(path)), device=str(device) if device else None)

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

    def _sample_bar(self, observation_id: int, rng: np.random.Generator) -> BarRecord:
        pool = self.bundle.observation_to_bars.get(int(observation_id), [])
        if not pool:
            raise ValueError(f"No bar pool for observation_id {observation_id}")
        return pool[int(rng.integers(0, len(pool)))]


