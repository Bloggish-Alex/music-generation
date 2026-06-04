#!/usr/bin/env python3
"""Persistence and rendering helpers for the HMM experiment model."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigView
from grid_tokenizer import BarGrid
from section_hmm import DiscreteSectionHMM


@dataclass(frozen=True)
class MidiRenderConfig:
    tempo: int = 120
    time_signature_num: int = 4
    time_signature_den: int = 4
    velocity: int = 72
    channel: int = 0


class HMMMusicModel:
    """Saved model bundle: config, bar prototypes, labels, and HMM matrices."""

    def __init__(
        self,
        config: Dict[str, Any],
        hmm: DiscreteSectionHMM,
        prototypes: Dict[int, List[int]],
        label_to_source_indices: Dict[int, List[int]],
        label_to_bars: Dict[int, List[List[int]]],
        training_summary: Dict[str, Any],
    ) -> None:
        self.config = config
        self.hmm = hmm
        self.prototypes = prototypes
        self.label_to_source_indices = label_to_source_indices
        self.label_to_bars = label_to_bars
        self.training_summary = training_summary

    @classmethod
    def from_training(
        cls,
        config: Dict[str, Any],
        bars: Sequence[BarGrid],
        labels: Sequence[int],
        hmm: DiscreteSectionHMM,
    ) -> "HMMMusicModel":
        grouped: Dict[int, List[BarGrid]] = defaultdict(list)
        source_indices: Dict[int, List[int]] = defaultdict(list)
        for bar, label in zip(bars, labels):
            grouped[int(label)].append(bar)
            source_indices[int(label)].append(int(bar.source_index))
        prototypes = {}
        label_to_bars = {}
        for label, label_bars in grouped.items():
            prototype = Counter(tuple(bar.tokens) for bar in label_bars).most_common(1)[0][0]
            prototypes[label] = list(prototype)
            label_to_bars[label] = [list(bar.tokens) for bar in label_bars]
        summary = {
            "n_bars": len(bars),
            "n_bar_labels": len(grouped),
            "label_counts": {str(label): len(items) for label, items in grouped.items()},
        }
        return cls(config, hmm, prototypes, dict(source_indices), label_to_bars, summary)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "hmm": self.hmm.to_dict(),
            "prototypes": {str(k): v for k, v in self.prototypes.items()},
            "label_to_source_indices": {str(k): v for k, v in self.label_to_source_indices.items()},
            "label_to_bars": {str(k): v for k, v in self.label_to_bars.items()},
            "training_summary": self.training_summary,
        }
        (model_dir / "hmm_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "HMMMusicModel":
        payload = json.loads((Path(model_dir) / "hmm_model.json").read_text(encoding="utf-8"))
        return cls(
            config=payload["config"],
            hmm=_load_hmm(payload["hmm"]),
            prototypes={int(k): [int(x) for x in v] for k, v in payload["prototypes"].items()},
            label_to_source_indices={
                int(k): [int(x) for x in v] for k, v in payload["label_to_source_indices"].items()
            },
            label_to_bars={
                int(k): [[int(x) for x in tokens] for tokens in bars]
                for k, bars in payload.get("label_to_bars", {}).items()
            },
            training_summary=payload.get("training_summary", {}),
        )


class HMMGenerator:
    """Sample section/bar-label timelines and render prototype grids to MIDI."""

    def __init__(self, model: HMMMusicModel) -> None:
        self.model = model

    def generate(self, measures: int, seed: Optional[int] = None) -> Dict[str, Any]:
        sample = self._sample_timeline(measures, seed)
        rng = np.random.default_rng(seed)
        bars = []
        previous_tokens: Optional[List[int]] = None
        for index, (state, label) in enumerate(zip(sample["section_states"], sample["bar_labels"])):
            tokens = self._sample_bar_tokens(int(label), rng, previous_tokens)
            previous_tokens = tokens
            bars.append({
                "bar_index": index,
                "section_state": int(state),
                "bar_label": int(label),
                "tokens": [int(x) for x in tokens],
            })
        return {
            "requested_measures": measures,
            "seed": seed,
            "training_summary": self.model.training_summary,
            "sample": bars,
            "hmm": {
                "startprob": self.model.hmm.startprob.tolist(),
                "transmat": self.model.hmm.transmat.tolist(),
                "emissionprob": self.model.hmm.emissionprob.tolist(),
            },
        }

    def _sample_timeline(self, measures: int, seed: Optional[int]) -> Dict[str, List[int]]:
        rng = np.random.default_rng(self.model.hmm.config.random_seed if seed is None else seed)
        generation_config = ConfigView(self.model.config).section("hmm_generation")
        transition_temperature = float(generation_config.get("transition_temperature", 1.0))
        emission_temperature = float(generation_config.get("emission_temperature", 1.0))
        max_same_label_run = generation_config.get("max_same_label_run")
        max_same_state_run = generation_config.get("max_same_state_run")
        max_same_label_run = int(max_same_label_run) if max_same_label_run is not None else None
        max_same_state_run = int(max_same_state_run) if max_same_state_run is not None else None

        states: List[int] = []
        labels: List[int] = []
        state = self._choice(rng, self.model.hmm.startprob, temperature=transition_temperature)
        state_run = 0
        label_run = 0
        previous_state: Optional[int] = None
        previous_label: Optional[int] = None

        for _ in range(measures):
            if state == previous_state:
                state_run += 1
            else:
                state_run = 1
            emission = self.model.hmm.emissionprob[state].copy()
            if (
                previous_label is not None
                and max_same_label_run is not None
                and label_run >= max_same_label_run
                and len(emission) > 1
            ):
                emission[previous_label] = 0.0
            label = self._choice(rng, emission, temperature=emission_temperature)
            if label == previous_label:
                label_run += 1
            else:
                label_run = 1

            states.append(int(state))
            labels.append(int(label))

            transition = self.model.hmm.transmat[state].copy()
            if max_same_state_run is not None and state_run >= max_same_state_run and len(transition) > 1:
                transition[state] = 0.0
            previous_state = state
            previous_label = label
            state = self._choice(rng, transition, temperature=transition_temperature)
        return {"section_states": states, "bar_labels": labels}

    def _choice(self, rng: np.random.Generator, probabilities: np.ndarray, temperature: float) -> int:
        probs = np.asarray(probabilities, dtype=np.float64).copy()
        probs = np.maximum(probs, 0.0)
        if probs.sum() <= 0:
            probs = np.ones_like(probs, dtype=np.float64)
        if temperature > 0 and abs(temperature - 1.0) > 1e-9:
            probs = np.power(np.maximum(probs, 1e-12), 1.0 / temperature)
        probs = probs / probs.sum()
        return int(rng.choice(len(probs), p=probs))

    def _sample_bar_tokens(
        self,
        label: int,
        rng: np.random.Generator,
        previous_tokens: Optional[List[int]],
    ) -> List[int]:
        generation_config = ConfigView(self.model.config).section("hmm_generation")
        avoid_exact = bool(generation_config.get("avoid_exact_bar_repeat", True))
        candidates = self.model.label_to_bars.get(label) or [self.model.prototypes.get(label)]
        candidates = [tokens for tokens in candidates if tokens is not None]
        if not candidates:
            candidates = list(self.model.prototypes.values())
        if avoid_exact and previous_tokens is not None and len(candidates) > 1:
            filtered = [tokens for tokens in candidates if list(tokens) != previous_tokens]
            if filtered:
                candidates = filtered
        return [int(x) for x in candidates[int(rng.integers(0, len(candidates)))]]

    def write_json(self, generation: Dict[str, Any], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(generation, indent=2), encoding="utf-8")

    def write_midi(
        self,
        generation: Dict[str, Any],
        output_path: str | Path,
        render_config: MidiRenderConfig,
    ) -> None:
        import mido

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ticks_per_beat = 480
        bar_length_ql = render_config.time_signature_num * (4.0 / render_config.time_signature_den)
        steps_per_bar = len(generation["sample"][0]["tokens"]) if generation["sample"] else 16
        step_ql = bar_length_ql / steps_per_bar
        us_per_beat = int(60_000_000 / render_config.tempo)

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))
        track.append(mido.MetaMessage(
            "time_signature",
            numerator=render_config.time_signature_num,
            denominator=render_config.time_signature_den,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        ))

        events: List[tuple[int, str, int, int]] = []
        for bar in generation["sample"]:
            bar_index = int(bar["bar_index"])
            for start_slot, pitch, duration_slots in self._tokens_to_note_spans(bar["tokens"]):
                start_ql = bar_index * bar_length_ql + start_slot * step_ql
                end_ql = start_ql + duration_slots * step_ql
                start_tick = int(round(start_ql * ticks_per_beat))
                end_tick = int(round(end_ql * ticks_per_beat))
                if end_tick > start_tick:
                    events.append((start_tick, "on", int(pitch), render_config.velocity))
                    events.append((end_tick, "off", int(pitch), 0))

        events.sort(key=lambda item: (item[0], 0 if item[1] == "off" else 1))
        previous_tick = 0
        for tick, kind, pitch, velocity in events:
            delta = max(0, tick - previous_tick)
            msg_type = "note_on" if kind == "on" else "note_off"
            track.append(mido.Message(
                msg_type,
                note=pitch,
                velocity=velocity,
                channel=render_config.channel,
                time=delta,
            ))
            previous_tick = tick
        mid.save(str(output_path))

    def _tokens_to_note_spans(self, tokens: Sequence[int]) -> List[tuple[int, int, int]]:
        spans: List[tuple[int, int, int]] = []
        current_pitch: Optional[int] = None
        current_start = 0
        current_len = 0
        for slot, token in enumerate(tokens):
            token = int(token)
            if token > 0:
                if current_pitch is not None:
                    spans.append((current_start, current_pitch, current_len))
                current_pitch = token
                current_start = slot
                current_len = 1
            elif token == 0 and current_pitch is not None:
                current_len += 1
            else:
                if current_pitch is not None:
                    spans.append((current_start, current_pitch, current_len))
                current_pitch = None
                current_len = 0
        if current_pitch is not None:
            spans.append((current_start, current_pitch, current_len))
        return spans


def _load_hmm(payload: Dict[str, Any]) -> Any:
    backend = payload.get("backend", "numpy")
    if backend == "hmmlearn":
        from hmmlearn_section_hmm import HmmlearnSectionHMM

        return HmmlearnSectionHMM.from_dict(payload)
    return DiscreteSectionHMM.from_dict(payload)
