#!/usr/bin/env python3
"""Form-aware left-to-right HMM training and sampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigLoader, ConfigView
from core_data import ObservationVocab, SongRecord


EPS = 1e-12


@dataclass(frozen=True)
class FormSection:
    name: str
    length: int
    source: Optional[str] = None
    pitch_offset: int = 0
    cadence: str = "none"
    start_degree: Optional[str] = None


@dataclass(frozen=True)
class FormTemplate:
    name: str
    sections: List[FormSection]


@dataclass(frozen=True)
class FormHMMConfig:
    max_iter: int = 80
    tol: float = 1e-4
    emission_smoothing: float = 0.2
    transition_smoothing: float = 0.5
    random_seed: int = 42
    max_forward_jump: int = 1
    allow_self_loop: bool = True
    warm_start_strength: float = 2.0


class FormTemplateLibrary:
    """Load form section templates from style config."""

    def __init__(self, templates: Dict[str, FormTemplate]) -> None:
        self.templates = templates

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "FormTemplateLibrary":
        forms = ConfigView(config).section("forms")
        templates = {}
        for form_name, payload in forms.items():
            sections = [
                FormSection(
                    name=str(item["name"]),
                    length=int(item["length"]),
                    source=item.get("source"),
                    pitch_offset=int(item.get("pitch_offset", 0) or 0),
                    cadence=str(item.get("cadence", "none")),
                    start_degree=item.get("start_degree"),
                )
                for item in payload.get("sections", [])
            ]
            templates[str(form_name)] = FormTemplate(str(form_name), sections)
        return cls(templates)

    def require(self, form_name: str) -> FormTemplate:
        if form_name not in self.templates:
            raise ValueError(f"Unknown form '{form_name}'. Available: {sorted(self.templates)}")
        return self.templates[form_name]


class LeftToRightFormHMM:
    """Baum-Welch HMM with a hard left-to-right transition mask."""

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        config: FormHMMConfig,
        name: str = "",
        state_role_map: Optional[Dict[int, str]] = None,
        section_lengths: Optional[List[int]] = None,
    ) -> None:
        self.name = name
        self.config = config
        self.n_states = n_states
        self.n_observations = n_observations
        self.section_lengths = section_lengths or [1] * n_states
        self.startprob = np.zeros(self.n_states, dtype=np.float64)
        self.transmat = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        self.emissionprob = np.zeros((self.n_states, self.n_observations), dtype=np.float64)
        self.state_role_map: Dict[int, str] = state_role_map or {
            index: f"State_{index}" for index in range(n_states)
        }
        self.training_log: List[Dict[str, float]] = []
        self.diagnostics: Dict[str, Any] = {}

    @classmethod
    def from_style_config(
        cls,
        config: Dict[str, Any],
        n_states: int,
        n_observations: int,
        name: str = "",
        state_role_map: Optional[Dict[int, str]] = None,
        section_lengths: Optional[List[int]] = None,
    ) -> "LeftToRightFormHMM":
        section = ConfigView(config).section("section_hmm")
        return cls(n_states, n_observations, FormHMMConfig(
            max_iter=int(section.get("max_iter", 80)),
            tol=float(section.get("tol", 1e-4)),
            emission_smoothing=float(section.get("emission_smoothing", 0.2)),
            transition_smoothing=float(section.get("transition_smoothing", 0.5)),
            random_seed=int(section.get("random_seed", 42)),
            max_forward_jump=int(section.get("max_forward_jump", 1)),
            allow_self_loop=bool(section.get("allow_self_loop", True)),
            warm_start_strength=float(section.get("warm_start_strength", 2.0)),
        ), name=name, state_role_map=state_role_map, section_lengths=section_lengths)

    def fit(self, sequences: Sequence[Sequence[int]]) -> "LeftToRightFormHMM":
        clean = [np.asarray(seq, dtype=int) for seq in sequences if len(seq) > 0]
        if not clean:
            raise ValueError(f"No observation sequences for HMM '{self.name}'.")
        self._initialize(clean)
        previous_ll: Optional[float] = None
        mask = self._transition_mask()
        for iteration in range(self.config.max_iter):
            start_counts = np.full(self.n_states, EPS)
            trans_counts = np.full((self.n_states, self.n_states), self.config.transition_smoothing)
            emit_counts = np.full((self.n_states, self.n_observations), self.config.emission_smoothing)
            total_ll = 0.0
            for observations in clean:
                alpha, scales, log_likelihood = self._forward(observations)
                beta = self._backward(observations, scales)
                gamma = alpha * beta
                gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), EPS)
                start_counts += gamma[0]
                for t, obs in enumerate(observations):
                    emit_counts[:, obs] += gamma[t]
                for t in range(len(observations) - 1):
                    xi = (
                        alpha[t, :, None]
                        * self.transmat
                        * self.emissionprob[:, observations[t + 1]][None, :]
                        * beta[t + 1, :][None, :]
                    )
                    xi /= max(float(xi.sum()), EPS)
                    trans_counts += xi
                total_ll += log_likelihood
            self.startprob = self._normalize(start_counts)
            self.transmat = self._normalize_rows(trans_counts * mask, mask)
            self.emissionprob = self._normalize_rows(emit_counts)
            delta = 0.0 if previous_ll is None else total_ll - previous_ll
            self.training_log.append({"iteration": float(iteration), "log_likelihood": float(total_ll), "delta": float(delta)})
            if previous_ll is not None and abs(delta) < self.config.tol:
                break
            previous_ll = total_ll
        self._build_diagnostics(clean)
        return self

    def sample_from_state(self, state: int, rng: np.random.Generator) -> tuple[int, float]:
        probs = self.emissionprob[int(state)]
        obs = int(rng.choice(self.n_observations, p=probs))
        return obs, float(probs[obs])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": asdict(self.config),
            "n_states": self.n_states,
            "n_observations": self.n_observations,
            "section_lengths": self.section_lengths,
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "emissionprob": self.emissionprob.tolist(),
            "state_role_map": {str(k): v for k, v in self.state_role_map.items()},
            "training_log": self.training_log,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LeftToRightFormHMM":
        model = cls(
            int(payload["n_states"]),
            int(payload["n_observations"]),
            FormHMMConfig(**payload["config"]),
            name=str(payload.get("name", "")),
            state_role_map={int(k): v for k, v in payload.get("state_role_map", {}).items()},
            section_lengths=[int(x) for x in payload.get("section_lengths", [])],
        )
        model.startprob = np.asarray(payload["startprob"], dtype=np.float64)
        model.transmat = np.asarray(payload["transmat"], dtype=np.float64)
        model.emissionprob = np.asarray(payload["emissionprob"], dtype=np.float64)
        model.state_role_map = {int(k): v for k, v in payload.get("state_role_map", {}).items()}
        model.training_log = payload.get("training_log", [])
        model.diagnostics = payload.get("diagnostics", {})
        return model

    def _initialize(self, sequences: Sequence[np.ndarray]) -> None:
        self.startprob = np.zeros(self.n_states, dtype=np.float64)
        self.startprob[0] = 1.0
        mask = self._transition_mask()
        self.transmat = self._normalize_rows(mask.copy(), mask)
        self.emissionprob = np.full(
            (self.n_states, self.n_observations),
            self.config.emission_smoothing,
            dtype=np.float64,
        )
        for seq in sequences:
            for state, start, end in self._template_regions(len(seq)):
                for obs in seq[start:end]:
                    self.emissionprob[state, obs] += self.config.warm_start_strength
        self.emissionprob = self._normalize_rows(self.emissionprob)

    def _template_regions(self, sequence_length: int) -> List[tuple[int, int, int]]:
        total_template = sum(self.section_lengths)
        if total_template <= 0:
            total_template = sequence_length
        regions = []
        cursor = 0
        for state, section_length in enumerate(self.section_lengths):
            length = int(round(sequence_length * section_length / total_template))
            end = min(sequence_length, cursor + max(1, length))
            if state == self.n_states - 1:
                end = sequence_length
            regions.append((state, cursor, end))
            cursor = end
        return regions

    def _transition_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for i in range(self.n_states):
            start = i if self.config.allow_self_loop else i + 1
            end = min(self.n_states, i + self.config.max_forward_jump + 1)
            mask[i, start:end] = 1.0
        return mask

    def _forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        alpha = np.zeros((len(observations), self.n_states), dtype=np.float64)
        scales = np.zeros(len(observations), dtype=np.float64)
        alpha[0] = self.startprob * self.emissionprob[:, observations[0]]
        scales[0] = max(float(alpha[0].sum()), EPS)
        alpha[0] /= scales[0]
        for t in range(1, len(observations)):
            alpha[t] = alpha[t - 1] @ self.transmat * self.emissionprob[:, observations[t]]
            scales[t] = max(float(alpha[t].sum()), EPS)
            alpha[t] /= scales[t]
        return alpha, scales, float(np.log(scales).sum())

    def _backward(self, observations: np.ndarray, scales: np.ndarray) -> np.ndarray:
        beta = np.zeros((len(observations), self.n_states), dtype=np.float64)
        beta[-1] = 1.0
        for t in range(len(observations) - 2, -1, -1):
            beta[t] = self.transmat @ (self.emissionprob[:, observations[t + 1]] * beta[t + 1])
            beta[t] /= max(float(scales[t + 1]), EPS)
        return beta

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        total = float(values.sum())
        return values / max(total, EPS)

    def _normalize_rows(self, matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        result = np.asarray(matrix, dtype=np.float64).copy()
        if mask is not None:
            result *= mask
        for row in range(result.shape[0]):
            total = float(result[row].sum())
            if total <= EPS:
                if mask is None:
                    result[row] = 1.0 / result.shape[1]
                else:
                    allowed = mask[row] > 0
                    result[row, allowed] = 1.0 / max(1, int(allowed.sum()))
            else:
                result[row] /= total
        return result

    def _build_diagnostics(self, sequences: Sequence[np.ndarray]) -> None:
        self.diagnostics = {
            "name": self.name,
            "state_role_map": {str(k): v for k, v in self.state_role_map.items()},
            "section_lengths": self.section_lengths,
            "transition_mask": self._transition_mask().tolist(),
            "transition_matrix": self.transmat.tolist(),
            "emission_entropy": [
                float(-np.sum(row * np.log(np.maximum(row, EPS))))
                for row in self.emissionprob
            ],
            "sequence_count": len(sequences),
            "training_log": self.training_log,
        }


class FormHMMTrainer:
    """Train one left-to-right HMM per form."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.library = FormTemplateLibrary.from_style_config(config)
        self.diagnostics: Dict[str, Any] = {}

    def train(self, songs: Sequence[SongRecord], vocab: ObservationVocab) -> Dict[str, LeftToRightFormHMM]:
        songs_by_form: Dict[str, List[SongRecord]] = {}
        for song in songs:
            form_name = song.form or "ternary"
            songs_by_form.setdefault(form_name, []).append(song)
        models = {}
        for form_name, form_songs in songs_by_form.items():
            template = self.library.require(form_name)
            sequences = [
                [int(bar.observation_id) for bar in song.bars if bar.observation_id is not None]
                for song in form_songs
            ]
            model = LeftToRightFormHMM.from_style_config(
                self.config,
                len(template.sections),
                len(vocab.composite_to_observation),
                name=form_name,
                state_role_map={
                    index: section.name for index, section in enumerate(template.sections)
                },
                section_lengths=[section.length for section in template.sections],
            ).fit(sequences)
            models[form_name] = model
            self.diagnostics[form_name] = model.diagnostics
        return models


class FormHMMCLI:
    """Standalone CLI for training form HMMs from clustered songs."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train form-aware HMM models.")
        parser.add_argument("--clustered-songs", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        payload = json.loads(args.clustered_songs.read_text(encoding="utf-8"))
        songs = [SongRecord.from_dict(item) for item in payload.get("songs", [])]
        vocab = ObservationVocab.from_dict(payload["observation_vocab"])
        trainer = FormHMMTrainer(config)
        models = trainer.train(songs, vocab)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({name: model.to_dict() for name, model in models.items()}, indent=2),
            encoding="utf-8",
        )
        if args.diagnostics_output:
            args.diagnostics_output.write_text(json.dumps(trainer.diagnostics, indent=2), encoding="utf-8")
        print(f"Wrote {len(models)} form HMM models -> {args.output}")


def main() -> None:
    FormHMMCLI().run()


if __name__ == "__main__":
    main()
