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
    backend: str = "numpy"
    max_iter: int = 80
    tol: float = 1e-4
    emission_smoothing: float = 0.2
    transition_smoothing: float = 0.5
    duration_smoothing: float = 0.2
    random_seed: int = 42
    max_forward_jump: int = 1
    allow_self_loop: bool = True
    warm_start_strength: float = 2.0
    duration_warm_start_strength: float = 8.0
    max_duration: Optional[int] = None
    require_final_state: bool = True
    max_expected_segment_operations: int = 20_000_000


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
            backend=str(section.get("backend", "numpy")),
            max_iter=int(section.get("max_iter", 80)),
            tol=float(section.get("tol", 1e-4)),
            emission_smoothing=float(section.get("emission_smoothing", 0.2)),
            transition_smoothing=float(section.get("transition_smoothing", 0.5)),
            duration_smoothing=float(section.get("duration_smoothing", 0.2)),
            random_seed=int(section.get("random_seed", 42)),
            max_forward_jump=int(section.get("max_forward_jump", 1)),
            allow_self_loop=bool(section.get("allow_self_loop", True)),
            warm_start_strength=float(section.get("warm_start_strength", 2.0)),
            duration_warm_start_strength=float(section.get("duration_warm_start_strength", 8.0)),
            max_duration=(
                int(section["max_duration"])
                if section.get("max_duration") is not None
                else None
            ),
            require_final_state=bool(section.get("require_final_state", True)),
            max_expected_segment_operations=int(section.get("max_expected_segment_operations", 20_000_000)),
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
            "model_type": "left_to_right_hmm",
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
            "model_type": "left_to_right_hmm",
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


class ExplicitDurationFormHSMM:
    """Discrete-observation HSMM trained with explicit-duration Baum-Welch."""

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
        self.max_duration = max(1, int(config.max_duration or max(self.section_lengths or [1])))
        self.startprob = np.zeros(self.n_states, dtype=np.float64)
        self.transmat = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        self.durationprob = np.zeros((self.n_states, self.max_duration + 1), dtype=np.float64)
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
    ) -> "ExplicitDurationFormHSMM":
        section = ConfigView(config).section("section_hmm")
        hmm_config = FormHMMConfig(
            backend=str(section.get("backend", "hsmm")),
            max_iter=int(section.get("max_iter", 80)),
            tol=float(section.get("tol", 1e-4)),
            emission_smoothing=float(section.get("emission_smoothing", 0.2)),
            transition_smoothing=float(section.get("transition_smoothing", 0.5)),
            duration_smoothing=float(section.get("duration_smoothing", 0.2)),
            random_seed=int(section.get("random_seed", 42)),
            max_forward_jump=int(section.get("max_forward_jump", 1)),
            allow_self_loop=bool(section.get("hsmm_allow_self_loop", False)),
            warm_start_strength=float(section.get("warm_start_strength", 2.0)),
            duration_warm_start_strength=float(section.get("duration_warm_start_strength", 8.0)),
            max_duration=(
                int(section["max_duration"])
                if section.get("max_duration") is not None
                else None
            ),
            require_final_state=bool(section.get("require_final_state", True)),
            max_expected_segment_operations=int(section.get("max_expected_segment_operations", 20_000_000)),
        )
        return cls(
            n_states,
            n_observations,
            hmm_config,
            name=name,
            state_role_map=state_role_map,
            section_lengths=section_lengths,
        )

    def fit(self, sequences: Sequence[Sequence[int]]) -> "ExplicitDurationFormHSMM":
        clean = [np.asarray(seq, dtype=int) for seq in sequences if len(seq) > 0]
        if not clean:
            raise ValueError(f"No observation sequences for HSMM '{self.name}'.")
        self._validate_sequence_lengths(clean)
        self._initialize(clean)
        previous_ll: Optional[float] = None
        mask = self._transition_mask()
        for iteration in range(self.config.max_iter):
            start_counts = np.full(self.n_states, EPS)
            trans_counts = np.full((self.n_states, self.n_states), self.config.transition_smoothing)
            duration_counts = np.full(
                (self.n_states, self.max_duration + 1),
                self.config.duration_smoothing,
                dtype=np.float64,
            )
            duration_counts[:, 0] = 0.0
            emit_counts = np.full(
                (self.n_states, self.n_observations),
                self.config.emission_smoothing,
                dtype=np.float64,
            )
            total_ll = 0.0
            for observations in clean:
                alpha, beta, log_prob, segment_loglik = self._forward_backward(observations)
                total_ll += log_prob
                self._accumulate_expectations(
                    observations,
                    alpha,
                    beta,
                    log_prob,
                    segment_loglik,
                    start_counts,
                    trans_counts,
                    duration_counts,
                    emit_counts,
                )
            self.startprob = self._normalize(start_counts)
            self.transmat = self._normalize_rows(trans_counts * mask, mask)
            self.durationprob = self._normalize_duration_rows(duration_counts)
            self.emissionprob = self._normalize_rows(emit_counts)
            delta = 0.0 if previous_ll is None else total_ll - previous_ll
            self.training_log.append({
                "iteration": float(iteration),
                "log_likelihood": float(total_ll),
                "delta": float(delta),
            })
            if previous_ll is not None and abs(delta) < self.config.tol:
                break
            previous_ll = total_ll
        self._build_diagnostics(clean)
        return self

    def sample_from_state(self, state: int, rng: np.random.Generator) -> tuple[int, float]:
        probs = self.emissionprob[int(state)]
        obs = int(rng.choice(self.n_observations, p=probs))
        return obs, float(probs[obs])

    def sample_duration(self, state: int, rng: np.random.Generator) -> tuple[int, float]:
        probs = self.durationprob[int(state)].copy()
        probs[0] = 0.0
        probs /= max(float(probs.sum()), EPS)
        duration = int(rng.choice(len(probs), p=probs))
        return duration, float(probs[duration])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "explicit_duration_hsmm",
            "name": self.name,
            "config": asdict(self.config),
            "n_states": self.n_states,
            "n_observations": self.n_observations,
            "section_lengths": self.section_lengths,
            "max_duration": self.max_duration,
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "durationprob": self.durationprob.tolist(),
            "emissionprob": self.emissionprob.tolist(),
            "state_role_map": {str(k): v for k, v in self.state_role_map.items()},
            "training_log": self.training_log,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExplicitDurationFormHSMM":
        model = cls(
            int(payload["n_states"]),
            int(payload["n_observations"]),
            FormHMMConfig(**payload["config"]),
            name=str(payload.get("name", "")),
            state_role_map={int(k): v for k, v in payload.get("state_role_map", {}).items()},
            section_lengths=[int(x) for x in payload.get("section_lengths", [])],
        )
        model.max_duration = int(payload.get("max_duration", model.max_duration))
        model.startprob = np.asarray(payload["startprob"], dtype=np.float64)
        model.transmat = np.asarray(payload["transmat"], dtype=np.float64)
        model.durationprob = np.asarray(payload["durationprob"], dtype=np.float64)
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
        self.durationprob = np.full(
            (self.n_states, self.max_duration + 1),
            self.config.duration_smoothing,
            dtype=np.float64,
        )
        self.durationprob[:, 0] = 0.0
        for state, section_length in enumerate(self.section_lengths):
            duration = min(self.max_duration, max(1, int(section_length)))
            self.durationprob[state, duration] += self.config.duration_warm_start_strength
        self.durationprob = self._normalize_duration_rows(self.durationprob)
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

    def _validate_sequence_lengths(self, sequences: Sequence[np.ndarray]) -> None:
        operation_estimate = sum(
            self._expected_segment_operations(len(seq))
            for seq in sequences
        )
        if operation_estimate > self.config.max_expected_segment_operations:
            raise ValueError(
                f"HSMM '{self.name}' expected segment operation count is too high: "
                f"{operation_estimate} > {self.config.max_expected_segment_operations}. "
                "Reduce section_hmm.max_duration, split long pieces into form-level excerpts, "
                "or increase section_hmm.max_expected_segment_operations explicitly."
            )
        if not self.config.require_final_state:
            return
        if self.config.allow_self_loop:
            return
        min_length = self.n_states
        max_length = self.n_states * self.max_duration
        invalid = [
            int(len(seq))
            for seq in sequences
            if len(seq) < min_length or len(seq) > max_length
        ]
        if invalid:
            examples = sorted(set(invalid))[:10]
            raise ValueError(
                f"HSMM '{self.name}' cannot cover training sequence lengths with the current "
                f"left-to-right duration support. Valid length range is [{min_length}, {max_length}] "
                f"bars, but found examples {examples}. Increase section_hmm.max_duration, split long "
                "training pieces into form-level excerpts, or use a form template whose durations match "
                "the training corpus."
            )

    def _expected_segment_operations(self, length: int) -> int:
        duration_work = sum(min(self.max_duration, end) for end in range(1, int(length) + 1))
        transition_work = sum(
            min(self.max_duration, int(length) - start)
            for start in range(int(length))
        )
        allowed_transitions = int(np.count_nonzero(self._transition_mask()))
        return int(
            self.n_states * duration_work
            + allowed_transitions * transition_work
        )

    def _forward_backward(
        self,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        length = len(observations)
        segment_loglik = self._segment_loglik(observations)
        log_start = self._safe_log(self.startprob)
        log_trans = self._safe_log(self.transmat)
        log_duration = self._safe_log(self.durationprob)

        alpha = np.full((length + 1, self.n_states), -np.inf, dtype=np.float64)
        for end in range(1, length + 1):
            for state in range(self.n_states):
                values = []
                max_d = min(self.max_duration, end)
                for duration in range(1, max_d + 1):
                    start = end - duration
                    segment_score = (
                        log_duration[state, duration]
                        + segment_loglik[state, start, end]
                    )
                    if start == 0:
                        values.append(log_start[state] + segment_score)
                    else:
                        prev = self._logsumexp(alpha[start] + log_trans[:, state])
                        values.append(prev + segment_score)
                alpha[end, state] = self._logsumexp_array(values)
        if self.config.require_final_state:
            log_prob = float(alpha[length, self.n_states - 1])
        else:
            log_prob = self._logsumexp(alpha[length])
        if not np.isfinite(log_prob):
            raise ValueError(
                f"HSMM '{self.name}' has no valid left-to-right segmentation for sequence length {length}. "
                "Increase section_hmm.max_duration or review form/state topology."
            )

        beta = np.full((length + 1, self.n_states), -np.inf, dtype=np.float64)
        if self.config.require_final_state:
            beta[length, self.n_states - 1] = 0.0
        else:
            beta[length, :] = 0.0
        for start in range(length - 1, -1, -1):
            for prev_state in range(self.n_states):
                values = []
                max_d = min(self.max_duration, length - start)
                for next_state in range(self.n_states):
                    transition = log_trans[prev_state, next_state]
                    if not np.isfinite(transition):
                        continue
                    for duration in range(1, max_d + 1):
                        end = start + duration
                        values.append(
                            transition
                            + log_duration[next_state, duration]
                            + segment_loglik[next_state, start, end]
                            + beta[end, next_state]
                        )
                beta[start, prev_state] = self._logsumexp_array(values)
        return alpha, beta, float(log_prob), segment_loglik

    def _accumulate_expectations(
        self,
        observations: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        log_prob: float,
        segment_loglik: np.ndarray,
        start_counts: np.ndarray,
        trans_counts: np.ndarray,
        duration_counts: np.ndarray,
        emit_counts: np.ndarray,
    ) -> None:
        length = len(observations)
        log_start = self._safe_log(self.startprob)
        log_trans = self._safe_log(self.transmat)
        log_duration = self._safe_log(self.durationprob)
        for start in range(length):
            max_d = min(self.max_duration, length - start)
            for state in range(self.n_states):
                for duration in range(1, max_d + 1):
                    end = start + duration
                    segment_score = (
                        log_duration[state, duration]
                        + segment_loglik[state, start, end]
                        + beta[end, state]
                    )
                    if start == 0:
                        log_weight = log_start[state] + segment_score - log_prob
                        weight = float(np.exp(log_weight)) if np.isfinite(log_weight) else 0.0
                        if weight > 0.0:
                            start_counts[state] += weight
                            duration_counts[state, duration] += weight
                            for obs in observations[start:end]:
                                emit_counts[state, obs] += weight
                    else:
                        for prev_state in range(self.n_states):
                            transition = log_trans[prev_state, state]
                            if not np.isfinite(transition):
                                continue
                            log_weight = alpha[start, prev_state] + transition + segment_score - log_prob
                            weight = float(np.exp(log_weight)) if np.isfinite(log_weight) else 0.0
                            if weight <= 0.0:
                                continue
                            trans_counts[prev_state, state] += weight
                            duration_counts[state, duration] += weight
                            for obs in observations[start:end]:
                                emit_counts[state, obs] += weight

    def _segment_loglik(self, observations: np.ndarray) -> np.ndarray:
        length = len(observations)
        log_emit = self._safe_log(self.emissionprob)
        cumulative = np.zeros((self.n_states, length + 1), dtype=np.float64)
        for state in range(self.n_states):
            cumulative[state, 1:] = np.cumsum(log_emit[state, observations])
        segment = np.full((self.n_states, length + 1, length + 1), -np.inf, dtype=np.float64)
        for start in range(length):
            max_d = min(self.max_duration, length - start)
            for duration in range(1, max_d + 1):
                end = start + duration
                segment[:, start, end] = cumulative[:, end] - cumulative[:, start]
        return segment

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
            if start < end:
                mask[i, start:end] = 1.0
        if self.n_states > 0 and not mask[-1].any():
            if not self.config.require_final_state:
                mask[-1, -1] = 1.0
        return mask

    def _safe_log(self, values: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(values, dtype=np.float64))

    def _logsumexp(self, values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return -np.inf
        max_value = float(np.max(finite))
        return max_value + float(np.log(np.sum(np.exp(finite - max_value))))

    def _logsumexp_array(self, values: Sequence[float]) -> float:
        if not values:
            return -np.inf
        return self._logsumexp(np.asarray(values, dtype=np.float64))

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

    def _normalize_duration_rows(self, matrix: np.ndarray) -> np.ndarray:
        result = np.asarray(matrix, dtype=np.float64).copy()
        result[:, 0] = 0.0
        for row in range(result.shape[0]):
            total = float(result[row].sum())
            if total <= EPS:
                result[row, 1:] = 1.0 / max(1, result.shape[1] - 1)
            else:
                result[row] /= total
        return result

    def _build_diagnostics(self, sequences: Sequence[np.ndarray]) -> None:
        expected_duration = []
        mode_duration = []
        for state in range(self.n_states):
            durations = np.arange(self.durationprob.shape[1], dtype=np.float64)
            expected_duration.append(float(np.sum(durations * self.durationprob[state])))
            mode_duration.append(int(np.argmax(self.durationprob[state])))
        self.diagnostics = {
            "model_type": "explicit_duration_hsmm",
            "name": self.name,
            "state_role_map": {str(k): v for k, v in self.state_role_map.items()},
            "section_lengths": self.section_lengths,
            "max_duration": int(self.max_duration),
            "require_final_state": bool(self.config.require_final_state),
            "transition_mask": self._transition_mask().tolist(),
            "transition_matrix": self.transmat.tolist(),
            "duration_probability": self.durationprob.tolist(),
            "duration_expected_by_state": expected_duration,
            "duration_mode_by_state": mode_duration,
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

    def train(self, songs: Sequence[SongRecord], vocab: ObservationVocab) -> Dict[str, Any]:
        songs_by_form: Dict[str, List[SongRecord]] = {}
        for song in songs:
            form_name = song.form or "ternary"
            songs_by_form.setdefault(form_name, []).append(song)
        models = {}
        for form_name, form_songs in songs_by_form.items():
            template = self.library.require(form_name)
            backend = str(ConfigView(self.config).section("section_hmm").get("backend", "numpy"))
            if backend == "hsmm":
                sequences, section_lengths, section_diagnostics = self._hsmm_training_sequences(
                    form_name,
                    form_songs,
                    template,
                )
                model_cls = ExplicitDurationFormHSMM
            elif backend == "numpy":
                sequences = [
                    [int(bar.observation_id) for bar in song.bars if bar.observation_id is not None]
                    for song in form_songs
                ]
                section_lengths = [section.length for section in template.sections]
                section_diagnostics = {
                    "section_source": "generation_template",
                    "requires_form_json_boundaries": False,
                }
                model_cls = LeftToRightFormHMM
            else:
                raise ValueError("section_hmm.backend must be 'numpy' or 'hsmm'.")
            model = model_cls.from_style_config(
                self.config,
                len(template.sections),
                len(vocab.composite_to_observation),
                name=form_name,
                state_role_map={
                    index: section.name for index, section in enumerate(template.sections)
                },
                section_lengths=section_lengths,
            ).fit(sequences)
            models[form_name] = model
            self.diagnostics[form_name] = {
                **model.diagnostics,
                "training_section_instances": section_diagnostics,
            }
        return models

    def _hsmm_training_sequences(
        self,
        form_name: str,
        songs: Sequence[SongRecord],
        template: FormTemplate,
    ) -> tuple[List[List[int]], List[int], Dict[str, Any]]:
        sequences: List[List[int]] = []
        durations_by_state: List[List[int]] = [[] for _ in template.sections]
        song_diagnostics: List[Dict[str, Any]] = []
        for song in songs:
            sections = list(song.metadata.get("sections") or [])
            if not sections:
                raise ValueError(
                    f"HSMM '{form_name}' requires form.json section boundaries for song "
                    f"'{song.song_id}'. Do not train HSMM duration from generation template lengths."
                )
            if len(sections) != len(template.sections):
                raise ValueError(
                    f"HSMM '{form_name}' song '{song.song_id}' has {len(sections)} annotated sections, "
                    f"but template has {len(template.sections)} states."
                )
            sequence: List[int] = []
            section_rows: List[Dict[str, Any]] = []
            for index, section in enumerate(sections):
                start, end = self._metadata_section_range(form_name, song.song_id, section)
                bars = [
                    bar for bar in song.bars
                    if start <= int(bar.bar_index) < end and bar.observation_id is not None
                ]
                if not bars:
                    raise ValueError(
                        f"HSMM '{form_name}' song '{song.song_id}' section "
                        f"'{section.get('name', index)}' has no observed bars in [{start}, {end})."
                    )
                sequence.extend(int(bar.observation_id) for bar in bars)
                durations_by_state[index].append(len(bars))
                section_rows.append({
                    "state": index,
                    "template_name": template.sections[index].name,
                    "annotated_name": section.get("name"),
                    "start_bar": start,
                    "end_bar": end,
                    "duration": len(bars),
                })
            sequences.append(sequence)
            song_diagnostics.append({
                "song_id": song.song_id,
                "file_path": song.file_path,
                "sequence_length": len(sequence),
                "sections": section_rows,
            })
        section_duration_median = [
            max(1, int(round(float(np.median(values)))))
            if values else max(1, int(template.sections[index].length))
            for index, values in enumerate(durations_by_state)
        ]
        section_lengths = [
            max(values) if values else max(1, int(template.sections[index].length))
            for index, values in enumerate(durations_by_state)
        ]
        return sequences, section_lengths, {
            "section_source": "form_json_boundaries",
            "requires_form_json_boundaries": True,
            "state_duration_support": section_lengths,
            "state_duration_median": section_duration_median,
            "state_duration_min": [min(values) if values else None for values in durations_by_state],
            "state_duration_max": [max(values) if values else None for values in durations_by_state],
            "songs": song_diagnostics,
        }

    def _metadata_section_range(
        self,
        form_name: str,
        song_id: str,
        section: Dict[str, Any],
    ) -> tuple[int, int]:
        start = section.get("start_bar", section.get("start"))
        end = section.get("end_bar", section.get("end"))
        length = section.get("length")
        if start is None:
            raise ValueError(
                f"HSMM '{form_name}' song '{song_id}' section '{section.get('name')}' "
                "must provide start_bar or start."
            )
        start_int = int(start)
        if end is not None:
            end_int = int(end)
            if length is not None and int(length) != end_int - start_int:
                raise ValueError(
                    f"HSMM '{form_name}' song '{song_id}' section '{section.get('name')}' "
                    "has inconsistent length and start_bar/end_bar."
                )
        elif length is not None:
            end_int = start_int + int(length)
        else:
            raise ValueError(
                f"HSMM '{form_name}' song '{song_id}' section '{section.get('name')}' "
                "must provide end_bar/end or length."
            )
        if start_int < 0 or end_int <= start_int:
            raise ValueError(
                f"HSMM '{form_name}' song '{song_id}' section '{section.get('name')}' "
                "has invalid section range."
            )
        return start_int, end_int


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
