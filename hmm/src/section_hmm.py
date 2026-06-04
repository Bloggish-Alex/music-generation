#!/usr/bin/env python3
"""Discrete-observation HMM for macro section inference and sampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigLoader, ConfigView


EPS = 1e-12


@dataclass(frozen=True)
class SectionHMMConfig:
    n_sections: int = 4
    max_iter: int = 80
    tol: float = 1e-4
    self_transition_bias: float = 8.0
    emission_smoothing: float = 0.2
    transition_smoothing: float = 0.2
    random_seed: int = 42


class DiscreteSectionHMM:
    """Baum-Welch trained HMM with integer observations."""

    backend = "numpy"

    def __init__(
        self,
        config: SectionHMMConfig,
        startprob: Optional[np.ndarray] = None,
        transmat: Optional[np.ndarray] = None,
        emissionprob: Optional[np.ndarray] = None,
        n_observations: Optional[int] = None,
    ) -> None:
        self.config = config
        self.startprob = startprob
        self.transmat = transmat
        self.emissionprob = emissionprob
        self.n_observations = n_observations
        self.training_log: List[Dict[str, float]] = []

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "DiscreteSectionHMM":
        section = ConfigView(config).section("section_hmm")
        return cls(SectionHMMConfig(
            n_sections=int(section.get("n_sections", 4)),
            max_iter=int(section.get("max_iter", 80)),
            tol=float(section.get("tol", 1e-4)),
            self_transition_bias=float(section.get("self_transition_bias", 8.0)),
            emission_smoothing=float(section.get("emission_smoothing", 0.2)),
            transition_smoothing=float(section.get("transition_smoothing", 0.2)),
            random_seed=int(section.get("random_seed", 42)),
        ))

    def fit(self, sequences: Sequence[Sequence[int]]) -> "DiscreteSectionHMM":
        clean = [np.asarray(seq, dtype=int) for seq in sequences if len(seq) > 0]
        if not clean:
            raise ValueError("Cannot train HMM without observations.")
        self.n_observations = int(max(int(seq.max()) for seq in clean) + 1)
        self._initialize_parameters(clean)
        previous_ll: Optional[float] = None
        for iteration in range(self.config.max_iter):
            start_counts = np.full(self.config.n_sections, EPS, dtype=np.float64)
            trans_counts = np.full(
                (self.config.n_sections, self.config.n_sections),
                self.config.transition_smoothing,
                dtype=np.float64,
            )
            emit_counts = np.full(
                (self.config.n_sections, self.n_observations),
                self.config.emission_smoothing,
                dtype=np.float64,
            )
            total_ll = 0.0
            for observations in clean:
                alpha, scales, log_likelihood = self._forward(observations)
                beta = self._backward(observations, scales)
                gamma = alpha * beta
                gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), EPS)
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
                    xi = xi / max(float(xi.sum()), EPS)
                    trans_counts += xi
                total_ll += log_likelihood
            self.startprob = self._normalize(start_counts)
            self.transmat = self._normalize_rows(trans_counts)
            self.emissionprob = self._normalize_rows(emit_counts)
            delta = 0.0 if previous_ll is None else total_ll - previous_ll
            self.training_log.append({"iteration": float(iteration), "log_likelihood": total_ll, "delta": delta})
            if previous_ll is not None and abs(delta) < self.config.tol:
                break
            previous_ll = total_ll
        return self

    def decode(self, observations: Sequence[int]) -> List[int]:
        obs = np.asarray(observations, dtype=int)
        n_states = self.config.n_sections
        log_start = np.log(np.maximum(self.startprob, EPS))
        log_trans = np.log(np.maximum(self.transmat, EPS))
        log_emit = np.log(np.maximum(self.emissionprob, EPS))
        scores = np.full((len(obs), n_states), -np.inf)
        backptr = np.zeros((len(obs), n_states), dtype=int)
        scores[0] = log_start + log_emit[:, obs[0]]
        for t in range(1, len(obs)):
            values = scores[t - 1, :, None] + log_trans
            backptr[t] = values.argmax(axis=0)
            scores[t] = values.max(axis=0) + log_emit[:, obs[t]]
        states = [int(scores[-1].argmax())]
        for t in range(len(obs) - 1, 0, -1):
            states.append(int(backptr[t, states[-1]]))
        states.reverse()
        return states

    def sample(self, length: int, seed: Optional[int] = None) -> Dict[str, List[int]]:
        rng = np.random.default_rng(self.config.random_seed if seed is None else seed)
        states: List[int] = []
        observations: List[int] = []
        state = int(rng.choice(self.config.n_sections, p=self.startprob))
        for _ in range(length):
            obs = int(rng.choice(self.n_observations, p=self.emissionprob[state]))
            states.append(state)
            observations.append(obs)
            state = int(rng.choice(self.config.n_sections, p=self.transmat[state]))
        return {"section_states": states, "bar_labels": observations}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "config": asdict(self.config),
            "n_observations": self.n_observations,
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "emissionprob": self.emissionprob.tolist(),
            "training_log": self.training_log,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DiscreteSectionHMM":
        config = SectionHMMConfig(**payload["config"])
        model = cls(
            config=config,
            startprob=np.asarray(payload["startprob"], dtype=np.float64),
            transmat=np.asarray(payload["transmat"], dtype=np.float64),
            emissionprob=np.asarray(payload["emissionprob"], dtype=np.float64),
            n_observations=int(payload["n_observations"]),
        )
        model.training_log = payload.get("training_log", [])
        return model

    def _initialize_parameters(self, sequences: Sequence[np.ndarray]) -> None:
        rng = np.random.default_rng(self.config.random_seed)
        n_states = self.config.n_sections
        n_obs = int(self.n_observations)
        self.startprob = self._normalize(np.ones(n_states, dtype=np.float64))
        trans = np.ones((n_states, n_states), dtype=np.float64)
        for state in range(n_states):
            trans[state, state] += self.config.self_transition_bias
        self.transmat = self._normalize_rows(trans)
        emit = rng.random((n_states, n_obs)) + self.config.emission_smoothing
        counts = np.bincount(np.concatenate(sequences), minlength=n_obs).astype(np.float64)
        emit += counts[None, :] / max(1.0, counts.sum())
        self.emissionprob = self._normalize_rows(emit)

    def _forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        alpha = np.zeros((len(observations), self.config.n_sections), dtype=np.float64)
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
        beta = np.zeros((len(observations), self.config.n_sections), dtype=np.float64)
        beta[-1] = 1.0
        for t in range(len(observations) - 2, -1, -1):
            beta[t] = self.transmat @ (self.emissionprob[:, observations[t + 1]] * beta[t + 1])
            beta[t] /= max(float(scales[t + 1]), EPS)
        return beta

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        return values / max(float(values.sum()), EPS)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), EPS)


class SectionHMMCLI:
    """CLI for training/testing the HMM from a labels JSON file."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train a discrete section HMM from bar labels.")
        parser.add_argument("--labels", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--n-sections", type=int, default=None)
        parser.add_argument("--max-iter", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        model_config = DiscreteSectionHMM.from_style_config(config).config
        model_config = SectionHMMConfig(
            n_sections=args.n_sections if args.n_sections is not None else model_config.n_sections,
            max_iter=args.max_iter if args.max_iter is not None else model_config.max_iter,
            tol=model_config.tol,
            self_transition_bias=model_config.self_transition_bias,
            emission_smoothing=model_config.emission_smoothing,
            transition_smoothing=model_config.transition_smoothing,
            random_seed=args.seed if args.seed is not None else model_config.random_seed,
        )
        payload = json.loads(args.labels.read_text(encoding="utf-8"))
        sequences = _sequences_from_label_payload(payload)
        hmm = DiscreteSectionHMM(model_config).fit(sequences)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hmm.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote HMM -> {args.output}")


def _sequences_from_label_payload(payload: Dict[str, Any]) -> List[List[int]]:
    by_file: Dict[str, List[tuple[int, int]]] = {}
    for item in payload.get("bars", []):
        by_file.setdefault(item["file_path"], []).append((int(item["bar_index"]), int(item["label"])))
    sequences = []
    for items in by_file.values():
        sequences.append([label for _, label in sorted(items)])
    if not sequences and payload.get("labels"):
        sequences.append([int(x) for x in payload["labels"]])
    return sequences


def main() -> None:
    SectionHMMCLI().run()


if __name__ == "__main__":
    main()
