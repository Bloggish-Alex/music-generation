#!/usr/bin/env python3
"""hmmlearn-backed discrete-observation HMM for section inference.

This file intentionally stays independent from ``section_hmm.py``.  Both
classes expose the same small interface used by training/generation:

    fit(sequences)
    decode(observations)
    sample(length, seed)
    to_dict()
    from_dict(payload)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigLoader, ConfigView
from section_hmm import _sequences_from_label_payload


@dataclass(frozen=True)
class HmmlearnSectionHMMConfig:
    n_sections: int = 4
    max_iter: int = 80
    tol: float = 1e-4
    self_transition_bias: float = 3.0
    random_seed: int = 42
    algorithm: str = "viterbi"
    implementation: str = "log"


class HmmlearnSectionHMM:
    """Adapter around hmmlearn.CategoricalHMM for integer bar-label observations."""

    backend = "hmmlearn"

    def __init__(
        self,
        config: HmmlearnSectionHMMConfig,
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
    def from_style_config(cls, config: Dict[str, Any]) -> "HmmlearnSectionHMM":
        section = ConfigView(config).section("section_hmm")
        hmmlearn_section = ConfigView(config).section("hmmlearn_section_hmm")
        return cls(HmmlearnSectionHMMConfig(
            n_sections=int(section.get("n_sections", 4)),
            max_iter=int(section.get("max_iter", 80)),
            tol=float(section.get("tol", 1e-4)),
            self_transition_bias=float(section.get("self_transition_bias", 3.0)),
            random_seed=int(section.get("random_seed", 42)),
            algorithm=str(hmmlearn_section.get("algorithm", "viterbi")),
            implementation=str(hmmlearn_section.get("implementation", "log")),
        ))

    def fit(self, sequences: Sequence[Sequence[int]]) -> "HmmlearnSectionHMM":
        clean = [np.asarray(seq, dtype=int) for seq in sequences if len(seq) > 0]
        if not clean:
            raise ValueError("Cannot train HMM without observations.")
        self.n_observations = int(max(int(seq.max()) for seq in clean) + 1)
        observations = np.concatenate(clean).reshape(-1, 1)
        lengths = [len(seq) for seq in clean]

        model = self._new_model()
        model.startprob_ = np.full(self.config.n_sections, 1.0 / self.config.n_sections)
        model.transmat_ = self._initial_transmat()
        model.emissionprob_ = self._initial_emissionprob(clean)
        model.fit(observations, lengths=lengths)

        self.startprob = np.asarray(model.startprob_, dtype=np.float64)
        self.transmat = np.asarray(model.transmat_, dtype=np.float64)
        self.emissionprob = np.asarray(model.emissionprob_, dtype=np.float64)
        history = list(getattr(model.monitor_, "history", []))
        self.training_log = [
            {
                "iteration": float(index),
                "log_likelihood": float(value),
                "delta": 0.0 if index == 0 else float(value - history[index - 1]),
            }
            for index, value in enumerate(history)
        ]
        return self

    def decode(self, observations: Sequence[int]) -> List[int]:
        obs = np.asarray(observations, dtype=int).reshape(-1, 1)
        model = self._model_from_matrices()
        _, states = model.decode(obs, algorithm=self.config.algorithm)
        return [int(x) for x in states]

    def sample(self, length: int, seed: Optional[int] = None) -> Dict[str, List[int]]:
        model = self._model_from_matrices()
        if seed is not None:
            model.random_state = seed
        observations, states = model.sample(length)
        return {
            "section_states": [int(x) for x in states],
            "bar_labels": [int(x[0]) for x in observations],
        }

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
    def from_dict(cls, payload: Dict[str, Any]) -> "HmmlearnSectionHMM":
        config = HmmlearnSectionHMMConfig(**payload["config"])
        model = cls(
            config=config,
            startprob=np.asarray(payload["startprob"], dtype=np.float64),
            transmat=np.asarray(payload["transmat"], dtype=np.float64),
            emissionprob=np.asarray(payload["emissionprob"], dtype=np.float64),
            n_observations=int(payload["n_observations"]),
        )
        model.training_log = payload.get("training_log", [])
        return model

    def _new_model(self) -> Any:
        from hmmlearn import hmm

        model_cls = getattr(hmm, "CategoricalHMM", None)
        if model_cls is None:
            raise ImportError(
                "hmmlearn.CategoricalHMM is required for HmmlearnSectionHMM. "
                "Please install hmmlearn >= 0.3."
            )
        return model_cls(
            n_components=self.config.n_sections,
            n_features=self.n_observations,
            n_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=self.config.random_seed,
            algorithm=self.config.algorithm,
            implementation=self.config.implementation,
            init_params="",
            params="ste",
        )

    def _model_from_matrices(self) -> Any:
        model = self._new_model()
        model.startprob_ = np.asarray(self.startprob, dtype=np.float64)
        model.transmat_ = np.asarray(self.transmat, dtype=np.float64)
        model.emissionprob_ = np.asarray(self.emissionprob, dtype=np.float64)
        return model

    def _initial_transmat(self) -> np.ndarray:
        matrix = np.ones((self.config.n_sections, self.config.n_sections), dtype=np.float64)
        for state in range(self.config.n_sections):
            matrix[state, state] += self.config.self_transition_bias
        return matrix / matrix.sum(axis=1, keepdims=True)

    def _initial_emissionprob(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_seed)
        counts = np.bincount(np.concatenate(sequences), minlength=int(self.n_observations)).astype(np.float64)
        base = counts / max(1.0, float(counts.sum()))
        matrix = rng.random((self.config.n_sections, int(self.n_observations))) * 0.05
        matrix += base[None, :] + 1e-3
        return matrix / matrix.sum(axis=1, keepdims=True)


class HmmlearnSectionHMMCLI:
    """CLI for standalone hmmlearn HMM training from bar-label JSON."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train hmmlearn CategoricalHMM from bar labels.")
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
        model_config = HmmlearnSectionHMM.from_style_config(config).config
        model_config = HmmlearnSectionHMMConfig(
            n_sections=args.n_sections if args.n_sections is not None else model_config.n_sections,
            max_iter=args.max_iter if args.max_iter is not None else model_config.max_iter,
            tol=model_config.tol,
            self_transition_bias=model_config.self_transition_bias,
            random_seed=args.seed if args.seed is not None else model_config.random_seed,
            algorithm=model_config.algorithm,
            implementation=model_config.implementation,
        )
        payload = json.loads(args.labels.read_text(encoding="utf-8"))
        hmm = HmmlearnSectionHMM(model_config).fit(_sequences_from_label_payload(payload))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hmm.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote hmmlearn HMM -> {args.output}")


def main() -> None:
    HmmlearnSectionHMMCLI().run()


if __name__ == "__main__":
    main()
