#!/usr/bin/env python3
"""Model-bundle analysis helpers for HMM diagnostics reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class DistributionComparison:
    """Distance metrics between two observation distributions."""

    l1: float
    total_variation: float
    kl_actual_to_model: float
    kl_model_to_actual: float
    js_divergence: float
    cosine_similarity: float
    pearson_correlation: float
    top_overlap_ratio: float


@dataclass(frozen=True)
class FormDistributionAnalysis:
    """Computed distribution views for one form HMM."""

    form_name: str
    n_states: int
    n_observations: int
    section_lengths: List[int]
    state_roles: Dict[int, str]
    startprob: np.ndarray
    transmat: np.ndarray
    emissionprob: np.ndarray
    section_weighted_distribution: np.ndarray
    transition_occupancy_distribution: np.ndarray
    transition_state_occupancy: np.ndarray
    section_state_weights: np.ndarray
    section_weighted_metrics: DistributionComparison
    transition_occupancy_metrics: DistributionComparison
    emission_entropy: List[float]
    emission_effective_counts: List[float]
    emission_pairwise_js: Dict[str, float]
    training_log: List[Dict[str, Any]]


class ModelBundleAnalyzer:
    """Analyze saved model_bundle.json without requiring retraining."""

    def __init__(self, payload: Dict[str, Any], model_path: Path) -> None:
        self.payload = payload
        self.model_path = model_path
        self.observation_vocab = payload.get("observation_vocab", {})
        self.observation_to_bars = payload.get("observation_to_bars", {})
        self.training_summary = payload.get("training_summary", {})
        self.form_models = payload.get("form_models", {})

    @classmethod
    def load(cls, model_bundle: Path) -> "ModelBundleAnalyzer":
        payload = json.loads(model_bundle.read_text(encoding="utf-8"))
        return cls(payload, model_bundle)

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "ModelBundleAnalyzer":
        return cls.load(model_dir / "model_bundle.json")

    def actual_observation_counts(self, n_observations: Optional[int] = None) -> np.ndarray:
        if n_observations is None:
            n_observations = self.observation_count()
        counts = np.zeros(int(n_observations), dtype=np.float64)
        if self.observation_to_bars:
            for key, bars in self.observation_to_bars.items():
                obs = int(key)
                if 0 <= obs < len(counts):
                    counts[obs] = len(bars)
            return counts
        pool_sizes = (
            self.training_summary
            .get("observation_bar_pools", {})
            .get("pool_size_by_observation", {})
        )
        for key, count in pool_sizes.items():
            obs = int(key)
            if 0 <= obs < len(counts):
                counts[obs] = float(count)
        return counts

    def actual_observation_distribution(self, n_observations: Optional[int] = None) -> np.ndarray:
        counts = self.actual_observation_counts(n_observations)
        total = float(np.sum(counts))
        if total <= EPS:
            return np.full(len(counts), 1.0 / max(1, len(counts)), dtype=np.float64)
        return counts / total

    def observation_count(self) -> int:
        summary_count = self.training_summary.get("observation_count")
        if summary_count is not None:
            return int(summary_count)
        mapping = self.observation_vocab.get("composite_to_observation", {})
        if mapping:
            return len(mapping)
        return len(self.observation_to_bars)

    def bar_count(self) -> int:
        summary_count = self.training_summary.get("bar_count")
        if summary_count is not None:
            return int(summary_count)
        return int(sum(len(bars) for bars in self.observation_to_bars.values()))

    def observation_label(self, observation_id: int) -> str:
        mapping = self.observation_vocab.get("observation_to_composite", {})
        return str(mapping.get(str(observation_id), mapping.get(observation_id, observation_id)))

    def form_names(self) -> List[str]:
        return sorted(str(name) for name in self.form_models)

    def analyze_form(self, form_name: str) -> FormDistributionAnalysis:
        payload = self.form_models[form_name]
        emissionprob = np.asarray(payload["emissionprob"], dtype=np.float64)
        transmat = np.asarray(payload["transmat"], dtype=np.float64)
        startprob = np.asarray(payload["startprob"], dtype=np.float64)
        section_lengths = [int(value) for value in payload.get("section_lengths", [])]
        if not section_lengths:
            section_lengths = [1] * int(payload.get("n_states", emissionprob.shape[0]))
        state_roles = {
            int(key): str(value)
            for key, value in payload.get("state_role_map", {}).items()
        }
        n_states, n_observations = emissionprob.shape
        actual = self.actual_observation_distribution(n_observations)
        section_weights = self._section_state_weights(section_lengths, n_states)
        section_weighted = section_weights @ emissionprob
        transition_occupancy = self._transition_state_occupancy(startprob, transmat, sum(section_lengths))
        transition_weighted = transition_occupancy @ emissionprob
        emission_entropy = [self._entropy(row) for row in emissionprob]
        pairwise_js_values = self._pairwise_js(emissionprob)
        return FormDistributionAnalysis(
            form_name=form_name,
            n_states=n_states,
            n_observations=n_observations,
            section_lengths=section_lengths,
            state_roles=state_roles,
            startprob=startprob,
            transmat=transmat,
            emissionprob=emissionprob,
            section_weighted_distribution=section_weighted,
            transition_occupancy_distribution=transition_weighted,
            transition_state_occupancy=transition_occupancy,
            section_state_weights=section_weights,
            section_weighted_metrics=self.compare_distributions(actual, section_weighted),
            transition_occupancy_metrics=self.compare_distributions(actual, transition_weighted),
            emission_entropy=emission_entropy,
            emission_effective_counts=[float(math.exp(value)) for value in emission_entropy],
            emission_pairwise_js=self._pairwise_summary(pairwise_js_values),
            training_log=list(payload.get("training_log", [])),
        )

    def compare_distributions(self, actual: np.ndarray, model: np.ndarray, top_n: int = 20) -> DistributionComparison:
        actual = self._normalize_array(actual)
        model = self._normalize_array(model)
        l1 = float(np.sum(np.abs(actual - model)))
        kl_am = self._kl(actual, model)
        kl_ma = self._kl(model, actual)
        js = self._js(actual, model)
        denom = max(float(np.linalg.norm(actual) * np.linalg.norm(model)), EPS)
        cosine = float(np.dot(actual, model) / denom)
        if float(np.std(actual)) <= EPS or float(np.std(model)) <= EPS:
            pearson = 0.0
        else:
            pearson = float(np.corrcoef(actual, model)[0, 1])
        actual_top = set(int(index) for index in np.argsort(-actual)[:top_n])
        model_top = set(int(index) for index in np.argsort(-model)[:top_n])
        overlap = len(actual_top & model_top) / max(1, top_n)
        return DistributionComparison(
            l1=l1,
            total_variation=l1 / 2.0,
            kl_actual_to_model=kl_am,
            kl_model_to_actual=kl_ma,
            js_divergence=js,
            cosine_similarity=cosine,
            pearson_correlation=pearson,
            top_overlap_ratio=float(overlap),
        )

    def top_observations(self, distribution: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
        distribution = self._normalize_array(distribution)
        result = []
        for index in np.argsort(-distribution)[:top_n]:
            obs = int(index)
            result.append({
                "observation_id": obs,
                "label": self.observation_label(obs),
                "probability": float(distribution[obs]),
                "count": int(self.actual_observation_counts(len(distribution))[obs]),
            })
        return result

    def pool_summary(self) -> Dict[str, Any]:
        counts = self.actual_observation_counts(self.observation_count())
        used = counts[counts > 0]
        total = int(np.sum(counts))
        singleton = int(np.sum(counts == 1))
        return {
            "bar_count": self.bar_count(),
            "observation_count": self.observation_count(),
            "used_observation_count": int(len(used)),
            "singleton_observation_count": singleton,
            "singleton_ratio": float(singleton / max(1, len(counts))),
            "max_pool_size": int(np.max(used)) if len(used) else 0,
            "min_pool_size": int(np.min(used)) if len(used) else 0,
            "mean_pool_size": float(np.mean(used)) if len(used) else 0.0,
            "bar_count_from_pools": total,
            "observation_per_bar_ratio": float(len(counts) / max(1, self.bar_count())),
        }

    def layered_pool_summaries(self) -> List[Dict[str, Any]]:
        bars = self._training_bars()
        return [
            self._pool_summary_for_keys("codebook_id", [self._codebook_key(bar) for bar in bars]),
            self._pool_summary_for_keys("base_composite", [self._base_composite_key(bar) for bar in bars]),
            self._pool_summary_for_keys("observation_id", [self._observation_key(bar) for bar in bars]),
        ]

    def _training_bars(self) -> List[Dict[str, Any]]:
        bars: List[Dict[str, Any]] = []
        for pool in self.observation_to_bars.values():
            bars.extend(item for item in pool if isinstance(item, dict))
        return bars

    def _pool_summary_for_keys(self, layer: str, keys: Sequence[Optional[str]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        missing = 0
        for key in keys:
            if key is None:
                missing += 1
                continue
            counts[str(key)] = counts.get(str(key), 0) + 1
        values = np.asarray(list(counts.values()), dtype=np.float64)
        singleton_count = int(np.sum(values == 1)) if values.size else 0
        total_bars = int(np.sum(values)) if values.size else 0
        return {
            "layer": layer,
            "bar_count": total_bars,
            "missing_bar_count": int(missing),
            "pool_count": int(len(counts)),
            "singleton_pool_count": singleton_count,
            "singleton_ratio": float(singleton_count / max(1, len(counts))),
            "min_pool_size": int(np.min(values)) if values.size else 0,
            "max_pool_size": int(np.max(values)) if values.size else 0,
            "mean_pool_size": float(np.mean(values)) if values.size else 0.0,
            "pool_per_bar_ratio": float(len(counts) / max(1, total_bars)),
            "top_pools": [
                {"key": key, "count": int(count)}
                for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
            ],
        }

    def _codebook_key(self, bar: Dict[str, Any]) -> Optional[str]:
        codebook_id = bar.get("codebook_id")
        if codebook_id is None:
            return None
        return f"C{int(codebook_id)}"

    def _base_composite_key(self, bar: Dict[str, Any]) -> Optional[str]:
        codebook_id = bar.get("codebook_id")
        if codebook_id is None:
            return None
        kmeans_id = bar.get("kmeans_id")
        if kmeans_id is None:
            return f"C{int(codebook_id)}"
        return f"C{int(codebook_id)}_K{int(kmeans_id)}"

    def _observation_key(self, bar: Dict[str, Any]) -> Optional[str]:
        observation_id = bar.get("observation_id")
        if observation_id is not None:
            return f"O{int(observation_id)}"
        composite_key = bar.get("composite_key")
        if composite_key is not None:
            return str(composite_key)
        return None

    def _section_state_weights(self, section_lengths: Sequence[int], n_states: int) -> np.ndarray:
        weights = np.zeros(n_states, dtype=np.float64)
        for index, length in enumerate(section_lengths[:n_states]):
            weights[index] = max(0, int(length))
        if float(np.sum(weights)) <= EPS:
            weights[:] = 1.0
        return weights / float(np.sum(weights))

    def _transition_state_occupancy(self, startprob: np.ndarray, transmat: np.ndarray, steps: int) -> np.ndarray:
        state = self._normalize_array(startprob)
        occupancy = np.zeros_like(state, dtype=np.float64)
        for _ in range(max(1, int(steps))):
            occupancy += state
            state = self._normalize_array(state @ transmat)
        return self._normalize_array(occupancy)

    def _pairwise_js(self, matrix: np.ndarray) -> List[float]:
        values = []
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[0]):
                values.append(self._js(matrix[i], matrix[j]))
        return values

    def _pairwise_summary(self, values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "mean": float(np.mean(array)),
            "max": float(np.max(array)),
        }

    def _normalize_array(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        total = float(np.sum(result))
        if total <= EPS:
            return np.full(len(result), 1.0 / max(1, len(result)), dtype=np.float64)
        return result / total

    def _entropy(self, values: np.ndarray) -> float:
        probs = self._normalize_array(values)
        return float(-np.sum(probs * np.log(np.maximum(probs, EPS))))

    def _kl(self, left: np.ndarray, right: np.ndarray) -> float:
        left = self._normalize_array(left)
        right = self._normalize_array(right)
        return float(np.sum(left * np.log(np.maximum(left, EPS) / np.maximum(right, EPS))))

    def _js(self, left: np.ndarray, right: np.ndarray) -> float:
        left = self._normalize_array(left)
        right = self._normalize_array(right)
        midpoint = 0.5 * (left + right)
        return float(0.5 * self._kl(left, midpoint) + 0.5 * self._kl(right, midpoint))


class ModelAnalysisCharts:
    """Optional PNG chart rendering for markdown reports."""

    def __init__(self, analyzer: ModelBundleAnalyzer, charts_dir: Path) -> None:
        self.analyzer = analyzer
        self.charts_dir = charts_dir
        self.available = True
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self.plt = plt
        except Exception:
            self.available = False
            self.plt = None

    def render_form_charts(self, analysis: FormDistributionAnalysis, top_n: int = 30) -> Dict[str, Path]:
        if not self.available:
            return {}
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        charts = {
            "top_distribution": self._plot_top_distribution(analysis, top_n),
            "transition_matrix": self._plot_matrix(
                analysis.transmat,
                f"{analysis.form_name} transition matrix",
                f"{analysis.form_name}_transition_matrix.png",
                x_labels=[analysis.state_roles.get(i, str(i)) for i in range(analysis.n_states)],
                y_labels=[analysis.state_roles.get(i, str(i)) for i in range(analysis.n_states)],
            ),
            "emission_top": self._plot_emission_top(analysis, top_n),
        }
        return {key: value for key, value in charts.items() if value is not None}

    def _plot_top_distribution(self, analysis: FormDistributionAnalysis, top_n: int) -> Optional[Path]:
        actual = self.analyzer.actual_observation_distribution(analysis.n_observations)
        model = analysis.section_weighted_distribution
        top = list(dict.fromkeys(
            [int(i) for i in np.argsort(-actual)[:top_n]]
            + [int(i) for i in np.argsort(-model)[:top_n]]
        ))[:top_n]
        labels = [str(obs) for obs in top]
        x = np.arange(len(top))
        width = 0.42
        fig, ax = self.plt.subplots(figsize=(max(10, len(top) * 0.35), 4.8))
        ax.bar(x - width / 2, [actual[i] for i in top], width, label="Actual")
        ax.bar(x + width / 2, [model[i] for i in top], width, label="HMM section-weighted")
        ax.set_title(f"{analysis.form_name} actual vs HMM observation distribution")
        ax.set_xlabel("observation_id")
        ax.set_ylabel("probability")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70)
        ax.legend()
        fig.tight_layout()
        path = self.charts_dir / f"{analysis.form_name}_observation_distribution_top.png"
        fig.savefig(path, dpi=150)
        self.plt.close(fig)
        return path

    def _plot_emission_top(self, analysis: FormDistributionAnalysis, top_n: int) -> Optional[Path]:
        aggregate = analysis.section_weighted_distribution
        top = [int(i) for i in np.argsort(-aggregate)[:top_n]]
        matrix = analysis.emissionprob[:, top]
        x_labels = [str(obs) for obs in top]
        y_labels = [analysis.state_roles.get(i, str(i)) for i in range(analysis.n_states)]
        return self._plot_matrix(
            matrix,
            f"{analysis.form_name} emission probabilities for top observations",
            f"{analysis.form_name}_emission_top.png",
            x_labels=x_labels,
            y_labels=y_labels,
        )

    def _plot_matrix(
        self,
        matrix: np.ndarray,
        title: str,
        filename: str,
        x_labels: Sequence[str],
        y_labels: Sequence[str],
    ) -> Optional[Path]:
        fig, ax = self.plt.subplots(figsize=(max(6, len(x_labels) * 0.28), max(3.5, len(y_labels) * 0.65)))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=70)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = self.charts_dir / filename
        fig.savefig(path, dpi=150)
        self.plt.close(fig)
        return path


class MarkdownReport:
    """Small markdown builder used by analysis CLIs."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def heading(self, text: str, level: int = 1) -> None:
        self.lines.append(f"{'#' * level} {text}")
        self.lines.append("")

    def paragraph(self, text: str) -> None:
        self.lines.append(text)
        self.lines.append("")

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(self._cell(value) for value in row) + " |")
        self.lines.append("")

    def image(self, title: str, path: Path, output_path: Path) -> None:
        try:
            link = path.relative_to(output_path.parent).as_posix()
        except ValueError:
            link = path.as_posix()
        self.lines.append(f"![{title}]({link})")
        self.lines.append("")

    def write(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")

    def _cell(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")
