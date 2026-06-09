#!/usr/bin/env python3
"""Denoising VAE encoder for bar-token latent-space experiments."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from core_data import BarRecord


REST_TOKEN = -1
SUSTAIN_TOKEN = -2
NOTE_TYPE_REST = 0
NOTE_TYPE_SUSTAIN = 1
NOTE_TYPE_ON = 2


@dataclass(frozen=True)
class DenoisingVAEConfig:
    """Config for denoising bar-token VAE experiments."""

    steps_per_bar: int = 16
    hidden_dim: int = 32
    latent_dim: int = 8
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 0.001
    beta_kl: float = 0.001
    kl_warmup_epochs: int = 10
    pitch_weight: float = 1.0
    note_drop_prob: float = 0.15
    sustain_fill_prob: float = 0.10
    drop_to_rest_prob: float = 0.5
    ornament_pitch_radius: int = 2
    pitch_scale: float = 24.0
    random_seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class LatentClusteringConfig:
    """Config for clustering z_mu vectors."""

    method: str = "kmeans"
    n_clusters: int = 384
    distance_threshold: float = 1.0
    linkage_method: str = "average"
    random_seed: int = 42


class BarTokenCodec:
    """Convert relative bar tokens to multi-head training targets."""

    def __init__(self, config: DenoisingVAEConfig) -> None:
        self.config = config

    def clean_tokens(self, bar: BarRecord) -> List[int]:
        tokens = [int(token) for token in bar.relative_tokens[: self.config.steps_per_bar]]
        if len(tokens) < self.config.steps_per_bar:
            tokens.extend([REST_TOKEN] * (self.config.steps_per_bar - len(tokens)))
        return tokens

    def input_vector(self, tokens: Sequence[int]) -> np.ndarray:
        return np.asarray([float(token) / self.config.pitch_scale for token in tokens], dtype=np.float32)

    def type_targets(self, tokens: Sequence[int]) -> np.ndarray:
        result = []
        for token in tokens:
            token = int(token)
            if token == REST_TOKEN:
                result.append(NOTE_TYPE_REST)
            elif token == SUSTAIN_TOKEN:
                result.append(NOTE_TYPE_SUSTAIN)
            else:
                result.append(NOTE_TYPE_ON)
        return np.asarray(result, dtype=np.int64)

    def pitch_targets(self, tokens: Sequence[int]) -> np.ndarray:
        return np.asarray([
            float(token) / self.config.pitch_scale if int(token) >= 0 else 0.0
            for token in tokens
        ], dtype=np.float32)

    def note_mask(self, tokens: Sequence[int]) -> np.ndarray:
        return np.asarray([1.0 if int(token) >= 0 else 0.0 for token in tokens], dtype=np.float32)


class BarNoiseInjector:
    """Apply token-level variation noise to relative-token bars."""

    def __init__(self, config: DenoisingVAEConfig) -> None:
        self.config = config

    def apply(self, tokens: Sequence[int], rng: np.random.Generator) -> List[int]:
        clean = [int(token) for token in tokens]
        noisy = list(clean)
        note_indices = [index for index, token in enumerate(clean) if token >= 0]
        for index in note_indices:
            if rng.random() < self.config.note_drop_prob:
                noisy[index] = REST_TOKEN if rng.random() < self.config.drop_to_rest_prob else SUSTAIN_TOKEN
        note_values = [token for token in clean if token >= 0]
        if note_values:
            for index, token in enumerate(clean):
                if token == SUSTAIN_TOKEN and rng.random() < self.config.sustain_fill_prob:
                    base = int(rng.choice(note_values))
                    delta = int(rng.integers(-self.config.ornament_pitch_radius, self.config.ornament_pitch_radius + 1))
                    noisy[index] = max(0, base + delta)
        if note_indices and not any(token >= 0 for token in noisy):
            restore_index = int(rng.choice(note_indices))
            noisy[restore_index] = clean[restore_index]
        return noisy


class DenoisingBarDataset:
    """Small torch Dataset wrapper that creates fresh noise each epoch."""

    def __init__(self, bars: Sequence[BarRecord], config: DenoisingVAEConfig) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, outer: "DenoisingBarDataset") -> None:
                self.outer = outer

            def __len__(self) -> int:
                return len(self.outer.clean_tokens)

            def __getitem__(self, index: int) -> Dict[str, Any]:
                outer = self.outer
                clean = outer.clean_tokens[index]
                noisy = outer.noise.apply(clean, outer.rng)
                return {
                    "x": torch.tensor(outer.codec.input_vector(noisy), dtype=torch.float32),
                    "type_target": torch.tensor(outer.codec.type_targets(clean), dtype=torch.long),
                    "pitch_target": torch.tensor(outer.codec.pitch_targets(clean), dtype=torch.float32),
                    "note_mask": torch.tensor(outer.codec.note_mask(clean), dtype=torch.float32),
                }

        self.codec = BarTokenCodec(config)
        self.noise = BarNoiseInjector(config)
        self.rng = np.random.default_rng(config.random_seed)
        self.clean_tokens = [self.codec.clean_tokens(bar) for bar in bars]
        self.torch_dataset = _Dataset(self)


class DenoisingBarVAE:
    """Factory for the torch module to keep torch import local to training."""

    @staticmethod
    def build(config: DenoisingVAEConfig):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(config.steps_per_bar, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
                self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(config.latent_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                self.type_head = nn.Linear(config.hidden_dim, config.steps_per_bar * 3)
                self.pitch_head = nn.Linear(config.hidden_dim, config.steps_per_bar)

            def encode(self, x):
                hidden = self.encoder(x)
                return self.mu(hidden), self.logvar(hidden)

            def reparameterize(self, mu, logvar):
                if not self.training:
                    return mu
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                hidden = self.decoder(z)
                type_logits = self.type_head(hidden).view(-1, config.steps_per_bar, 3)
                pitch = self.pitch_head(hidden)
                return type_logits, pitch, mu, logvar

        return _Model()


class DenoisingVAETrainer:
    """Train the denoising VAE and expose z_mu latent features."""

    def __init__(self, config: DenoisingVAEConfig) -> None:
        self.config = config
        self.training_log: List[Dict[str, float]] = []
        self.model = None

    def fit(self, bars: Sequence[BarRecord]) -> "DenoisingVAETrainer":
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader

        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        device = torch.device(self.config.device)
        dataset = DenoisingBarDataset(bars, self.config)
        loader = DataLoader(dataset.torch_dataset, batch_size=self.config.batch_size, shuffle=True)
        model = DenoisingBarVAE.build(self.config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epochs):
            model.train()
            totals = Counter()
            for batch in loader:
                x = batch["x"].to(device)
                type_target = batch["type_target"].to(device)
                pitch_target = batch["pitch_target"].to(device)
                note_mask = batch["note_mask"].to(device)
                type_logits, pitch, mu, logvar = model(x)
                type_loss = functional.cross_entropy(
                    type_logits.reshape(-1, 3),
                    type_target.reshape(-1),
                )
                pitch_error = (pitch - pitch_target) ** 2
                pitch_loss = (pitch_error * note_mask).sum() / torch.clamp(note_mask.sum(), min=1.0)
                kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                kl_weight = self._kl_weight(epoch)
                loss = type_loss + self.config.pitch_weight * pitch_loss + kl_weight * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = int(x.shape[0])
                totals["count"] += batch_size
                totals["loss"] += float(loss.detach().cpu()) * batch_size
                totals["type_loss"] += float(type_loss.detach().cpu()) * batch_size
                totals["pitch_loss"] += float(pitch_loss.detach().cpu()) * batch_size
                totals["kl_loss"] += float(kl_loss.detach().cpu()) * batch_size
            count = max(1, int(totals["count"]))
            self.training_log.append({
                "epoch": float(epoch),
                "loss": float(totals["loss"] / count),
                "type_loss": float(totals["type_loss"] / count),
                "pitch_loss": float(totals["pitch_loss"] / count),
                "kl_loss": float(totals["kl_loss"] / count),
                "kl_weight": float(self._kl_weight(epoch)),
            })
        self.model = model
        return self

    def encode(self, bars: Sequence[BarRecord]) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("VAE model has not been trained.")
        codec = BarTokenCodec(self.config)
        x = np.asarray([codec.input_vector(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32, device=device)
            mu, _ = self.model.encode(tensor)
        return mu.detach().cpu().numpy().astype(np.float32)

    def evaluate_reconstruction(self, bars: Sequence[BarRecord]) -> Dict[str, Any]:
        import torch

        if self.model is None:
            raise RuntimeError("VAE model has not been trained.")
        codec = BarTokenCodec(self.config)
        x = np.asarray([codec.input_vector(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        type_targets = np.asarray([codec.type_targets(codec.clean_tokens(bar)) for bar in bars], dtype=np.int64)
        pitch_targets = np.asarray([codec.pitch_targets(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        note_masks = np.asarray([codec.note_mask(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32, device=device)
            type_logits, pitch, mu, logvar = self.model(tensor)
            type_pred = torch.argmax(type_logits, dim=-1).detach().cpu().numpy()
            pitch_pred = pitch.detach().cpu().numpy()
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        type_accuracy = float(np.mean(type_pred == type_targets)) if type_targets.size else 0.0
        note_positions = note_masks > 0
        note_type_accuracy = float(np.mean(type_pred[note_positions] == NOTE_TYPE_ON)) if np.any(note_positions) else 0.0
        special_positions = ~note_positions
        special_type_accuracy = float(np.mean(type_pred[special_positions] == type_targets[special_positions])) if np.any(special_positions) else 0.0
        pitch_mse = float(np.mean(((pitch_pred - pitch_targets) ** 2)[note_positions])) if np.any(note_positions) else 0.0
        return {
            "type_accuracy": type_accuracy,
            "note_on_type_accuracy": note_type_accuracy,
            "rest_sustain_type_accuracy": special_type_accuracy,
            "pitch_mse_note_on": pitch_mse,
            "kl_loss": float(kl_loss.detach().cpu()),
        }

    def save(self, path: Path) -> None:
        import torch

        if self.model is None:
            raise RuntimeError("VAE model has not been trained.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "state_dict": self.model.state_dict(),
            "training_log": self.training_log,
        }, path)

    def _kl_weight(self, epoch: int) -> float:
        warmup = max(0, int(self.config.kl_warmup_epochs))
        if warmup <= 0:
            return float(self.config.beta_kl)
        return float(self.config.beta_kl * min(1.0, float(epoch + 1) / float(warmup)))


class LatentClusterAnalyzer:
    """Cluster latent vectors and summarize label distribution."""

    def __init__(self, config: LatentClusteringConfig) -> None:
        self.config = config

    def cluster(self, latents: np.ndarray) -> np.ndarray:
        if len(latents) == 0:
            return np.asarray([], dtype=int)
        if self.config.method == "kmeans":
            n_clusters = min(max(1, int(self.config.n_clusters)), len(latents))
            return KMeans(
                n_clusters=n_clusters,
                random_state=self.config.random_seed,
                n_init="auto",
            ).fit_predict(latents).astype(int)
        if self.config.method == "agglomerative":
            if len(latents) == 1:
                return np.asarray([0], dtype=int)
            distances = pdist(latents, metric="euclidean")
            z_matrix = linkage(distances, method=self.config.linkage_method)
            raw = fcluster(z_matrix, t=float(self.config.distance_threshold), criterion="distance")
            return self._compact(raw)
        raise ValueError("latent clustering method must be 'kmeans' or 'agglomerative'.")

    def diagnostics(self, bars: Sequence[BarRecord], latents: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        counts = Counter(int(label) for label in labels)
        values = np.asarray([int(value) for value in counts.values() if int(value) > 0], dtype=np.float64)
        total = int(np.sum(values)) if values.size else 0
        probabilities = values / max(1.0, float(total)) if values.size else np.asarray([], dtype=np.float64)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))) if values.size else 0.0
        singleton_count = int(np.sum(values == 1)) if values.size else 0
        max_label, max_count = self._max_count(counts)
        return {
            "config": asdict(self.config),
            "total_assignments": total,
            "used_label_count": int(len(values)),
            "singleton_label_count": singleton_count,
            "singleton_ratio": float(singleton_count / max(1, len(values))),
            "max_label": str(max_label) if max_label is not None else None,
            "max_label_count": int(max_count),
            "max_label_ratio": float(max_count / max(1, total)),
            "entropy": entropy,
            "normalized_entropy": float(entropy / math.log(len(values))) if len(values) > 1 else 0.0,
            "effective_label_count": float(math.exp(entropy)) if values.size else 0.0,
            "latent": self._latent_summary(latents),
            "top_labels": self._top_labels(counts, total),
            "top_label_examples": self._top_label_examples(bars, labels, counts),
            "nearest_neighbors": self._nearest_neighbors(bars, latents),
        }

    def _compact(self, labels: Sequence[int]) -> np.ndarray:
        mapping: Dict[int, int] = {}
        result: List[int] = []
        for label in labels:
            value = int(label)
            if value not in mapping:
                mapping[value] = len(mapping)
            result.append(mapping[value])
        return np.asarray(result, dtype=int)

    def _max_count(self, counts: Counter) -> tuple[Optional[int], int]:
        if not counts:
            return None, 0
        label, count = max(counts.items(), key=lambda item: int(item[1]))
        return int(label), int(count)

    def _latent_summary(self, latents: np.ndarray) -> Dict[str, Any]:
        if latents.size == 0:
            return {}
        return {
            "shape": [int(item) for item in latents.shape],
            "mean_abs": float(np.mean(np.abs(latents))),
            "std": float(np.std(latents)),
            "dim_mean": [float(value) for value in np.mean(latents, axis=0)],
            "dim_std": [float(value) for value in np.std(latents, axis=0)],
        }

    def _top_labels(self, counts: Counter, total: int, limit: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "label": str(label),
                "count": int(count),
                "ratio": float(count / max(1, total)),
            }
            for label, count in counts.most_common(limit)
        ]

    def _top_label_examples(
        self,
        bars: Sequence[BarRecord],
        labels: np.ndarray,
        counts: Counter,
        limit: int = 8,
        examples_per_label: int = 4,
    ) -> List[Dict[str, Any]]:
        bars_by_label: Dict[int, List[BarRecord]] = defaultdict(list)
        for bar, label in zip(bars, labels):
            bars_by_label[int(label)].append(bar)
        result = []
        for label, count in counts.most_common(limit):
            result.append({
                "label": str(label),
                "count": int(count),
                "examples": [
                    self._bar_example(bar)
                    for bar in bars_by_label[int(label)][:examples_per_label]
                ],
            })
        return result

    def _nearest_neighbors(
        self,
        bars: Sequence[BarRecord],
        latents: np.ndarray,
        anchor_count: int = 8,
        neighbor_count: int = 5,
    ) -> List[Dict[str, Any]]:
        if len(bars) <= 1:
            return []
        anchors = np.linspace(0, len(bars) - 1, num=min(anchor_count, len(bars)), dtype=int)
        result = []
        for anchor in anchors:
            distances = np.linalg.norm(latents - latents[int(anchor)], axis=1)
            nearest = [
                int(index)
                for index in np.argsort(distances)
                if int(index) != int(anchor)
            ][:neighbor_count]
            result.append({
                "anchor": self._bar_example(bars[int(anchor)]),
                "neighbors": [
                    {
                        **self._bar_example(bars[index]),
                        "latent_distance": float(distances[index]),
                    }
                    for index in nearest
                ],
            })
        return result

    def _bar_example(self, bar: BarRecord) -> Dict[str, Any]:
        return {
            "song_id": bar.song_id,
            "bar_index": int(bar.bar_index),
            "relative_tokens": list(bar.relative_tokens),
            "absolute_tokens": list(bar.absolute_tokens),
            "token_variance": round(float(bar.token_variance), 6),
            "sharing_score": round(float(bar.sharing_score), 6),
        }
