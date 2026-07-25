#!/usr/bin/env python3
"""Offline contrastive training for DVAE-latent theme embeddings."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.config_loader import ConfigView
from diagnostics.diagnostics import DiagnosticsBase
from model.theme_encoder import BiLSTMAttentionThemeEncoder, ThemeEncoderConfig


@dataclass(frozen=True)
class ThemeEncoderTrainingConfig:
    """Configuration for contrastive theme encoder training."""

    theme_bars: int = 8
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    temperature: float = 0.1
    jitter_std: float = 0.02
    time_mask_ratio: float = 0.10
    validation_ratio: float = 0.20
    random_seed: int = 42
    device: str = "cpu"
    diagnostics_top_k: int = 5

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ThemeEncoderTrainingConfig":
        """Build training config from style config."""
        section = ConfigView(config).section("theme_encoder_training")
        return cls(
            theme_bars=int(section.get("theme_bars", 8)),
            epochs=int(section.get("epochs", 50)),
            batch_size=int(section.get("batch_size", 64)),
            learning_rate=float(section.get("learning_rate", 1.0e-3)),
            weight_decay=float(section.get("weight_decay", 1.0e-4)),
            temperature=float(section.get("temperature", 0.1)),
            jitter_std=float(section.get("jitter_std", 0.02)),
            time_mask_ratio=float(section.get("time_mask_ratio", 0.10)),
            validation_ratio=float(section.get("validation_ratio", 0.20)),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
            diagnostics_top_k=int(section.get("diagnostics_top_k", 5)),
        )


@dataclass
class ThemeExample:
    """One opening-theme latent sequence from one song variant."""

    song_id: str
    base_song_id: str
    sequence: np.ndarray
    start_bar_index: int
    end_bar_index: int


@dataclass
class ThemeDatasetBuildResult:
    """Theme examples and label metadata."""

    examples: List[ThemeExample]
    base_id_to_label: Dict[str, int]
    summary: Dict[str, Any]


@dataclass
class ThemeEncoderTrainingResult:
    """Paths and diagnostics produced by training."""

    model_path: Path
    diagnostics_path: Path
    report_path: Path
    embeddings_path: Path
    index_path: Path
    diagnostics: Dict[str, Any]


class ThemeSequenceDataset(Dataset):
    """Torch dataset of opening latent theme sequences."""

    def __init__(self, examples: Sequence[ThemeExample], base_id_to_label: Dict[str, int]) -> None:
        self.examples = list(examples)
        self.base_id_to_label = dict(base_id_to_label)

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sequence, base label, and example index."""
        example = self.examples[index]
        return (
            torch.from_numpy(example.sequence).float(),
            torch.tensor(self.base_id_to_label[example.base_song_id], dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )


class LatentThemeDatasetReader:
    """Read latent_mu.npy and latent_index.json into opening theme examples."""

    TRANSPOSE_SUFFIX = re.compile(r"_T[+-]\d+$")

    def load(self, latent_dir: str | Path, theme_bars: int) -> ThemeDatasetBuildResult:
        """Load opening theme windows from exported latent data."""
        directory = Path(latent_dir)
        mu_path = directory / "latent_mu.npy"
        index_path = directory / "latent_index.json"
        if not mu_path.exists():
            raise FileNotFoundError(f"Missing latent_mu.npy: {mu_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing latent_index.json: {index_path}")
        mu = np.load(mu_path).astype(np.float32)
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        if len(rows) != int(mu.shape[0]):
            raise ValueError("latent_index.json row count must match latent_mu.npy rows.")
        grouped = self._group_rows(rows)
        examples: List[ThemeExample] = []
        skipped_short = 0
        n_bars = max(1, int(theme_bars))
        for song_id, indices in grouped.items():
            ordered = sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            if len(ordered) < n_bars:
                skipped_short += 1
                continue
            selected = ordered[:n_bars]
            examples.append(ThemeExample(
                song_id=song_id,
                base_song_id=self.base_song_id(song_id),
                sequence=mu[selected].astype(np.float32),
                start_bar_index=int(rows[selected[0]].get("bar_index", 0)),
                end_bar_index=int(rows[selected[-1]].get("bar_index", 0)),
            ))
        if not examples:
            raise ValueError("No theme examples were built from latent data.")
        base_ids = sorted({example.base_song_id for example in examples})
        base_id_to_label = {base_id: index for index, base_id in enumerate(base_ids)}
        counts: Dict[str, int] = {}
        for example in examples:
            counts[example.base_song_id] = counts.get(example.base_song_id, 0) + 1
        summary = {
            "latent_dir": str(directory),
            "latent_dim": int(mu.shape[1]),
            "theme_bars": int(n_bars),
            "song_variant_count": int(len(examples)),
            "base_song_count": int(len(base_ids)),
            "skipped_short_song_count": int(skipped_short),
            "min_variants_per_base": int(min(counts.values())) if counts else 0,
            "max_variants_per_base": int(max(counts.values())) if counts else 0,
            "mean_variants_per_base": float(np.mean(list(counts.values()))) if counts else 0.0,
        }
        return ThemeDatasetBuildResult(examples=examples, base_id_to_label=base_id_to_label, summary=summary)

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group latent rows by song_id."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return grouped

    def base_song_id(self, song_id: str) -> str:
        """Remove transposition suffix from song_id."""
        return self.TRANSPOSE_SUFFIX.sub("", str(song_id))


class ThemeAugmenter:
    """Create two lightly perturbed views of a latent theme sequence."""

    def __init__(self, jitter_std: float, time_mask_ratio: float) -> None:
        self.jitter_std = float(jitter_std)
        self.time_mask_ratio = float(time_mask_ratio)

    def two_views(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views."""
        return self._augment(sequence), self._augment(sequence)

    def _augment(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply latent jitter and mild time-step masking."""
        output = sequence
        if self.jitter_std > 0.0:
            output = output + torch.randn_like(output) * self.jitter_std
        if self.time_mask_ratio > 0.0 and output.shape[1] > 1:
            keep = torch.rand(output.shape[:2], device=output.device) > self.time_mask_ratio
            keep[:, 0] = True
            fallback = output.mean(dim=1, keepdim=True)
            output = torch.where(keep.unsqueeze(-1), output, fallback)
        return output


class SupervisedContrastiveLoss(torch.nn.Module):
    """Supervised contrastive loss over two augmented views."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return contrastive loss where same base_song_id views are positives."""
        z = torch.cat([z1, z2], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        similarity = torch.matmul(z, z.T) / self.temperature
        self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
        logits = similarity.masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        loss = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count)
        return loss.mean()


class ThemeEncoderTrainingPipeline:
    """Train and evaluate a frozen-backbone candidate theme encoder offline."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.training_config = self._with_overrides(ThemeEncoderTrainingConfig.from_config(config), overrides or {})
        self.model_config = ThemeEncoderConfig.from_config(config)
        self.reader = LatentThemeDatasetReader()
        self.diagnostics = DiagnosticsBase("theme_encoder_training")

    def run(self, latent_dir: str | Path, output_dir: str | Path) -> ThemeEncoderTrainingResult:
        """Train theme encoder and write diagnostics/report artifacts."""
        self._set_seed()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        data = self.reader.load(latent_dir, int(self.training_config.theme_bars))
        self.model_config = self._resolved_model_config(data)
        train_dataset, val_dataset, split_diag = self._split_dataset(data)
        self.diagnostics.record_stage("input", data.summary)
        self.diagnostics.record_stage("dataset_split", split_diag)
        model = BiLSTMAttentionThemeEncoder(self.model_config).to(self.training_config.device)
        loss_fn = SupervisedContrastiveLoss(float(self.training_config.temperature))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        history = self._train(model, loss_fn, optimizer, train_dataset, val_dataset)
        all_dataset = ThemeSequenceDataset(data.examples, data.base_id_to_label)
        embeddings = self._encode(model, all_dataset)
        train_embeddings = self._encode(model, train_dataset)
        val_embeddings = self._encode(model, val_dataset)
        diagnostics = {
            "all": self._evaluate_embeddings(data.examples, data.base_id_to_label, embeddings),
            "train": self._evaluate_embeddings(train_dataset.examples, data.base_id_to_label, train_embeddings),
            "validation": self._evaluate_embeddings(val_dataset.examples, data.base_id_to_label, val_embeddings),
        }
        paths = self._write_outputs(output, model, data, embeddings, history, diagnostics)
        self.diagnostics.record_stage("training", {
            "config": self.training_config.__dict__,
            "model_config": self.model_config.to_dict(),
            "history": history,
        })
        self.diagnostics.record_stage("embedding_eval", diagnostics)
        self.diagnostics.record_stage("paths", paths)
        diagnostics_path = output / "theme_encoder_training_diagnostics.json"
        self.diagnostics.write(diagnostics_path)
        return ThemeEncoderTrainingResult(
            model_path=Path(paths["model"]),
            diagnostics_path=diagnostics_path,
            report_path=Path(paths["report"]),
            embeddings_path=Path(paths["embeddings"]),
            index_path=Path(paths["index"]),
            diagnostics=self.diagnostics.to_dict(),
        )

    def _resolved_model_config(self, data: ThemeDatasetBuildResult) -> ThemeEncoderConfig:
        """Resolve latent_dim from dataset."""
        values = dict(self.model_config.__dict__)
        values["latent_dim"] = int(data.summary["latent_dim"])
        return ThemeEncoderConfig(**values)

    def _with_overrides(
        self,
        base: ThemeEncoderTrainingConfig,
        overrides: Dict[str, Any],
    ) -> ThemeEncoderTrainingConfig:
        """Apply CLI overrides and resolve device."""
        values = dict(base.__dict__)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        values["device"] = self._resolve_device(str(values.get("device", "cpu")))
        return ThemeEncoderTrainingConfig(**values)

    def _resolve_device(self, requested: str) -> str:
        """Return an available torch device."""
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _set_seed(self) -> None:
        """Seed Python, numpy, and torch RNGs."""
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _split_dataset(
        self,
        data: ThemeDatasetBuildResult,
    ) -> tuple[ThemeSequenceDataset, ThemeSequenceDataset, Dict[str, Any]]:
        """Split by base_song_id so song variants do not leak across train/validation."""
        rng = np.random.default_rng(int(self.training_config.random_seed))
        base_ids = np.asarray(sorted(data.base_id_to_label.keys()), dtype=object)
        rng.shuffle(base_ids)
        val_count = int(round(len(base_ids) * max(0.0, min(0.9, float(self.training_config.validation_ratio)))))
        if len(base_ids) > 1:
            val_count = max(1, min(val_count, len(base_ids) - 1))
        val_base_ids = {str(value) for value in base_ids[:val_count].tolist()}
        train_examples = [example for example in data.examples if example.base_song_id not in val_base_ids]
        val_examples = [example for example in data.examples if example.base_song_id in val_base_ids]
        return (
            ThemeSequenceDataset(train_examples, data.base_id_to_label),
            ThemeSequenceDataset(val_examples, data.base_id_to_label),
            {
                "split_unit": "base_song_id",
                "base_song_count": int(len(base_ids)),
                "train_base_song_count": int(len(base_ids) - len(val_base_ids)),
                "validation_base_song_count": int(len(val_base_ids)),
                "train_size": int(len(train_examples)),
                "validation_size": int(len(val_examples)),
                "validation_base_song_ids": sorted(val_base_ids),
            },
        )

    def _train(
        self,
        model: BiLSTMAttentionThemeEncoder,
        loss_fn: SupervisedContrastiveLoss,
        optimizer: torch.optim.Optimizer,
        train_dataset: ThemeSequenceDataset,
        val_dataset: ThemeSequenceDataset,
    ) -> List[Dict[str, float]]:
        """Run fixed-epoch contrastive training."""
        generator = torch.Generator().manual_seed(int(self.training_config.random_seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.training_config.batch_size),
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        augmenter = ThemeAugmenter(float(self.training_config.jitter_std), float(self.training_config.time_mask_ratio))
        history: List[Dict[str, float]] = []
        for epoch in range(int(self.training_config.epochs)):
            train_loss = self._run_epoch(model, loss_fn, train_loader, augmenter, optimizer)
            val_loss = self._run_epoch(model, loss_fn, val_loader, augmenter, None)
            history.append({
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })
        return history

    def _run_epoch(
        self,
        model: BiLSTMAttentionThemeEncoder,
        loss_fn: SupervisedContrastiveLoss,
        loader: DataLoader,
        augmenter: ThemeAugmenter,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> float:
        """Run one training or validation epoch."""
        is_train = optimizer is not None
        model.train(is_train)
        total = 0.0
        count = 0
        for sequence, labels, _ in loader:
            sequence = sequence.to(self.training_config.device)
            labels = labels.to(self.training_config.device)
            view_a, view_b = augmenter.two_views(sequence)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_train):
                z_a = model(view_a)
                z_b = model(view_b)
                loss = loss_fn(z_a, z_b, labels)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            batch_size = int(sequence.shape[0])
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
        return total / max(1, count)

    def _encode(self, model: BiLSTMAttentionThemeEncoder, dataset: ThemeSequenceDataset) -> np.ndarray:
        """Encode clean theme examples."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        model.eval()
        rows: List[np.ndarray] = []
        with torch.no_grad():
            for sequence, _, _ in loader:
                sequence = sequence.to(self.training_config.device)
                rows.append(model(sequence).detach().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(np.float32)

    def _evaluate_embeddings(
        self,
        examples: Sequence[ThemeExample],
        base_id_to_label: Dict[str, int],
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute retrieval and geometry diagnostics."""
        labels = np.asarray([base_id_to_label[example.base_song_id] for example in examples], dtype=np.int64)
        similarity = np.matmul(embeddings, embeddings.T)
        self_mask = np.eye(len(labels), dtype=bool)
        positive_mask = labels[:, None] == labels[None, :]
        positive_mask = positive_mask & ~self_mask
        negative_mask = labels[:, None] != labels[None, :]
        positive_values = similarity[positive_mask]
        negative_values = similarity[negative_mask]
        top = self._retrieval(labels, similarity)
        neighbors = self._nearest_neighbors(examples, similarity)
        errors = self._retrieval_errors(examples, labels, similarity)
        margin = float(np.mean(positive_values) - np.mean(negative_values)) if len(positive_values) and len(negative_values) else 0.0
        return {
            "sample_count": int(len(examples)),
            "base_song_count": int(len({example.base_song_id for example in examples})),
            "embedding_norm_mean": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "embedding_norm_std": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "positive_cosine_mean": float(np.mean(positive_values)) if len(positive_values) else 0.0,
            "positive_cosine_std": float(np.std(positive_values)) if len(positive_values) else 0.0,
            "negative_cosine_mean": float(np.mean(negative_values)) if len(negative_values) else 0.0,
            "negative_cosine_std": float(np.std(negative_values)) if len(negative_values) else 0.0,
            "pos_neg_margin": margin,
            "retrieval_top1_accuracy": top["top1"],
            "retrieval_top5_accuracy": top["top5"],
            "transpose_invariance_score": float(np.mean(positive_values)) if len(positive_values) else 0.0,
            "nearest_neighbors": neighbors,
            "retrieval_top1_errors": errors,
        }

    def _retrieval(self, labels: np.ndarray, similarity: np.ndarray) -> Dict[str, float]:
        """Return top-k same-base retrieval accuracy."""
        ranked = np.argsort(-similarity, axis=1)
        top1_hits = 0
        top5_hits = 0
        for row_index, order in enumerate(ranked):
            order = [int(index) for index in order if int(index) != row_index]
            top1_hits += int(bool(order) and labels[order[0]] == labels[row_index])
            top5_hits += int(any(labels[index] == labels[row_index] for index in order[:5]))
        total = max(1, int(len(labels)))
        return {"top1": float(top1_hits / total), "top5": float(top5_hits / total)}

    def _nearest_neighbors(self, examples: Sequence[ThemeExample], similarity: np.ndarray) -> List[Dict[str, Any]]:
        """Return nearest-neighbor examples for diagnostics."""
        top_k = max(1, int(self.training_config.diagnostics_top_k))
        output: List[Dict[str, Any]] = []
        for index, example in enumerate(examples[: min(50, len(examples))]):
            order = np.argsort(-similarity[index])
            neighbors = []
            for neighbor_index in order:
                neighbor_index = int(neighbor_index)
                if neighbor_index == index:
                    continue
                neighbor = examples[neighbor_index]
                neighbors.append({
                    "song_id": neighbor.song_id,
                    "base_song_id": neighbor.base_song_id,
                    "cosine": float(similarity[index, neighbor_index]),
                    "same_base_song": bool(neighbor.base_song_id == example.base_song_id),
                })
                if len(neighbors) >= top_k:
                    break
            output.append({
                "song_id": example.song_id,
                "base_song_id": example.base_song_id,
                "neighbors": neighbors,
            })
        return output

    def _retrieval_errors(
        self,
        examples: Sequence[ThemeExample],
        labels: np.ndarray,
        similarity: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Return top-1 nearest-neighbor mistakes across base_song_id."""
        ranked = np.argsort(-similarity, axis=1)
        errors: List[Dict[str, Any]] = []
        for row_index, order in enumerate(ranked):
            order = [int(index) for index in order if int(index) != row_index]
            if not order:
                continue
            neighbor_index = order[0]
            if labels[neighbor_index] == labels[row_index]:
                continue
            query = examples[row_index]
            neighbor = examples[neighbor_index]
            errors.append({
                "song_id": query.song_id,
                "base_song_id": query.base_song_id,
                "nearest_song_id": neighbor.song_id,
                "nearest_base_song_id": neighbor.base_song_id,
                "cosine": float(similarity[row_index, neighbor_index]),
            })
        errors.sort(key=lambda row: float(row["cosine"]), reverse=True)
        return errors

    def _write_outputs(
        self,
        output_dir: Path,
        model: BiLSTMAttentionThemeEncoder,
        data: ThemeDatasetBuildResult,
        embeddings: np.ndarray,
        history: Sequence[Dict[str, float]],
        diagnostics: Dict[str, Any],
    ) -> Dict[str, str]:
        """Write model, embeddings, diagnostics report, and plots."""
        model_path = output_dir / "theme_encoder.pt"
        embeddings_path = output_dir / "theme_embeddings.npy"
        index_path = output_dir / "theme_embedding_index.json"
        report_path = output_dir / "theme_encoder_report.md"
        pca_path = output_dir / "theme_embeddings_pca.png"
        tsne_path = output_dir / "theme_embeddings_tsne.png"
        torch.save({
            "model_type": "BiLSTMAttentionThemeEncoder",
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "base_id_to_label": data.base_id_to_label,
            "state_dict": model.state_dict(),
        }, model_path)
        np.save(embeddings_path, embeddings.astype(np.float32))
        index_path.write_text(json.dumps([
            {
                "row_index": index,
                "song_id": example.song_id,
                "base_song_id": example.base_song_id,
                "label": data.base_id_to_label[example.base_song_id],
                "start_bar_index": int(example.start_bar_index),
                "end_bar_index": int(example.end_bar_index),
            }
            for index, example in enumerate(data.examples)
        ], indent=2), encoding="utf-8")
        self._write_plots(embeddings, np.asarray([data.base_id_to_label[e.base_song_id] for e in data.examples]), pca_path, tsne_path)
        report_path.write_text(self._markdown_report(data, history, diagnostics, pca_path, tsne_path), encoding="utf-8")
        return {
            "model": str(model_path),
            "embeddings": str(embeddings_path),
            "index": str(index_path),
            "report": str(report_path),
            "pca_plot": str(pca_path) if pca_path.exists() else "",
            "tsne_plot": str(tsne_path) if tsne_path.exists() else "",
        }

    def _write_plots(self, embeddings: np.ndarray, labels: np.ndarray, pca_path: Path, tsne_path: Path) -> None:
        """Write PCA and optional t-SNE scatter plots."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        pca = self._pca_2d(embeddings)
        self._scatter_plot(pca, labels, pca_path, "Theme Embeddings PCA")
        try:
            from sklearn.manifold import TSNE

            perplexity = max(5, min(30, int((len(embeddings) - 1) / 3)))
            tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=int(self.training_config.random_seed))
            coords = tsne.fit_transform(embeddings)
            self._scatter_plot(coords, labels, tsne_path, "Theme Embeddings t-SNE")
        except Exception:
            return

    def _pca_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """Return first two PCA coordinates using numpy SVD."""
        centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T

    def _scatter_plot(self, coords: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
        """Write one scatter plot."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=18, cmap="tab20", alpha=0.85)
        plt.title(title)
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def _markdown_report(
        self,
        data: ThemeDatasetBuildResult,
        history: Sequence[Dict[str, float]],
        diagnostics: Dict[str, Any],
        pca_path: Path,
        tsne_path: Path,
    ) -> str:
        """Render a compact Markdown report."""
        final = history[-1] if history else {}
        lines = [
            "# Theme Encoder Report",
            "",
            "## Dataset",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| base_song_count | {data.summary['base_song_count']} |",
            f"| song_variant_count | {data.summary['song_variant_count']} |",
            f"| theme_bars | {data.summary['theme_bars']} |",
            f"| latent_dim | {data.summary['latent_dim']} |",
            "",
            "## Training",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| epochs | {len(history)} |",
            f"| final_train_loss | {float(final.get('train_loss', math.nan)):.6f} |",
            f"| final_val_loss | {float(final.get('val_loss', math.nan)):.6f} |",
            "",
            "## Embedding Evaluation",
            "",
        ]
        for split_name in ["all", "train", "validation"]:
            split = diagnostics.get(split_name, {})
            lines.extend([
                f"### {split_name.title()}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| sample_count | {int(split.get('sample_count', 0))} |",
                f"| base_song_count | {int(split.get('base_song_count', 0))} |",
            ])
            for key in [
                "positive_cosine_mean",
                "negative_cosine_mean",
                "pos_neg_margin",
                "retrieval_top1_accuracy",
                "retrieval_top5_accuracy",
                "transpose_invariance_score",
                "embedding_norm_mean",
                "embedding_norm_std",
            ]:
                lines.append(f"| {key} | {float(split.get(key, 0.0)):.6f} |")
            lines.append("")
        lines.extend(["", "## Plots", ""])
        if pca_path.exists():
            lines.append(f"![PCA]({pca_path})")
        if tsne_path.exists():
            lines.append(f"![t-SNE]({tsne_path})")
        lines.extend(["", "## Validation Top-1 Retrieval Errors", ""])
        validation_errors = diagnostics.get("validation", {}).get("retrieval_top1_errors", [])
        if not validation_errors:
            lines.append("No validation top-1 errors.")
        for row in validation_errors[:20]:
            lines.append(
                "- `{song}` -> `{neighbor}` cosine={cosine:.4f}".format(
                    song=row["song_id"],
                    neighbor=row["nearest_song_id"],
                    cosine=float(row["cosine"]),
                )
            )
        lines.extend(["", "## Validation Nearest Neighbor Samples", ""])
        for row in diagnostics.get("validation", {}).get("nearest_neighbors", [])[:10]:
            lines.append(f"- `{row['song_id']}`")
            for neighbor in row.get("neighbors", [])[:3]:
                flag = "same" if neighbor["same_base_song"] else "different"
                lines.append(f"  - {flag}: `{neighbor['song_id']}` cosine={neighbor['cosine']:.4f}")
        lines.extend(["", "## All Nearest Neighbor Samples", ""])
        for row in diagnostics.get("all", {}).get("nearest_neighbors", [])[:10]:
            lines.append(f"- `{row['song_id']}`")
            for neighbor in row.get("neighbors", [])[:3]:
                flag = "same" if neighbor["same_base_song"] else "different"
                lines.append(f"  - {flag}: `{neighbor['song_id']}` cosine={neighbor['cosine']:.4f}")
        lines.append("")
        return "\n".join(lines)
