#!/usr/bin/env python3
"""Next-bar retrieval benchmark for MidiTok-style event sequences."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.miditok_bar_sequence_encoder import MidiTokBarSequenceEncoderConfig
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader
from pipeline.miditok_bar_sequence_training_pipeline import MidiTokBarTokenBuilder


@dataclass(frozen=True)
class NextBarSample:
    """One song-local next-bar retrieval sample."""

    context_indices: List[int]
    target_index: int
    song_id: str
    base_song_id: str
    target_bar_index: int


class NextBarRetrievalDataset(Dataset):
    """Dataset for in-batch next-bar retrieval."""

    def __init__(
        self,
        samples: Sequence[NextBarSample],
        mu: np.ndarray,
        tokens: np.ndarray,
        token_mask: np.ndarray,
        context_bars: int,
    ) -> None:
        self.samples = list(samples)
        self.mu = mu.astype(np.float32)
        self.tokens = tokens.astype(np.int64)
        self.token_mask = token_mask.astype(bool)
        self.context_bars = int(context_bars)

    def __len__(self) -> int:
        """Return sample count."""
        return int(len(self.samples))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one padded retrieval sample."""
        sample = self.samples[index]
        latent_dim = int(self.mu.shape[1])
        max_events = int(self.tokens.shape[1])
        context_mu = np.zeros((self.context_bars, latent_dim), dtype=np.float32)
        context_tokens = np.zeros((self.context_bars, max_events, 5), dtype=np.int64)
        context_token_mask = np.ones((self.context_bars, max_events), dtype=bool)
        context_pad_mask = np.ones((self.context_bars,), dtype=bool)
        recent = sample.context_indices[-self.context_bars:]
        offset = self.context_bars - len(recent)
        for local, row_index in enumerate(recent):
            slot = offset + local
            context_mu[slot] = self.mu[row_index]
            context_tokens[slot] = self.tokens[row_index]
            context_token_mask[slot] = self.token_mask[row_index]
            context_pad_mask[slot] = False
        return {
            "context_mu": torch.from_numpy(context_mu).float(),
            "context_tokens": torch.from_numpy(context_tokens).long(),
            "context_token_mask": torch.from_numpy(context_token_mask).bool(),
            "context_pad_mask": torch.from_numpy(context_pad_mask).bool(),
            "target_mu": torch.from_numpy(self.mu[sample.target_index]).float(),
            "target_tokens": torch.from_numpy(self.tokens[sample.target_index]).long(),
            "target_token_mask": torch.from_numpy(self.token_mask[sample.target_index]).bool(),
        }


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positions to batch-first sequence."""
        return x + self.pe[:, : x.shape[1], :]


class EventBarEncoder(nn.Module):
    """Encode one bar's MidiTok-style event token sequence."""

    PAD_ID = 0

    def __init__(self, config: MidiTokBarSequenceEncoderConfig) -> None:
        super().__init__()
        self.config = config
        emb_dim = int(config.field_embedding_dim)
        self.position_embedding = nn.Embedding(int(config.position_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.track_embedding = nn.Embedding(int(config.track_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.pitch_embedding = nn.Embedding(int(config.pitch_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.duration_embedding = nn.Embedding(int(config.duration_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.velocity_embedding = nn.Embedding(int(config.velocity_bins) + 1, emb_dim, padding_idx=self.PAD_ID)
        self.event_projection = nn.Linear(emb_dim * 5, int(config.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config.d_model)))
        self.position_encoding = SinusoidalPositionEncoding(int(config.d_model), int(config.max_events) + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(config.n_layers))
        self.norm = nn.LayerNorm(int(config.d_model))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return CLS embedding for tokens [B, E, 5]."""
        position = self.position_embedding(tokens[..., 0].long())
        track = self.track_embedding(tokens[..., 1].long())
        pitch = self.pitch_embedding(tokens[..., 2].long())
        duration = self.duration_embedding(tokens[..., 3].long())
        velocity = self.velocity_embedding(tokens[..., 4].long())
        events = self.event_projection(torch.cat([position, track, pitch, duration, velocity], dim=-1))
        batch_size = int(events.shape[0])
        cls = self.cls_token.expand(batch_size, 1, -1)
        sequence = self.position_encoding(torch.cat([cls, events], dim=1))
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask.bool()], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=full_mask)
        return self.norm(encoded[:, 0])


class ContextEncoder(nn.Module):
    """Causal-ish context encoder with CLS pooling."""

    def __init__(self, d_model: int, context_bars: int, n_layers: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.position_encoding = SinusoidalPositionEncoding(int(d_model), int(context_bars) + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_model) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, context: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Encode context sequence [B, C, D]."""
        batch_size = int(context.shape[0])
        cls = self.cls_token.expand(batch_size, 1, -1)
        sequence = self.position_encoding(torch.cat([cls, context], dim=1))
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([cls_mask, pad_mask.bool()], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=full_mask)
        return self.norm(encoded[:, 0])


class RetrievalModel(nn.Module):
    """In-batch next-bar retrieval model."""

    def __init__(
        self,
        backend: str,
        latent_dim: int,
        context_bars: int,
        event_config: MidiTokBarSequenceEncoderConfig,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backend = str(backend)
        self.d_model = int(d_model)
        self.event_encoder = EventBarEncoder(event_config) if "miditok" in self.backend else None
        if self.backend == "latent_retrieval":
            input_dim = int(latent_dim)
        elif self.backend == "miditok_sequence_retrieval":
            input_dim = int(event_config.d_model)
        elif self.backend == "latent_plus_miditok_retrieval":
            input_dim = int(latent_dim) + int(event_config.d_model)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.input_projection = nn.Linear(input_dim, int(d_model))
        self.target_projection = nn.Linear(input_dim, int(d_model))
        self.context_encoder = ContextEncoder(d_model, context_bars, n_layers, n_heads, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized context and target embeddings."""
        context_features = self._context_features(batch)
        target_features = self._target_features(batch)
        context_hidden = self.input_projection(context_features)
        context_embedding = self.context_encoder(context_hidden, batch["context_pad_mask"])
        target_embedding = self.target_projection(target_features)
        return F.normalize(context_embedding, dim=-1), F.normalize(target_embedding, dim=-1)

    def _context_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build context feature sequence."""
        if self.backend == "latent_retrieval":
            return batch["context_mu"]
        if self.event_encoder is None:
            raise RuntimeError("event_encoder is missing")
        b, c, e, f = batch["context_tokens"].shape
        flat_tokens = batch["context_tokens"].reshape(b * c, e, f)
        flat_mask = batch["context_token_mask"].reshape(b * c, e)
        event = self.event_encoder(flat_tokens, flat_mask).reshape(b, c, -1)
        if self.backend == "miditok_sequence_retrieval":
            return event
        return torch.cat([batch["context_mu"], event], dim=-1)

    def _target_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build target feature matrix."""
        if self.backend == "latent_retrieval":
            return batch["target_mu"]
        if self.event_encoder is None:
            raise RuntimeError("event_encoder is missing")
        event = self.event_encoder(batch["target_tokens"], batch["target_token_mask"])
        if self.backend == "miditok_sequence_retrieval":
            return event
        return torch.cat([batch["target_mu"], event], dim=-1)


class NextBarBenchmark:
    """Train and evaluate next-bar retrieval backends."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = self._resolve_device(str(args.device))

    def run(self) -> Dict[str, Any]:
        """Run benchmark."""
        self._set_seed(int(self.args.seed))
        model_dir = Path(self.args.model_dir)
        latent_dir = Path(self.args.latent_dir) if self.args.latent_dir else model_dir / "latent"
        encoded_dir = Path(self.args.encoded_dir) if self.args.encoded_dir else model_dir / "encoded"
        output_dir = Path(self.args.output_dir) if self.args.output_dir else model_dir / "miditok_next_bar_benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        mu, rows, latent_summary = LatentDatasetReader().load(latent_dir)
        selected_indices = self._select_indices(rows)
        selected_rows = [rows[index] for index in selected_indices]
        selected_mu = mu[selected_indices].astype(np.float32)
        event_config = MidiTokBarSequenceEncoderConfig(
            latent_dim=int(selected_mu.shape[1]),
            max_events=int(self.args.max_events),
            d_model=int(self.args.event_d_model),
            n_layers=int(self.args.event_layers),
            n_heads=int(self.args.event_heads),
            dropout=float(self.args.dropout),
            field_embedding_dim=int(self.args.field_embedding_dim),
        )
        tokens, token_mask, token_summary = self._tokenize_rows(selected_rows, encoded_dir, event_config)
        samples = self._build_samples(selected_rows)
        train_samples, val_samples, split_summary = self._split_samples(samples)
        datasets = {
            "train": NextBarRetrievalDataset(train_samples, selected_mu, tokens, token_mask, int(self.args.context_bars)),
            "val": NextBarRetrievalDataset(val_samples, selected_mu, tokens, token_mask, int(self.args.context_bars)),
        }
        backends = [item.strip() for item in str(self.args.backends).split(",") if item.strip()]
        results: Dict[str, Any] = {}
        for backend in backends:
            results[backend] = self._run_backend(backend, datasets, event_config, int(selected_mu.shape[1]))
        report = {
            "analysis_type": "miditok_next_bar_retrieval_benchmark",
            "model_dir": str(model_dir),
            "latent_dir": str(latent_dir),
            "encoded_dir": str(encoded_dir),
            "output_dir": str(output_dir),
            "config": self._json_safe_args(),
            "device": self.device,
            "latent_summary": latent_summary,
            "data": {
                "selected_row_count": int(len(selected_rows)),
                "sample_count": int(len(samples)),
                "train_sample_count": int(len(train_samples)),
                "validation_sample_count": int(len(val_samples)),
                "latent_dim": int(selected_mu.shape[1]),
                "token_summary": token_summary,
                "split": split_summary,
            },
            "results": results,
            "interpretation": self._interpret(results),
        }
        (output_dir / "miditok_next_bar_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output_dir / "miditok_next_bar_benchmark.md").write_text(self._markdown(report), encoding="utf-8")
        print(f"Miditok next-bar benchmark -> {output_dir / 'miditok_next_bar_benchmark.json'}")
        print(f"Report -> {output_dir / 'miditok_next_bar_benchmark.md'}")
        return report

    def _run_backend(
        self,
        backend: str,
        datasets: Dict[str, NextBarRetrievalDataset],
        event_config: MidiTokBarSequenceEncoderConfig,
        latent_dim: int,
    ) -> Dict[str, Any]:
        """Train one backend."""
        model = RetrievalModel(
            backend=backend,
            latent_dim=latent_dim,
            context_bars=int(self.args.context_bars),
            event_config=event_config,
            d_model=int(self.args.d_model),
            n_layers=int(self.args.context_layers),
            n_heads=int(self.args.context_heads),
            dropout=float(self.args.dropout),
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.args.learning_rate), weight_decay=float(self.args.weight_decay))
        train_loader = DataLoader(datasets["train"], batch_size=int(self.args.batch_size), shuffle=True, drop_last=True)
        history: List[Dict[str, Any]] = []
        best_state = None
        best_mrr = -1.0
        best_epoch = -1
        for epoch in range(int(self.args.epochs)):
            model.train()
            losses = []
            for batch in train_loader:
                batch = self._batch_to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                context, target = model(batch)
                logits = context @ target.T / float(self.args.temperature)
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_metrics = self._evaluate(model, datasets["val"])
            row = {"epoch": int(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            if float(val_metrics["mrr"]) > best_mrr:
                best_mrr = float(val_metrics["mrr"])
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        return {
            "backend": backend,
            "best_epoch": int(best_epoch),
            "best_val_mrr": float(best_mrr),
            "history": history,
            "train_eval": self._evaluate(model, datasets["train"]),
            "val_eval": self._evaluate(model, datasets["val"]),
        }

    def _evaluate(self, model: RetrievalModel, dataset: NextBarRetrievalDataset) -> Dict[str, float]:
        """Evaluate with in-batch retrieval ranks."""
        loader = DataLoader(dataset, batch_size=int(self.args.batch_size), shuffle=False, drop_last=False)
        model.eval()
        ranks: List[int] = []
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                if int(batch["target_mu"].shape[0]) < 2:
                    continue
                batch = self._batch_to_device(batch)
                context, target = model(batch)
                logits = context @ target.T / float(self.args.temperature)
                labels = torch.arange(logits.shape[0], device=logits.device)
                losses.append(float(F.cross_entropy(logits, labels).detach().cpu()))
                order = torch.argsort(logits, dim=1, descending=True)
                for row in range(int(logits.shape[0])):
                    rank = int((order[row] == row).nonzero(as_tuple=False)[0, 0].item()) + 1
                    ranks.append(rank)
        if not ranks:
            return {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0, "mean_rank": 0.0, "median_rank": 0.0}
        ranks_array = np.asarray(ranks, dtype=np.float64)
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "top1": float(np.mean(ranks_array <= 1)),
            "top5": float(np.mean(ranks_array <= 5)),
            "top10": float(np.mean(ranks_array <= 10)),
            "mrr": float(np.mean(1.0 / ranks_array)),
            "mean_rank": float(np.mean(ranks_array)),
            "median_rank": float(np.median(ranks_array)),
            "candidate_count_mean": float(self.args.batch_size),
        }

    def _tokenize_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        encoded_dir: Path,
        event_config: MidiTokBarSequenceEncoderConfig,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Tokenize selected rows from bar_tensors.npz."""
        tensor_path = encoded_dir / "bar_tensors.npz"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing bar_tensors.npz: {tensor_path}")
        builder = MidiTokBarTokenBuilder(event_config)
        archive = np.load(tensor_path)
        tokens: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        event_counts: List[int] = []
        truncated = 0
        try:
            for row in rows:
                key = str(row.get("tensor_key", ""))
                if key not in archive.files:
                    raise KeyError(f"Missing tensor_key in archive: {key}")
                token, mask, diag = builder.build(np.asarray(archive[key], dtype=np.float32))
                tokens.append(token)
                masks.append(mask)
                event_counts.append(int(diag["event_count"]))
                truncated += int(diag["truncated_event_count"])
        finally:
            archive.close()
        return (
            np.stack(tokens, axis=0).astype(np.int64),
            np.stack(masks, axis=0).astype(bool),
            {
                "max_events": int(event_config.max_events),
                "event_count_mean": float(np.mean(event_counts)) if event_counts else 0.0,
                "event_count_median": float(np.median(event_counts)) if event_counts else 0.0,
                "event_count_max": int(max(event_counts)) if event_counts else 0,
                "truncated_event_total": int(truncated),
                "truncated_bar_count": int(sum(1 for value in event_counts if value > int(event_config.max_events))),
            },
        )

    def _build_samples(self, rows: Sequence[Dict[str, Any]]) -> List[NextBarSample]:
        """Build next-bar samples inside songs."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        samples: List[NextBarSample] = []
        for song_id, indices in grouped.items():
            ordered = sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            if len(ordered) < 2:
                continue
            for pos in range(1, len(ordered)):
                target = int(ordered[pos])
                context = [int(item) for item in ordered[max(0, pos - int(self.args.context_bars)):pos]]
                samples.append(NextBarSample(
                    context_indices=context,
                    target_index=target,
                    song_id=song_id,
                    base_song_id=self._base_song_id(song_id),
                    target_bar_index=int(rows[target].get("bar_index", pos)),
                ))
        return samples

    def _split_samples(self, samples: Sequence[NextBarSample]) -> Tuple[List[NextBarSample], List[NextBarSample], Dict[str, Any]]:
        """Split samples by base_song_id."""
        bases = sorted({sample.base_song_id for sample in samples})
        rng = np.random.default_rng(int(self.args.seed))
        shuffled = list(bases)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(self.args.validation_ratio)))) if len(shuffled) > 1 else 1
        val_bases = set(shuffled[:val_count])
        train = [sample for sample in samples if sample.base_song_id not in val_bases]
        val = [sample for sample in samples if sample.base_song_id in val_bases]
        if not train:
            train = val[:]
        return train, val, {
            "train_samples": int(len(train)),
            "validation_samples": int(len(val)),
            "base_song_count": int(len(bases)),
            "validation_base_song_ids": sorted(val_bases),
        }

    def _select_indices(self, rows: Sequence[Dict[str, Any]]) -> List[int]:
        """Select rows by song."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        song_ids = sorted(grouped)
        rng = np.random.default_rng(int(self.args.seed))
        if self.args.max_songs is not None:
            shuffled = list(song_ids)
            rng.shuffle(shuffled)
            song_ids = sorted(shuffled[: max(1, int(self.args.max_songs))])
        selected: List[int] = []
        for song_id in song_ids:
            selected.extend(sorted(grouped[song_id], key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx)))))
        if self.args.max_rows is not None:
            selected = selected[: max(2, int(self.args.max_rows))]
        return [int(item) for item in selected]

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {key: value.to(self.device) for key, value in batch.items()}

    def _interpret(self, results: Dict[str, Any]) -> List[str]:
        """Return concise interpretation."""
        messages: List[str] = []
        latent = results.get("latent_retrieval", {}).get("val_eval", {})
        miditok = results.get("miditok_sequence_retrieval", {}).get("val_eval", {})
        hybrid = results.get("latent_plus_miditok_retrieval", {}).get("val_eval", {})
        if miditok and latent and float(miditok.get("mrr", 0.0)) > float(latent.get("mrr", 0.0)):
            messages.append("MidiTok sequence retrieval beats latent retrieval on validation MRR.")
        if hybrid and latent and float(hybrid.get("mrr", 0.0)) > float(latent.get("mrr", 0.0)):
            messages.append("Latent plus MidiTok sequence retrieval beats latent-only retrieval on validation MRR.")
        if not messages:
            messages.append("No MidiTok retrieval backend beat latent-only retrieval on validation MRR.")
        return messages

    def _markdown(self, report: Dict[str, Any]) -> str:
        """Render markdown report."""
        lines = [
            "# MidiTok Next-Bar Retrieval Benchmark",
            "",
            "This benchmark tests whether MidiTok-style event sequences directly support next-bar prediction without forcing the task through VAE latent delta regression.",
            "",
            "## Data",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Selected rows | {report['data']['selected_row_count']} |",
            f"| Samples | {report['data']['sample_count']} |",
            f"| Train samples | {report['data']['train_sample_count']} |",
            f"| Validation samples | {report['data']['validation_sample_count']} |",
            f"| Mean events per bar | {report['data']['token_summary']['event_count_mean']:.3f} |",
            "",
            "## Results",
            "",
            "| Backend | Val Top1 | Val Top5 | Val Top10 | Val MRR | Mean Rank | Best Epoch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for backend, result in report["results"].items():
            val = result["val_eval"]
            lines.append(
                f"| {backend} | {val['top1']:.6f} | {val['top5']:.6f} | {val['top10']:.6f} | "
                f"{val['mrr']:.6f} | {val['mean_rank']:.3f} | {result['best_epoch']} |"
            )
        lines.extend(["", "## Interpretation", ""])
        lines.extend(f"- {item}" for item in report["interpretation"])
        lines.append("")
        return "\n".join(lines)

    def _base_song_id(self, song_id: str) -> str:
        """Remove transposition suffix."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _resolve_device(self, requested: str) -> str:
        """Resolve CUDA fallback."""
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _json_safe_args(self) -> Dict[str, Any]:
        """Return CLI args in a JSON-serializable form."""
        values: Dict[str, Any] = {}
        for key, value in vars(self.args).items():
            if isinstance(value, Path):
                values[key] = str(value)
            else:
                values[key] = value
        return values

    def _set_seed(self, seed: int) -> None:
        """Set RNG seeds."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark MidiTok-style sequence next-bar retrieval.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, default=None)
    parser.add_argument("--encoded-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backends", type=str, default="latent_retrieval,miditok_sequence_retrieval,latent_plus_miditok_retrieval")
    parser.add_argument("--context-bars", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--event-d-model", type=int, default=128)
    parser.add_argument("--event-layers", type=int, default=2)
    parser.add_argument("--event-heads", type=int, default=4)
    parser.add_argument("--field-embedding-dim", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def main() -> None:
    """Run CLI."""
    args = build_parser().parse_args()
    NextBarBenchmark(args).run()


if __name__ == "__main__":
    main()
