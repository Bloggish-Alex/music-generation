#!/usr/bin/env python3
"""Training and generation pipeline for hybrid MidiTok next-bar retrieval."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from model.base_pitch_motion import BasePitchMotionConfig, BasePitchMotionModel
from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from model.hybrid_miditok_retrieval import HybridMidiTokRetrievalConfig, HybridMidiTokRetrievalModel
from model.miditok_bar_sequence_encoder import MidiTokBarSequenceEncoderConfig
from pipeline.latent_generation_pipeline import SequenceTensorMidiRenderer
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader
from pipeline.miditok_bar_sequence_training_pipeline import MidiTokBarTokenBuilder


@dataclass(frozen=True)
class HybridMidiTokTrainingConfig:
    """Training config for the hybrid retrieval model."""

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.2
    random_seed: int = 42
    device: str = "cpu"
    early_stopping_patience: int = 5
    max_rows: Optional[int] = None


@dataclass(frozen=True)
class HybridMidiTokGenerationConfig:
    """Generation config for hybrid retrieval."""

    bars: int = 32
    primer_bars: int = 8
    top_k: int = 24
    temperature: float = 0.35
    recent_penalty: float = 2.0
    recent_window: int = 8
    seed: int = 42
    device: str = "cpu"
    base_pitch: int = 60
    tempo_bpm: int = 100
    decode_latent: bool = True
    candidate_limit: Optional[int] = None
    candidate_transpose_mode: str = "all"
    base_pitch_mode: str = "source"  # source / fixed / learned
    base_pitch_motion_path: Optional[str] = None
    render_base_pitch_min: int = 36
    render_base_pitch_max: int = 84
    base_pitch_delta_min: Optional[int] = None
    base_pitch_delta_max: Optional[int] = None


@dataclass(frozen=True)
class HybridMidiTokResult:
    """Paths produced by a pipeline run."""

    model_path: Path
    diagnostics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class HybridMidiTokGenerationResult:
    """Generation artifact paths."""

    json_path: Path
    midi_path: Path
    tensor_path: Path


@dataclass(frozen=True)
class RetrievalSample:
    """One song-local next-bar sample."""

    context_indices: List[int]
    target_index: int
    song_id: str
    base_song_id: str
    target_bar_index: int


class HybridMidiTokDataset(Dataset):
    """Torch dataset for hybrid next-bar retrieval."""

    def __init__(
        self,
        samples: Sequence[RetrievalSample],
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
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one left-padded sample."""
        sample = self.samples[index]
        latent_dim = int(self.mu.shape[1])
        max_events = int(self.tokens.shape[1])
        context_mu = np.zeros((self.context_bars, latent_dim), dtype=np.float32)
        context_tokens = np.zeros((self.context_bars, max_events, 5), dtype=np.int64)
        context_token_mask = np.ones((self.context_bars, max_events), dtype=bool)
        context_padding_mask = np.ones((self.context_bars,), dtype=bool)
        recent = sample.context_indices[-self.context_bars:]
        offset = self.context_bars - len(recent)
        for local_index, row_index in enumerate(recent):
            slot = offset + local_index
            context_mu[slot] = self.mu[row_index]
            context_tokens[slot] = self.tokens[row_index]
            context_token_mask[slot] = self.token_mask[row_index]
            context_padding_mask[slot] = False
        return {
            "context_mu": torch.from_numpy(context_mu).float(),
            "context_tokens": torch.from_numpy(context_tokens).long(),
            "context_token_mask": torch.from_numpy(context_token_mask).bool(),
            "context_padding_mask": torch.from_numpy(context_padding_mask).bool(),
            "target_mu": torch.from_numpy(self.mu[sample.target_index]).float(),
            "target_tokens": torch.from_numpy(self.tokens[sample.target_index]).long(),
            "target_token_mask": torch.from_numpy(self.token_mask[sample.target_index]).bool(),
        }


class HybridMidiTokDataBuilder:
    """Load latent rows and build MidiTok token arrays."""

    def __init__(self, model_config: HybridMidiTokRetrievalConfig, event_config: MidiTokBarSequenceEncoderConfig) -> None:
        self.model_config = model_config
        self.event_config = event_config

    def load(
        self,
        latent_dir: str | Path,
        encoded_dir: str | Path,
        max_rows: Optional[int] = None,
        transpose_mode: str = "all",
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load latent rows and tokenized bar tensors."""
        mu, rows, latent_summary = LatentDatasetReader().load(latent_dir)
        original_row_count = int(len(rows))
        selected = self._indices_for_transpose_mode(rows, transpose_mode)
        if len(selected) != len(rows):
            mu = mu[selected]
            rows = [rows[int(index)] for index in selected]
        if max_rows is not None:
            limit = max(2, int(max_rows))
            mu = mu[:limit]
            rows = rows[:limit]
        if int(mu.shape[1]) != int(self.model_config.latent_dim):
            raise ValueError(f"latent_dim mismatch: model={self.model_config.latent_dim}, data={mu.shape[1]}")
        tokens, token_mask, token_summary = self._tokenize(rows, encoded_dir)
        return mu.astype(np.float32), rows, tokens, token_mask, {
            "latent": latent_summary,
            "tokens": token_summary,
            "row_filter": {
                "transpose_mode": str(transpose_mode),
                "original_row_count": int(original_row_count),
                "filtered_row_count": int(len(rows)),
                "max_rows": None if max_rows is None else int(max_rows),
            },
        }

    def build_samples(self, rows: Sequence[Dict[str, Any]]) -> List[RetrievalSample]:
        """Build next-bar samples inside each song."""
        grouped = self.group_rows(rows)
        samples: List[RetrievalSample] = []
        for song_id, ordered in grouped.items():
            if len(ordered) < 2:
                continue
            for position in range(1, len(ordered)):
                target = int(ordered[position])
                context = [int(item) for item in ordered[max(0, position - int(self.model_config.context_bars)):position]]
                samples.append(RetrievalSample(
                    context_indices=context,
                    target_index=target,
                    song_id=song_id,
                    base_song_id=self.base_song_id(song_id),
                    target_bar_index=int(rows[target].get("bar_index", position)),
                ))
        if not samples:
            raise ValueError("No retrieval samples built. Need songs with at least two bars.")
        return samples

    def group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group rows by song_id and sort by bar_index."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return {
            song_id: sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            for song_id, indices in grouped.items()
        }

    def _tokenize(self, rows: Sequence[Dict[str, Any]], encoded_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Tokenize all bar tensors into MidiTok-style event fields."""
        tensor_path = Path(encoded_dir) / "bar_tensors.npz"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing bar_tensors.npz: {tensor_path}")
        builder = MidiTokBarTokenBuilder(self.event_config)
        archive = np.load(tensor_path)
        tokens: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        event_counts: List[int] = []
        truncated_total = 0
        try:
            for row in rows:
                key = str(row.get("tensor_key", ""))
                if key not in archive.files:
                    raise KeyError(f"Missing tensor_key in bar_tensors.npz: {key}")
                token, mask, diag = builder.build(np.asarray(archive[key], dtype=np.float32))
                tokens.append(token)
                masks.append(mask)
                event_counts.append(int(diag["event_count"]))
                truncated_total += int(diag["truncated_event_count"])
        finally:
            archive.close()
        return np.stack(tokens, axis=0), np.stack(masks, axis=0).astype(bool), {
            "row_count": int(len(rows)),
            "max_events": int(self.event_config.max_events),
            "event_count_mean": float(np.mean(event_counts)) if event_counts else 0.0,
            "event_count_median": float(np.median(event_counts)) if event_counts else 0.0,
            "event_count_max": int(max(event_counts)) if event_counts else 0,
            "truncated_event_total": int(truncated_total),
        }

    @staticmethod
    def base_song_id(song_id: str) -> str:
        """Strip transposition suffix from song id."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _indices_for_transpose_mode(self, rows: Sequence[Dict[str, Any]], transpose_mode: str) -> np.ndarray:
        """Return row indices allowed by generation-time transpose mode."""
        mode = str(transpose_mode).strip().lower()
        if mode in {"all", "", "none"}:
            return np.arange(len(rows), dtype=np.int64)
        if mode in {"canonical", "canonical_only", "no_transpose", "original_only"}:
            return np.asarray([
                index
                for index, row in enumerate(rows)
                if not re.search(r"_T[+-]?\d+$", str(row.get("song_id", "")))
            ], dtype=np.int64)
        raise ValueError(
            f"Unsupported candidate_transpose_mode: {transpose_mode}. "
            "Use all or canonical_only."
        )


class HybridMidiTokTrainingPipeline:
    """Train hybrid latent + MidiTok next-bar retrieval."""

    def __init__(
        self,
        model_config: HybridMidiTokRetrievalConfig,
        event_config: MidiTokBarSequenceEncoderConfig,
        training_config: HybridMidiTokTrainingConfig,
    ) -> None:
        self.model_config = model_config
        self.event_config = event_config
        self.training_config = training_config
        self.builder = HybridMidiTokDataBuilder(model_config, event_config)

    def run(self, model_dir: str | Path, latent_dir: Optional[str | Path] = None, encoded_dir: Optional[str | Path] = None) -> HybridMidiTokResult:
        """Train and write model artifacts."""
        self._set_seed()
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        latent_path = Path(latent_dir) if latent_dir else output_dir / "latent"
        encoded_path = Path(encoded_dir) if encoded_dir else output_dir / "encoded"
        mu, rows, tokens, token_mask, source_summary = self.builder.load(
            latent_path,
            encoded_path,
            max_rows=self.training_config.max_rows,
        )
        samples = self.builder.build_samples(rows)
        train_samples, val_samples, split = self._split_samples(samples)
        train_dataset = HybridMidiTokDataset(train_samples, mu, tokens, token_mask, int(self.model_config.context_bars))
        val_dataset = HybridMidiTokDataset(val_samples, mu, tokens, token_mask, int(self.model_config.context_bars))
        model = HybridMidiTokRetrievalModel(self.model_config, self.event_config).to(self.training_config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        fit = self._fit(model, optimizer, train_dataset, val_dataset)
        if fit["best_state_dict"] is not None:
            model.load_state_dict(fit["best_state_dict"])
        train_eval = self._evaluate(model, train_dataset)
        val_eval = self._evaluate(model, val_dataset)
        model_path = output_dir / "hybrid_miditok_retrieval.pt"
        torch.save({
            "model_type": "HybridMidiTokRetrievalModel",
            "checkpoint_role": "best",
            "model_config": self.model_config.to_dict(),
            "event_config": self.event_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "state_dict": model.state_dict(),
        }, model_path)
        diagnostics = {
            "model_path": str(model_path),
            "latent_dir": str(latent_path),
            "encoded_dir": str(encoded_path),
            "model_config": self.model_config.to_dict(),
            "event_config": self.event_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "source_summary": source_summary,
            "sample_count": int(len(samples)),
            "split": split,
            "history": fit["history"],
            "model_selection": {
                "best_epoch": int(fit["best_epoch"]),
                "best_val_mrr": float(fit["best_val_mrr"]),
                "early_stopped": bool(fit["early_stopped"]),
            },
            "train_eval": train_eval,
            "val_eval": val_eval,
        }
        diagnostics_path = output_dir / "hybrid_miditok_retrieval_diagnostics.json"
        summary_path = output_dir / "hybrid_miditok_retrieval_summary.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(self._summary(diagnostics), indent=2), encoding="utf-8")
        return HybridMidiTokResult(model_path=model_path, diagnostics_path=diagnostics_path, summary_path=summary_path)

    def _fit(
        self,
        model: HybridMidiTokRetrievalModel,
        optimizer: torch.optim.Optimizer,
        train_dataset: HybridMidiTokDataset,
        val_dataset: HybridMidiTokDataset,
    ) -> Dict[str, Any]:
        """Train with in-batch contrastive retrieval loss."""
        train_loader = DataLoader(train_dataset, batch_size=int(self.training_config.batch_size), shuffle=True, drop_last=True)
        history: List[Dict[str, float]] = []
        best_state = None
        best_mrr = -1.0
        best_epoch = 0
        stale = 0
        early_stopped = False
        for epoch in range(1, int(self.training_config.epochs) + 1):
            model.train()
            losses: List[float] = []
            for batch in train_loader:
                prepared = self._batch_to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                context, target = model(prepared)
                logits = context @ target.T / float(self.model_config.temperature)
                labels = torch.arange(int(logits.shape[0]), device=logits.device)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_eval = self._evaluate(model, val_dataset)
            row = {"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, **{f"val_{k}": float(v) for k, v in val_eval.items()}}
            history.append(row)
            if float(val_eval["mrr"]) > best_mrr:
                best_mrr = float(val_eval["mrr"])
                best_epoch = int(epoch)
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if int(self.training_config.early_stopping_patience) > 0 and stale >= int(self.training_config.early_stopping_patience):
                early_stopped = True
                break
        return {
            "history": history,
            "best_state_dict": best_state,
            "best_epoch": int(best_epoch),
            "best_val_mrr": float(best_mrr),
            "early_stopped": bool(early_stopped),
        }

    def _evaluate(self, model: HybridMidiTokRetrievalModel, dataset: HybridMidiTokDataset) -> Dict[str, float]:
        """Evaluate in-batch retrieval ranks."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False, drop_last=False)
        model.eval()
        ranks: List[int] = []
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                if int(batch["target_mu"].shape[0]) < 2:
                    continue
                prepared = self._batch_to_device(batch)
                context, target = model(prepared)
                logits = context @ target.T / float(self.model_config.temperature)
                labels = torch.arange(int(logits.shape[0]), device=logits.device)
                losses.append(float(F.cross_entropy(logits, labels).detach().cpu()))
                order = torch.argsort(logits, dim=1, descending=True)
                for row in range(int(logits.shape[0])):
                    ranks.append(int((order[row] == row).nonzero(as_tuple=False)[0, 0].item()) + 1)
        if not ranks:
            return {"loss": 0.0, "top1": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0, "mean_rank": 0.0, "median_rank": 0.0}
        values = np.asarray(ranks, dtype=np.float64)
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "top1": float(np.mean(values <= 1)),
            "top5": float(np.mean(values <= 5)),
            "top10": float(np.mean(values <= 10)),
            "mrr": float(np.mean(1.0 / values)),
            "mean_rank": float(np.mean(values)),
            "median_rank": float(np.median(values)),
        }

    def _split_samples(self, samples: Sequence[RetrievalSample]) -> Tuple[List[RetrievalSample], List[RetrievalSample], Dict[str, Any]]:
        """Split by base_song_id."""
        base_ids = sorted({sample.base_song_id for sample in samples})
        rng = np.random.default_rng(int(self.training_config.random_seed))
        shuffled = list(base_ids)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(self.training_config.validation_ratio)))) if len(shuffled) > 1 else 1
        val_bases = set(shuffled[:val_count])
        train = [sample for sample in samples if sample.base_song_id not in val_bases]
        val = [sample for sample in samples if sample.base_song_id in val_bases]
        if not train:
            train = val[:]
        return train, val, {
            "train_samples": int(len(train)),
            "validation_samples": int(len(val)),
            "base_song_count": int(len(base_ids)),
            "validation_base_song_ids": sorted(val_bases),
        }

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch to the configured device."""
        return {key: value.to(self.training_config.device) for key, value in batch.items()}

    def _summary(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Return compact summary."""
        return {
            "model_path": diagnostics["model_path"],
            "sample_count": diagnostics["sample_count"],
            "model_selection": diagnostics["model_selection"],
            "train_eval": diagnostics["train_eval"],
            "val_eval": diagnostics["val_eval"],
        }

    def _set_seed(self) -> None:
        """Set random seeds."""
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class HybridMidiTokGenerationPipeline:
    """Generate by hybrid retrieval and DVAE decode."""

    def __init__(self, config: HybridMidiTokGenerationConfig) -> None:
        self.config = config

    def run(
        self,
        model_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        latent_dir: Optional[str | Path] = None,
        encoded_dir: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
        dvae_path: Optional[str | Path] = None,
        seed_song_id: Optional[str] = None,
    ) -> HybridMidiTokGenerationResult:
        """Generate MIDI and diagnostics."""
        self._set_seed()
        rng = np.random.default_rng(int(self.config.seed))
        model_directory = Path(model_dir)
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else model_directory / "hybrid_miditok_retrieval.pt"
        checkpoint = torch.load(checkpoint_file, map_location=self.config.device, weights_only=False)
        model_config = HybridMidiTokRetrievalConfig(**checkpoint["model_config"])
        event_config = MidiTokBarSequenceEncoderConfig(**checkpoint["event_config"])
        builder = HybridMidiTokDataBuilder(model_config, event_config)
        latent_path = Path(latent_dir) if latent_dir else model_directory / "latent"
        encoded_path = Path(encoded_dir) if encoded_dir else model_directory / "encoded"
        mu, rows, tokens, token_mask, source_summary = builder.load(
            latent_path,
            encoded_path,
            max_rows=self.config.candidate_limit,
            transpose_mode=self.config.candidate_transpose_mode,
        )
        model = HybridMidiTokRetrievalModel(model_config, event_config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        motion_model, motion_config = self._load_base_pitch_motion(model_directory)
        dvae = self._load_dvae(Path(dvae_path) if dvae_path else model_directory / "dvae.pt")
        grouped = builder.group_rows(rows)
        selected_song_id = self._select_song_id(grouped, seed_song_id)
        ordered = grouped[selected_song_id]
        primer_count = max(1, min(int(self.config.primer_bars), int(self.config.bars), len(ordered)))
        selected_indices = [int(item) for item in ordered[:primer_count]]
        base_pitch_lookup = self._base_pitch_lookup(encoded_path)
        generated_mu = [mu[index].astype(np.float32) for index in selected_indices]
        generated_tokens = [tokens[index].astype(np.int64) for index in selected_indices]
        generated_masks = [token_mask[index].astype(bool) for index in selected_indices]
        source_base_pitches = [
            self._row_base_pitch(rows[index], base_pitch_lookup, int(self.config.base_pitch))
            for index in selected_indices
        ]
        render_base_pitches = self._initial_render_base_pitches(source_base_pitches)
        candidate_indices = self._candidate_indices(len(rows))
        candidate_embeddings = self._candidate_embeddings(model, mu, tokens, token_mask, candidate_indices)
        steps: List[Dict[str, Any]] = []
        while len(generated_mu) < int(self.config.bars):
            context = self._context_tensors(generated_mu, generated_tokens, generated_masks, int(model_config.context_bars))
            with torch.no_grad():
                query = model.encode_context(
                    context_mu=context["context_mu"].to(self.config.device),
                    context_tokens=context["context_tokens"].to(self.config.device),
                    context_token_mask=context["context_token_mask"].to(self.config.device),
                    context_padding_mask=context["context_padding_mask"].to(self.config.device),
                )
                scores = (query @ candidate_embeddings.T).detach().cpu().numpy()[0]
            selected, diag = self._select_candidate(scores, candidate_indices, selected_indices, rng)
            selected_indices.append(int(selected))
            generated_mu.append(mu[selected].astype(np.float32))
            generated_tokens.append(tokens[selected].astype(np.int64))
            generated_masks.append(token_mask[selected].astype(bool))
            source_base_pitch = self._row_base_pitch(rows[selected], base_pitch_lookup, int(self.config.base_pitch))
            source_base_pitches.append(int(source_base_pitch))
            render_base_pitch, base_pitch_diag = self._next_render_base_pitch(
                generated_mu=generated_mu[:-1],
                generated_tokens=generated_tokens[:-1],
                generated_masks=generated_masks[:-1],
                render_base_pitches=render_base_pitches,
                source_base_pitch=int(source_base_pitch),
                motion_model=motion_model,
                motion_config=motion_config,
            )
            render_base_pitches.append(int(render_base_pitch))
            steps.append({
                "bar_index": int(len(generated_mu) - 1),
                "selected_row_index": int(selected),
                "selected_song_id": str(rows[selected].get("song_id", "")),
                "selected_bar_index": int(rows[selected].get("bar_index", 0)),
                "source_base_pitch": int(source_base_pitch),
                "render_base_pitch": int(render_base_pitch),
                "base_pitch_motion": base_pitch_diag,
                **diag,
            })
        latent_array = np.stack(generated_mu, axis=0).astype(np.float32)
        tensors = self._decode_tensors(dvae, latent_array)
        tensor_path = Path(output_json).with_suffix(".bar_tensors.npz")
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tensor_path,
            bars=tensors.astype(np.float32),
            latent_mu=latent_array,
            selected_row_indices=np.asarray(selected_indices, dtype=np.int64),
            source_base_pitches=np.asarray(source_base_pitches, dtype=np.int64),
            render_base_pitches=np.asarray(render_base_pitches, dtype=np.int64),
        )
        midi_diag = SequenceTensorMidiRenderer(DVAEMidiRenderConfig(
            tempo_bpm=int(self.config.tempo_bpm),
            default_base_pitch=int(self.config.base_pitch),
        )).render(
            tensors,
            output_midi,
            base_pitch=int(self.config.base_pitch),
            base_pitches=render_base_pitches,
        )
        diagnostics = {
            "model_dir": str(model_directory),
            "checkpoint": str(checkpoint_file),
            "latent_dir": str(latent_path),
            "encoded_dir": str(encoded_path),
            "dvae_checkpoint": str(Path(dvae_path) if dvae_path else model_directory / "dvae.pt"),
            "config": self.config.__dict__,
            "model_config": model_config.to_dict(),
            "event_config": event_config.to_dict(),
            "source_summary": source_summary,
            "selected_song_id": selected_song_id,
            "primer_row_indices": [int(item) for item in selected_indices[:primer_count]],
            "candidate_pool": {
                "total_rows": int(len(rows)),
                "candidate_count": int(len(candidate_indices)),
                "candidate_limit": None if self.config.candidate_limit is None else int(self.config.candidate_limit),
                "candidate_transpose_mode": str(self.config.candidate_transpose_mode),
                "row_filter": source_summary.get("row_filter", {}),
            },
            "selected_row_indices": [int(item) for item in selected_indices],
            "source_base_pitches": [int(item) for item in source_base_pitches],
            "render_base_pitches": [int(item) for item in render_base_pitches],
            "steps": steps,
            "midi": midi_diag,
            "tensor_path": str(tensor_path),
            "json_path": str(output_json),
            "midi_path": str(output_midi),
        }
        Path(output_json).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        self._verify_final_midi_tempo(Path(output_midi), int(self.config.tempo_bpm))
        return HybridMidiTokGenerationResult(json_path=Path(output_json), midi_path=Path(output_midi), tensor_path=tensor_path)

    def _verify_final_midi_tempo(self, midi_path: Path, expected_bpm: int) -> None:
        """Fail fast if a later step rewrote the MIDI tempo."""
        import mido

        midi = mido.MidiFile(str(midi_path))
        for track in midi.tracks:
            for message in track:
                if message.type == "set_tempo":
                    actual_bpm = float(mido.tempo2bpm(message.tempo))
                    if int(round(actual_bpm)) != int(expected_bpm):
                        raise RuntimeError(
                            f"Final MIDI tempo mismatch: expected {int(expected_bpm)} BPM, "
                            f"but {midi_path} contains {actual_bpm:.3f} BPM."
                        )
                    return
        raise RuntimeError(f"Final MIDI tempo mismatch: no set_tempo meta event found in {midi_path}.")

    def _base_pitch_lookup(self, encoded_dir: str | Path) -> Dict[str, int]:
        """Load source tensor base pitches from encoded bar diagnostics."""
        index_path = Path(encoded_dir) / "bar_tensor_index.json"
        if not index_path.exists():
            return {}
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        lookup: Dict[str, int] = {}
        for row in rows:
            key = str(row.get("tensor_key", ""))
            diagnostics = row.get("diagnostics", {}) if isinstance(row.get("diagnostics", {}), dict) else {}
            value = diagnostics.get("base_pitch")
            if value is not None:
                lookup[key] = int(value)
        return lookup

    def _row_base_pitch(self, row: Dict[str, Any], base_pitch_lookup: Dict[str, int], fallback: int) -> int:
        """Return source bar base pitch for a latent row."""
        key = str(row.get("tensor_key", ""))
        return int(base_pitch_lookup.get(key, int(fallback)))

    def _load_base_pitch_motion(self, model_dir: Path) -> Tuple[Optional[BasePitchMotionModel], Optional[BasePitchMotionConfig]]:
        """Load learned base-pitch motion model when requested."""
        mode = str(self.config.base_pitch_mode).strip().lower()
        if mode != "learned":
            return None, None
        path = Path(self.config.base_pitch_motion_path) if self.config.base_pitch_motion_path else model_dir / "base_pitch_motion.pt"
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        model_config = BasePitchMotionConfig(**checkpoint["model_config"])
        event_config = MidiTokBarSequenceEncoderConfig(**checkpoint["event_config"])
        model = BasePitchMotionModel(model_config, event_config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, model_config

    def _initial_render_base_pitches(self, source_base_pitches: Sequence[int]) -> List[int]:
        """Return render base-pitch trajectory for primer bars."""
        mode = str(self.config.base_pitch_mode).strip().lower()
        if mode == "fixed":
            return [int(self.config.base_pitch) for _ in source_base_pitches]
        return [self._clip_render_base_pitch(int(value)) for value in source_base_pitches]

    def _next_render_base_pitch(
        self,
        generated_mu: Sequence[np.ndarray],
        generated_tokens: Sequence[np.ndarray],
        generated_masks: Sequence[np.ndarray],
        render_base_pitches: Sequence[int],
        source_base_pitch: int,
        motion_model: Optional[BasePitchMotionModel],
        motion_config: Optional[BasePitchMotionConfig],
    ) -> Tuple[int, Dict[str, Any]]:
        """Return next render base pitch according to configured strategy."""
        mode = str(self.config.base_pitch_mode).strip().lower()
        if mode == "fixed":
            return int(self.config.base_pitch), {"mode": "fixed", "delta": 0}
        if mode != "learned":
            return self._clip_render_base_pitch(int(source_base_pitch)), {"mode": "source", "delta": None}
        if motion_model is None or motion_config is None:
            raise ValueError("base_pitch_mode=learned requires a loaded base-pitch motion model.")
        context = self._base_pitch_motion_context(generated_mu, generated_tokens, generated_masks, render_base_pitches, int(motion_config.context_bars))
        with torch.no_grad():
            logits = motion_model({
                key: value.to(self.config.device)
                for key, value in context.items()
            })
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        class_id = int(np.argmax(probs))
        delta = int(motion_config.class_to_delta(class_id))
        raw_delta = int(delta)
        delta = self._clip_base_pitch_delta(delta)
        previous = int(render_base_pitches[-1]) if render_base_pitches else int(self.config.base_pitch)
        predicted = self._clip_render_base_pitch(previous + delta)
        top = np.argsort(-probs)[:5]
        return predicted, {
            "mode": "learned",
            "previous_base_pitch": int(previous),
            "raw_predicted_delta": int(raw_delta),
            "predicted_delta": int(delta),
            "predicted_class": int(class_id),
            "confidence": float(probs[class_id]),
            "delta_clamped": bool(delta != raw_delta),
            "delta_clip_range": self._base_pitch_delta_clip_range(),
            "top_deltas": [
                {"delta": int(motion_config.class_to_delta(int(item))), "probability": float(probs[int(item)])}
                for item in top
            ],
        }

    def _clip_render_base_pitch(self, value: int) -> int:
        """Clip rendered base pitch to configured usable range."""
        return max(int(self.config.render_base_pitch_min), min(int(self.config.render_base_pitch_max), int(value)))

    def _base_pitch_delta_clip_range(self) -> Optional[Dict[str, int]]:
        """Return generation-time base-pitch delta clip range if configured."""
        if self.config.base_pitch_delta_min is None and self.config.base_pitch_delta_max is None:
            return None
        return {
            "min": int(self.config.base_pitch_delta_min) if self.config.base_pitch_delta_min is not None else -999,
            "max": int(self.config.base_pitch_delta_max) if self.config.base_pitch_delta_max is not None else 999,
        }

    def _clip_base_pitch_delta(self, delta: int) -> int:
        """Apply optional generation-time delta clamp."""
        low = int(self.config.base_pitch_delta_min) if self.config.base_pitch_delta_min is not None else -999
        high = int(self.config.base_pitch_delta_max) if self.config.base_pitch_delta_max is not None else 999
        return max(low, min(high, int(delta)))

    def _base_pitch_motion_context(
        self,
        generated_mu: Sequence[np.ndarray],
        generated_tokens: Sequence[np.ndarray],
        generated_masks: Sequence[np.ndarray],
        render_base_pitches: Sequence[int],
        context_bars: int,
    ) -> Dict[str, torch.Tensor]:
        """Build one left-padded context batch for base-pitch motion prediction."""
        latent_dim = int(generated_mu[0].shape[0])
        max_events = int(generated_tokens[0].shape[0])
        context_mu = np.zeros((1, context_bars, latent_dim), dtype=np.float32)
        context_tokens = np.zeros((1, context_bars, max_events, 5), dtype=np.int64)
        context_token_mask = np.ones((1, context_bars, max_events), dtype=bool)
        context_padding_mask = np.ones((1, context_bars), dtype=bool)
        context_base_pitch = np.zeros((1, context_bars), dtype=np.float32)
        recent = list(range(max(0, len(generated_mu) - context_bars), len(generated_mu)))
        offset = context_bars - len(recent)
        for local_index, source_index in enumerate(recent):
            slot = offset + local_index
            context_mu[0, slot] = generated_mu[source_index]
            context_tokens[0, slot] = generated_tokens[source_index]
            context_token_mask[0, slot] = generated_masks[source_index]
            context_padding_mask[0, slot] = False
            context_base_pitch[0, slot] = float(render_base_pitches[source_index])
        return {
            "context_mu": torch.from_numpy(context_mu).float(),
            "context_tokens": torch.from_numpy(context_tokens).long(),
            "context_token_mask": torch.from_numpy(context_token_mask).bool(),
            "context_padding_mask": torch.from_numpy(context_padding_mask).bool(),
            "context_base_pitch": torch.from_numpy(context_base_pitch).float(),
        }

    def _candidate_embeddings(
        self,
        model: HybridMidiTokRetrievalModel,
        mu: np.ndarray,
        tokens: np.ndarray,
        token_mask: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> torch.Tensor:
        """Precompute all candidate embeddings."""
        embeddings: List[torch.Tensor] = []
        batch_size = 512
        with torch.no_grad():
            for start in range(0, int(candidate_indices.shape[0]), batch_size):
                end = min(int(candidate_indices.shape[0]), start + batch_size)
                indices = candidate_indices[start:end]
                embeddings.append(model.encode_targets(
                    target_mu=torch.from_numpy(mu[indices].astype(np.float32)).to(self.config.device),
                    target_tokens=torch.from_numpy(tokens[indices].astype(np.int64)).to(self.config.device),
                    target_token_mask=torch.from_numpy(token_mask[indices].astype(bool)).to(self.config.device),
                ).detach())
        return torch.cat(embeddings, dim=0)

    def _context_tensors(
        self,
        generated_mu: Sequence[np.ndarray],
        generated_tokens: Sequence[np.ndarray],
        generated_masks: Sequence[np.ndarray],
        context_bars: int,
    ) -> Dict[str, torch.Tensor]:
        """Build one left-padded context batch."""
        latent_dim = int(generated_mu[0].shape[0])
        max_events = int(generated_tokens[0].shape[0])
        context_mu = np.zeros((1, context_bars, latent_dim), dtype=np.float32)
        context_tokens = np.zeros((1, context_bars, max_events, 5), dtype=np.int64)
        context_token_mask = np.ones((1, context_bars, max_events), dtype=bool)
        context_padding_mask = np.ones((1, context_bars), dtype=bool)
        recent = list(range(max(0, len(generated_mu) - context_bars), len(generated_mu)))
        offset = context_bars - len(recent)
        for local_index, source_index in enumerate(recent):
            slot = offset + local_index
            context_mu[0, slot] = generated_mu[source_index]
            context_tokens[0, slot] = generated_tokens[source_index]
            context_token_mask[0, slot] = generated_masks[source_index]
            context_padding_mask[0, slot] = False
        return {
            "context_mu": torch.from_numpy(context_mu).float(),
            "context_tokens": torch.from_numpy(context_tokens).long(),
            "context_token_mask": torch.from_numpy(context_token_mask).bool(),
            "context_padding_mask": torch.from_numpy(context_padding_mask).bool(),
        }

    def _candidate_indices(self, total_rows: int) -> np.ndarray:
        """Return candidate row indices, optionally capped for debug generation."""
        if self.config.candidate_limit is None:
            return np.arange(int(total_rows), dtype=np.int64)
        limit = max(1, min(int(self.config.candidate_limit), int(total_rows)))
        return np.arange(limit, dtype=np.int64)

    def _select_candidate(
        self,
        scores: np.ndarray,
        candidate_indices: np.ndarray,
        selected_indices: Sequence[int],
        rng: np.random.Generator,
    ) -> Tuple[int, Dict[str, Any]]:
        """Sample one candidate from similarity scores."""
        adjusted = np.asarray(scores, dtype=np.float64).copy()
        recent = set(int(item) for item in selected_indices[-max(0, int(self.config.recent_window)):])
        for local_index, row_index in enumerate(candidate_indices.tolist()):
            if int(row_index) in recent:
                adjusted[int(local_index)] -= float(self.config.recent_penalty)
        order = np.argsort(-adjusted)
        top = order[: max(1, min(int(self.config.top_k), len(order)))]
        if len(top) <= 1 or float(self.config.temperature) <= 0.0:
            local = 0
        else:
            logits = adjusted[top] / max(1.0e-6, float(self.config.temperature))
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            probs = probs / float(np.sum(probs))
            local = int(rng.choice(len(top), p=probs))
        selected_local = int(top[local])
        selected_row = int(candidate_indices[selected_local])
        return selected_row, {
            "selection_backend": "hybrid_miditok_retrieval",
            "candidate_count": int(len(scores)),
            "top_k": int(len(top)),
            "selected_score": float(scores[selected_local]),
            "selected_adjusted_score": float(adjusted[selected_local]),
            "selected_rank": int(np.where(order == selected_local)[0][0]) + 1,
            "top_candidates": [
                {
                    "row_index": int(candidate_indices[int(item)]),
                    "score": float(scores[int(item)]),
                    "adjusted_score": float(adjusted[int(item)]),
                }
                for item in top[:10]
            ],
        }

    def _load_dvae(self, path: Path) -> DenoisingMusicVAE:
        """Load DVAE checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        model = DenoisingMusicVAE(DVAEMusicConfig(**checkpoint["config"])).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _decode_tensors(self, dvae: DenoisingMusicVAE, latent_mu: np.ndarray) -> np.ndarray:
        """Decode latent vectors into bar tensors."""
        with torch.no_grad():
            z = torch.from_numpy(latent_mu.astype(np.float32)).to(self.config.device)
            pitch, state_logits, velocity, chord = dvae.decoder(z)
            state = torch.argmax(state_logits, dim=-1)
            state_one_hot = torch.nn.functional.one_hot(state, num_classes=3).float()
            tensor = torch.zeros(
                (z.shape[0], int(dvae.config.tracks), int(dvae.config.steps_per_bar), int(dvae.config.feature_dim)),
                dtype=torch.float32,
                device=z.device,
            )
            tensor[..., 0] = pitch
            tensor[..., 1:4] = state_one_hot
            tensor[..., 4] = velocity
            tensor[..., 5:5 + chord.shape[-1]] = chord
        return tensor.detach().cpu().numpy().astype(np.float32)

    def _select_song_id(self, grouped: Dict[str, List[int]], seed_song_id: Optional[str]) -> str:
        """Select primer song."""
        if seed_song_id:
            if seed_song_id in grouped:
                return seed_song_id
            pattern = re.compile(str(seed_song_id))
            matches = [song_id for song_id in grouped if pattern.search(song_id)]
            if matches:
                return sorted(matches)[0]
            raise ValueError(f"seed_song_id not found: {seed_song_id}")
        rng = random.Random(int(self.config.seed))
        return rng.choice(sorted(grouped.keys()))

    def _set_seed(self) -> None:
        """Set random seeds."""
        seed = int(self.config.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
