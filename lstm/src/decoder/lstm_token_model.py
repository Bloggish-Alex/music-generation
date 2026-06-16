#!/usr/bin/env python3
"""Small LSTM next-token model over encoder latent vectors."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from decoder.latent_token_model import LatentTokenContext, TokenLogitMasker, TokenLogitResult
from decoder.temporal_graph import CandidateTokenMask


@dataclass(frozen=True)
class LSTMDecoderConfig:
    context_size: int = 8
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 128
    epochs: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    validation_ratio: float = 0.15
    random_seed: int = 42
    device: str = "cpu"
    condition_on_hidden_state: bool = True
    state_embedding_dim: int = 16

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LSTMDecoderConfig":
        section = config.get("lstm_decoder", {})
        section = section if isinstance(section, dict) else {}
        return cls(
            context_size=int(section.get("context_size", 8)),
            hidden_dim=int(section.get("hidden_dim", 128)),
            num_layers=int(section.get("num_layers", 1)),
            dropout=float(section.get("dropout", 0.0)),
            batch_size=int(section.get("batch_size", 128)),
            epochs=int(section.get("epochs", 60)),
            learning_rate=float(section.get("learning_rate", 0.001)),
            weight_decay=float(section.get("weight_decay", 0.0)),
            validation_ratio=float(section.get("validation_ratio", 0.15)),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
            condition_on_hidden_state=bool(section.get("condition_on_hidden_state", True)),
            state_embedding_dim=int(section.get("state_embedding_dim", 16)),
        )


@dataclass(frozen=True)
class LSTMDecoderMetadata:
    config: LSTMDecoderConfig
    vocab_size: int
    latent_dim: int
    input_dim: int
    hidden_state_count: int
    training_log: List[Dict[str, float]]
    dataset_summary: Dict[str, Any]
    validation_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "vocab_size": int(self.vocab_size),
            "latent_dim": int(self.latent_dim),
            "input_dim": int(self.input_dim),
            "hidden_state_count": int(self.hidden_state_count),
            "training_log": self.training_log,
            "dataset_summary": self.dataset_summary,
            "validation_metrics": self.validation_metrics,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LSTMDecoderMetadata":
        return cls(
            config=LSTMDecoderConfig(**payload["config"]),
            vocab_size=int(payload["vocab_size"]),
            latent_dim=int(payload["latent_dim"]),
            input_dim=int(payload.get("input_dim", payload["latent_dim"])),
            hidden_state_count=int(payload.get("hidden_state_count", 0)),
            training_log=list(payload.get("training_log", [])),
            dataset_summary=dict(payload.get("dataset_summary", {})),
            validation_metrics=dict(payload.get("validation_metrics", {})),
        )


class LSTMTokenTorchModel:
    """Lazy wrapper around the PyTorch module to keep imports optional."""

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        config: LSTMDecoderConfig,
        hidden_state_count: int = 0,
    ) -> None:
        torch, nn = self._torch()
        dropout = float(config.dropout) if int(config.num_layers) > 1 else 0.0
        self.module = nn.Sequential()
        self.config = config
        self.hidden_state_count = int(hidden_state_count)
        self.state_embedding_dim = (
            int(config.state_embedding_dim)
            if bool(config.condition_on_hidden_state) and self.hidden_state_count > 0
            else 0
        )
        self.lstm = nn.LSTM(
            input_size=int(latent_dim),
            hidden_size=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            batch_first=True,
            dropout=dropout,
        )
        if self.state_embedding_dim > 0:
            self.state_embedding = nn.Embedding(self.hidden_state_count, self.state_embedding_dim)
        else:
            self.state_embedding = None
        self.projection = nn.Linear(int(config.hidden_dim) + self.state_embedding_dim, int(vocab_size))

    def __call__(self, inputs: Any, state_ids: Optional[Any] = None) -> Any:
        output, _ = self.lstm(inputs)
        features = output[:, -1, :]
        if self.state_embedding is not None:
            if state_ids is None:
                import torch
                state_ids = torch.zeros((inputs.shape[0],), dtype=torch.long, device=inputs.device)
            state_ids = state_ids.clamp(min=0, max=self.hidden_state_count - 1)
            features = self._concat(features, self.state_embedding(state_ids))
        return self.projection(features)

    def parameters(self) -> Any:
        params = list(self.lstm.parameters()) + list(self.projection.parameters())
        if self.state_embedding is not None:
            params.extend(list(self.state_embedding.parameters()))
        return params

    def state_dict(self) -> Dict[str, Any]:
        return {
            "lstm": self.lstm.state_dict(),
            "projection": self.projection.state_dict(),
            "state_embedding": (
                self.state_embedding.state_dict()
                if self.state_embedding is not None
                else None
            ),
        }

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        self.lstm.load_state_dict(payload["lstm"])
        self.projection.load_state_dict(payload["projection"])
        if self.state_embedding is not None and payload.get("state_embedding") is not None:
            self.state_embedding.load_state_dict(payload["state_embedding"])

    def train(self) -> None:
        self.lstm.train()
        self.projection.train()
        if self.state_embedding is not None:
            self.state_embedding.train()

    def eval(self) -> None:
        self.lstm.eval()
        self.projection.eval()
        if self.state_embedding is not None:
            self.state_embedding.eval()

    def to(self, device: str) -> "LSTMTokenTorchModel":
        self.lstm.to(device)
        self.projection.to(device)
        if self.state_embedding is not None:
            self.state_embedding.to(device)
        return self

    @staticmethod
    def _concat(left: Any, right: Any) -> Any:
        import torch
        return torch.cat([left, right], dim=-1)

    @staticmethod
    def _torch() -> tuple[Any, Any]:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("LSTM decoder requires PyTorch.") from exc
        return torch, nn


class LSTMTokenModel:
    """Runtime next-token scorer used by future generation integration."""

    def __init__(self, metadata: LSTMDecoderMetadata, model: LSTMTokenTorchModel, device: Optional[str] = None) -> None:
        self.metadata = metadata
        self.model = model
        self.device = device or metadata.config.device
        self.masker = TokenLogitMasker()

    @classmethod
    def load(cls, model_dir: str | Path, device: Optional[str] = None) -> "LSTMTokenModel":
        torch, _ = LSTMTokenTorchModel._torch()
        model_dir = Path(model_dir)
        metadata = LSTMDecoderMetadata.from_dict(
            json.loads((model_dir / "lstm_decoder.json").read_text(encoding="utf-8"))
        )
        runtime_device = device or metadata.config.device
        module = LSTMTokenTorchModel(
            metadata.input_dim,
            metadata.vocab_size,
            metadata.config,
            hidden_state_count=metadata.hidden_state_count,
        ).to(runtime_device)
        state = torch.load(model_dir / "lstm_decoder.pt", map_location=runtime_device)
        module.load_state_dict(state)
        module.eval()
        return cls(metadata, module, device=runtime_device)

    def score_next(
        self,
        context: LatentTokenContext,
        candidate_mask: CandidateTokenMask,
    ) -> TokenLogitResult:
        torch, _ = LSTMTokenTorchModel._torch()
        latents = self._padded_latents(context.latent_sequence)
        state_id = self._state_id(context.metadata)
        device = self.device
        with torch.no_grad():
            tensor = torch.tensor([latents], dtype=torch.float32, device=device)
            state_tensor = torch.tensor([state_id], dtype=torch.long, device=device)
            logits = self.model(tensor, state_tensor).detach().cpu().numpy()[0].tolist()
        result = self.masker.apply(logits, candidate_mask)
        diagnostics = dict(result.diagnostics)
        diagnostics.update({
            "model": self.__class__.__name__,
            "context_size": self.metadata.config.context_size,
            "context_token_count": len(context.token_ids),
            "latent_dim": self.metadata.latent_dim,
            "input_dim": self.metadata.input_dim,
            "hidden_state_count": self.metadata.hidden_state_count,
            "condition_on_hidden_state": bool(self.metadata.config.condition_on_hidden_state),
            "state_condition_mode": "output_embedding",
            "state_embedding_dim": int(self.metadata.config.state_embedding_dim),
            "target_state_id": int(state_id),
        })
        return TokenLogitResult(
            vocab_size=result.vocab_size,
            logits=result.logits,
            masked_logits=result.masked_logits,
            allowed_token_ids=result.allowed_token_ids,
            top_tokens=result.top_tokens,
            diagnostics=diagnostics,
        )

    def _padded_latents(
        self,
        latent_sequence: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        context_size = int(self.metadata.config.context_size)
        latent_dim = int(self.metadata.latent_dim)
        input_dim = int(self.metadata.input_dim)
        values = []
        for row in latent_sequence[-context_size:]:
            latent = [float(x) for x in row[:latent_dim]]
            latent = latent + [0.0] * max(0, latent_dim - len(latent))
            features = latent[:input_dim] + [0.0] * max(0, input_dim - len(latent))
            values.append(features)
        padding = [[0.0 for _ in range(input_dim)] for _ in range(max(0, context_size - len(values)))]
        return padding + values

    def _state_id(self, metadata: Dict[str, Any]) -> int:
        state_id = metadata.get("hidden_state")
        if state_id is None:
            return 0
        index = int(state_id)
        hidden_state_count = max(1, int(self.metadata.hidden_state_count))
        return max(0, min(hidden_state_count - 1, index))


class LSTMDecoderTrainer:
    """Train and evaluate the next-token LSTM."""

    def __init__(self, config: LSTMDecoderConfig) -> None:
        self.config = config
        self.training_log: List[Dict[str, float]] = []

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        vocab_size: int,
        dataset_summary: Dict[str, Any],
        target_states: Optional[np.ndarray] = None,
    ) -> LSTMDecoderMetadata:
        torch, nn = LSTMTokenTorchModel._torch()
        if inputs.size == 0 or len(targets) == 0:
            raise ValueError("No LSTM decoder training windows were built.")
        self._seed(torch)
        train_idx, val_idx = self._split_indices(len(targets))
        input_dim = int(inputs.shape[-1])
        latent_dim = int(dataset_summary.get("latent_dim", input_dim))
        hidden_state_count = int(dataset_summary.get("hidden_state_count", 0))
        model = LSTMTokenTorchModel(
            input_dim,
            int(vocab_size),
            self.config,
            hidden_state_count=hidden_state_count,
        ).to(self.config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        loss_fn = nn.CrossEntropyLoss()
        x = torch.tensor(inputs, dtype=torch.float32, device=self.config.device)
        y = torch.tensor(targets, dtype=torch.long, device=self.config.device)
        states = self._state_tensor(torch, target_states, len(targets), hidden_state_count)
        for epoch in range(int(self.config.epochs)):
            model.train()
            losses = []
            for batch in self._batches(train_idx):
                optimizer.zero_grad()
                logits = model(x[batch], states[batch])
                loss = loss_fn(logits, y[batch])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            train_loss = float(np.mean(losses)) if losses else math.nan
            val_metrics = self._evaluate_tensors(model, x, y, val_idx, loss_fn, states)
            self.training_log.append({
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": float(val_metrics["loss"]),
                "validation_top1": float(val_metrics["top1_accuracy"]),
                "validation_top5": float(val_metrics["top5_accuracy"]),
            })
        validation_metrics = self._evaluate_tensors(model, x, y, val_idx, loss_fn, states)
        self.model = model
        return LSTMDecoderMetadata(
            config=self.config,
            vocab_size=int(vocab_size),
            latent_dim=latent_dim,
            input_dim=input_dim,
            hidden_state_count=hidden_state_count,
            training_log=self.training_log,
            dataset_summary=dataset_summary,
            validation_metrics=validation_metrics,
        )

    def save(self, output_dir: str | Path, metadata: LSTMDecoderMetadata) -> None:
        if not hasattr(self, "model"):
            raise RuntimeError("No trained model to save.")
        torch, _ = LSTMTokenTorchModel._torch()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "lstm_decoder.json").write_text(
            json.dumps(metadata.to_dict(), indent=2),
            encoding="utf-8",
        )
        torch.save(self.model.state_dict(), output_dir / "lstm_decoder.pt")

    def evaluate(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        model: Optional[LSTMTokenTorchModel] = None,
        target_states: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        torch, nn = LSTMTokenTorchModel._torch()
        runtime_model = model or getattr(self, "model")
        runtime_model.eval()
        x = torch.tensor(inputs, dtype=torch.float32, device=self.config.device)
        y = torch.tensor(targets, dtype=torch.long, device=self.config.device)
        states = self._state_tensor(
            torch,
            target_states,
            len(targets),
            int(getattr(runtime_model, "hidden_state_count", 0)),
        )
        return self._evaluate_tensors(runtime_model, x, y, list(range(len(targets))), nn.CrossEntropyLoss(), states)

    def _split_indices(self, size: int) -> tuple[List[int], List[int]]:
        rng = np.random.default_rng(int(self.config.random_seed))
        indices = np.arange(size, dtype=int)
        rng.shuffle(indices)
        val_size = int(round(size * max(0.0, min(0.9, float(self.config.validation_ratio)))))
        val_idx = indices[:val_size].tolist()
        train_idx = indices[val_size:].tolist()
        if not train_idx:
            train_idx = val_idx
        if not val_idx:
            val_idx = train_idx
        return train_idx, val_idx

    def _batches(self, indices: Sequence[int]) -> List[List[int]]:
        rng = np.random.default_rng(int(self.config.random_seed) + len(self.training_log))
        shuffled = np.asarray(indices, dtype=int)
        rng.shuffle(shuffled)
        batch_size = max(1, int(self.config.batch_size))
        return [
            shuffled[start:start + batch_size].tolist()
            for start in range(0, len(shuffled), batch_size)
        ]

    def _evaluate_tensors(
        self,
        model: LSTMTokenTorchModel,
        x: Any,
        y: Any,
        indices: Sequence[int],
        loss_fn: Any,
        states: Optional[Any] = None,
    ) -> Dict[str, Any]:
        torch, _ = LSTMTokenTorchModel._torch()
        if not indices:
            return {
                "loss": math.nan,
                "perplexity": math.nan,
                "top1_accuracy": math.nan,
                "top3_accuracy": math.nan,
                "top5_accuracy": math.nan,
                "sample_count": 0,
            }
        model.eval()
        with torch.no_grad():
            idx = torch.tensor(list(indices), dtype=torch.long, device=x.device)
            state_batch = states[idx] if states is not None else None
            logits = model(x[idx], state_batch)
            targets = y[idx]
            loss = float(loss_fn(logits, targets).detach().cpu().item())
            metrics = {
                "loss": loss,
                "perplexity": float(math.exp(min(20.0, loss))),
                "sample_count": int(len(indices)),
            }
            for k in (1, 3, 5):
                limit = min(k, int(logits.shape[-1]))
                top = torch.topk(logits, k=limit, dim=-1).indices
                correct = (top == targets[:, None]).any(dim=1).float().mean()
                metrics[f"top{k}_accuracy"] = float(correct.detach().cpu().item())
            return metrics

    def _state_tensor(
        self,
        torch: Any,
        target_states: Optional[np.ndarray],
        size: int,
        hidden_state_count: int,
    ) -> Any:
        if target_states is None:
            values = np.zeros((int(size),), dtype=np.int64)
        else:
            values = np.asarray(target_states, dtype=np.int64)
        if values.shape[0] != int(size):
            raise ValueError("target_states length must match targets length.")
        if int(hidden_state_count) > 0:
            values = np.clip(values, 0, int(hidden_state_count) - 1)
        else:
            values = np.zeros_like(values)
        return torch.tensor(values, dtype=torch.long, device=self.config.device)

    def _seed(self, torch: Any) -> None:
        seed = int(self.config.random_seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if str(self.config.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
