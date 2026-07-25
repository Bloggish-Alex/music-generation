#!/usr/bin/env python3
"""Generation pipeline for Latent-Transformer MDN + DVAE decoder."""

from __future__ import annotations

import json
import random
import re
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from diagnostics.dvae_midi_render import DVAEMidiRenderConfig, TensorMidiRenderer
from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from model.latent_transformer import LatentTransformerConfig, LatentTransformerMDN, MDNConfig
from model.memory_latent_transformer import MemoryLatentTransformer, MemoryLatentTransformerConfig
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader
from pipeline.latent_retrieval import LatentCandidateIndex, RetrievalConfig
from pipeline.theme_embedding_provider import FrozenThemeEmbeddingProvider


@dataclass(frozen=True)
class LatentGenerationConfig:
    """Configuration for one generated sequence."""

    bars: int = 32
    primer_bars: int = 8
    temperature: float = 0.9
    sample_std_scale: float = 0.0
    backend: str = "direct_mdn"
    model_variant: str = "root"
    default_action: str = "VARY"
    action_plan: str = "source"
    action_sections: str = ""
    seed: int = 42
    device: str = "cpu"
    base_pitch: int = 60
    tempo_bpm: int = 120
    retrieval_top_k: int = 24
    retrieval_temperature: float = 0.35
    retrieval_distance_weight: float = 1.0
    retrieval_energy_weight: float = 1.25
    retrieval_position_weight: float = 0.25
    retrieval_recent_penalty: float = 2.0
    retrieval_recent_window: int = 8
    retrieval_use_retrieved_tensors: Optional[bool] = None
    energy_curve: str = "INTRODUCE:0.35,VARY:0.55,DEVELOP:0.9,RETURN:0.7,CADENCE:0.45"
    energy_arc_strength: float = 0.15
    memory_scope_enabled: bool = True
    memory_scope_top_n: int = 4
    memory_scope_develop_top_n: int = 12

    def normalized_model_variant(self) -> str:
        """Return the canonical model variant name."""
        value = str(self.model_variant).strip().lower().replace("-", "_")
        return "root" if value in {"", "default"} else value

    def retrieval_config(self) -> RetrievalConfig:
        """Return retrieval backend config."""
        return RetrievalConfig(
            enabled=str(self.backend).lower() in {"retrieval_mdn", "memory_latent"},
            top_k=int(self.retrieval_top_k),
            temperature=float(self.retrieval_temperature),
            distance_weight=float(self.retrieval_distance_weight),
            energy_weight=float(self.retrieval_energy_weight),
            position_weight=float(self.retrieval_position_weight),
            recent_penalty=float(self.retrieval_recent_penalty),
            recent_window=int(self.retrieval_recent_window),
            use_retrieved_tensors=bool(self.resolved_use_retrieved_tensors()),
            energy_curve=str(self.energy_curve),
            energy_arc_strength=float(self.energy_arc_strength),
        )

    def resolved_use_retrieved_tensors(self) -> bool:
        """Return whether selected source tensors should bypass the DVAE decoder."""
        if self.retrieval_use_retrieved_tensors is not None:
            return bool(self.retrieval_use_retrieved_tensors)
        return str(self.backend).lower() == "retrieval_mdn"


@dataclass
class LatentGenerationResult:
    """Paths and diagnostics produced by generation."""

    json_path: Path
    midi_path: Path
    tensor_path: Path
    diagnostics: Dict[str, Any]


class LatentGenerationPipeline:
    """Generate bar tensors and MIDI from a trained latent Transformer and DVAE."""

    def __init__(self, config: LatentGenerationConfig) -> None:
        self.config = config
        self.reader = LatentDatasetReader()

    def run(
        self,
        model_dir: str | Path,
        latent_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        seed_song_id: Optional[str] = None,
        transformer_path: Optional[str | Path] = None,
        dvae_path: Optional[str | Path] = None,
    ) -> LatentGenerationResult:
        """Run generation and write JSON, tensor NPZ, and MIDI outputs."""
        self._set_seed()
        model_directory = Path(model_dir)
        backend = str(self.config.backend).lower()
        transformer_checkpoint_path = (
            Path(transformer_path)
            if transformer_path
            else self._default_transformer_checkpoint_path(model_directory, backend, self.config.normalized_model_variant())
        )
        dvae_checkpoint_path = Path(dvae_path) if dvae_path else model_directory / "dvae.pt"
        mu, rows, latent_summary = self.reader.load(latent_dir)
        dvae = self._load_dvae(dvae_checkpoint_path)
        if backend == "memory_latent":
            transformer, transformer_checkpoint = self._load_memory_transformer(transformer_checkpoint_path)
            model_config = transformer.config
        else:
            transformer, transformer_checkpoint = self._load_transformer(transformer_checkpoint_path)
            model_config = transformer.transformer_config
        action_to_id = {str(k): int(v) for k, v in transformer_checkpoint["action_to_id"].items()}
        id_to_action = {int(v): str(k) for k, v in action_to_id.items()}
        grouped = self._group_rows(rows)
        selected_song_id = self._select_song_id(grouped, seed_song_id)
        ordered_indices = grouped[selected_song_id]
        retrieval_index = self._retrieval_index(
            model_directory,
            mu,
            rows,
            action_to_id,
            int(model_config.position_vocab_size),
        )
        theme_embedding, theme_tokens, theme_diag = self._theme_context(
            transformer_checkpoint,
            mu,
            rows,
            selected_song_id,
            model_config,
        )
        if backend == "memory_latent":
            generated_mu, generation_steps, selected_row_indices, retrieved_tensors = self._generate_memory_latents(
                transformer=transformer,
                seed_mu=mu,
                rows=rows,
                ordered_indices=ordered_indices,
                action_to_id=action_to_id,
                id_to_action=id_to_action,
                theme_embedding=theme_embedding,
                theme_tokens=theme_tokens,
                action_planner=GenerationActionPlanner(self.config, action_to_id),
                retrieval_index=retrieval_index,
            )
        else:
            generated_mu, generation_steps, selected_row_indices, retrieved_tensors = self._generate_latents(
                transformer=transformer,
                seed_mu=mu,
                rows=rows,
                ordered_indices=ordered_indices,
                action_to_id=action_to_id,
                id_to_action=id_to_action,
                theme_embedding=theme_embedding,
                theme_tokens=theme_tokens,
                action_planner=GenerationActionPlanner(self.config, action_to_id),
                retrieval_index=retrieval_index,
            )
        tensors = self._output_tensors(dvae, generated_mu, retrieved_tensors)
        sequence_diagnostics = self._sequence_diagnostics(tensors, generation_steps)
        tensor_path = Path(output_json).with_suffix(".bar_tensors.npz")
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tensor_path,
            bars=tensors.astype(np.float32),
            latent_mu=generated_mu.astype(np.float32),
            selected_row_indices=np.asarray(selected_row_indices, dtype=np.int64),
        )
        midi_diag = SequenceTensorMidiRenderer(DVAEMidiRenderConfig(
            tempo_bpm=int(self.config.tempo_bpm),
            default_base_pitch=int(self.config.base_pitch),
        )).render(tensors, output_midi, base_pitch=int(self.config.base_pitch))
        diagnostics = {
            "model_dir": str(model_directory),
            "latent_dir": str(latent_dir),
            "transformer_checkpoint": str(transformer_checkpoint_path),
            "dvae_checkpoint": str(dvae_checkpoint_path),
            "checkpoint_role": transformer_checkpoint.get("checkpoint_role"),
            "latent_summary": latent_summary,
            "config": self.config.__dict__,
            "render_source": {
                "use_retrieved_tensors": bool(self.config.resolved_use_retrieved_tensors()),
                "mode": "retrieved_tensors" if self.config.resolved_use_retrieved_tensors() else "dvae_decoded_selected_latents",
            },
            "selected_song_id": selected_song_id,
            "seed_song_id": seed_song_id,
            "primer_bars": int(min(self.config.primer_bars, len(ordered_indices))),
            "generated_bar_count": int(generated_mu.shape[0]),
            "action_to_id": action_to_id,
            "action_plan": GenerationActionPlanner(self.config, action_to_id).diagnostics(target_bars=int(self.config.bars)),
            "generation_backend": str(self.config.backend),
            "model_variant": self.config.normalized_model_variant(),
            "retrieval": retrieval_index.diagnostics() if retrieval_index is not None else {"enabled": False},
            "theme_context": theme_diag,
            "sequence_diagnostics": sequence_diagnostics,
            "steps": generation_steps,
            "midi": midi_diag,
            "tensor_path": str(tensor_path),
            "json_path": str(output_json),
            "midi_path": str(output_midi),
        }
        Path(output_json).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return LatentGenerationResult(
            json_path=Path(output_json),
            midi_path=Path(output_midi),
            tensor_path=tensor_path,
            diagnostics=diagnostics,
        )

    def _load_transformer(self, path: Path) -> tuple[LatentTransformerMDN, Dict[str, Any]]:
        """Load a trained LatentTransformerMDN checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self._validate_checkpoint_keys(
            checkpoint=checkpoint,
            path=path,
            required_keys=("transformer_config", "mdn_config", "state_dict"),
            expected_kind="LatentTransformerMDN",
            hint="Use latent_transformer_mdn.pt for direct_mdn/retrieval_mdn backends.",
        )
        transformer_config = LatentTransformerConfig(**checkpoint["transformer_config"])
        mdn_config = MDNConfig(**checkpoint["mdn_config"])
        model = LatentTransformerMDN(transformer_config, mdn_config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint

    def _load_memory_transformer(self, path: Path) -> tuple[MemoryLatentTransformer, Dict[str, Any]]:
        """Load a trained MemoryLatentTransformer checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self._validate_checkpoint_keys(
            checkpoint=checkpoint,
            path=path,
            required_keys=("model_config", "state_dict"),
            expected_kind="MemoryLatentTransformer",
            hint="Use the memory model directory produced by train_memory_latent_transformer.py, or pass --transformer-path explicitly.",
        )
        model_config = MemoryLatentTransformerConfig(**checkpoint["model_config"])
        model = MemoryLatentTransformer(model_config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint

    def _default_transformer_checkpoint_path(self, model_directory: Path, backend: str, model_variant: str) -> Path:
        """Return the default transformer checkpoint path for the selected backend."""
        if backend == "memory_latent":
            return self._first_existing_path((
                model_directory / "memory_latent_transformer.pt",
                model_directory / "memory_latent_transformer.final.pt",
            ))
        if model_variant in {"theme_fusion", "theme-fusion", "theme"}:
            return self._first_existing_path((
                model_directory / "fold_1_theme" / "latent_transformer_mdn.pt",
                model_directory / "latent_transformer_mdn.pt",
                model_directory / "fold_1_theme" / "latent_transformer_mdn.final.pt",
                model_directory / "latent_transformer_mdn.final.pt",
            ))
        if model_variant not in {"root", "default", ""}:
            raise ValueError(f"Unsupported model_variant: {model_variant}")
        return model_directory / "latent_transformer_mdn.pt"

    @staticmethod
    def _first_existing_path(candidates: Sequence[Path]) -> Path:
        """Return the first existing path, or the preferred candidate for a useful downstream error."""
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _validate_checkpoint_keys(
        checkpoint: Dict[str, Any],
        path: Path,
        required_keys: Sequence[str],
        expected_kind: str,
        hint: str,
    ) -> None:
        """Raise a clear error if the checkpoint does not match the selected backend."""
        missing = [key for key in required_keys if key not in checkpoint]
        if not missing:
            return
        available = ", ".join(sorted(str(key) for key in checkpoint.keys()))
        raise ValueError(
            f"Checkpoint '{path}' is not a valid {expected_kind} checkpoint. "
            f"Missing keys: {missing}. Available keys: [{available}]. {hint}"
        )

    def _load_dvae(self, path: Path) -> DenoisingMusicVAE:
        """Load a trained DVAE checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        config = DVAEMusicConfig(**checkpoint["config"])
        model = DenoisingMusicVAE(config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _theme_context(
        self,
        checkpoint: Dict[str, Any],
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        selected_song_id: str,
        transformer_config: Any,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Build theme embedding/token tensors for generation."""
        if not bool(transformer_config.theme_fusion_enabled):
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), {"enabled": False}
        provider_config = {"theme_fusion": dict(checkpoint.get("theme_fusion_config", {}))}
        provider_config["theme_fusion"]["enabled"] = True
        provider = FrozenThemeEmbeddingProvider.from_config(provider_config, device=self.config.device)
        embeddings, token_sequences, diagnostics = provider.theme_contexts_by_song(mu, rows)
        embedding = embeddings.get(selected_song_id)
        if embedding is None:
            embedding = np.zeros((int(transformer_config.theme_embedding_dim),), dtype=np.float32)
            diagnostics["generation_missing_embedding"] = True
        tokens = token_sequences.get(selected_song_id)
        if tokens is None:
            tokens = np.zeros((int(transformer_config.theme_token_bars), int(transformer_config.latent_dim)), dtype=np.float32)
            diagnostics["generation_missing_tokens"] = True
        return embedding.astype(np.float32), tokens.astype(np.float32), diagnostics

    def _generate_latents(
        self,
        transformer: LatentTransformerMDN,
        seed_mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        ordered_indices: Sequence[int],
        action_to_id: Dict[str, int],
        id_to_action: Dict[int, str],
        theme_embedding: np.ndarray,
        theme_tokens: np.ndarray,
        action_planner: "GenerationActionPlanner",
        retrieval_index: Optional[LatentCandidateIndex],
    ) -> tuple[np.ndarray, List[Dict[str, Any]], List[int], List[Optional[np.ndarray]]]:
        """Autoregressively sample latent means."""
        cfg = transformer.transformer_config
        context_bars = int(cfg.context_bars)
        target_bars = max(1, int(self.config.bars))
        primer_count = max(1, min(int(self.config.primer_bars), len(ordered_indices), target_bars))
        generated: List[np.ndarray] = [seed_mu[index].astype(np.float32) for index in ordered_indices[:primer_count]]
        selected_row_indices: List[int] = [int(index) for index in ordered_indices[:primer_count]]
        retrieved_tensors: List[Optional[np.ndarray]] = [
            retrieval_index.tensor_for_row(int(index)) if retrieval_index is not None else None
            for index in ordered_indices[:primer_count]
        ]
        generated_actions: List[int] = [
            self._action_id(rows[index], action_to_id)
            for index in ordered_indices[:primer_count]
        ]
        generated_positions: List[int] = [int(i) % int(cfg.position_vocab_size) for i in range(primer_count)]
        steps: List[Dict[str, Any]] = [
            {
                "bar_index": int(i),
                "source": "primer",
                "action_id": int(generated_actions[i]),
                "action": id_to_action.get(int(generated_actions[i]), "UNKNOWN"),
                "position_id": int(generated_positions[i]),
            }
            for i in range(primer_count)
        ]
        rng = np.random.default_rng(int(self.config.seed))
        while len(generated) < target_bars:
            bar_index = int(len(generated))
            target_row = rows[ordered_indices[bar_index]] if bar_index < len(ordered_indices) else {}
            target_action = action_planner.action_id(bar_index=bar_index, source_row=target_row)
            target_position = int(bar_index) % int(cfg.position_vocab_size)
            prepared = self._prepare_context(generated, generated_actions, generated_positions, context_bars, int(cfg.latent_dim))
            with torch.no_grad():
                output = transformer(
                    context_mu=torch.from_numpy(prepared["context_mu"]).unsqueeze(0).float().to(self.config.device),
                    context_action_ids=torch.from_numpy(prepared["context_action_ids"]).unsqueeze(0).long().to(self.config.device),
                    context_position_ids=torch.from_numpy(prepared["context_position_ids"]).unsqueeze(0).long().to(self.config.device),
                    target_action_ids=torch.tensor([target_action], dtype=torch.long, device=self.config.device),
                    target_position_ids=torch.tensor([target_position], dtype=torch.long, device=self.config.device),
                    padding_mask=torch.from_numpy(prepared["padding_mask"]).unsqueeze(0).bool().to(self.config.device),
                    theme_embedding=torch.from_numpy(theme_embedding).unsqueeze(0).float().to(self.config.device),
                    theme_tokens=torch.from_numpy(theme_tokens).unsqueeze(0).float().to(self.config.device),
                )
            raw_logits = output.pi_logits[0] / max(1.0e-6, float(self.config.temperature))
            invalid_logits = not bool(torch.isfinite(raw_logits).all())
            logits = torch.nan_to_num(
                raw_logits,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            )
            probs = torch.softmax(logits, dim=-1)
            invalid_probs = not bool(torch.isfinite(probs).all()) or float(probs.sum().detach().cpu()) <= 0.0
            if invalid_probs:
                probs = torch.ones_like(probs) / float(probs.numel())
            component = int(torch.multinomial(probs, num_samples=1).detach().cpu().item())
            raw_component_mu = output.mu[0, component]
            raw_component_sigma = output.sigma[0, component]
            invalid_mu = not bool(torch.isfinite(raw_component_mu).all())
            invalid_sigma = not bool(torch.isfinite(raw_component_sigma).all())
            component_mu = torch.nan_to_num(raw_component_mu, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy().astype(np.float32)
            component_sigma = torch.nan_to_num(raw_component_sigma, nan=1.0, posinf=1.0, neginf=1.0).detach().cpu().numpy().astype(np.float32)
            if float(self.config.sample_std_scale) > 0.0:
                next_mu = component_mu + rng.normal(0.0, float(self.config.sample_std_scale), size=component_mu.shape).astype(np.float32) * component_sigma
            else:
                next_mu = component_mu
            generated.append(next_mu.astype(np.float32))
            generated_actions.append(int(target_action))
            generated_positions.append(int(target_position))
            steps.append({
                "bar_index": int(bar_index),
                "source": "generated",
                "action_id": int(target_action),
                "action": id_to_action.get(int(target_action), "UNKNOWN"),
                "position_id": int(target_position),
                "component": int(component),
                "component_probability": float(probs[component].detach().cpu()),
                "max_component_probability": float(probs.max().detach().cpu()),
                "sigma_mean": float(np.mean(component_sigma)),
                "invalid_logits_fallback": bool(invalid_logits),
                "invalid_probs_fallback": bool(invalid_probs),
                "invalid_mu_fallback": bool(invalid_mu),
                "invalid_sigma_fallback": bool(invalid_sigma),
            })
            if str(self.config.backend).lower() == "retrieval_mdn":
                if retrieval_index is None:
                    raise ValueError("retrieval_mdn backend requires a retrieval index.")
                action_name = id_to_action.get(int(target_action), "UNKNOWN")
                selection = retrieval_index.select(
                    predicted_mu=next_mu,
                    action_id=int(target_action),
                    action_name=action_name,
                    position_id=int(target_position),
                    bar_index=int(bar_index),
                    total_bars=int(target_bars),
                    recent_row_indices=selected_row_indices,
                    config=self.config.retrieval_config(),
                    rng=rng,
                )
                next_mu = selection.mu
                generated[-1] = next_mu.astype(np.float32)
                selected_row_indices.append(int(selection.row_index))
                retrieved_tensors.append(selection.tensor)
                steps[-1].update(selection.diagnostics)
            else:
                selected_row_indices.append(-1)
                retrieved_tensors.append(None)
        return np.stack(generated, axis=0).astype(np.float32), steps, selected_row_indices, retrieved_tensors

    def _generate_memory_latents(
        self,
        transformer: MemoryLatentTransformer,
        seed_mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        ordered_indices: Sequence[int],
        action_to_id: Dict[str, int],
        id_to_action: Dict[int, str],
        theme_embedding: np.ndarray,
        theme_tokens: np.ndarray,
        action_planner: "GenerationActionPlanner",
        retrieval_index: Optional[LatentCandidateIndex],
    ) -> tuple[np.ndarray, List[Dict[str, Any]], List[int], List[Optional[np.ndarray]]]:
        """Autoregressively select real memory latents from a learned query model."""
        if retrieval_index is None:
            raise ValueError("memory_latent backend requires a retrieval index.")
        cfg = transformer.config
        context_bars = int(cfg.context_bars)
        target_bars = max(1, int(self.config.bars))
        primer_count = max(1, min(int(self.config.primer_bars), len(ordered_indices), target_bars))
        generated: List[np.ndarray] = [seed_mu[index].astype(np.float32) for index in ordered_indices[:primer_count]]
        selected_row_indices: List[int] = [int(index) for index in ordered_indices[:primer_count]]
        retrieved_tensors: List[Optional[np.ndarray]] = [retrieval_index.tensor_for_row(int(index)) for index in ordered_indices[:primer_count]]
        generated_actions: List[int] = [self._action_id(rows[index], action_to_id) for index in ordered_indices[:primer_count]]
        generated_positions: List[int] = [int(i) % int(cfg.position_vocab_size) for i in range(primer_count)]
        steps: List[Dict[str, Any]] = [
            {
                "bar_index": int(i),
                "source": "primer",
                "action_id": int(generated_actions[i]),
                "action": id_to_action.get(int(generated_actions[i]), "UNKNOWN"),
                "position_id": int(generated_positions[i]),
                "selected_row_index": int(selected_row_indices[i]),
            }
            for i in range(primer_count)
        ]
        rng = np.random.default_rng(int(self.config.seed))
        memory_scope = MemoryScopeController(
            enabled=bool(self.config.memory_scope_enabled),
            top_n=int(self.config.memory_scope_top_n),
            develop_top_n=int(self.config.memory_scope_develop_top_n),
        )
        while len(generated) < target_bars:
            bar_index = int(len(generated))
            target_row = rows[ordered_indices[bar_index]] if bar_index < len(ordered_indices) else {}
            target_action = action_planner.action_id(bar_index=bar_index, source_row=target_row)
            target_action_name = id_to_action.get(int(target_action), "UNKNOWN")
            target_position = int(bar_index) % int(cfg.position_vocab_size)
            prepared = self._prepare_context(generated, generated_actions, generated_positions, context_bars, int(cfg.latent_dim))
            with torch.no_grad():
                query = transformer(
                    context_mu=torch.from_numpy(prepared["context_mu"]).unsqueeze(0).float().to(self.config.device),
                    context_action_ids=torch.from_numpy(prepared["context_action_ids"]).unsqueeze(0).long().to(self.config.device),
                    context_position_ids=torch.from_numpy(prepared["context_position_ids"]).unsqueeze(0).long().to(self.config.device),
                    target_action_ids=torch.tensor([target_action], dtype=torch.long, device=self.config.device),
                    target_position_ids=torch.tensor([target_position], dtype=torch.long, device=self.config.device),
                    padding_mask=torch.from_numpy(prepared["padding_mask"]).unsqueeze(0).bool().to(self.config.device),
                    theme_embedding=torch.from_numpy(theme_embedding).unsqueeze(0).float().to(self.config.device),
                    theme_tokens=torch.from_numpy(theme_tokens).unsqueeze(0).float().to(self.config.device),
                )
            query_mu = torch.nan_to_num(query[0], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy().astype(np.float32)
            expected_row_index = int(ordered_indices[bar_index]) if bar_index < len(ordered_indices) else None
            scope = memory_scope.scope_for_step(
                retrieval_index=retrieval_index,
                query_mu=query_mu,
                action_id=int(target_action),
                action_name=target_action_name,
                position_id=int(target_position),
            )
            selection = retrieval_index.select_memory(
                query_mu=query_mu,
                action_id=int(target_action),
                position_id=int(target_position),
                expected_row_index=expected_row_index,
                config=self.config.retrieval_config(),
                rng=rng,
                allowed_source_base_ids=scope.source_base_ids,
            )
            generated.append(selection.mu.astype(np.float32))
            generated_actions.append(int(target_action))
            generated_positions.append(int(target_position))
            selected_row_indices.append(int(selection.row_index))
            retrieved_tensors.append(selection.tensor)
            steps.append({
                "bar_index": int(bar_index),
                "source": "generated",
                "generation_model": "memory_latent",
                "action_id": int(target_action),
                "action": target_action_name,
                "position_id": int(target_position),
                "query_norm": float(np.linalg.norm(query_mu)),
                "memory_scope": scope.diagnostics,
                **selection.diagnostics,
            })
        return np.stack(generated, axis=0).astype(np.float32), steps, selected_row_indices, retrieved_tensors

    def _sequence_diagnostics(self, tensors: np.ndarray, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize whole-sequence continuity and source consistency."""
        bars = np.asarray(tensors, dtype=np.float32)
        note_counts: List[int] = []
        active_counts: List[int] = []
        pitch_ranges: List[float] = []
        first_pitches: List[Optional[float]] = []
        last_pitches: List[Optional[float]] = []
        for bar in bars:
            note_mask = bar[..., 2] > 0.5
            active_mask = (bar[..., 2] > 0.5) | (bar[..., 3] > 0.5)
            pitches = bar[..., 0][note_mask] * 24.0 + float(self.config.base_pitch)
            note_counts.append(int(note_mask.sum()))
            active_counts.append(int(active_mask.sum()))
            pitch_ranges.append(float(np.max(pitches) - np.min(pitches)) if len(pitches) else 0.0)
            first_pitch, last_pitch = self._bar_boundary_pitches(bar)
            first_pitches.append(first_pitch)
            last_pitches.append(last_pitch)

        boundary_jumps: List[float] = []
        adjacent_bar_l2: List[float] = []
        for index in range(1, len(bars)):
            adjacent_bar_l2.append(float(np.linalg.norm(bars[index] - bars[index - 1])))
            if last_pitches[index - 1] is not None and first_pitches[index] is not None:
                jump = float(abs(float(first_pitches[index]) - float(last_pitches[index - 1])))
                boundary_jumps.append(jump)
                if index < len(steps):
                    steps[index]["boundary_jump_from_previous"] = jump
            elif index < len(steps):
                steps[index]["boundary_jump_from_previous"] = None

        generated_steps = [step for step in steps if step.get("source") == "generated"]
        selected_song_ids = [str(step.get("selected_song_id")) for step in generated_steps if step.get("selected_song_id")]
        selected_source_base_ids = [
            str(step.get("selected_source_base_id"))
            for step in generated_steps
            if step.get("selected_source_base_id")
        ]
        source_switch_count = sum(
            1
            for index in range(1, len(selected_song_ids))
            if selected_song_ids[index] != selected_song_ids[index - 1]
        )
        source_base_switch_count = sum(
            1
            for index in range(1, len(selected_source_base_ids))
            if selected_source_base_ids[index] != selected_source_base_ids[index - 1]
        )
        expected_ranks = [
            int(step["expected_row_rank"])
            for step in generated_steps
            if isinstance(step.get("expected_row_rank"), int)
        ]
        similarities = [
            float(step["selected_memory_similarity"])
            for step in generated_steps
            if isinstance(step.get("selected_memory_similarity"), (int, float))
        ]
        return {
            "note_count": self._numeric_summary(note_counts),
            "active_slot_count": self._numeric_summary(active_counts),
            "pitch_range": self._numeric_summary(pitch_ranges),
            "adjacent_bar_l2": self._numeric_summary(adjacent_bar_l2),
            "boundary_jump_abs": self._numeric_summary(boundary_jumps),
            "boundary_jump_gt12_count": int(sum(1 for value in boundary_jumps if value > 12.0)),
            "boundary_jump_gt24_count": int(sum(1 for value in boundary_jumps if value > 24.0)),
            "generated_source_song_count": int(len(selected_song_ids)),
            "generated_unique_source_song_count": int(len(set(selected_song_ids))),
            "source_switch_count": int(source_switch_count),
            "generated_unique_source_base_count": int(len(set(selected_source_base_ids))),
            "source_base_switch_count": int(source_base_switch_count),
            "memory_similarity": self._numeric_summary(similarities),
            "expected_row_rank": self._numeric_summary(expected_ranks),
        }

    def _bar_boundary_pitches(self, bar: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        """Return first and last note-on pitch for one bar tensor."""
        events: List[tuple[int, int, float]] = []
        for slot_index in range(bar.shape[1]):
            for track_index in range(bar.shape[0]):
                if float(bar[track_index, slot_index, 2]) > 0.5:
                    pitch = float(bar[track_index, slot_index, 0] * 24.0 + float(self.config.base_pitch))
                    events.append((int(slot_index), int(track_index), pitch))
        if not events:
            return None, None
        return float(events[0][2]), float(events[-1][2])

    def _numeric_summary(self, values: Sequence[float | int]) -> Dict[str, Any]:
        """Return compact numeric summary for diagnostics."""
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    def _output_tensors(
        self,
        dvae: DenoisingMusicVAE,
        latent_mu: np.ndarray,
        retrieved_tensors: Sequence[Optional[np.ndarray]],
    ) -> np.ndarray:
        """Return final tensors using retrieval tensors when configured and available."""
        if str(self.config.backend).lower() in {"retrieval_mdn", "memory_latent"} and bool(self.config.resolved_use_retrieved_tensors()):
            if retrieved_tensors and all(tensor is not None for tensor in retrieved_tensors):
                return np.stack([np.asarray(tensor, dtype=np.float32) for tensor in retrieved_tensors], axis=0)
        return self._decode_tensors(dvae, latent_mu)

    def _retrieval_index(
        self,
        model_dir: Path,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        action_to_id: Dict[str, int],
        position_vocab_size: int,
    ) -> Optional[LatentCandidateIndex]:
        """Build retrieval index only when requested."""
        if str(self.config.backend).lower() not in {"retrieval_mdn", "memory_latent"}:
            return None
        return LatentCandidateIndex.from_model_dir(
            model_dir=model_dir,
            mu=mu,
            rows=rows,
            action_to_id=action_to_id,
            position_vocab_size=position_vocab_size,
        )

    def _decode_tensors(self, dvae: DenoisingMusicVAE, latent_mu: np.ndarray) -> np.ndarray:
        """Decode latent vectors into [bars, tracks, slots, features] tensors."""
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

    def _prepare_context(
        self,
        generated: Sequence[np.ndarray],
        action_ids: Sequence[int],
        position_ids: Sequence[int],
        context_bars: int,
        latent_dim: int,
    ) -> Dict[str, np.ndarray]:
        """Build left-padded context arrays for one next-bar prediction."""
        context_mu = np.zeros((context_bars, latent_dim), dtype=np.float32)
        context_action_ids = np.zeros((context_bars,), dtype=np.int64)
        context_position_ids = np.zeros((context_bars,), dtype=np.int64)
        padding_mask = np.ones((context_bars,), dtype=bool)
        selected = list(range(max(0, len(generated) - context_bars), len(generated)))
        for local, source_index in enumerate(selected):
            slot = local
            context_mu[slot] = np.asarray(generated[source_index], dtype=np.float32)
            context_action_ids[slot] = int(action_ids[source_index])
            context_position_ids[slot] = int(position_ids[source_index])
            padding_mask[slot] = False
        return {
            "context_mu": context_mu,
            "context_action_ids": context_action_ids,
            "context_position_ids": context_position_ids,
            "padding_mask": padding_mask,
        }

    def _action_id(self, row: Dict[str, Any], action_to_id: Dict[str, int]) -> int:
        """Return an action id from row metadata or fallback config."""
        action = str(row.get("action") or self.config.default_action)
        if action in action_to_id:
            return int(action_to_id[action])
        return int(action_to_id.get(self.config.default_action, action_to_id.get("UNKNOWN", 1)))

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group row indices by song_id and sort by bar_index."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return {
            song_id: sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            for song_id, indices in grouped.items()
        }

    def _select_song_id(self, grouped: Dict[str, List[int]], seed_song_id: Optional[str]) -> str:
        """Choose the source song for primer/theme context."""
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
        """Seed Python, numpy, and torch RNGs."""
        seed = int(self.config.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class MemoryScopeResult:
    """Source neighborhood selected for one memory-latent step."""

    source_base_ids: Optional[List[str]]
    diagnostics: Dict[str, Any]


class MemoryScopeController:
    """Maintain section-level source neighborhoods for memory-latent generation."""

    def __init__(self, enabled: bool, top_n: int, develop_top_n: int) -> None:
        self.enabled = bool(enabled)
        self.top_n = max(1, int(top_n))
        self.develop_top_n = max(self.top_n, int(develop_top_n))
        self.active_action: Optional[str] = None
        self.active_scope: Optional[List[str]] = None
        self.theme_scope: Optional[List[str]] = None

    def scope_for_step(
        self,
        retrieval_index: LatentCandidateIndex,
        query_mu: np.ndarray,
        action_id: int,
        action_name: str,
        position_id: int,
    ) -> MemoryScopeResult:
        """Return current source scope, refreshing at action boundaries."""
        if not self.enabled:
            return MemoryScopeResult(
                source_base_ids=None,
                diagnostics={"enabled": False, "source_base_ids": []},
            )
        action = str(action_name)
        section_changed = action != self.active_action
        if section_changed:
            self.active_action = action
            if action == "RETURN" and self.theme_scope:
                self.active_scope = list(self.theme_scope)
                return MemoryScopeResult(
                    source_base_ids=list(self.active_scope),
                    diagnostics={
                        "enabled": True,
                        "section_changed": True,
                        "mode": "return_to_theme_scope",
                        "source_base_ids": list(self.active_scope),
                    },
                )
            top_n = self.develop_top_n if action == "DEVELOP" else self.top_n
            neighborhood = retrieval_index.source_neighborhood(
                query_mu=query_mu,
                action_id=int(action_id),
                position_id=int(position_id),
                top_n=int(top_n),
            )
            self.active_scope = list(neighborhood["source_base_ids"])
            if self.theme_scope is None and action in {"INTRODUCE", "VARY", "REPEAT"}:
                self.theme_scope = list(self.active_scope)
            return MemoryScopeResult(
                source_base_ids=list(self.active_scope),
                diagnostics={
                    "enabled": True,
                    "section_changed": True,
                    "mode": "new_section_scope",
                    "top_n": int(top_n),
                    **neighborhood,
                },
            )
        return MemoryScopeResult(
            source_base_ids=list(self.active_scope or []),
            diagnostics={
                "enabled": True,
                "section_changed": False,
                "mode": "reuse_section_scope",
                "source_base_ids": list(self.active_scope or []),
            },
        )


class GenerationActionPlanner:
    """Small macro-action planner for generation-time conditioning."""

    DEFAULT_SECTIONS = "INTRODUCE:8,VARY:8,DEVELOP:8,RETURN:6,CADENCE:2"

    def __init__(self, config: LatentGenerationConfig, action_to_id: Dict[str, int]) -> None:
        self.config = config
        self.action_to_id = action_to_id

    def action_id(self, bar_index: int, source_row: Dict[str, Any]) -> int:
        """Return the planned action id for one generated bar."""
        action = self.action_name(bar_index=bar_index, source_row=source_row)
        if action in self.action_to_id:
            return int(self.action_to_id[action])
        return int(self.action_to_id.get(self.config.default_action, self.action_to_id.get("UNKNOWN", 1)))

    def action_name(self, bar_index: int, source_row: Dict[str, Any]) -> str:
        """Return the planned action name for one generated bar."""
        mode = str(self.config.action_plan).lower()
        if mode == "source":
            return str(source_row.get("action") or self.config.default_action)
        if mode in {"form", "macro", "sections"}:
            return self._section_action(int(bar_index), self._section_plan())
        if mode == "cycle":
            sequence = [name for name, _length in self._section_plan()]
            return sequence[int(bar_index) % len(sequence)] if sequence else str(self.config.default_action)
        raise ValueError(f"Unsupported latent_generation.action_plan: {self.config.action_plan}")

    def diagnostics(self, target_bars: int) -> Dict[str, Any]:
        """Return JSON-safe action plan diagnostics."""
        planned = [self.action_name(index, {}) for index in range(max(0, int(target_bars)))]
        counts: Dict[str, int] = {}
        for action in planned:
            counts[action] = counts.get(action, 0) + 1
        return {
            "mode": str(self.config.action_plan),
            "sections": [{"action": action, "length": int(length)} for action, length in self._section_plan()],
            "planned_counts": counts,
        }

    def _section_action(self, bar_index: int, sections: Sequence[tuple[str, int]]) -> str:
        """Map bar index into section action ranges."""
        cursor = 0
        for action, length in sections:
            cursor += max(0, int(length))
            if bar_index < cursor:
                return str(action)
        return str(sections[-1][0]) if sections else str(self.config.default_action)

    def _section_plan(self) -> List[tuple[str, int]]:
        """Parse configured action sections."""
        text = str(self.config.action_sections or self.DEFAULT_SECTIONS)
        sections: List[tuple[str, int]] = []
        for raw in text.split(","):
            item = raw.strip()
            if not item:
                continue
            if ":" not in item:
                sections.append((item, 1))
                continue
            action, length = item.split(":", 1)
            sections.append((action.strip(), max(1, int(length.strip()))))
        return sections or [(str(self.config.default_action), 1)]


class SequenceTensorMidiRenderer:
    """Render a sequence of generated bar tensors into one MIDI file."""

    def __init__(self, config: DVAEMidiRenderConfig) -> None:
        self.config = config
        self.renderer = TensorMidiRenderer(config)

    def render(
        self,
        bars: np.ndarray,
        output_path: str | Path,
        base_pitch: int,
        base_pitches: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Render [bars, tracks, slots, features] tensors to MIDI."""
        import mido

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mid = mido.MidiFile(ticks_per_beat=int(self.config.ticks_per_beat))
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(int(self.config.tempo_bpm)), time=0))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        mid.tracks.append(meta)
        bar_ticks = int(round(float(self.config.bar_length_ql) * int(self.config.ticks_per_beat)))
        all_events: List[Dict[str, int]] = []
        for bar_index, tensor in enumerate(bars):
            current_base_pitch = int(base_pitch)
            if base_pitches is not None and bar_index < len(base_pitches):
                current_base_pitch = int(base_pitches[bar_index])
            all_events.extend(self.renderer._tensor_events(tensor, current_base_pitch, start_tick=int(bar_index * bar_ticks)))
        for track_index in range(3):
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name=f"track_{track_index}", time=0))
            events = [event for event in all_events if int(event["track"]) == int(track_index)]
            self.renderer._append_events(track, events)
            mid.tracks.append(track)
        mid.save(str(output))
        written_tempo_bpm = self._midi_tempo_bpm(output)
        midi_sha256 = self._file_sha256(output)
        expected_tempo_bpm = int(self.config.tempo_bpm)
        if written_tempo_bpm is not None and int(round(written_tempo_bpm)) != expected_tempo_bpm:
            raise RuntimeError(
                f"MIDI tempo mismatch: expected {expected_tempo_bpm} BPM, "
                f"but wrote {written_tempo_bpm:.3f} BPM to {output}."
            )
        if bool(getattr(self.config, "audio_quality_enabled", True)):
            audio_quality = self._run_audio_quality_subprocess(output, bars, int(base_pitch), base_pitches)
        else:
            audio_quality = {"enabled": False, "status": "skipped", "reason": "disabled_by_render_config"}
        pitches = [int(event["pitch"]) for event in all_events if event["type"] == "on"]
        return {
            "output_path": str(output),
            "tempo_bpm": expected_tempo_bpm,
            "written_tempo_bpm": float(written_tempo_bpm) if written_tempo_bpm is not None else None,
            "midi_sha256": midi_sha256,
            "note_count": int(len(pitches)),
            "min_pitch": int(min(pitches)) if pitches else None,
            "max_pitch": int(max(pitches)) if pitches else None,
            "base_pitch_mode": "per_bar" if base_pitches is not None else "fixed",
            "fallback_base_pitch": int(base_pitch),
            "audio_quality": audio_quality,
        }

    def _midi_tempo_bpm(self, midi_path: Path) -> Optional[float]:
        """Return the first MIDI tempo meta event as BPM."""
        import mido

        midi = mido.MidiFile(str(midi_path))
        for track in midi.tracks:
            for message in track:
                if message.type == "set_tempo":
                    return float(mido.tempo2bpm(message.tempo))
        return None

    def _file_sha256(self, path: Path) -> str:
        """Return SHA256 for a generated artifact."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _run_audio_quality_subprocess(
        self,
        midi_path: Path,
        bars: np.ndarray,
        fallback_base_pitch: int,
        base_pitches: Optional[Sequence[int]],
    ) -> Dict[str, Any]:
        """Run librosa-based analysis in a separate process to isolate OpenMP runtimes."""
        project_root = Path(__file__).resolve().parents[2]
        analyzer = project_root / "bin" / "analyze_generation_quality.py"
        if not analyzer.exists():
            return {"enabled": True, "status": "skipped", "reason": f"Missing analyzer CLI: {analyzer}"}

        tensor_path = midi_path.parent / f"{midi_path.stem}.audio_quality.input.npz"
        payload: Dict[str, Any] = {"bars": np.asarray(bars, dtype=np.float32)}
        if base_pitches is not None:
            payload["source_base_pitches"] = np.asarray(base_pitches, dtype=np.int64)
        np.savez_compressed(tensor_path, **payload)
        command = [
            sys.executable,
            str(analyzer),
            "--midi",
            str(midi_path),
            "--bar-tensors",
            str(tensor_path),
            "--base-pitch",
            str(int(fallback_base_pitch)),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            parsed = self._parse_audio_quality_stdout(completed.stdout)
            if parsed is not None:
                return parsed
            return {
                "enabled": True,
                "status": "ok",
                "subprocess_stdout": completed.stdout[-4000:],
                "subprocess_stderr": completed.stderr[-4000:],
            }
        except Exception as exc:
            stderr = ""
            stdout = ""
            if isinstance(exc, subprocess.CalledProcessError):
                stderr = exc.stderr or ""
                stdout = exc.stdout or ""
            parsed = self._parse_audio_quality_stdout(stdout)
            if parsed is not None:
                parsed.setdefault("subprocess_stderr", stderr[-4000:])
                return parsed
            return {
                "enabled": True,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "subprocess_stdout": stdout[-4000:],
                "subprocess_stderr": stderr[-4000:],
            }
        finally:
            try:
                tensor_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _parse_audio_quality_stdout(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Parse analyzer JSON even when FluidSynth writes banner text to stdout."""
        text = (stdout or "").strip()
        if not text:
            return None
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
