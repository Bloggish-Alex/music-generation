#!/usr/bin/env python3
"""Encoder layer facade.

The encoder owns method-independent bar features, global codebook construction,
and SymbolID vocabulary construction. Current implementation delegates to the
legacy bar clustering pipeline while exposing the stable architecture objects.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from architecture import EncoderModel
from bar_clustering import BarClusteringPipeline
from bar_density import TokenDensityAnalyzer
from config_loader import ConfigView
from core_data import ObservationVocab, SongRecord
from generation_data import CodebookCandidate, CodebookEntry
from vae_bar_encoder import DenoisingVAEConfig, DenoisingVAETrainer, LatentClusterAnalyzer, LatentClusteringConfig


@dataclass
class EncodingResult:
    """Output of the encoder layer used by downstream training."""

    observation_vocab: ObservationVocab
    encoder_model: EncoderModel
    global_codebook: Dict[int, CodebookEntry]
    diagnostics: Dict[str, Any]


class LegacySymbolEncoder:
    """Adapter around the current clustering implementation.

    This class is the boundary that future encoder experiments should replace
    or extend. Decoder and renderer layers should not depend on
    BarClusteringPipeline directly.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def fit(self, songs: Sequence[SongRecord]) -> EncodingResult:
        pipeline = BarClusteringPipeline(self.config)
        result = pipeline.run(songs)
        vocab = result.observation_vocab
        encoder_model = EncoderModel.from_legacy(
            result.global_codebook,
            vocab,
            metadata={
                "encoder": self.__class__.__name__,
                "symbol_id_field": "observation_id",
                "codebook_id_field": "codebook_id",
            },
        )
        return EncodingResult(
            observation_vocab=vocab,
            encoder_model=encoder_model,
            global_codebook=result.global_codebook,
            diagnostics=result.diagnostics,
        )


class VAELatentSymbolEncoder:
    """Build SymbolID vocabulary from denoising VAE latent clusters."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.density = TokenDensityAnalyzer.from_style_config(config)

    def fit(self, songs: Sequence[SongRecord]) -> EncodingResult:
        bars = [bar for song in songs for bar in song.bars]
        vae_config = self._vae_config()
        cluster_config = self._cluster_config()
        trainer = DenoisingVAETrainer(vae_config).fit(bars)
        latents = trainer.encode(bars)
        labels = LatentClusterAnalyzer(cluster_config).cluster(latents)
        labels = self._compact_labels(labels)
        for bar, label in zip(bars, labels):
            bar.codebook_id = int(label)
            bar.kmeans_id = None
            bar.composite_key = f"V{int(label)}"
            bar.observation_id = int(label)
        vocab = self._build_vocab(labels)
        global_codebook = self._build_codebook(bars, latents, labels)
        reconstruction = trainer.evaluate_reconstruction(bars)
        clustering = LatentClusterAnalyzer(cluster_config).diagnostics(bars, latents, labels)
        diagnostics = {
            "backend": "vae_latent",
            "vae_config": asdict(vae_config),
            "cluster_config": asdict(cluster_config),
            "training_log": trainer.training_log,
            "reconstruction": reconstruction,
            "clustering": clustering,
            "observation_vocab": self._vocab_diagnostics(vocab, bars),
        }
        encoder_model = EncoderModel.from_legacy(
            global_codebook,
            vocab,
            metadata={
                "encoder": self.__class__.__name__,
                "backend": "vae_latent",
                "symbol_id_field": "observation_id",
                "codebook_id_field": "codebook_id",
                "symbol_policy": "observation_id_equals_latent_cluster_id",
            },
        )
        return EncodingResult(
            observation_vocab=vocab,
            encoder_model=encoder_model,
            global_codebook=global_codebook,
            diagnostics=diagnostics,
        )

    def _vae_config(self) -> DenoisingVAEConfig:
        section = ConfigView(self.config).section("vae_encoder")
        return DenoisingVAEConfig(
            steps_per_bar=int(section.get("steps_per_bar", 16)),
            hidden_dim=int(section.get("hidden_dim", 32)),
            latent_dim=int(section.get("latent_dim", 8)),
            epochs=int(section.get("epochs", 80)),
            batch_size=int(section.get("batch_size", 128)),
            learning_rate=float(section.get("learning_rate", 0.001)),
            beta_kl=float(section.get("beta_kl", 0.001)),
            kl_warmup_epochs=int(section.get("kl_warmup_epochs", 10)),
            pitch_weight=float(section.get("pitch_weight", 1.0)),
            note_drop_prob=float(section.get("note_drop_prob", 0.15)),
            sustain_fill_prob=float(section.get("sustain_fill_prob", 0.10)),
            drop_to_rest_prob=float(section.get("drop_to_rest_prob", 0.5)),
            ornament_pitch_radius=int(section.get("ornament_pitch_radius", 2)),
            pitch_scale=float(section.get("pitch_scale", 24.0)),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
        )

    def _cluster_config(self) -> LatentClusteringConfig:
        section = ConfigView(self.config).section("vae_encoder")
        clustering = section.get("clustering", {})
        clustering = clustering if isinstance(clustering, dict) else {}
        return LatentClusteringConfig(
            method=str(clustering.get("method", "kmeans")),
            n_clusters=int(clustering.get("n_clusters", 192)),
            distance_threshold=float(clustering.get("distance_threshold", 1.0)),
            linkage_method=str(clustering.get("linkage_method", "average")),
            random_seed=int(clustering.get("random_seed", section.get("random_seed", 42))),
        )

    def _compact_labels(self, labels: Sequence[int]) -> np.ndarray:
        mapping: Dict[int, int] = {}
        compact: List[int] = []
        for label in labels:
            value = int(label)
            if value not in mapping:
                mapping[value] = len(mapping)
            compact.append(mapping[value])
        return np.asarray(compact, dtype=int)

    def _build_vocab(self, labels: Sequence[int]) -> ObservationVocab:
        unique_labels = sorted(set(int(label) for label in labels))
        composite_to_observation = {f"V{label}": int(label) for label in unique_labels}
        observation_to_composite = {int(label): f"V{label}" for label in unique_labels}
        composite_parts = {
            f"V{label}": {
                "codebook_id": int(label),
                "kmeans_id": None,
                "latent_cluster_id": int(label),
                "encoder_backend": "vae_latent",
            }
            for label in unique_labels
        }
        return ObservationVocab(composite_to_observation, observation_to_composite, composite_parts)

    def _build_codebook(
        self,
        bars: Sequence[Any],
        latents: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, CodebookEntry]:
        bars_by_label: Dict[int, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            bars_by_label[int(label)].append(int(index))
        entries: Dict[int, CodebookEntry] = {}
        for label, indices in sorted(bars_by_label.items()):
            medoid_index = self._latent_medoid(latents, indices)
            representative = bars[medoid_index]
            relative_tokens = representative.tokens_for_edit_distance("relative")
            entries[int(label)] = CodebookEntry(
                codebook_id=int(label),
                source_song=representative.song_id,
                source_file=representative.file_path,
                source_bar_index=int(representative.bar_index),
                relative_tokens=relative_tokens,
                absolute_tokens=list(representative.absolute_tokens),
                density=self.density.analyze(relative_tokens),
                token_variance=float(representative.token_variance),
                sharing_score=float(representative.sharing_score),
                candidates=[
                    self._candidate_for_bar(bars[index])
                    for index in indices
                ],
            )
        return entries

    def _candidate_for_bar(self, bar: Any) -> CodebookCandidate:
        relative_tokens = bar.tokens_for_edit_distance("relative")
        return CodebookCandidate(
            source_song=bar.song_id,
            source_file=bar.file_path,
            source_bar_index=int(bar.bar_index),
            relative_tokens=relative_tokens,
            absolute_tokens=list(bar.absolute_tokens),
            density=self.density.analyze(relative_tokens),
            token_variance=float(bar.token_variance),
            sharing_score=float(bar.sharing_score),
            kmeans_id=None,
            observation_id=int(bar.observation_id) if bar.observation_id is not None else None,
            position_ratio=self._position_ratio(bar),
        )

    def _latent_medoid(self, latents: np.ndarray, indices: Sequence[int]) -> int:
        if len(indices) == 1:
            return int(indices[0])
        cluster = latents[np.asarray(indices, dtype=int)]
        center = np.mean(cluster, axis=0)
        distances = np.linalg.norm(cluster - center, axis=1)
        return int(indices[int(np.argmin(distances))])

    def _position_ratio(self, bar: Any) -> float:
        if bar.source_bar_count is None or int(bar.source_bar_count) <= 1:
            return 0.0
        return float(int(bar.bar_index) / max(1, int(bar.source_bar_count) - 1))

    def _vocab_diagnostics(self, vocab: ObservationVocab, bars: Sequence[Any]) -> Dict[str, Any]:
        counts = Counter(int(bar.observation_id) for bar in bars if bar.observation_id is not None)
        return {
            "config": {
                "strategy": "vae_latent",
                "position_conditioning": False,
                "kmeans_conditioning": False,
            },
            "observation_count": len(vocab.composite_to_observation),
            "base_composite_count": len(vocab.composite_to_observation),
            "observation_expansion_ratio": 1.0,
            "observation_counts": {str(key): int(value) for key, value in sorted(counts.items())},
            "rare_observation_count": sum(1 for value in counts.values() if int(value) == 1),
            "vocab": vocab.to_dict(),
        }


class SymbolEncoderFactory:
    """Create the configured encoder backend."""

    def from_config(self, config: Dict[str, Any]) -> Any:
        section = ConfigView(config).section("encoder")
        backend = str(section.get("backend", "legacy"))
        if backend == "legacy":
            return LegacySymbolEncoder(config)
        if backend == "vae_latent":
            return VAELatentSymbolEncoder(config)
        raise ValueError("encoder.backend must be 'legacy' or 'vae_latent'.")
