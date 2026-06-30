#!/usr/bin/env python3
"""Encoder layer facade.

The encoder owns method-independent bar features, global codebook construction,
and SymbolID vocabulary construction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.architecture import EncoderModel
from data.bar_density import TokenDensityAnalyzer
from common.config_loader import ConfigView
from data.core_data import ObservationVocab, SongRecord
from data.generation_data import CodebookCandidate, CodebookEntry
from encoder.vae_bar_encoder import (
    DenoisingVAEConfig,
    DenoisingVAETrainer,
    GLOBAL_FEATURE_NAMES,
    LatentClusterAnalyzer,
    LatentClusteringConfig,
    LatentFeatureBuilder,
)
from encoder.opening_seed import OpeningSeedSelector


@dataclass
class EncodingResult:
    """Output of the encoder layer used by downstream training."""

    observation_vocab: ObservationVocab
    encoder_model: EncoderModel
    global_codebook: Dict[int, CodebookEntry]
    diagnostics: Dict[str, Any]


class VAELatentSymbolEncoder:
    """Build SymbolID vocabulary from denoising VAE latent clusters."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.density = TokenDensityAnalyzer.from_style_config(config)
        self.trainer: Optional[DenoisingVAETrainer] = None

    def fit(self, songs: Sequence[SongRecord]) -> EncodingResult:
        bars = [bar for song in songs for bar in song.bars]
        vae_config = self._vae_config()
        cluster_config = self._cluster_config()
        trainer = DenoisingVAETrainer(vae_config).fit(bars)
        self.trainer = trainer
        latent_mu, latent_logvar = trainer.encode_distribution(bars)
        feature_builder = LatentFeatureBuilder(cluster_config)
        cluster_features = feature_builder.build(latent_mu, latent_logvar)
        clusterer = LatentClusterAnalyzer(cluster_config)
        labels = clusterer.cluster(cluster_features)
        labels = self._compact_labels(labels)
        for bar, label in zip(bars, labels):
            bar.codebook_id = int(label)
            bar.kmeans_id = None
            bar.composite_key = f"V{int(label)}"
            bar.observation_id = int(label)
        vocab = self._build_vocab(labels)
        global_codebook = self._build_codebook(bars, latent_mu, cluster_features, labels)
        encoder_model = EncoderModel.from_legacy(
            global_codebook,
            vocab,
            metadata={
                "encoder": self.__class__.__name__,
                "backend": "vae_latent",
                "symbol_id_field": "observation_id",
                "codebook_id_field": "codebook_id",
                "symbol_policy": "observation_id_equals_latent_cluster_id",
                "vae_config": asdict(vae_config),
                "global_feature_names": list(GLOBAL_FEATURE_NAMES),
                "global_feature_dim": int(vae_config.global_feature_dim()),
            },
        )
        opening_seed = self._opening_seed_diagnostics(global_codebook, encoder_model)
        opening_pipelines = self._opening_theme_pipeline_diagnostics(songs, opening_seed)
        encoder_model.metadata["opening_seed_candidates"] = opening_seed
        encoder_model.metadata["opening_theme_pipelines"] = opening_pipelines
        reconstruction = trainer.evaluate_reconstruction(bars)
        clustering = clusterer.diagnostics(bars, cluster_features, labels)
        diagnostics = {
            "backend": "vae_latent",
            "vae_config": asdict(vae_config),
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
            "global_feature_dim": int(vae_config.global_feature_dim()),
            "cluster_config": asdict(cluster_config),
            "latent_features": feature_builder.diagnostics(latent_mu, latent_logvar, cluster_features),
            "training_log": trainer.training_log,
            "reconstruction": reconstruction,
            "clustering": clustering,
            "opening_seed_candidates": opening_seed,
            "opening_theme_pipelines": opening_pipelines,
            "observation_vocab": self._vocab_diagnostics(vocab, bars),
        }
        return EncodingResult(
            observation_vocab=vocab,
            encoder_model=encoder_model,
            global_codebook=global_codebook,
            diagnostics=diagnostics,
        )

    def save_artifacts(self, model_dir: str | Path) -> Dict[str, Any]:
        if self.trainer is None:
            return {"saved": False, "reason": "encoder_not_fitted"}
        artifact_path = Path(model_dir) / "vae_decoder.pt"
        self.trainer.save(artifact_path)
        return {
            "saved": True,
            "vae_decoder_path": str(artifact_path),
            "artifact": "vae_decoder.pt",
        }

    def _vae_config(self) -> DenoisingVAEConfig:
        section = ConfigView(self.config).section("vae_encoder")
        return DenoisingVAEConfig(
            steps_per_bar=int(section.get("steps_per_bar", 16)),
            input_mode=str(section.get("input_mode", "multi_channel")),
            hidden_dim=int(section.get("hidden_dim", 64)),
            latent_dim=int(section.get("latent_dim", 12)),
            epochs=int(section.get("epochs", 80)),
            batch_size=int(section.get("batch_size", 128)),
            learning_rate=float(section.get("learning_rate", 0.001)),
            beta_kl=float(section.get("beta_kl", 0.001)),
            kl_warmup_epochs=int(section.get("kl_warmup_epochs", 10)),
            pitch_weight=float(section.get("pitch_weight", 1.0)),
            onset_weight=float(section.get("onset_weight", 0.5)),
            sustain_weight=float(section.get("sustain_weight", 0.5)),
            global_weight=float(section.get("global_weight", 0.25)),
            note_drop_prob=float(section.get("note_drop_prob", 0.15)),
            sustain_fill_prob=float(section.get("sustain_fill_prob", 0.10)),
            drop_to_rest_prob=float(section.get("drop_to_rest_prob", 0.5)),
            ornament_pitch_radius=int(section.get("ornament_pitch_radius", 2)),
            pitch_scale=float(section.get("pitch_scale", 24.0)),
            condition_on_previous_last_pitch=bool(section.get("condition_on_previous_last_pitch", True)),
            previous_last_pitch_scale=float(section.get("previous_last_pitch_scale", section.get("pitch_scale", 24.0))),
            decoder_head_architecture=str(section.get("decoder_head_architecture", "split_rhythm_pitch")),
            pitch_activation=str(section.get("pitch_activation", "hardtanh")),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
        )

    def _cluster_config(self) -> LatentClusteringConfig:
        section = ConfigView(self.config).section("vae_encoder")
        clustering = section.get("clustering", {})
        clustering = clustering if isinstance(clustering, dict) else {}
        return LatentClusteringConfig(
            method=str(clustering.get("method", "kmeans")),
            feature_mode=str(clustering.get("feature_mode", "mu")),
            logvar_weight=float(clustering.get("logvar_weight", 0.25)),
            n_clusters=int(clustering.get("n_clusters", 192)),
            distance_threshold=float(clustering.get("distance_threshold", 1.0)),
            linkage_method=str(clustering.get("linkage_method", "average")),
            covariance_type=str(clustering.get("covariance_type", "full")),
            reg_covar=float(clustering.get("reg_covar", 1e-6)),
            max_iter=int(clustering.get("max_iter", 200)),
            random_seed=int(clustering.get("random_seed", section.get("random_seed", 42))),
        )

    def _opening_seed_diagnostics(
        self,
        global_codebook: Dict[int, CodebookEntry],
        encoder_model: EncoderModel,
    ) -> Dict[str, Any]:
        decoder_config = ConfigView(self.config).section("decoder")
        selector = OpeningSeedSelector.from_config_dict(decoder_config.get("opening_seed", {}))
        return selector.analyze_codebook(global_codebook, encoder_model)

    def _opening_theme_pipeline_diagnostics(
        self,
        songs: Sequence[SongRecord],
        opening_seed: Dict[str, Any],
    ) -> Dict[str, Any]:
        decoder_config = ConfigView(self.config).section("decoder")
        selector = OpeningSeedSelector.from_config_dict(decoder_config.get("opening_seed", {}))
        return selector.analyze_theme_pipelines(songs, opening_seed)

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
        content_latents: np.ndarray,
        cluster_features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, CodebookEntry]:
        bars_by_label: Dict[int, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            bars_by_label[int(label)].append(int(index))
        entries: Dict[int, CodebookEntry] = {}
        for label, indices in sorted(bars_by_label.items()):
            medoid_index = self._latent_medoid(cluster_features, indices)
            representative = bars[medoid_index]
            relative_tokens = representative.tokens_for_encoder("relative")
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
                    self._candidate_for_bar(bars[index], content_latents[index])
                    for index in indices
                ],
                latent_vector=[float(value) for value in np.asarray(content_latents[medoid_index], dtype=float).tolist()],
                position_ratio=self._position_ratio(representative),
            )
        return entries

    def _candidate_for_bar(self, bar: Any, latent: Optional[np.ndarray] = None) -> CodebookCandidate:
        relative_tokens = bar.tokens_for_encoder("relative")
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
            latent_vector=(
                [float(value) for value in np.asarray(latent, dtype=float).tolist()]
                if latent is not None
                else None
            ),
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
        backend = str(section.get("backend", "vae_latent"))
        if backend == "vae_latent":
            return VAELatentSymbolEncoder(config)
        raise ValueError("encoder.backend must be 'vae_latent'.")
