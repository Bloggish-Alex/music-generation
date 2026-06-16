#!/usr/bin/env python3
"""Train and evaluate a denoising VAE bar encoder as an isolated experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.config_loader import ConfigLoader
from data.core_data import BarRecord, SongRecord
from diagnostics.encoder_input_loader import EncoderInputLoader
from diagnostics.markdown_report import MarkdownReport
from encoder.vae_bar_encoder import (
    DenoisingVAEConfig,
    DenoisingVAETrainer,
    LatentClusterAnalyzer,
    LatentClusteringConfig,
    LatentFeatureBuilder,
)


class VAEEncoderReport:
    """Render markdown diagnostics for the denoising VAE encoder."""

    def __init__(
        self,
        songs: Sequence[SongRecord],
        vae_config: DenoisingVAEConfig,
        cluster_config: LatentClusteringConfig,
        training_log: Sequence[Dict[str, float]],
        reconstruction: Dict[str, Any],
        clustering: Dict[str, Any],
    ) -> None:
        self.songs = list(songs)
        self.bars = [bar for song in songs for bar in song.bars]
        self.vae_config = vae_config
        self.cluster_config = cluster_config
        self.training_log = list(training_log)
        self.reconstruction = reconstruction
        self.clustering = clustering

    def write(self, output_path: Path) -> None:
        report = MarkdownReport()
        report.heading("Denoising VAE Encoder Report")
        report.paragraph(
            "This report evaluates a denoising VAE encoder independently from the main training pipeline. "
            "The decoder is trained with multi-head loss: slot type classification, note-on pitch regression, "
            "onset reconstruction, sustain reconstruction, global bar-feature reconstruction, and KL regularization. "
            "Clustering uses z_mu latent vectors."
        )
        report.table([
            "Metric",
            "Value",
        ], [
            ["song_count", len(self.songs)],
            ["bar_count", len(self.bars)],
            ["latent_dim", self.vae_config.latent_dim],
            ["hidden_dim", self.vae_config.hidden_dim],
            ["epochs", self.vae_config.epochs],
            ["cluster_method", self.cluster_config.method],
            ["n_clusters", self.cluster_config.n_clusters],
        ])

        report.heading("VAE Config", 2)
        report.table(["Parameter", "Value"], [[key, value] for key, value in asdict(self.vae_config).items()])
        report.heading("Latent Clustering Config", 2)
        report.table(["Parameter", "Value"], [[key, value] for key, value in asdict(self.cluster_config).items()])

        cluster_model = self.clustering.get("cluster_model", {})
        if cluster_model:
            report.heading("Latent Clustering Model", 2)
            report.table(["Metric", "Value"], [[key, value] for key, value in cluster_model.items()])

        latent_features = self.clustering.get("latent_features", {})
        if latent_features:
            report.heading("Latent Clustering Features", 2)
            report.table(["Metric", "Value"], [[key, value] for key, value in latent_features.items()])

        report.heading("Training Loss", 2)
        report.table([
            "Epoch",
            "Loss",
            "Type loss",
            "Pitch loss",
            "Onset loss",
            "Sustain loss",
            "Global loss",
            "KL loss",
            "KL weight",
        ], [
            [
                int(item["epoch"]),
                item["loss"],
                item["type_loss"],
                item["pitch_loss"],
                item.get("onset_loss", 0.0),
                item.get("sustain_loss", 0.0),
                item.get("global_loss", 0.0),
                item["kl_loss"],
                item["kl_weight"],
            ]
            for item in self._sampled_training_log()
        ])

        report.heading("Reconstruction / Multi-head Metrics", 2)
        report.table(["Metric", "Value"], [[key, value] for key, value in self.reconstruction.items()])

        report.heading("Latent Cluster Distribution", 2)
        report.table([
            "Metric",
            "Value",
            "Meaning",
        ], [
            ["total_assignments", self.clustering.get("total_assignments"), "Number of bars assigned to latent clusters."],
            ["used_label_count", self.clustering.get("used_label_count"), "Number of non-empty latent labels."],
            ["singleton_label_count", self.clustering.get("singleton_label_count"), "Labels with one bar only."],
            ["singleton_ratio", self.clustering.get("singleton_ratio"), "Lower is better for reusable symbol memory."],
            ["max_label", self.clustering.get("max_label"), "Largest latent cluster ID."],
            ["max_label_count", self.clustering.get("max_label_count"), "Bars in largest latent cluster."],
            ["max_label_ratio", self.clustering.get("max_label_ratio"), "Too high can indicate cluster collapse."],
            ["normalized_entropy", self.clustering.get("normalized_entropy"), "Higher means more even label usage."],
            ["effective_label_count", self.clustering.get("effective_label_count"), "Entropy-equivalent number of used labels."],
        ])

        assignment_confidence = self.clustering.get("assignment_confidence", {})
        if assignment_confidence:
            report.heading("GMM Assignment Confidence", 2)
            report.table(["Metric", "Value"], [[key, value] for key, value in assignment_confidence.items()])

        label_confidence = self.clustering.get("label_confidence", [])
        if label_confidence:
            report.heading("Top Label Confidence", 2)
            report.table([
                "Label",
                "Count",
                "Confidence mean",
                "Confidence min",
                "Low confidence < 0.8",
                "Entropy mean",
            ], [
                [
                    item.get("label"),
                    item.get("count"),
                    item.get("confidence_mean"),
                    item.get("confidence_min"),
                    item.get("low_confidence_ratio_lt_0_80"),
                    item.get("entropy_mean"),
                ]
                for item in label_confidence
            ])

        latent = self.clustering.get("latent", {})
        report.heading("Latent Space Summary", 2)
        report.table([
            "Metric",
            "Value",
        ], [
            ["shape", latent.get("shape")],
            ["mean_abs", latent.get("mean_abs")],
            ["std", latent.get("std")],
            ["dim_std", latent.get("dim_std")],
        ])

        report.heading("Top Latent Labels", 2)
        report.table([
            "Label",
            "Count",
            "Ratio",
            "Confidence mean",
            "Confidence min",
            "Low confidence < 0.8",
        ], [
            [
                item["label"],
                item["count"],
                item["ratio"],
                item.get("confidence_mean"),
                item.get("confidence_min"),
                item.get("low_confidence_ratio_lt_0_80"),
            ]
            for item in self.clustering.get("top_labels", [])
        ])

        report.heading("Top Label Examples", 2)
        for item in self.clustering.get("top_label_examples", []):
            report.paragraph(f"Label `{item['label']}` count={item['count']}")
            report.table([
                "Song",
                "Bar",
                "Relative tokens",
                "Variance",
                "Sharing",
                "Confidence",
                "Entropy",
            ], [
                [
                    example["song_id"],
                    example["bar_index"],
                    example["relative_tokens"],
                    example["token_variance"],
                    example["sharing_score"],
                    example.get("assignment_confidence"),
                    example.get("assignment_entropy"),
                ]
                for example in item.get("examples", [])
            ])

        report.heading("Latent Nearest Neighbors", 2)
        for item in self.clustering.get("nearest_neighbors", []):
            anchor = item["anchor"]
            report.paragraph(
                f"Anchor `{anchor['song_id']}` bar `{anchor['bar_index']}` tokens `{anchor['relative_tokens']}`"
            )
            report.table([
                "Song",
                "Bar",
                "Distance",
                "Relative tokens",
            ], [
                [
                    neighbor["song_id"],
                    neighbor["bar_index"],
                    neighbor["latent_distance"],
                    neighbor["relative_tokens"],
                ]
                for neighbor in item.get("neighbors", [])
            ])
        report.heading("Initial Interpretation", 2)
        report.table([
            "Signal",
            "How to read it",
        ], [
            ["singleton_ratio lower than edit-distance baseline", "VAE latent is producing a more reusable vocabulary."],
            ["type_accuracy high but singleton_ratio high", "Model reconstructs slot types but latent may still memorize surface patterns."],
            ["pitch_mse low but max_label_ratio high", "Pitch structure may be over-smoothed into dominant clusters."],
            ["latent dim_std near zero", "Possible posterior collapse; reduce beta_kl or use longer KL warmup."],
            ["nearest neighbors musically unrelated", "Representation is not yet meaningful even if distribution metrics look good."],
        ])
        report.write(output_path)

    def _sampled_training_log(self) -> List[Dict[str, float]]:
        if len(self.training_log) <= 20:
            return list(self.training_log)
        keep = set(np.linspace(0, len(self.training_log) - 1, num=20, dtype=int).tolist())
        return [item for index, item in enumerate(self.training_log) if index in keep]


class VAEEncoderCLI:
    """CLI entrypoint."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train and analyze a denoising VAE bar encoder.")
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--model-dir", type=Path)
        source.add_argument("--model-bundle", type=Path)
        source.add_argument("--songs-json", type=Path)
        source.add_argument("--music-dir", type=Path)
        parser.add_argument("--config", type=Path, default=None, help="Optional style config override. VAE defaults are read from the merged style config.")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--latents-output", type=Path, default=None)
        parser.add_argument("--checkpoint-output", type=Path, default=None)

        parser.add_argument("--steps-per-bar", type=int, default=None)
        parser.add_argument("--input-mode", choices=["token", "multi_channel"], default=None)
        parser.add_argument("--hidden-dim", type=int, default=None)
        parser.add_argument("--latent-dim", type=int, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--beta-kl", type=float, default=None)
        parser.add_argument("--kl-warmup-epochs", type=int, default=None)
        parser.add_argument("--pitch-weight", type=float, default=None)
        parser.add_argument("--onset-weight", type=float, default=None)
        parser.add_argument("--sustain-weight", type=float, default=None)
        parser.add_argument("--global-weight", type=float, default=None)
        parser.add_argument("--note-drop-prob", type=float, default=None)
        parser.add_argument("--sustain-fill-prob", type=float, default=None)
        parser.add_argument("--drop-to-rest-prob", type=float, default=None)
        parser.add_argument("--ornament-pitch-radius", type=int, default=None)
        parser.add_argument("--pitch-scale", type=float, default=None)
        parser.add_argument("--random-seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)

        parser.add_argument("--cluster-method", choices=["kmeans", "gmm", "agglomerative"], default=None)
        parser.add_argument("--feature-mode", choices=["mu", "mu_logvar"], default=None)
        parser.add_argument("--logvar-weight", type=float, default=None)
        parser.add_argument("--n-clusters", type=int, default=None)
        parser.add_argument("--distance-threshold", type=float, default=None)
        parser.add_argument("--linkage-method", type=str, default=None)
        parser.add_argument("--covariance-type", choices=["full", "tied", "diag", "spherical"], default=None)
        parser.add_argument("--reg-covar", type=float, default=None)
        parser.add_argument("--gmm-max-iter", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        songs = EncoderInputLoader(config).load(
            model_dir=args.model_dir,
            model_bundle=args.model_bundle,
            songs_json=args.songs_json,
            music_dir=args.music_dir,
        )
        bars = [bar for song in songs for bar in song.bars]
        vae_config = self._vae_config(args, config)
        cluster_config = self._cluster_config(args, config)
        trainer = DenoisingVAETrainer(vae_config).fit(bars)
        latent_mu, latent_logvar = trainer.encode_distribution(bars)
        feature_builder = LatentFeatureBuilder(cluster_config)
        cluster_features = feature_builder.build(latent_mu, latent_logvar)
        clusterer = LatentClusterAnalyzer(cluster_config)
        labels = clusterer.cluster(cluster_features)
        reconstruction = trainer.evaluate_reconstruction(bars)
        clustering = clusterer.diagnostics(bars, cluster_features, labels)
        clustering["latent_features"] = feature_builder.diagnostics(latent_mu, latent_logvar, cluster_features)

        VAEEncoderReport(
            songs=songs,
            vae_config=vae_config,
            cluster_config=cluster_config,
            training_log=trainer.training_log,
            reconstruction=reconstruction,
            clustering=clustering,
        ).write(args.output)

        diagnostics = {
            "vae_config": asdict(vae_config),
            "cluster_config": asdict(cluster_config),
            "training_log": trainer.training_log,
            "reconstruction": reconstruction,
            "clustering": clustering,
        }
        if args.diagnostics_output:
            args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics_output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        if args.latents_output:
            args.latents_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.latents_output,
                latents=latent_mu,
                latent_mu=latent_mu,
                latent_logvar=latent_logvar,
                cluster_features=cluster_features,
                labels=labels,
                song_id=np.asarray([bar.song_id for bar in bars]),
                bar_index=np.asarray([int(bar.bar_index) for bar in bars], dtype=np.int64),
            )
        if args.checkpoint_output:
            trainer.save(args.checkpoint_output)
        print(f"VAE encoder report -> {args.output}")

    def _vae_config(self, args: argparse.Namespace, config: Dict[str, Any]) -> DenoisingVAEConfig:
        section = self._section(config, "vae_encoder")
        return DenoisingVAEConfig(
            steps_per_bar=int(self._config_value(args, section, "steps_per_bar", 16)),
            input_mode=str(self._config_value(args, section, "input_mode", "multi_channel")),
            hidden_dim=int(self._config_value(args, section, "hidden_dim", 64)),
            latent_dim=int(self._config_value(args, section, "latent_dim", 12)),
            epochs=int(self._config_value(args, section, "epochs", 80)),
            batch_size=int(self._config_value(args, section, "batch_size", 128)),
            learning_rate=float(self._config_value(args, section, "learning_rate", 0.001)),
            beta_kl=float(self._config_value(args, section, "beta_kl", 0.001)),
            kl_warmup_epochs=int(self._config_value(args, section, "kl_warmup_epochs", 10)),
            pitch_weight=float(self._config_value(args, section, "pitch_weight", 1.0)),
            onset_weight=float(self._config_value(args, section, "onset_weight", 0.5)),
            sustain_weight=float(self._config_value(args, section, "sustain_weight", 0.5)),
            global_weight=float(self._config_value(args, section, "global_weight", 0.25)),
            note_drop_prob=float(self._config_value(args, section, "note_drop_prob", 0.15)),
            sustain_fill_prob=float(self._config_value(args, section, "sustain_fill_prob", 0.10)),
            drop_to_rest_prob=float(self._config_value(args, section, "drop_to_rest_prob", 0.5)),
            ornament_pitch_radius=int(self._config_value(args, section, "ornament_pitch_radius", 2)),
            pitch_scale=float(self._config_value(args, section, "pitch_scale", 24.0)),
            random_seed=int(self._config_value(args, section, "random_seed", 42)),
            device=str(self._config_value(args, section, "device", "cpu")),
        )

    def _cluster_config(self, args: argparse.Namespace, config: Dict[str, Any]) -> LatentClusteringConfig:
        section = self._section(config, "vae_encoder")
        clustering = section.get("clustering", {})
        clustering = clustering if isinstance(clustering, dict) else {}
        random_seed = self._config_value(args, section, "random_seed", 42)
        return LatentClusteringConfig(
            method=str(self._config_value(args, clustering, "method", "kmeans", "cluster_method")),
            feature_mode=str(self._config_value(args, clustering, "feature_mode", "mu")),
            logvar_weight=float(self._config_value(args, clustering, "logvar_weight", 0.25)),
            n_clusters=int(self._config_value(args, clustering, "n_clusters", 384)),
            distance_threshold=float(self._config_value(args, clustering, "distance_threshold", 1.0)),
            linkage_method=str(self._config_value(args, clustering, "linkage_method", "average")),
            covariance_type=str(self._config_value(args, clustering, "covariance_type", "full")),
            reg_covar=float(self._config_value(args, clustering, "reg_covar", 1e-6)),
            max_iter=int(self._config_value(args, clustering, "max_iter", 200, "gmm_max_iter")),
            random_seed=int(self._config_value(args, clustering, "random_seed", random_seed)),
        )

    def _section(self, config: Dict[str, Any], name: str) -> Dict[str, Any]:
        value = config.get(name, {})
        return value if isinstance(value, dict) else {}

    def _config_value(
        self,
        args: argparse.Namespace,
        section: Dict[str, Any],
        config_key: str,
        default: Any,
        arg_name: Optional[str] = None,
    ) -> Any:
        cli_value = getattr(args, arg_name or config_key, None)
        if cli_value is not None:
            return cli_value
        return section.get(config_key, default)


def main() -> None:
    VAEEncoderCLI().run()


if __name__ == "__main__":
    main()
