#!/usr/bin/env python3
"""Orchestrate the standard training stages for the MDN Transformer experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common.config_loader import ConfigView
from model.base_pitch_motion import BasePitchMotionConfig
from model.hybrid_miditok_retrieval import HybridMidiTokRetrievalConfig
from model.miditok_bar_sequence_encoder import MidiTokBarSequenceEncoderConfig
from pipeline.base_pitch_motion_pipeline import BasePitchMotionTrainingConfig, BasePitchMotionTrainingPipeline
from pipeline.dvae_training_pipeline import DVAETrainingPipeline
from pipeline.encoding_pipeline import EncodingPipeline
from pipeline.hybrid_miditok_retrieval_pipeline import (
    HybridMidiTokTrainingConfig,
    HybridMidiTokTrainingPipeline,
)
from pipeline.latent_dataset_exporter import LatentDatasetExporter, LatentExportConfig
from pipeline.miditok_bar_sequence_training_pipeline import MidiTokBarSequenceTrainingPipeline
from pipeline.remi_motion_pipeline import RemiMotionTrainingPipeline


STANDARD_STAGES = ("dvae", "latent", "miditok", "retrieval", "base_pitch", "remi_motion")


@dataclass(frozen=True)
class TrainingOrchestratorConfig:
    """High-level training orchestration controls."""

    stage: str = "dvae"
    device: Optional[str] = None
    max_rows: Optional[int] = None
    dvae_epochs: Optional[int] = None
    dvae_learning_rate: Optional[float] = None
    miditok_epochs: Optional[int] = None
    retrieval_epochs: Optional[int] = None
    base_pitch_epochs: Optional[int] = None
    remi_motion_epochs: Optional[int] = None
    remi_motion_batch_size: Optional[int] = None
    batch_size: Optional[int] = None
    random_seed: Optional[int] = None
    transpose_semitones: Optional[str] = None
    save_z: bool = False


@dataclass
class StageRecord:
    """One stage execution record."""

    name: str
    status: str
    outputs: Dict[str, Any]


@dataclass
class TrainingOrchestratorResult:
    """Result produced by the high-level training orchestrator."""

    model_dir: Path
    manifest_path: Path
    stages: List[StageRecord]


class TrainingOrchestrator:
    """Run stable training stages behind one model-dir contract."""

    def __init__(self, config: Dict[str, Any], options: TrainingOrchestratorConfig) -> None:
        self.config = config
        self.options = options

    def run(self, music_dir: str | Path, model_dir: str | Path) -> TrainingOrchestratorResult:
        """Run the requested training stages."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        stages = self._resolve_stages(self.options.stage)
        records: List[StageRecord] = []
        for stage in stages:
            if stage == "encode":
                records.append(self._run_encode(music_dir, model_path))
            elif stage == "dvae":
                records.append(self._run_dvae(music_dir, model_path))
            elif stage == "latent":
                records.append(self._run_latent(model_path))
            elif stage == "miditok":
                records.append(self._run_miditok(model_path))
            elif stage == "retrieval":
                records.append(self._run_retrieval(model_path))
            elif stage == "base_pitch":
                records.append(self._run_base_pitch(model_path))
            elif stage == "remi_motion":
                records.append(self._run_remi_motion(model_path))
            else:
                raise ValueError(f"Unsupported training stage: {stage}")
        manifest_path = self._write_manifest(model_path, music_dir, records)
        return TrainingOrchestratorResult(model_dir=model_path, manifest_path=manifest_path, stages=records)

    def _resolve_stages(self, stage: str) -> List[str]:
        """Resolve stage aliases to concrete stage names."""
        value = str(stage).strip().lower().replace("-", "_")
        if value == "all":
            return list(STANDARD_STAGES)
        if value in {"remi_direct", "direct_remi"}:
            return ["dvae", "latent", "remi_motion"]
        if value == "sequence":
            return ["miditok", "retrieval", "base_pitch", "remi_motion"]
        if value == "encoder":
            return ["dvae", "latent", "miditok"]
        if value == "hybrid":
            return ["retrieval", "base_pitch", "remi_motion"]
        if value == "best":
            return list(STANDARD_STAGES)
        if value in {"encode", *STANDARD_STAGES}:
            return [value]
        values = [item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip()]
        if values:
            unsupported = [item for item in values if item not in {"encode", *STANDARD_STAGES}]
            if unsupported:
                raise ValueError(f"Unsupported training stages: {unsupported}")
            return values
        raise ValueError(f"Unsupported training stage: {stage}")

    def _run_encode(self, music_dir: str | Path, model_dir: Path) -> StageRecord:
        """Encode music files into bar tensors without training DVAE."""
        encoded_dir = model_dir / "encoded"
        result = EncodingPipeline(self.config).run(music_dir, encoded_dir)
        return StageRecord(
            name="encode",
            status="ok",
            outputs={
                "encoded_dir": str(encoded_dir),
                "song_count": int(len(result.songs)),
                "bar_count": int(len(result.tensors)),
            },
        )

    def _run_dvae(self, music_dir: str | Path, model_dir: Path) -> StageRecord:
        """Train DVAE and write encoded artifacts."""
        overrides = {
            "device": self.options.device,
            "epochs": self.options.dvae_epochs,
            "batch_size": self.options.batch_size,
            "learning_rate": self.options.dvae_learning_rate,
            "random_seed": self.options.random_seed,
            "transpose_semitones": self.options.transpose_semitones,
        }
        result = DVAETrainingPipeline(self.config, overrides=overrides).run(music_dir, model_dir)
        return StageRecord(
            name="dvae",
            status="ok",
            outputs={
                "model_path": str(result.model_path),
                "diagnostics_path": str(result.diagnostics_path),
                "summary_path": str(result.summary_path),
                "encoded_dir": str(model_dir / "encoded"),
            },
        )

    def _run_latent(self, model_dir: Path) -> StageRecord:
        """Export reusable latent arrays from the trained DVAE."""
        output_dir = model_dir / "latent"
        summary = LatentDatasetExporter(LatentExportConfig(
            batch_size=int(self.options.batch_size or 512),
            device=str(self.options.device or "cpu"),
            save_z=bool(self.options.save_z),
        )).export_from_model_dir(model_dir, output_dir)
        return StageRecord(
            name="latent",
            status="ok",
            outputs={
                "latent_dir": str(output_dir),
                "sample_count": int(summary.get("sample_count", 0)),
                "latent_dim": int(summary.get("latent_dim", 0)),
            },
        )

    def _run_miditok(self, model_dir: Path) -> StageRecord:
        """Train the MidiTok-style bar sequence encoder."""
        overrides = {
            "device": self.options.device,
            "epochs": self.options.miditok_epochs,
            "batch_size": self.options.batch_size,
            "random_seed": self.options.random_seed,
            "max_rows": self.options.max_rows,
        }
        diagnostics = MidiTokBarSequenceTrainingPipeline(self.config, overrides=overrides).run(model_dir=model_dir)
        return StageRecord(
            name="miditok",
            status="ok",
            outputs={
                "checkpoint": diagnostics.get("checkpoint"),
                "diagnostics_path": str(model_dir / "miditok_bar_sequence_encoder_diagnostics.json"),
                "embedding_export": diagnostics.get("embedding_export", {}),
            },
        )

    def _run_retrieval(self, model_dir: Path) -> StageRecord:
        """Train the hybrid latent + MidiTok retrieval model."""
        model_config = HybridMidiTokRetrievalConfig.from_config(self.config)
        event_config = replace(MidiTokBarSequenceEncoderConfig.from_config(self.config), latent_dim=int(model_config.latent_dim))
        training_config = self._hybrid_training_config()
        result = HybridMidiTokTrainingPipeline(model_config, event_config, training_config).run(model_dir=model_dir)
        return StageRecord(
            name="retrieval",
            status="ok",
            outputs={
                "model_path": str(result.model_path),
                "diagnostics_path": str(result.diagnostics_path),
                "summary_path": str(result.summary_path),
            },
        )

    def _run_base_pitch(self, model_dir: Path) -> StageRecord:
        """Train learned base-pitch motion."""
        model_config = BasePitchMotionConfig.from_config(self.config)
        event_config = replace(MidiTokBarSequenceEncoderConfig.from_config(self.config), latent_dim=int(model_config.latent_dim))
        training_config = self._base_pitch_training_config()
        result = BasePitchMotionTrainingPipeline(model_config, event_config, training_config).run(model_dir=model_dir)
        return StageRecord(
            name="base_pitch",
            status="ok",
            outputs={
                "model_path": str(result.model_path),
                "diagnostics_path": str(result.diagnostics_path),
                "summary_path": str(result.summary_path),
            },
        )

    def _run_remi_motion(self, model_dir: Path) -> StageRecord:
        """Train REMI-context latent motion for the current main generation route."""
        overrides = {
            "device": self.options.device,
            "epochs": self.options.remi_motion_epochs,
            "batch_size": self.options.remi_motion_batch_size
            if self.options.remi_motion_batch_size is not None
            else self.options.batch_size,
            "random_seed": self.options.random_seed,
            "force_rebuild_tokens": True,
        }
        result = RemiMotionTrainingPipeline(self.config, overrides=overrides).run(model_dir=model_dir)
        remi_dir = model_dir / "remi_motion"
        return StageRecord(
            name="remi_motion",
            status="ok",
            outputs={
                "model_path": str(result["model_path"]),
                "diagnostics_path": str(remi_dir / "remi_motion_training_diagnostics.json"),
                "tokenizer_path": str(remi_dir / "tokenizer.json"),
                "token_cache_path": str(remi_dir / "remi_bar_tokens.json"),
                "best_val_loss": float(result.get("best_val_loss", 0.0)),
            },
        )

    def _hybrid_training_config(self) -> HybridMidiTokTrainingConfig:
        """Return hybrid retrieval training config with shared overrides."""
        value = HybridMidiTokTrainingConfig()
        updates = {
            "device": self.options.device,
            "epochs": self.options.retrieval_epochs,
            "batch_size": self.options.batch_size,
            "random_seed": self.options.random_seed,
            "max_rows": self.options.max_rows,
        }
        return self._replace_present(value, updates)

    def _base_pitch_training_config(self) -> BasePitchMotionTrainingConfig:
        """Return base-pitch motion training config with shared overrides."""
        value = BasePitchMotionTrainingConfig()
        updates = {
            "device": self.options.device,
            "epochs": self.options.base_pitch_epochs,
            "batch_size": self.options.batch_size,
            "random_seed": self.options.random_seed,
            "max_rows": self.options.max_rows,
        }
        return self._replace_present(value, updates)

    def _replace_present(self, value: Any, updates: Dict[str, Any]) -> Any:
        """Apply non-None dataclass updates."""
        clean = {key: update for key, update in updates.items() if update is not None}
        return replace(value, **clean) if clean else value

    def _write_manifest(self, model_dir: Path, music_dir: str | Path, records: Sequence[StageRecord]) -> Path:
        """Write the model directory manifest."""
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "music_dir": str(music_dir),
            "model_dir": str(model_dir),
            "codec_backend": str(ConfigView(self.config).section("bar_tensor").get("backend", "legacy_physical")),
            "requested_stage": str(self.options.stage),
            "executed_stages": [
                {
                    "name": record.name,
                    "status": record.status,
                    "outputs": record.outputs,
                }
                for record in records
            ],
            "model_contract": {
                "encoded_dir": str(model_dir / "encoded"),
                "latent_dir": str(model_dir / "latent"),
                "dvae": str(model_dir / "dvae.pt"),
                "miditok_sequence_encoder": str(model_dir / "miditok_bar_sequence_encoder.pt"),
                "hybrid_retrieval": str(model_dir / "hybrid_miditok_retrieval.pt"),
                "base_pitch_motion": str(model_dir / "base_pitch_motion.pt"),
                "remi_motion": str(model_dir / "remi_motion" / "remi_motion_predictor.pt"),
                "remi_tokenizer": str(model_dir / "remi_motion" / "tokenizer.json"),
            },
        }
        path = model_dir / "training_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path
