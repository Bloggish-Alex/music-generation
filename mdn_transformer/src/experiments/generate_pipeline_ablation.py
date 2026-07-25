#!/usr/bin/env python3
"""Generate the three-way VAE/composer/MDN ablation MIDI set."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.generate_benchmark_methods import BenchmarkGenerationConfig, BenchmarkMethodMidiGenerator
from pipeline.latent_generation_pipeline import LatentGenerationConfig, LatentGenerationPipeline


@dataclass(frozen=True)
class PipelineAblationConfig:
    """Configuration for the three-way pipeline ablation generation."""

    model_dir: Path
    output_dir: Path
    latent_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    bars: int = 32
    primer_bars: int = 8
    composer_epochs: int = 20
    batch_size: int = 256
    seed: int = 42
    device: str = "cpu"
    base_pitch: int = 60
    tempo_bpm: int = 120
    seed_song_id: Optional[str] = None
    validation_fold_count: int = 5
    validation_fold_index: int = 0
    composer_max_songs: Optional[int] = None
    composer_max_rows: Optional[int] = None


class PipelineAblationGenerator:
    """Generate A/B/C MIDI outputs with standard file names."""

    def __init__(self, config: PipelineAblationConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, Any]:
        """Run all ablation groups."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        report: Dict[str, Any] = {
            "config": self._config_dict(),
            "groups": {},
        }
        report["groups"]["A_transition_composer"] = self._run_composer_group(
            group_name="A_transition_composer",
            method="hybrid_transition_composer",
            context_bars=32,
            hidden_dim=256,
            composer_hidden_dim=None,
            composer_layers=1,
            dropout=0.2,
        )
        report["groups"]["B_anchor_motion_composer"] = self._run_composer_group(
            group_name="B_anchor_motion_composer",
            method="hybrid_anchor_motion_composer",
            context_bars=32,
            hidden_dim=512,
            composer_hidden_dim=512,
            composer_layers=2,
            dropout=0.2,
        )
        report["groups"]["C_direct_mdn_context_32"] = self._run_direct_mdn_group()
        report_path = self.config.output_dir / "ablation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _run_composer_group(
        self,
        group_name: str,
        method: str,
        context_bars: int,
        hidden_dim: int,
        composer_hidden_dim: Optional[int],
        composer_layers: int,
        dropout: float,
    ) -> Dict[str, Any]:
        """Train/generate one composer group and normalize output names."""
        group_dir = self.config.output_dir / group_name
        generator = BenchmarkMethodMidiGenerator(BenchmarkGenerationConfig(
            model_dir=self.config.model_dir,
            latent_dir=self._latent_dir(),
            output_dir=group_dir,
            methods=(method,),
            bars=int(self.config.bars),
            primer_bars=int(self.config.primer_bars),
            context_bars=int(context_bars),
            epochs=int(self.config.composer_epochs),
            batch_size=int(self.config.batch_size),
            hidden_dim=int(hidden_dim),
            composer_hidden_dim=composer_hidden_dim,
            composer_layers=int(composer_layers),
            dropout=float(dropout),
            validation_fold_count=int(self.config.validation_fold_count),
            validation_fold_index=int(self.config.validation_fold_index),
            max_songs=self.config.composer_max_songs,
            max_rows=self.config.composer_max_rows,
            seed_song_id=self.config.seed_song_id,
            random_seed=int(self.config.seed),
            device=str(self.config.device),
            tempo_bpm=int(self.config.tempo_bpm),
            base_pitch=int(self.config.base_pitch),
        ))
        raw_report = generator.run()
        method_report = raw_report["methods"][method]
        midi_path = self._copy_or_replace(Path(method_report["midi_path"]), group_dir / "generation.mid")
        json_path = self._copy_or_replace(group_dir / f"{method}.generation_diagnostics.json", group_dir / "generation.json")
        tensor_path = self._copy_or_replace(Path(method_report["tensor_path"]), group_dir / "bars.npz")
        return {
            "kind": "composer",
            "method": method,
            "context_bars": int(context_bars),
            "hidden_dim": int(hidden_dim),
            "composer_hidden_dim": int(composer_hidden_dim or hidden_dim),
            "composer_layers": int(composer_layers),
            "dropout": float(dropout),
            "midi_path": str(midi_path),
            "json_path": str(json_path),
            "tensor_path": str(tensor_path),
            "source_report": method_report,
        }

    def _run_direct_mdn_group(self) -> Dict[str, Any]:
        """Run the existing Direct MDN context-32 generation pipeline."""
        group_dir = self.config.output_dir / "C_direct_mdn_context_32"
        group_dir.mkdir(parents=True, exist_ok=True)
        output_json = group_dir / "generation.json"
        output_midi = group_dir / "generation.mid"
        generation_config = LatentGenerationConfig(
            bars=int(self.config.bars),
            primer_bars=int(self.config.primer_bars),
            backend="direct_mdn",
            model_variant="root",
            seed=int(self.config.seed),
            device=str(self.config.device),
            base_pitch=int(self.config.base_pitch),
            tempo_bpm=int(self.config.tempo_bpm),
            sample_std_scale=0.0,
            retrieval_use_retrieved_tensors=False,
        )
        result = LatentGenerationPipeline(generation_config).run(
            model_dir=self.config.model_dir,
            latent_dir=self._latent_dir(),
            output_json=output_json,
            output_midi=output_midi,
            seed_song_id=self.config.seed_song_id,
        )
        tensor_path = self._copy_or_replace(result.tensor_path, group_dir / "bars.npz")
        return {
            "kind": "direct_mdn",
            "context_bars": 32,
            "midi_path": str(output_midi),
            "json_path": str(output_json),
            "tensor_path": str(tensor_path),
            "source_report": result.diagnostics,
        }

    def _latent_dir(self) -> Path:
        """Return latent directory."""
        return Path(self.config.latent_dir) if self.config.latent_dir else self.config.model_dir / "latent"

    def _copy_or_replace(self, source: Path, target: Path) -> Path:
        """Copy source to target, replacing any existing target."""
        source = Path(source)
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == target.resolve():
            return target
        if target.exists():
            target.unlink()
        shutil.copy2(source, target)
        return target

    def _config_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config."""
        result = asdict(self.config)
        for key in ("model_dir", "output_dir", "latent_dir", "config_path"):
            if result.get(key) is not None:
                result[key] = str(result[key])
        return result


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineAblationConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate A/B/C MIDI outputs for composer-vs-MDN ablation.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Reserved for compatibility; current ablation uses model defaults.")
    parser.add_argument("--bars", type=int, default=32)
    parser.add_argument("--primer-bars", type=int, default=8)
    parser.add_argument("--composer-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base-pitch", type=int, default=60)
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--seed-song-id", type=str, default=None)
    parser.add_argument("--validation-fold-count", type=int, default=5)
    parser.add_argument("--validation-fold-index", type=int, default=0)
    parser.add_argument("--composer-max-songs", type=int, default=None)
    parser.add_argument("--composer-max-rows", type=int, default=None)
    args = parser.parse_args(argv)
    return PipelineAblationConfig(
        model_dir=Path(args.model_dir),
        latent_dir=Path(args.latent_dir) if args.latent_dir else None,
        output_dir=Path(args.output_dir),
        config_path=Path(args.config) if args.config else None,
        bars=int(args.bars),
        primer_bars=int(args.primer_bars),
        composer_epochs=int(args.composer_epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=str(args.device),
        base_pitch=int(args.base_pitch),
        tempo_bpm=int(args.tempo),
        seed_song_id=args.seed_song_id,
        validation_fold_count=int(args.validation_fold_count),
        validation_fold_index=int(args.validation_fold_index),
        composer_max_songs=args.composer_max_songs,
        composer_max_rows=args.composer_max_rows,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the ablation generator."""
    config = parse_args(argv)
    report = PipelineAblationGenerator(config).run()
    print(f"Pipeline ablation complete -> {config.output_dir}")
    for group, info in report["groups"].items():
        print(f"{group}: {info['midi_path']}")


if __name__ == "__main__":
    main()
