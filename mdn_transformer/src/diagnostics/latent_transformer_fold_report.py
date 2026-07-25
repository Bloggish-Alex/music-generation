#!/usr/bin/env python3
"""Aggregate Latent-Transformer fold summaries into JSON and Markdown reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class FoldReportConfig:
    """Configuration for fold summary aggregation."""

    input_dirs: tuple[Path, ...]
    output_dir: Path


class LatentTransformerFoldReport:
    """Build compact cross-fold diagnostics from training summaries."""

    SUMMARY_NAME = "latent_transformer_training_summary.json"

    def run(self, config: FoldReportConfig) -> Dict[str, Any]:
        """Read summaries and write aggregate report files."""
        summaries = self._load_summaries(config.input_dirs)
        report = self._build_report(summaries)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = config.output_dir / "latent_transformer_fold_report.json"
        md_path = config.output_dir / "latent_transformer_fold_report.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown(report), encoding="utf-8")
        return {
            **report,
            "paths": {
                "json": str(json_path),
                "markdown": str(md_path),
            },
        }

    def _load_summaries(self, input_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
        """Load all summary files from explicit dirs or recursive roots."""
        paths: List[Path] = []
        for directory in input_dirs:
            directory = Path(directory)
            direct = directory / self.SUMMARY_NAME
            if direct.exists():
                paths.append(direct)
            elif directory.exists():
                paths.extend(sorted(directory.rglob(self.SUMMARY_NAME)))
        unique_paths = sorted({path.resolve() for path in paths})
        if not unique_paths:
            raise FileNotFoundError("No latent_transformer_training_summary.json files found.")
        return [self._load_one(path) for path in unique_paths]

    def _load_one(self, path: Path) -> Dict[str, Any]:
        """Load one summary and attach source path metadata."""
        data = json.loads(path.read_text(encoding="utf-8"))
        split = data.get("dataset_split") or {}
        fold = data.get("model_selection") or data.get("early_stopping", {})
        data["_source_path"] = str(path)
        data["_fold_index"] = split.get("validation_fold_index")
        data["_fold_count"] = split.get("validation_fold_count")
        data["_best_epoch"] = fold.get("best_epoch")
        return data

    def _build_report(self, summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Build aggregate metric summaries."""
        rows = [self._row(summary) for summary in summaries]
        return {
            "fold_count": int(len(rows)),
            "rows": rows,
            "aggregate": {
                "val_nll": self._metric_stats(rows, "val_nll"),
                "train_nll": self._metric_stats(rows, "train_nll"),
                "generalization_gap_nll": self._metric_stats(rows, "generalization_gap_nll"),
                "val_avg_max_pi": self._metric_stats(rows, "val_avg_max_pi"),
                "val_component_entropy": self._metric_stats(rows, "val_component_entropy"),
                "val_latent_l2_to_best_component": self._metric_stats(rows, "val_latent_l2_to_best_component"),
                "best_epoch": self._metric_stats(rows, "best_epoch"),
            },
        }

    def _row(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Extract one fold row."""
        train_eval = summary.get("train_eval", {})
        val_eval = summary.get("val_eval", {})
        selection = summary.get("model_selection") or summary.get("early_stopping", {})
        split = summary.get("dataset_split", {})
        train_nll = float(train_eval.get("nll", np.nan))
        val_nll = float(val_eval.get("nll", np.nan))
        return {
            "source_path": summary.get("_source_path"),
            "fold_index": split.get("validation_fold_index"),
            "fold_count": split.get("validation_fold_count"),
            "validation_base_song_ids": split.get("validation_base_song_ids", []),
            "train_size": split.get("train_size"),
            "validation_size": split.get("validation_size"),
            "best_epoch": selection.get("best_epoch"),
            "best_metric": selection.get("best_metric"),
            "train_nll": train_nll,
            "val_nll": val_nll,
            "generalization_gap_nll": float(val_nll - train_nll),
            "val_component_entropy": float(val_eval.get("component_entropy", np.nan)),
            "val_avg_max_pi": float(val_eval.get("avg_max_pi", np.nan)),
            "val_latent_l2_to_best_component": float(val_eval.get("latent_l2_to_best_component", np.nan)),
            "val_pi_argmax_ratios": val_eval.get("component_usage", {}).get("pi_argmax_ratios", []),
            "val_nearest_component_ratios": val_eval.get("component_usage", {}).get("nearest_component_ratios", []),
        }

    def _metric_stats(self, rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
        """Return mean/std/min/max for a numeric row key."""
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def _markdown(self, report: Dict[str, Any]) -> str:
        """Render report as Markdown."""
        lines = [
            "# Latent Transformer Fold Report",
            "",
            "## Aggregate",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for key, stats in report["aggregate"].items():
            lines.append(
                f"| {key} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |"
            )
        lines.extend([
            "",
            "## Folds",
            "",
            "| Fold | Best Epoch | Train NLL | Val NLL | Gap | Val Entropy | Val Avg Max Pi | Val L2 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in report["rows"]:
            lines.append(
                "| {fold} | {best} | {train:.6f} | {val:.6f} | {gap:.6f} | {ent:.6f} | {max_pi:.6f} | {l2:.6f} |".format(
                    fold=row.get("fold_index"),
                    best=row.get("best_epoch"),
                    train=float(row.get("train_nll", np.nan)),
                    val=float(row.get("val_nll", np.nan)),
                    gap=float(row.get("generalization_gap_nll", np.nan)),
                    ent=float(row.get("val_component_entropy", np.nan)),
                    max_pi=float(row.get("val_avg_max_pi", np.nan)),
                    l2=float(row.get("val_latent_l2_to_best_component", np.nan)),
                )
            )
        lines.extend(["", "## Validation Songs", ""])
        for row in report["rows"]:
            songs = ", ".join(str(item) for item in row.get("validation_base_song_ids", []))
            lines.append(f"- Fold {row.get('fold_index')}: {songs}")
        lines.append("")
        return "\n".join(lines)
