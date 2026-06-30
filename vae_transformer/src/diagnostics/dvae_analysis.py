#!/usr/bin/env python3
"""Offline diagnostics for trained DVAE reconstruction quality."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.dvae import DVAEMusicConfig, DenoisingMusicVAE


STATE_NAMES = ["REST", "NOTE_ON", "HOLD"]


@dataclass(frozen=True)
class DVAEAnalysisConfig:
    """Configuration for offline DVAE diagnostics."""

    batch_size: int = 256
    device: str = "cpu"
    sample_count: int = 32
    random_seed: int = 42


class DVAEArtifactLoader:
    """Load a trained DVAE and encoded tensor artifacts."""

    def load_model(self, model_path: str | Path, device: str) -> DenoisingMusicVAE:
        """Load a DVAE model from a saved torch payload."""
        payload = torch.load(Path(model_path), map_location=device)
        config = DVAEMusicConfig(**payload["config"])
        model = DenoisingMusicVAE(config).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def load_tensor_array(self, tensor_path: str | Path) -> tuple[np.ndarray, List[str]]:
        """Load bar tensors from npz into a stacked float32 array."""
        payload = np.load(Path(tensor_path))
        keys = sorted(payload.files)
        if not keys:
            raise ValueError("No tensors found in bar_tensors.npz.")
        values = np.stack([np.asarray(payload[key], dtype=np.float32) for key in keys], axis=0)
        return values, keys

    def load_index(self, index_path: str | Path) -> Dict[str, Dict[str, Any]]:
        """Load tensor index rows keyed by tensor_key."""
        rows = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return {str(row["tensor_key"]): dict(row) for row in rows}

    def load_action_map(self, songs_path: str | Path) -> Dict[tuple[str, int], str]:
        """Load action labels keyed by song_id and bar_index."""
        songs = json.loads(Path(songs_path).read_text(encoding="utf-8"))
        result: Dict[tuple[str, int], str] = {}
        for song in songs:
            song_id = str(song.get("song_id"))
            for bar in song.get("bars", []):
                result[(song_id, int(bar.get("bar_index", 0)))] = str(bar.get("action", "UNKNOWN"))
        return result


class DVAEReconstructionAnalyzer:
    """Compute reconstruction diagnostics from a trained DVAE."""

    def __init__(self, config: DVAEAnalysisConfig) -> None:
        self.config = config

    def analyze(
        self,
        model: DenoisingMusicVAE,
        tensors: np.ndarray,
        keys: Sequence[str],
        index: Dict[str, Dict[str, Any]],
        action_map: Dict[tuple[str, int], str],
    ) -> Dict[str, Any]:
        """Run model reconstruction and aggregate diagnostics."""
        dataset = TensorDataset(torch.from_numpy(tensors).float())
        loader = DataLoader(dataset, batch_size=int(self.config.batch_size), shuffle=False)
        accum = _MetricAccumulator()
        latent_mu: List[np.ndarray] = []
        latent_log_var: List[np.ndarray] = []
        offset = 0
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.config.device)
                output = model(batch, add_noise=False)
                batch_result = self._batch_result(output, batch)
                mu, log_var = model.encoder(batch)
                latent_mu.append(mu.detach().cpu().numpy())
                latent_log_var.append(log_var.detach().cpu().numpy())
                batch_keys = list(keys[offset: offset + int(batch.shape[0])])
                accum.add(batch_result, batch_keys, index, action_map)
                offset += int(batch.shape[0])
        mu_array = np.concatenate(latent_mu, axis=0)
        log_var_array = np.concatenate(latent_log_var, axis=0)
        return {
            "summary": accum.summary(),
            "state_confusion": accum.state_confusion_report(),
            "track_metrics": accum.track_report(),
            "action_metrics": accum.action_report(),
            "worst_bars": accum.worst_bars(20),
            "latent": self._latent_report(mu_array, log_var_array),
        }

    def sample_reconstructions(
        self,
        model: DenoisingMusicVAE,
        tensors: np.ndarray,
        keys: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        """Return sampled target and reconstructed tensors for inspection."""
        rng = random.Random(int(self.config.random_seed))
        indices = list(range(len(keys)))
        rng.shuffle(indices)
        selected = indices[: max(0, min(int(self.config.sample_count), len(indices)))]
        if not selected:
            return {}
        batch = torch.from_numpy(tensors[selected]).float().to(self.config.device)
        model.eval()
        with torch.no_grad():
            output = model(batch, add_noise=False)
        reconstructed = self._reconstructed_tensor(output).detach().cpu().numpy()
        result: Dict[str, np.ndarray] = {}
        for local_index, source_index in enumerate(selected):
            key = str(keys[source_index])
            result[f"{key}__target"] = tensors[source_index]
            result[f"{key}__reconstructed"] = reconstructed[local_index]
        return result

    def _batch_result(self, output: Any, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build per-slot tensors used by metric accumulation."""
        pred_state = torch.argmax(output.state_logits, dim=-1)
        target_state = torch.argmax(target[..., 1:4], dim=-1)
        pitch_error = (output.pitch - target[..., 0]).pow(2)
        velocity_error = (output.velocity - target[..., 4]).pow(2)
        chord_error = torch.mean((output.chord - target[..., 5:16]).pow(2), dim=-1)
        state_error = (pred_state != target_state).float()
        bar_error = torch.mean(pitch_error + velocity_error + chord_error + state_error, dim=(1, 2))
        return {
            "pred_state": pred_state.detach().cpu(),
            "target_state": target_state.detach().cpu(),
            "pitch_error": pitch_error.detach().cpu(),
            "velocity_error": velocity_error.detach().cpu(),
            "chord_error": chord_error.detach().cpu(),
            "state_error": state_error.detach().cpu(),
            "bar_error": bar_error.detach().cpu(),
        }

    def _reconstructed_tensor(self, output: Any) -> torch.Tensor:
        """Reassemble multi-head output into [B, 3, 16, 16] tensor form."""
        state = torch.softmax(output.state_logits, dim=-1)
        return torch.cat([
            output.pitch.unsqueeze(-1),
            state,
            output.velocity.unsqueeze(-1),
            output.chord,
        ], dim=-1)

    def _latent_report(self, mu: np.ndarray, log_var: np.ndarray) -> Dict[str, Any]:
        """Summarize latent mean and variance distributions."""
        dim_std = np.std(mu, axis=0)
        return {
            "sample_count": int(mu.shape[0]),
            "latent_dim": int(mu.shape[1]),
            "mu_mean_abs": float(np.mean(np.abs(mu))),
            "mu_std_mean": float(np.mean(dim_std)),
            "mu_std_min": float(np.min(dim_std)),
            "mu_std_max": float(np.max(dim_std)),
            "active_dim_count_std_gt_0_05": int(np.sum(dim_std > 0.05)),
            "active_dim_count_std_gt_0_10": int(np.sum(dim_std > 0.10)),
            "log_var_mean": float(np.mean(log_var)),
            "log_var_std": float(np.std(log_var)),
        }


class _MetricAccumulator:
    """Accumulate reconstruction metrics across batches."""

    def __init__(self) -> None:
        self.count = 0
        self.slot_count = 0
        self.pitch_error_sum = 0.0
        self.velocity_error_sum = 0.0
        self.chord_error_sum = 0.0
        self.state_error_sum = 0.0
        self.confusion = np.zeros((3, 3), dtype=np.int64)
        self.track: Dict[int, Dict[str, float]] = {}
        self.action: Dict[str, Dict[str, float]] = {}
        self.bar_rows: List[Dict[str, Any]] = []

    def add(
        self,
        result: Dict[str, torch.Tensor],
        keys: Sequence[str],
        index: Dict[str, Dict[str, Any]],
        action_map: Dict[tuple[str, int], str],
    ) -> None:
        """Add one batch result to the accumulator."""
        batch_size = int(result["bar_error"].shape[0])
        self.count += batch_size
        self.slot_count += int(np.prod(result["target_state"].shape))
        self.pitch_error_sum += float(result["pitch_error"].sum())
        self.velocity_error_sum += float(result["velocity_error"].sum())
        self.chord_error_sum += float(result["chord_error"].sum())
        self.state_error_sum += float(result["state_error"].sum())
        self._add_confusion(result["target_state"].numpy(), result["pred_state"].numpy())
        self._add_track(result)
        self._add_bars(result, keys, index, action_map)

    def summary(self) -> Dict[str, Any]:
        """Return global metric summary."""
        denom = max(1, self.slot_count)
        return {
            "bar_count": int(self.count),
            "slot_count": int(self.slot_count),
            "pitch_mse": float(self.pitch_error_sum / denom),
            "velocity_mse": float(self.velocity_error_sum / denom),
            "chord_mse": float(self.chord_error_sum / denom),
            "state_error_rate": float(self.state_error_sum / denom),
            "state_accuracy": float(1.0 - self.state_error_sum / denom),
        }

    def state_confusion_report(self) -> Dict[str, Any]:
        """Return state confusion matrix and per-class accuracy."""
        rows = []
        for target_index, name in enumerate(STATE_NAMES):
            total = int(self.confusion[target_index].sum())
            correct = int(self.confusion[target_index, target_index])
            rows.append({
                "target_state": name,
                "total": total,
                "correct": correct,
                "accuracy": float(correct / total) if total else 0.0,
                "predicted": {
                    STATE_NAMES[pred_index]: int(self.confusion[target_index, pred_index])
                    for pred_index in range(3)
                },
            })
        return {"matrix": self.confusion.tolist(), "rows": rows}

    def track_report(self) -> List[Dict[str, Any]]:
        """Return metrics grouped by track index."""
        return [self._finalize_group({"track_index": key}, value) for key, value in sorted(self.track.items())]

    def action_report(self) -> List[Dict[str, Any]]:
        """Return metrics grouped by action label."""
        return [self._finalize_group({"action": key}, value) for key, value in sorted(self.action.items())]

    def worst_bars(self, limit: int) -> List[Dict[str, Any]]:
        """Return bars with the largest reconstruction error."""
        return sorted(self.bar_rows, key=lambda row: float(row["bar_error"]), reverse=True)[: max(0, int(limit))]

    def _add_confusion(self, target: np.ndarray, pred: np.ndarray) -> None:
        """Update state confusion counts."""
        for target_index in range(3):
            for pred_index in range(3):
                self.confusion[target_index, pred_index] += int(np.sum((target == target_index) & (pred == pred_index)))

    def _add_track(self, result: Dict[str, torch.Tensor]) -> None:
        """Update per-track metrics."""
        for track_index in range(result["target_state"].shape[1]):
            group = self.track.setdefault(int(track_index), self._new_group())
            self._add_group_slice(group, result, track_index=track_index)

    def _add_bars(
        self,
        result: Dict[str, torch.Tensor],
        keys: Sequence[str],
        index: Dict[str, Dict[str, Any]],
        action_map: Dict[tuple[str, int], str],
    ) -> None:
        """Update per-action metrics and worst-bar rows."""
        for batch_index, key in enumerate(keys):
            row = index.get(str(key), {})
            song_id = str(row.get("song_id", "UNKNOWN"))
            bar_index = int(row.get("bar_index", 0))
            action = action_map.get((song_id, bar_index), "UNKNOWN")
            group = self.action.setdefault(action, self._new_group())
            self._add_group_bar(group, result, batch_index)
            self.bar_rows.append({
                "tensor_key": str(key),
                "song_id": song_id,
                "bar_index": bar_index,
                "action": action,
                "bar_error": float(result["bar_error"][batch_index]),
            })

    def _new_group(self) -> Dict[str, float]:
        """Create an empty metric group."""
        return {
            "slot_count": 0.0,
            "pitch_error_sum": 0.0,
            "velocity_error_sum": 0.0,
            "chord_error_sum": 0.0,
            "state_error_sum": 0.0,
        }

    def _add_group_slice(self, group: Dict[str, float], result: Dict[str, torch.Tensor], track_index: int) -> None:
        """Add all bars for one track into a metric group."""
        slots = int(np.prod(result["state_error"][:, track_index, :].shape))
        group["slot_count"] += float(slots)
        group["pitch_error_sum"] += float(result["pitch_error"][:, track_index, :].sum())
        group["velocity_error_sum"] += float(result["velocity_error"][:, track_index, :].sum())
        group["chord_error_sum"] += float(result["chord_error"][:, track_index, :].sum())
        group["state_error_sum"] += float(result["state_error"][:, track_index, :].sum())

    def _add_group_bar(self, group: Dict[str, float], result: Dict[str, torch.Tensor], batch_index: int) -> None:
        """Add one bar into a metric group."""
        slots = int(np.prod(result["state_error"][batch_index].shape))
        group["slot_count"] += float(slots)
        group["pitch_error_sum"] += float(result["pitch_error"][batch_index].sum())
        group["velocity_error_sum"] += float(result["velocity_error"][batch_index].sum())
        group["chord_error_sum"] += float(result["chord_error"][batch_index].sum())
        group["state_error_sum"] += float(result["state_error"][batch_index].sum())

    def _finalize_group(self, prefix: Dict[str, Any], group: Dict[str, float]) -> Dict[str, Any]:
        """Finalize one metric group into averages."""
        denom = max(1.0, group["slot_count"])
        return {
            **prefix,
            "slot_count": int(group["slot_count"]),
            "pitch_mse": float(group["pitch_error_sum"] / denom),
            "velocity_mse": float(group["velocity_error_sum"] / denom),
            "chord_mse": float(group["chord_error_sum"] / denom),
            "state_error_rate": float(group["state_error_sum"] / denom),
            "state_accuracy": float(1.0 - group["state_error_sum"] / denom),
        }


class DVAEAnalysisWriter:
    """Write DVAE analysis artifacts."""

    def write(self, output_dir: str | Path, report: Dict[str, Any], samples: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Write JSON, Markdown, CSV, and sampled reconstruction NPZ files."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        json_path = path / "dvae_reconstruction_report.json"
        md_path = path / "dvae_reconstruction_report.md"
        csv_path = path / "dvae_worst_bars.csv"
        sample_path = path / "dvae_reconstruction_samples.npz"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown(report), encoding="utf-8")
        self._write_worst_csv(csv_path, report.get("worst_bars", []))
        if samples:
            np.savez_compressed(sample_path, **samples)
        return {
            "json": str(json_path),
            "markdown": str(md_path),
            "worst_bars_csv": str(csv_path),
            "samples_npz": str(sample_path) if samples else "",
        }

    def _write_worst_csv(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        """Write worst reconstruction rows as CSV."""
        fields = ["tensor_key", "song_id", "bar_index", "action", "bar_error"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})

    def _markdown(self, report: Dict[str, Any]) -> str:
        """Create a compact Markdown summary."""
        lines = ["# DVAE Reconstruction Report", ""]
        summary = report.get("summary", {})
        lines.extend([
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ])
        for key in ["bar_count", "state_accuracy", "state_error_rate", "pitch_mse", "velocity_mse", "chord_mse"]:
            lines.append(f"| {key} | {summary.get(key, 0)} |")
        lines.extend(["", "## State Confusion", "", "| Target | Accuracy | REST | NOTE_ON | HOLD |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in report.get("state_confusion", {}).get("rows", []):
            pred = row.get("predicted", {})
            lines.append(
                f"| {row.get('target_state')} | {row.get('accuracy', 0):.6f} | "
                f"{pred.get('REST', 0)} | {pred.get('NOTE_ON', 0)} | {pred.get('HOLD', 0)} |"
            )
        lines.extend(["", "## Track Metrics", "", "| Track | State Acc | Pitch MSE | Velocity MSE | Chord MSE |", "| ---: | ---: | ---: | ---: | ---: |"])
        for row in report.get("track_metrics", []):
            lines.append(
                f"| {row.get('track_index')} | {row.get('state_accuracy', 0):.6f} | "
                f"{row.get('pitch_mse', 0):.6f} | {row.get('velocity_mse', 0):.6f} | {row.get('chord_mse', 0):.6f} |"
            )
        lines.extend(["", "## Action Metrics", "", "| Action | State Acc | Pitch MSE | Velocity MSE | Chord MSE |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in report.get("action_metrics", []):
            lines.append(
                f"| {row.get('action')} | {row.get('state_accuracy', 0):.6f} | "
                f"{row.get('pitch_mse', 0):.6f} | {row.get('velocity_mse', 0):.6f} | {row.get('chord_mse', 0):.6f} |"
            )
        latent = report.get("latent", {})
        lines.extend(["", "## Latent", "", "| Metric | Value |", "| --- | ---: |"])
        for key, value in latent.items():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines) + "\n"
