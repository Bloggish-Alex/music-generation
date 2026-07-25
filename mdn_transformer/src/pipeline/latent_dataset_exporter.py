#!/usr/bin/env python3
"""Export trained DVAE latent vectors as a reusable sequence dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.core import LatentBarRecord
from diagnostics.dvae_analysis import DVAEArtifactLoader
from model.dvae import DenoisingMusicVAE


@dataclass(frozen=True)
class LatentExportConfig:
    """Configuration for latent dataset export."""

    batch_size: int = 512
    device: str = "cpu"
    save_z: bool = False


@dataclass
class LatentDataset:
    """In-memory latent dataset for downstream sequence models."""

    mu: np.ndarray
    log_var: np.ndarray
    records: List[LatentBarRecord]
    z: Optional[np.ndarray] = None


class LatentMetadataReader:
    """Read bar metadata for latent rows from encoded artifacts."""

    def load_records(self, index_path: str | Path, songs_path: Optional[str | Path] = None) -> List[LatentBarRecord]:
        """Build latent row records ordered by tensor_key."""
        index_rows = json.loads(Path(index_path).read_text(encoding="utf-8"))
        song_lookup = self._load_song_lookup(songs_path) if songs_path and Path(songs_path).exists() else {}
        ordered = sorted(index_rows, key=lambda row: str(row["tensor_key"]))
        records: List[LatentBarRecord] = []
        for row_index, row in enumerate(ordered):
            song_id = str(row.get("song_id", "UNKNOWN"))
            bar_index = int(row.get("bar_index", 0))
            extra = song_lookup.get((song_id, bar_index), {})
            records.append(LatentBarRecord(
                row_index=row_index,
                tensor_key=str(row["tensor_key"]),
                song_id=song_id,
                bar_index=bar_index,
                action=extra.get("action"),
                form=extra.get("form"),
                section_label=extra.get("section_label"),
                section_index=extra.get("section_index"),
            ))
        return records

    def _load_song_lookup(self, songs_path: str | Path) -> Dict[tuple[str, int], Dict[str, Any]]:
        """Load optional action/form/section metadata from songs.json."""
        songs = json.loads(Path(songs_path).read_text(encoding="utf-8"))
        result: Dict[tuple[str, int], Dict[str, Any]] = {}
        for song in songs:
            song_id = str(song.get("song_id", "UNKNOWN"))
            form = song.get("form")
            for bar in song.get("bars", []):
                result[(song_id, int(bar.get("bar_index", 0)))] = {
                    "action": bar.get("action"),
                    "form": bar.get("form", form),
                    "section_label": bar.get("section_label"),
                    "section_index": bar.get("section_index"),
                }
        return result


class LatentDatasetExporter:
    """Encode tensors with a trained DVAE encoder and export latent arrays."""

    def __init__(self, config: LatentExportConfig) -> None:
        self.config = config
        self.loader = DVAEArtifactLoader()
        self.metadata_reader = LatentMetadataReader()

    def export_from_model_dir(self, model_dir: str | Path, output_dir: str | Path) -> Dict[str, Any]:
        """Load standard model-dir artifacts and export latent dataset files."""
        model_path = Path(model_dir) / "dvae.pt"
        encoded_dir = Path(model_dir) / "encoded"
        model = self.loader.load_model(model_path, self.config.device)
        tensors, keys = self.loader.load_tensor_array(encoded_dir / "bar_tensors.npz")
        records = self.metadata_reader.load_records(
            encoded_dir / "bar_tensor_index.json",
            encoded_dir / "songs.json",
        )
        dataset = self.encode_tensor_array(model, tensors, keys, records)
        return self.write_dataset(dataset, output_dir, model_path=model_path)

    def encode_tensor_array(
        self,
        model: DenoisingMusicVAE,
        tensors: np.ndarray,
        keys: Sequence[str],
        records: Optional[List[LatentBarRecord]] = None,
    ) -> LatentDataset:
        """Encode a tensor array into latent means and log variances."""
        if records is None:
            records = [
                LatentBarRecord(row_index=index, tensor_key=str(key), song_id=str(key), bar_index=index)
                for index, key in enumerate(keys)
            ]
        if len(records) != int(tensors.shape[0]):
            raise ValueError("records length must match tensor count.")
        dataset = TensorDataset(torch.from_numpy(np.asarray(tensors, dtype=np.float32)).float())
        loader = DataLoader(dataset, batch_size=int(self.config.batch_size), shuffle=False)
        mu_rows: List[np.ndarray] = []
        log_var_rows: List[np.ndarray] = []
        z_rows: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.config.device)
                mu, log_var = model.encoder(batch)
                mu_rows.append(mu.detach().cpu().numpy())
                log_var_rows.append(log_var.detach().cpu().numpy())
                if self.config.save_z:
                    z_rows.append(model.reparameterize(mu, log_var).detach().cpu().numpy())
        return LatentDataset(
            mu=np.concatenate(mu_rows, axis=0),
            log_var=np.concatenate(log_var_rows, axis=0),
            z=np.concatenate(z_rows, axis=0) if z_rows else None,
            records=records,
        )

    def write_dataset(self, dataset: LatentDataset, output_dir: str | Path, model_path: str | Path) -> Dict[str, Any]:
        """Write latent arrays, row index, and summary to output_dir."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        mu_path = output / "latent_mu.npy"
        log_var_path = output / "latent_log_var.npy"
        z_path = output / "latent_z.npy"
        index_path = output / "latent_index.json"
        summary_path = output / "latent_summary.json"
        np.save(mu_path, dataset.mu.astype(np.float32))
        np.save(log_var_path, dataset.log_var.astype(np.float32))
        if dataset.z is not None:
            np.save(z_path, dataset.z.astype(np.float32))
        index_path.write_text(
            json.dumps([record.to_dict() for record in dataset.records], indent=2),
            encoding="utf-8",
        )
        summary = self._summary(dataset, model_path, {
            "latent_mu": str(mu_path),
            "latent_log_var": str(log_var_path),
            "latent_z": str(z_path) if dataset.z is not None else None,
            "latent_index": str(index_path),
            "latent_summary": str(summary_path),
        })
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _summary(self, dataset: LatentDataset, model_path: str | Path, paths: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """Create latent dataset summary statistics."""
        dim_std = np.std(dataset.mu, axis=0)
        action_counts: Dict[str, int] = {}
        for record in dataset.records:
            action = str(record.action or "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1
        return {
            "model_path": str(model_path),
            "sample_count": int(dataset.mu.shape[0]),
            "latent_dim": int(dataset.mu.shape[1]),
            "save_z": bool(dataset.z is not None),
            "mu_mean_abs": float(np.mean(np.abs(dataset.mu))),
            "mu_std_mean": float(np.mean(dim_std)),
            "mu_std_min": float(np.min(dim_std)),
            "mu_std_max": float(np.max(dim_std)),
            "active_dim_count_std_gt_0_05": int(np.sum(dim_std > 0.05)),
            "active_dim_count_std_gt_0_10": int(np.sum(dim_std > 0.10)),
            "log_var_mean": float(np.mean(dataset.log_var)),
            "log_var_std": float(np.std(dataset.log_var)),
            "action_counts": action_counts,
            "paths": paths,
        }
