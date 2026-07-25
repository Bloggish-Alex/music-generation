#!/usr/bin/env python3
"""Measure how well frozen DVAE latents preserve audible bar harmony."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader


@dataclass(frozen=True)
class DVAEHarmonyOracleConfig:
    """Inputs and numerical settings for the frozen-latent harmony probe."""

    model_dir: Path
    latent_dir: Path | None = None
    encoded_dir: Path | None = None
    dvae_path: Path | None = None
    output_dir: Path | None = None
    batch_size: int = 256
    device: str = "cpu"
    pitch_scale: float = 24.0
    pitch_class_sigma: float = 0.35
    posterior_samples: int = 4
    max_rows: int | None = None


class DVAEHarmonyOracleAnalyzer:
    """Decode real latent means and compare physical chroma to source bars."""

    def __init__(self, config: DVAEHarmonyOracleConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, Any]:
        model_dir = Path(self.config.model_dir)
        latent_dir = Path(self.config.latent_dir) if self.config.latent_dir else model_dir / "latent"
        encoded_dir = Path(self.config.encoded_dir) if self.config.encoded_dir else model_dir / "encoded"
        output_dir = Path(self.config.output_dir) if self.config.output_dir else model_dir / "dvae_harmony_oracle"
        output_dir.mkdir(parents=True, exist_ok=True)
        dvae_path = Path(self.config.dvae_path) if self.config.dvae_path else model_dir / "dvae.pt"
        model = self._load_model(dvae_path)
        latent_mu, rows, latent_summary = LatentDatasetReader().load(latent_dir)
        limit = min(len(rows), int(self.config.max_rows)) if self.config.max_rows is not None else len(rows)
        latent_mu = latent_mu[:limit]
        rows = rows[:limit]
        archive_path = encoded_dir / "bar_tensors.npz"
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing encoded bar tensor archive: {archive_path}")
        archive = np.load(archive_path)
        source_chroma = self._source_chroma_by_bar(encoded_dir / "songs.json")
        target_physical: List[np.ndarray] = []
        decoded_physical_soft: List[np.ndarray] = []
        decoded_physical_hard: List[np.ndarray] = []
        target_head: List[np.ndarray] = []
        decoded_head: List[np.ndarray] = []
        target_active: List[bool] = []
        tensor_physical_target: List[np.ndarray] = []
        direct_mu_physical: List[np.ndarray] = []
        posterior_z_physical: List[List[np.ndarray]] = [[] for _ in range(max(1, int(self.config.posterior_samples)))]
        export_mu_squared_error = 0.0
        export_mu_value_count = 0
        export_mu_max_abs_error = 0.0
        for start in range(0, len(rows), int(self.config.batch_size)):
            end = min(len(rows), start + int(self.config.batch_size))
            tensors = self._tensor_batch(archive, rows[start:end])
            source = torch.from_numpy(tensors).to(self.config.device)
            latent = torch.from_numpy(latent_mu[start:end]).to(self.config.device)
            with torch.no_grad():
                pitch, state_logits, _velocity, chord = model.decoder(latent)
                state_probability = torch.softmax(state_logits, dim=-1)
                decoded_active = state_probability[..., 1] + state_probability[..., 2]
                decoded_soft = self._soft_physical_chroma(pitch, decoded_active, float(self.config.pitch_scale))
                decoded_hard_active = (torch.argmax(state_logits, dim=-1) != 0).to(dtype=pitch.dtype)
                decoded_hard = self._soft_physical_chroma(pitch, decoded_hard_active, float(self.config.pitch_scale))
                encoder_mu, encoder_log_var = model.encoder(source)
                direct_mu_physical.append(self._decoded_hard_physical_chroma(model, encoder_mu).cpu().numpy())
                target_state = torch.argmax(source[..., 1:4], dim=-1)
                target_tensor_active = (target_state != 0).to(dtype=source.dtype)
                tensor_physical_target.append(
                    self._soft_physical_chroma(source[..., 0], target_tensor_active, float(self.config.pitch_scale)).cpu().numpy()
                )
                for sample_index in range(len(posterior_z_physical)):
                    posterior_z = model.reparameterize(encoder_mu, encoder_log_var)
                    posterior_z_physical[sample_index].append(
                        self._decoded_hard_physical_chroma(model, posterior_z).cpu().numpy()
                    )
                export_delta = latent - encoder_mu
                export_mu_squared_error += float(export_delta.square().sum().cpu())
                export_mu_value_count += int(export_delta.numel())
                export_mu_max_abs_error = max(export_mu_max_abs_error, float(export_delta.abs().max().cpu()))
            target = self._source_chroma_batch(rows[start:end], source_chroma)
            target_physical.append(target.cpu().numpy())
            decoded_physical_soft.append(decoded_soft.cpu().numpy())
            decoded_physical_hard.append(decoded_hard.cpu().numpy())
            target_head.append(source[..., 7:18].mean(dim=(1, 2)).cpu().numpy())
            decoded_head.append(chord[..., 2:13].mean(dim=(1, 2)).cpu().numpy())
            target_active.extend((target.sum(dim=1) > 0.0).cpu().tolist())
        physical_target = np.concatenate(target_physical, axis=0)
        physical_decoded_soft = np.concatenate(decoded_physical_soft, axis=0)
        physical_decoded_hard = np.concatenate(decoded_physical_hard, axis=0)
        head_target = np.concatenate(target_head, axis=0)
        head_decoded = np.concatenate(decoded_head, axis=0)
        tensor_target = np.concatenate(tensor_physical_target, axis=0)
        direct_mu = np.concatenate(direct_mu_physical, axis=0)
        posterior_samples = [np.concatenate(values, axis=0) for values in posterior_z_physical]
        active_mask = np.asarray(target_active, dtype=bool)
        physical_soft = self._vector_metrics(physical_target, physical_decoded_soft, active_mask)
        physical_hard = self._vector_metrics(physical_target, physical_decoded_hard, active_mask)
        auxiliary_head = self._vector_metrics(head_target, head_decoded, active_mask)
        transitions = self._transition_metrics(physical_target, physical_decoded_hard, active_mask, rows)
        worst = self._worst_rows(physical_target, physical_decoded_hard, active_mask, rows)
        posterior_gap = self._posterior_inference_gap(
            physical_target,
            tensor_target,
            direct_mu,
            posterior_samples,
            active_mask,
            rows,
            export_mu_squared_error,
            export_mu_value_count,
            export_mu_max_abs_error,
        )
        conclusion = self._conclusion(physical_hard, transitions, posterior_gap)
        report = {
            "backend": "dvae_harmony_oracle",
            "model_dir": str(model_dir),
            "dvae_path": str(dvae_path),
            "latent_dir": str(latent_dir),
            "encoded_dir": str(encoded_dir),
            "source_chroma": "duration-weighted 12-bin relative chroma from encoded/songs.json raw notes",
            "sample_count": int(len(rows)),
            "active_bar_count": int(active_mask.sum()),
            "config": {
                "batch_size": int(self.config.batch_size),
                "device": str(self.config.device),
                "pitch_scale": float(self.config.pitch_scale),
                "pitch_class_sigma": float(self.config.pitch_class_sigma),
                "posterior_samples": int(self.config.posterior_samples),
            },
            "latent_summary": latent_summary,
            "hard_physical_chroma": physical_hard,
            "soft_physical_chroma": physical_soft,
            "auxiliary_chord_head": auxiliary_head,
            "physical_chroma_transitions": transitions,
            "posterior_inference_gap": posterior_gap,
            "worst_bars": worst,
            "conclusion": conclusion,
            "output_dir": str(output_dir),
        }
        diagnostics_path = output_dir / "dvae_harmony_oracle_diagnostics.json"
        report_path = output_dir / "dvae_harmony_oracle_report.md"
        diagnostics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_path.write_text(self._markdown(report), encoding="utf-8")
        return report

    def _load_model(self, path: Path) -> DenoisingMusicVAE:
        if not path.exists():
            raise FileNotFoundError(f"Missing DVAE checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        model = DenoisingMusicVAE(DVAEMusicConfig(**checkpoint["config"])).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _tensor_batch(self, archive: Any, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        values: List[np.ndarray] = []
        for row in rows:
            key = str(row.get("tensor_key", ""))
            if key not in archive:
                raise KeyError(f"Missing tensor_key in bar_tensors.npz: {key}")
            values.append(np.asarray(archive[key], dtype=np.float32))
        return np.stack(values, axis=0)

    def _source_chroma_by_bar(self, songs_path: Path) -> Dict[tuple[str, int], np.ndarray]:
        """Rebuild true 12-bin relative chroma from uncompressed source notes."""
        if not songs_path.exists():
            raise FileNotFoundError(f"Missing source note metadata: {songs_path}")
        songs = json.loads(songs_path.read_text(encoding="utf-8"))
        values: Dict[tuple[str, int], np.ndarray] = {}
        for song in songs:
            for bar in song.get("bars", []):
                notes = [note for track in bar.get("tracks", []) for note in track.get("notes", [])]
                chroma = np.zeros(12, dtype=np.float32)
                if notes:
                    pitches = np.asarray([int(note["pitch"]) for note in notes], dtype=np.int32)
                    base_pitch = int(np.median(pitches))
                    for note in notes:
                        chroma[(int(note["pitch"]) - base_pitch) % 12] += max(0.0, float(note.get("duration_ql", 0.0)))
                    total = float(chroma.sum())
                    if total > 0.0:
                        chroma /= total
                values[(str(bar.get("song_id", song.get("song_id", "UNKNOWN"))), int(bar.get("bar_index", 0)))] = chroma
        return values

    def _source_chroma_batch(self, rows: Sequence[Dict[str, Any]], source_chroma: Dict[tuple[str, int], np.ndarray]) -> torch.Tensor:
        values: List[np.ndarray] = []
        for row in rows:
            key = (str(row.get("song_id", "UNKNOWN")), int(row.get("bar_index", 0)))
            if key not in source_chroma:
                raise KeyError(f"Missing source chroma for song/bar: {key}")
            values.append(source_chroma[key])
        return torch.from_numpy(np.stack(values, axis=0)).to(self.config.device)

    def _soft_physical_chroma(self, pitch: torch.Tensor, active: torch.Tensor, pitch_scale: float) -> torch.Tensor:
        """Pool predicted physical pitches into a differentiable relative 12-bin chroma."""
        semitones = pitch * float(pitch_scale)
        pitch_classes = torch.arange(12, device=pitch.device, dtype=pitch.dtype)
        distance = torch.remainder(semitones.unsqueeze(-1) - pitch_classes + 6.0, 12.0) - 6.0
        logits = -0.5 * (distance / float(self.config.pitch_class_sigma)).square()
        membership = torch.softmax(logits, dim=-1)
        chroma = torch.sum(membership * active.unsqueeze(-1), dim=(1, 2))
        return chroma / chroma.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _decoded_hard_physical_chroma(self, model: DenoisingMusicVAE, latent: torch.Tensor) -> torch.Tensor:
        """Decode one latent batch through the same hard state path used for MIDI rendering."""
        pitch, state_logits, _velocity, _chord = model.decoder(latent)
        active = (torch.argmax(state_logits, dim=-1) != 0).to(dtype=pitch.dtype)
        return self._soft_physical_chroma(pitch, active, float(self.config.pitch_scale))

    def _posterior_inference_gap(
        self,
        raw_target: np.ndarray,
        tensor_target: np.ndarray,
        mu_decoded: np.ndarray,
        posterior_samples: Sequence[np.ndarray],
        active_mask: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        export_mu_squared_error: float,
        export_mu_value_count: int,
        export_mu_max_abs_error: float,
    ) -> Dict[str, Any]:
        """Compare the training posterior path to inference-time deterministic mu."""
        mu_raw_metrics = self._vector_metrics(raw_target, mu_decoded, active_mask)
        mu_tensor_metrics = self._vector_metrics(tensor_target, mu_decoded, active_mask)
        mu_raw_transitions = self._transition_metrics(raw_target, mu_decoded, active_mask, rows)
        mu_tensor_transitions = self._transition_metrics(tensor_target, mu_decoded, active_mask, rows)
        posterior_raw = [self._vector_metrics(raw_target, sample, active_mask) for sample in posterior_samples]
        posterior_tensor = [self._vector_metrics(tensor_target, sample, active_mask) for sample in posterior_samples]
        posterior_raw_transitions = [self._transition_metrics(raw_target, sample, active_mask, rows) for sample in posterior_samples]
        posterior_tensor_transitions = [self._transition_metrics(tensor_target, sample, active_mask, rows) for sample in posterior_samples]
        posterior_raw_summary = self._aggregate_metric_samples(posterior_raw)
        posterior_tensor_summary = self._aggregate_metric_samples(posterior_tensor)
        posterior_raw_transition_summary = self._aggregate_metric_samples(posterior_raw_transitions)
        posterior_tensor_transition_summary = self._aggregate_metric_samples(posterior_tensor_transitions)
        return {
            "posterior_samples": int(len(posterior_samples)),
            "latent_export_consistency": {
                "mu_mse": float(export_mu_squared_error / max(1, export_mu_value_count)),
                "mu_max_abs_error": float(export_mu_max_abs_error),
            },
            "deterministic_mu": {
                "raw_source_chroma": mu_raw_metrics,
                "tensor_physical_chroma": mu_tensor_metrics,
                "raw_source_transitions": mu_raw_transitions,
                "tensor_physical_transitions": mu_tensor_transitions,
            },
            "posterior_z": {
                "raw_source_chroma": posterior_raw_summary,
                "tensor_physical_chroma": posterior_tensor_summary,
                "raw_source_transitions": posterior_raw_transition_summary,
                "tensor_physical_transitions": posterior_tensor_transition_summary,
            },
            "z_minus_mu": {
                "raw_source_chroma_cosine_mean": float(posterior_raw_summary["mean"]["cosine_mean"] - mu_raw_metrics["cosine_mean"]),
                "tensor_physical_chroma_cosine_mean": float(posterior_tensor_summary["mean"]["cosine_mean"] - mu_tensor_metrics["cosine_mean"]),
                "raw_source_transition_cosine_mean": float(posterior_raw_transition_summary["mean"]["cosine_mean"] - mu_raw_transitions["cosine_mean"]),
                "tensor_physical_transition_cosine_mean": float(posterior_tensor_transition_summary["mean"]["cosine_mean"] - mu_tensor_transitions["cosine_mean"]),
            },
        }

    def _aggregate_metric_samples(self, samples: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Return mean and standard deviation for metrics over posterior draws."""
        if not samples:
            return {"mean": {}, "std": {}}
        keys = samples[0].keys()
        return {
            "mean": {key: float(np.mean([sample[key] for sample in samples])) for key in keys},
            "std": {key: float(np.std([sample[key] for sample in samples])) for key in keys},
        }

    def _vector_metrics(self, target: np.ndarray, decoded: np.ndarray, active_mask: np.ndarray) -> Dict[str, float]:
        target = target[active_mask]
        decoded = decoded[active_mask]
        cosine = self._cosine(target, decoded)
        return {
            "mse": float(np.mean(np.square(decoded - target))),
            "cosine_mean": float(np.mean(cosine)),
            "cosine_p10": float(np.quantile(cosine, 0.10)),
            "cosine_p50": float(np.quantile(cosine, 0.50)),
            "target_std": float(np.std(target)),
            "decoded_std": float(np.std(decoded)),
            "std_ratio": float(np.std(decoded) / max(float(np.std(target)), 1.0e-8)),
            "mean_l2": float(np.mean(np.linalg.norm(decoded - target, axis=1))),
        }

    def _transition_metrics(
        self, target: np.ndarray, decoded: np.ndarray, active_mask: np.ndarray, rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        target_delta: List[np.ndarray] = []
        decoded_delta: List[np.ndarray] = []
        for indices in grouped.values():
            ordered = sorted(indices, key=lambda index: int(rows[index].get("bar_index", index)))
            for left, right in zip(ordered, ordered[1:]):
                if int(rows[right].get("bar_index", right)) != int(rows[left].get("bar_index", left)) + 1:
                    continue
                if active_mask[left] and active_mask[right]:
                    target_delta.append(target[right] - target[left])
                    decoded_delta.append(decoded[right] - decoded[left])
        if not target_delta:
            return {"pair_count": 0, "mse": 0.0, "cosine_mean": 0.0, "target_delta_norm": 0.0, "decoded_delta_norm": 0.0}
        target_array = np.stack(target_delta)
        decoded_array = np.stack(decoded_delta)
        return {
            "pair_count": int(len(target_array)),
            "mse": float(np.mean(np.square(decoded_array - target_array))),
            "cosine_mean": float(np.mean(self._cosine(target_array, decoded_array))),
            "target_delta_norm": float(np.mean(np.linalg.norm(target_array, axis=1))),
            "decoded_delta_norm": float(np.mean(np.linalg.norm(decoded_array, axis=1))),
        }

    def _worst_rows(self, target: np.ndarray, decoded: np.ndarray, active_mask: np.ndarray, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cosine = self._cosine(target, decoded)
        order = np.argsort(cosine)
        result: List[Dict[str, Any]] = []
        for index in order:
            if not active_mask[index]:
                continue
            row = rows[int(index)]
            result.append({
                "song_id": str(row.get("song_id", "UNKNOWN")),
                "bar_index": int(row.get("bar_index", index)),
                "tensor_key": str(row.get("tensor_key", "")),
                "cosine": float(cosine[index]),
                "target_chroma": [float(value) for value in target[index]],
                "decoded_chroma": [float(value) for value in decoded[index]],
            })
            if len(result) >= 20:
                break
        return result

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return numerator / np.maximum(denominator, 1.0e-8)

    def _conclusion(
        self,
        physical: Dict[str, float],
        transitions: Dict[str, float],
        posterior_gap: Dict[str, Any],
    ) -> Dict[str, Any]:
        recovered = physical["cosine_mean"] >= 0.85 and physical["std_ratio"] >= 0.70
        substantially_recovered = physical["cosine_mean"] >= 0.80 and physical["std_ratio"] >= 0.60
        transition_recovered = transitions["cosine_mean"] >= 0.50 and transitions["decoded_delta_norm"] >= transitions["target_delta_norm"] * 0.60
        if recovered and transition_recovered:
            diagnosis = "DVAE oracle retains harmonic state and movement; the trajectory model is the primary failure point."
        elif substantially_recovered and transition_recovered:
            diagnosis = "DVAE now retains harmonic identity and movement, with remaining Chroma amplitude contraction. The trajectory model can be retrained on this representation before further DVAE tuning."
        elif recovered:
            diagnosis = "DVAE retains individual bar harmony but weakens harmonic movement; assess trajectory targets before changing DVAE architecture."
        else:
            diagnosis = "Frozen DVAE latent means do not reliably preserve physical chroma; repair or reweight DVAE representation before attributing the failure to trajectory prediction."
        z_gap = float(posterior_gap["z_minus_mu"]["raw_source_chroma_cosine_mean"])
        if z_gap >= 0.03:
            inference_diagnosis = "Posterior z is materially better than deterministic mu; the DVAE has a train/inference latent-path mismatch."
        elif z_gap <= -0.03:
            inference_diagnosis = "Deterministic mu is materially better than posterior z; sampling noise is not the source of the reconstruction failure."
        else:
            inference_diagnosis = "Posterior z and deterministic mu are similar; the reconstruction failure is not explained by a train/inference latent-path mismatch."
        return {
            "physical_state_recovered": recovered,
            "physical_state_substantially_recovered": substantially_recovered,
            "physical_transition_recovered": transition_recovered,
            "diagnosis": diagnosis,
            "posterior_inference_diagnosis": inference_diagnosis,
        }

    def _markdown(self, report: Dict[str, Any]) -> str:
        physical = report["hard_physical_chroma"]
        soft_physical = report["soft_physical_chroma"]
        transitions = report["physical_chroma_transitions"]
        auxiliary = report["auxiliary_chord_head"]
        posterior = report["posterior_inference_gap"]
        conclusion = report["conclusion"]
        return "\n".join([
            "# DVAE Harmony Oracle Report",
            "",
            "This probe decodes real `latent_mu` values with the frozen DVAE and compares physical pitch-class chroma to duration-weighted source notes.",
            "",
            "## Hard Physical Chroma",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in physical.items()],
            "",
            "## Soft Physical Chroma",
            "",
            "This is the differentiable pitch/state surrogate suitable for a future decoded-harmony training loss.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in soft_physical.items()],
            "",
            "## Harmonic Transitions",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in transitions.items()],
            "",
            "## Auxiliary Chord Head",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in auxiliary.items()],
            "",
            "## Posterior z vs Deterministic mu",
            "",
            "This compares the posterior samples used during DVAE training with the deterministic `mu` exported for downstream generation.",
            "",
            "| Metric | mu (raw source) | posterior z mean | z - mu |",
            "| --- | ---: | ---: | ---: |",
            f"| Chroma cosine mean | {posterior['deterministic_mu']['raw_source_chroma']['cosine_mean']:.6f} | {posterior['posterior_z']['raw_source_chroma']['mean']['cosine_mean']:.6f} | {posterior['z_minus_mu']['raw_source_chroma_cosine_mean']:.6f} |",
            f"| Transition cosine mean | {posterior['deterministic_mu']['raw_source_transitions']['cosine_mean']:.6f} | {posterior['posterior_z']['raw_source_transitions']['mean']['cosine_mean']:.6f} | {posterior['z_minus_mu']['raw_source_transition_cosine_mean']:.6f} |",
            f"| Tensor-target Chroma cosine | {posterior['deterministic_mu']['tensor_physical_chroma']['cosine_mean']:.6f} | {posterior['posterior_z']['tensor_physical_chroma']['mean']['cosine_mean']:.6f} | {posterior['z_minus_mu']['tensor_physical_chroma_cosine_mean']:.6f} |",
            f"| Tensor-target transition cosine | {posterior['deterministic_mu']['tensor_physical_transitions']['cosine_mean']:.6f} | {posterior['posterior_z']['tensor_physical_transitions']['mean']['cosine_mean']:.6f} | {posterior['z_minus_mu']['tensor_physical_transition_cosine_mean']:.6f} |",
            "",
            f"Exported mu consistency MSE: `{posterior['latent_export_consistency']['mu_mse']:.8f}`  ",
            f"Exported mu max absolute error: `{posterior['latent_export_consistency']['mu_max_abs_error']:.8f}`  ",
            f"Posterior sample count: `{posterior['posterior_samples']}`",
            "",
            "## Conclusion",
            "",
            conclusion["diagnosis"],
            "",
            conclusion["posterior_inference_diagnosis"],
            "",
            f"Physical harmonic state recovered: `{conclusion['physical_state_recovered']}`  ",
            f"Physical harmonic state substantially recovered: `{conclusion['physical_state_substantially_recovered']}`  ",
            f"Physical harmonic transitions recovered: `{conclusion['physical_transition_recovered']}`",
        ]) + "\n"
