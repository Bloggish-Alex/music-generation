#!/usr/bin/env python3
"""Measure harmonic information retained by the encoded semantic bar tensor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from codec.relative_chroma import relative_chromagram


@dataclass(frozen=True)
class BarCodecHarmonyOracleConfig:
    """Inputs and numerical settings for the codec-level harmony probe."""

    model_dir: Path
    encoded_dir: Path | None = None
    output_dir: Path | None = None
    pitch_scale: float = 24.0
    pitch_class_sigma: float = 0.35
    max_rows: int | None = None


class BarCodecHarmonyOracleAnalyzer:
    """Compare source-note chroma with semantic tensor representations."""

    def __init__(self, config: BarCodecHarmonyOracleConfig) -> None:
        """Store input paths and numerical reconstruction settings."""
        self.config = config

    def run(self) -> Dict[str, Any]:
        """Compare all indexed bars and write JSON and Markdown reports."""
        model_dir = Path(self.config.model_dir)
        encoded_dir = Path(self.config.encoded_dir) if self.config.encoded_dir else model_dir / "encoded"
        output_dir = Path(self.config.output_dir) if self.config.output_dir else model_dir / "bar_codec_harmony_oracle"
        output_dir.mkdir(parents=True, exist_ok=True)

        tensor_path = encoded_dir / "bar_tensors.npz"
        index_path = encoded_dir / "bar_tensor_index.json"
        songs_path = encoded_dir / "songs.json"
        for path in (tensor_path, index_path, songs_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing codec oracle input: {path}")

        rows = json.loads(index_path.read_text(encoding="utf-8"))
        if self.config.max_rows is not None:
            rows = rows[: min(len(rows), int(self.config.max_rows))]
        source_chroma = self._source_chroma_by_bar(songs_path)

        raw_target: List[np.ndarray] = []
        semantic_physical: List[np.ndarray] = []
        chroma_embedding: List[np.ndarray] = []
        active: List[bool] = []
        archive = np.load(tensor_path)
        try:
            for row in rows:
                tensor = self._tensor_for_row(archive, row)
                target = self._source_for_row(source_chroma, row)
                raw_target.append(target)
                semantic_physical.append(self._semantic_physical_chroma(tensor))
                chroma_embedding.append(self._expanded_chroma_embedding(tensor))
                active.append(bool(target.sum() > 0.0))
        finally:
            archive.close()

        target_array = np.stack(raw_target, axis=0)
        physical_array = np.stack(semantic_physical, axis=0)
        embedding_array = np.stack(chroma_embedding, axis=0)
        active_mask = np.asarray(active, dtype=bool)

        physical_metrics = self._vector_metrics(target_array, physical_array, active_mask)
        embedding_metrics = self._vector_metrics(target_array, embedding_array, active_mask)
        physical_transitions = self._transition_metrics(target_array, physical_array, active_mask, rows)
        embedding_transitions = self._transition_metrics(target_array, embedding_array, active_mask, rows)
        report = {
            "backend": "bar_codec_harmony_oracle",
            "model_dir": str(model_dir),
            "encoded_dir": str(encoded_dir),
            "source_chroma": "duration-and-velocity-weighted 12-bin relative chroma from encoded/songs.json raw notes",
            "semantic_physical_chroma": "active Melody/Harmony/Bass tensor pitches pooled by 16-slot duration",
            "tensor_chroma_embedding": "11D relative chroma embedding expanded back to 12 bins",
            "sample_count": int(len(rows)),
            "active_bar_count": int(active_mask.sum()),
            "config": {
                "pitch_scale": float(self.config.pitch_scale),
                "pitch_class_sigma": float(self.config.pitch_class_sigma),
            },
            "semantic_physical_chroma_metrics": physical_metrics,
            "tensor_chroma_embedding_metrics": embedding_metrics,
            "semantic_physical_chroma_transitions": physical_transitions,
            "tensor_chroma_embedding_transitions": embedding_transitions,
            "worst_semantic_physical_bars": self._worst_rows(target_array, physical_array, active_mask, rows),
            "conclusion": self._conclusion(physical_metrics, embedding_metrics, physical_transitions),
            "output_dir": str(output_dir),
        }
        diagnostics_path = output_dir / "bar_codec_harmony_oracle_diagnostics.json"
        report_path = output_dir / "bar_codec_harmony_oracle_report.md"
        diagnostics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_path.write_text(self._markdown(report), encoding="utf-8")
        return report

    def _tensor_for_row(self, archive: Any, row: Dict[str, Any]) -> np.ndarray:
        """Load one `float32[3, 16, 18]` tensor by its index-row key."""
        key = str(row.get("tensor_key", ""))
        if key not in archive.files:
            raise KeyError(f"Missing tensor_key in bar_tensors.npz: {key}")
        return np.asarray(archive[key], dtype=np.float32)

    def _source_chroma_by_bar(self, songs_path: Path) -> Dict[tuple[str, int], np.ndarray]:
        """Build duration-and-velocity-weighted source chroma for every bar."""
        songs = json.loads(songs_path.read_text(encoding="utf-8"))
        values: Dict[tuple[str, int], np.ndarray] = {}
        for song in songs:
            for bar in song.get("bars", []):
                notes = [note for track in bar.get("tracks", []) for note in track.get("notes", [])]
                chroma = relative_chromagram(notes)
                values[(str(bar.get("song_id", song.get("song_id", "UNKNOWN"))), int(bar.get("bar_index", 0)))] = chroma
        return values

    def _source_for_row(self, source_chroma: Dict[tuple[str, int], np.ndarray], row: Dict[str, Any]) -> np.ndarray:
        """Return the source `float32[12]` chroma matching one index row."""
        key = (str(row.get("song_id", "UNKNOWN")), int(row.get("bar_index", 0)))
        if key not in source_chroma:
            raise KeyError(f"Missing source chroma for song/bar: {key}")
        return source_chroma[key]

    def _semantic_physical_chroma(self, tensor: np.ndarray) -> np.ndarray:
        """Reconstruct the Chroma audible from the three semantic pitch tracks."""
        pitches = np.asarray(tensor[..., 0], dtype=np.float32) * float(self.config.pitch_scale)
        active = (np.asarray(tensor[..., 2]) > 0.5) | (np.asarray(tensor[..., 3]) > 0.5)
        if not bool(active.any()):
            return np.zeros(12, dtype=np.float32)
        pitch_classes = np.arange(12, dtype=np.float32)
        distance = np.remainder(pitches[..., None] - pitch_classes + 6.0, 12.0) - 6.0
        logits = -0.5 * np.square(distance / float(self.config.pitch_class_sigma))
        logits -= np.max(logits, axis=-1, keepdims=True)
        membership = np.exp(logits)
        membership /= np.maximum(membership.sum(axis=-1, keepdims=True), 1.0e-8)
        chroma = np.sum(membership * active[..., None], axis=(0, 1))
        return (chroma / max(float(chroma.sum()), 1.0e-8)).astype(np.float32)

    def _expanded_chroma_embedding(self, tensor: np.ndarray) -> np.ndarray:
        """Invert the deterministic 12-to-11 Chroma projection as far as possible."""
        embedding = np.asarray(tensor[..., 7:18], dtype=np.float32).mean(axis=(0, 1))
        chroma = np.zeros(12, dtype=np.float32)
        chroma[:10] = embedding[:10]
        chroma[10] = embedding[10]
        chroma[11] = embedding[10]
        return (chroma / max(float(chroma.sum()), 1.0e-8)).astype(np.float32)

    def _vector_metrics(self, target: np.ndarray, candidate: np.ndarray, active_mask: np.ndarray) -> Dict[str, float]:
        """Measure reconstruction accuracy over bars with non-empty chroma."""
        target = target[active_mask]
        candidate = candidate[active_mask]
        cosine = self._cosine(target, candidate)
        return {
            "mse": float(np.mean(np.square(candidate - target))),
            "cosine_mean": float(np.mean(cosine)),
            "cosine_p10": float(np.quantile(cosine, 0.10)),
            "cosine_p50": float(np.quantile(cosine, 0.50)),
            "target_std": float(np.std(target)),
            "decoded_std": float(np.std(candidate)),
            "std_ratio": float(np.std(candidate) / max(float(np.std(target)), 1.0e-8)),
            "mean_l2": float(np.mean(np.linalg.norm(candidate - target, axis=1))),
        }

    def _transition_metrics(
        self, target: np.ndarray, candidate: np.ndarray, active_mask: np.ndarray, rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Measure consecutive within-song chroma movement retention."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        target_delta: List[np.ndarray] = []
        candidate_delta: List[np.ndarray] = []
        for indices in grouped.values():
            ordered = sorted(indices, key=lambda index: int(rows[index].get("bar_index", index)))
            for left, right in zip(ordered, ordered[1:]):
                if int(rows[right].get("bar_index", right)) != int(rows[left].get("bar_index", left)) + 1:
                    continue
                if active_mask[left] and active_mask[right]:
                    target_delta.append(target[right] - target[left])
                    candidate_delta.append(candidate[right] - candidate[left])
        if not target_delta:
            return {"pair_count": 0, "mse": 0.0, "cosine_mean": 0.0, "target_delta_norm": 0.0, "decoded_delta_norm": 0.0}
        target_array = np.stack(target_delta)
        candidate_array = np.stack(candidate_delta)
        return {
            "pair_count": int(len(target_array)),
            "mse": float(np.mean(np.square(candidate_array - target_array))),
            "cosine_mean": float(np.mean(self._cosine(target_array, candidate_array))),
            "target_delta_norm": float(np.mean(np.linalg.norm(target_array, axis=1))),
            "decoded_delta_norm": float(np.mean(np.linalg.norm(candidate_array, axis=1))),
        }

    def _worst_rows(self, target: np.ndarray, candidate: np.ndarray, active_mask: np.ndarray, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return up to twenty active bars with the lowest cosine score."""
        cosine = self._cosine(target, candidate)
        result: List[Dict[str, Any]] = []
        for index in np.argsort(cosine):
            if not active_mask[int(index)]:
                continue
            row = rows[int(index)]
            result.append({
                "song_id": str(row.get("song_id", "UNKNOWN")),
                "bar_index": int(row.get("bar_index", index)),
                "tensor_key": str(row.get("tensor_key", "")),
                "cosine": float(cosine[index]),
                "target_chroma": [float(value) for value in target[index]],
                "semantic_physical_chroma": [float(value) for value in candidate[index]],
            })
            if len(result) == 20:
                break
        return result

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return row-wise cosine similarity with a zero-norm guard."""
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return numerator / np.maximum(denominator, 1.0e-8)

    def _conclusion(
        self,
        physical: Dict[str, float],
        embedding: Dict[str, float],
        physical_transitions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Classify whether harmony is retained as condition, sound, and motion."""
        embedding_retained = embedding["cosine_mean"] >= 0.95
        physical_retained = physical["cosine_mean"] >= 0.85 and physical["std_ratio"] >= 0.70
        movement_retained = physical_transitions["cosine_mean"] >= 0.50
        if embedding_retained and not physical_retained:
            diagnosis = "The tensor retains Chroma conditioning, but semantic Melody/Harmony/Bass pitches lose source polyphony before DVAE. A pitch/state decoder cannot recover full source Chroma without using the retained Chroma condition."
        elif not embedding_retained:
            diagnosis = "The deterministic tensor Chroma embedding itself loses substantial source-harmony information; revise the codec representation before changing DVAE capacity."
        elif physical_retained and not movement_retained:
            diagnosis = "The codec preserves per-bar harmony but weakens harmonic transitions; downstream motion learning needs an explicit harmonic trajectory target."
        else:
            diagnosis = "The semantic codec retains physical harmony and movement; DVAE reconstruction is the primary remaining bottleneck."
        return {
            "tensor_chroma_condition_retained": embedding_retained,
            "semantic_physical_harmony_retained": physical_retained,
            "semantic_physical_harmony_movement_retained": movement_retained,
            "diagnosis": diagnosis,
        }

    def _markdown(self, report: Dict[str, Any]) -> str:
        """Render the main oracle metrics and conclusion as Markdown."""
        physical = report["semantic_physical_chroma_metrics"]
        embedding = report["tensor_chroma_embedding_metrics"]
        physical_transitions = report["semantic_physical_chroma_transitions"]
        embedding_transitions = report["tensor_chroma_embedding_transitions"]
        conclusion = report["conclusion"]
        return "\n".join([
            "# BarCodec Harmony Oracle Report",
            "",
            "This probe separates Chroma stored in the semantic tensor from Chroma reconstructible through its three pitch/state tracks.",
            "",
            "## Semantic Physical Chroma",
            "",
            "This is the audible Chroma recoverable from Melody/Harmony/Bass relative-pitch and active-state slots.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in physical.items()],
            "",
            "## Tensor Chroma Condition",
            "",
            "This expands the stored deterministic 11D relative-Chroma embedding back to 12 bins.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in embedding.items()],
            "",
            "## Semantic Physical Chroma Transitions",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in physical_transitions.items()],
            "",
            "## Tensor Chroma Condition Transitions",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            *[f"| {key} | {value:.6f} |" for key, value in embedding_transitions.items()],
            "",
            "## Conclusion",
            "",
            conclusion["diagnosis"],
            "",
            f"Tensor Chroma condition retained: `{conclusion['tensor_chroma_condition_retained']}`  ",
            f"Semantic physical harmony retained: `{conclusion['semantic_physical_harmony_retained']}`  ",
            f"Semantic physical harmony movement retained: `{conclusion['semantic_physical_harmony_movement_retained']}`",
        ]) + "\n"
