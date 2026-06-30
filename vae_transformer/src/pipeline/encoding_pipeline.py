#!/usr/bin/env python3
"""Pipeline that parses music, encodes bar tensors, and labels actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from codec.action_labeler import ActionLabeler
from codec.bar_tensor_codec import BarTensorCodec
from data.core import BarTensorRecord, SongRecord
from data.music_parser import MusicDirectoryParser
from diagnostics.diagnostics import DiagnosticsBase


@dataclass
class EncodingPipelineResult:
    """All outputs produced by the encoding pipeline."""

    songs: List[SongRecord]
    tensors: List[BarTensorRecord]
    diagnostics: Dict[str, Any]


class EncodingPipeline:
    """Parse files, encode bars into tensors, label actions, and save outputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.diagnostics = DiagnosticsBase("encoding")

    def run(self, music_dir: str | Path, output_dir: str | Path) -> EncodingPipelineResult:
        """Run the encoding pipeline and write artifacts to output_dir."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        parser = MusicDirectoryParser.from_config(self.config)
        songs = parser.parse_directory(music_dir)
        self.diagnostics.record_stage("input", {
            "music_dir": str(music_dir),
            "parsed_file_count": int(len(songs)),
            "failed_file_count": int(len(parser.failed_files)),
            "failed_files": parser.failed_files,
            "bar_count": int(sum(len(song.bars) for song in songs)),
        })
        labeler = ActionLabeler.from_config(self.config)
        action_diagnostics = [labeler.label_song(song) for song in songs]
        self.diagnostics.record_stage("action_labeling", {
            "song_count": int(len(action_diagnostics)),
            "songs": action_diagnostics,
            "global_action_counts": self._global_action_counts(action_diagnostics),
        })
        codec = BarTensorCodec.from_config(self.config)
        tensors = self._encode_tensors(codec, songs)
        self.diagnostics.record_stage("bar_tensor_encoding", self._tensor_diagnostics(tensors))
        self._write_outputs(output_path, songs, tensors)
        diagnostics = self.diagnostics.to_dict()
        self.diagnostics.write(output_path / "encoding_diagnostics.json")
        return EncodingPipelineResult(songs=songs, tensors=tensors, diagnostics=diagnostics)

    def _encode_tensors(self, codec: BarTensorCodec, songs: List[SongRecord]) -> List[BarTensorRecord]:
        """Encode every parsed bar with diagnostics."""
        records: List[BarTensorRecord] = []
        for song in songs:
            for bar in song.bars:
                records.append(codec.encode(bar))
        return records

    def _write_outputs(self, output_dir: Path, songs: List[SongRecord], tensors: List[BarTensorRecord]) -> None:
        """Write JSON metadata and compressed tensor artifacts."""
        (output_dir / "songs.json").write_text(
            json.dumps([song.to_dict() for song in songs], indent=2),
            encoding="utf-8",
        )
        arrays = {
            f"{record.song_id}__bar_{record.bar_index:04d}": record.tensor
            for record in tensors
        }
        if arrays:
            np.savez_compressed(output_dir / "bar_tensors.npz", **arrays)
        else:
            np.savez_compressed(output_dir / "bar_tensors.npz")
        index_rows = [
            {
                "tensor_key": f"{record.song_id}__bar_{record.bar_index:04d}",
                "song_id": record.song_id,
                "bar_index": int(record.bar_index),
                "tensor_shape": record.tensor_shape,
                "diagnostics": record.diagnostics,
            }
            for record in tensors
        ]
        (output_dir / "bar_tensor_index.json").write_text(
            json.dumps(index_rows, indent=2),
            encoding="utf-8",
        )

    def _tensor_diagnostics(self, tensors: List[BarTensorRecord]) -> Dict[str, Any]:
        """Summarize tensor output shapes and density."""
        shapes = [tuple(record.tensor_shape) for record in tensors]
        nonzero_ratios = [
            float(np.count_nonzero(record.tensor) / max(1, int(np.prod(record.tensor.shape))))
            for record in tensors
        ]
        return {
            "bar_count": int(len(tensors)),
            "unique_shapes": [list(shape) for shape in sorted(set(shapes))],
            "mean_nonzero_ratio": float(np.mean(nonzero_ratios)) if nonzero_ratios else 0.0,
            "min_nonzero_ratio": float(np.min(nonzero_ratios)) if nonzero_ratios else 0.0,
            "max_nonzero_ratio": float(np.max(nonzero_ratios)) if nonzero_ratios else 0.0,
        }

    def _global_action_counts(self, action_diagnostics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate action counts across songs."""
        counts: Dict[str, int] = {}
        for song_diag in action_diagnostics:
            for action, count in song_diag.get("action_counts", {}).items():
                counts[str(action)] = counts.get(str(action), 0) + int(count)
        return counts
