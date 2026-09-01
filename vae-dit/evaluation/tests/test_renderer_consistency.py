from __future__ import annotations

import sys
from pathlib import Path

import mido
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.core.artifacts import VerifiedArtifactResolver
from evaluation_framework.evaluation_renderer_consistency import _markdown, _measure


def test_renderer_metrics_decode_pitch_through_the_declared_schema(tmp_path: Path) -> None:
    archive_path = tmp_path / "renderer_arrays.npz"
    bars = np.zeros((1, 1, 1, 5), dtype=np.float32)
    bars[..., 0] = 0.5
    bars[..., 2] = 1.0
    np.savez_compressed(archive_path, bar_tensors=bars, render_base_pitches=np.asarray([60.0], dtype=np.float32))
    midi_path = tmp_path / "rendered.mid"
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", note=72, velocity=80, time=0))
    track.append(mido.Message("note_off", note=72, velocity=0, time=120))
    midi.save(midi_path)
    observation = {
        "tensor_schema": {
            "schema_version": "bar_tensor_schema.v1",
            "axis_order": ["bar", "track", "step", "feature"],
            "feature_names": ["relative_pitch", "is_rest", "is_note_on", "is_hold", "velocity"],
            "track_names": ["melody"],
            "pitch_scale_semitones": 24.0,
        },
        "midi": _reference(midi_path),
        "bar_alignment": [{"bar_index": 0, "start_tick": 0, "end_tick": 480}],
    }

    arrays = np.load(archive_path, allow_pickle=False)
    try:
        metrics = _measure(observation, arrays, VerifiedArtifactResolver(tmp_path))
    finally:
        arrays.close()

    assert metrics["tensor_to_midi"]["chroma_cosine_mean"] == 1.0
    assert metrics["tensor_to_midi"]["register_median_absolute_error_semitones"] == 0.0


def test_renderer_unavailable_markdown_is_utf8_chinese_text() -> None:
    markdown = _markdown({"status": "UNAVAILABLE", "missing_inputs": [{"reason": "fixture"}]})

    assert markdown.startswith("# Renderer 一致性")
    assert "原始观察数据不可用" in markdown


def _reference(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": VerifiedArtifactResolver.sha256(path)}
