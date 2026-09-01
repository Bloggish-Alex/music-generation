from __future__ import annotations

import numpy as np

from evaluation_framework.evaluation_trajectory_history_swap import _measure


def test_history_swap_reports_condition_specific_predictions() -> None:
    tensors = np.zeros((3, 1, 1, 1, 5), dtype=np.float32)
    tensors[:, 0, 0, 0, 2] = 1.0
    tensors[1, 0, 0, 0, 0] = 0.25
    tensors[2, 0, 0, 0, 0] = 0.5
    result = _measure({
        "variant_names": np.asarray(["H1_real", "H2_transposed_plus_6", "H3_alternate_song"]),
        "prediction_states": np.asarray([[[1.0, 0.0]], [[2.0, 3.0]], [[1.0, 1.0]]], dtype=np.float32),
        "predicted_bar_tensors": tensors,
        "predicted_render_base_pitches": np.asarray([[60.0], [66.0], [55.0]], dtype=np.float32),
        "target_bar_tensor": tensors[0, 0],
        "target_render_base_pitch": np.asarray(60.0, dtype=np.float32),
        "target_source_stream_index": np.asarray(12, dtype=np.int64),
        "history_song_ids": np.asarray(["song-a", "song-a", "song-b"]),
        "history_bar_tensors": np.zeros((3, 2, 1, 1, 5), dtype=np.float32),
        "boundary_source_song_id": np.asarray("song-a"),
        "primer_bars": np.asarray(2, dtype=np.int64),
        "sampling_seed": np.asarray(7936, dtype=np.int64),
        "sampling_seed_algorithm": np.asarray("torch_global_seed.v1"),
        "sampling_seed_offset": np.asarray(7919, dtype=np.int64),
        "theme_memory_condition": np.asarray("variant_history"),
        "codec": {"pitch": {"pitch_scale": 24.0}},
    })
    variants = result["variants"]
    assert variants["H1_real"]["latent_l2_from_H1"] == 0.0
    assert variants["H2_transposed_plus_6"]["register_change_from_H1_semitones"] == 6.0
    assert variants["H3_alternate_song"]["absolute_register_error_to_target_semitones"] == -5.0
    assert result["experiment_context"]["sample_count"] == 1
    assert result["experiment_context"]["theme_memory_condition"] == "variant_history"
    assert result["experiment_context"]["sampling_seed"] == 7936
