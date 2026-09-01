from __future__ import annotations

import numpy as np

from evaluation_framework.evaluation_trajectory_reference_frame import _measure_first_future_register


def test_reference_frame_decomposition_separates_native_anchor_gap() -> None:
    data = {
        "predicted_render_base_pitches": np.asarray([[70.0], [64.0]], dtype=np.float32),
        "context_render_base_pitches": np.asarray([59.0, 61.0], dtype=np.float32),
        "source_render_base_pitches": np.asarray([60.0, 62.0, 65.0], dtype=np.float32),
        "committed_source_stream_indices": np.asarray([1, 2], dtype=np.int64),
        "target_source_stream_indices": np.asarray([[1], [2]], dtype=np.int64),
    }

    result = _measure_first_future_register(data)

    # Absolute errors are +8 and -1. The first native history anchor is one
    # semitone below the true history, so its native-coordinate error is +9.
    assert result["valid_samples"] == 2
    assert result["absolute_register_rmse_semitones"] == np.sqrt((64.0 + 1.0) / 2.0)
    assert result["common_anchor_delta_rmse_semitones"] == result["absolute_register_rmse_semitones"]
    assert result["native_anchor_delta_rmse_semitones"] == np.sqrt(81.0 / 2.0)
    assert result["history_anchor_gap_rmse_semitones"] == 1.0
    assert result["decomposition_residual_rmse_semitones"] == 0.0


def test_reference_frame_requires_true_history_before_boundary() -> None:
    result = _measure_first_future_register({
        "predicted_render_base_pitches": np.asarray([[60.0]], dtype=np.float32),
        "context_render_base_pitches": np.asarray([60.0], dtype=np.float32),
        "source_render_base_pitches": np.asarray([60.0], dtype=np.float32),
        "committed_source_stream_indices": np.asarray([0], dtype=np.int64),
        "target_source_stream_indices": np.asarray([[0]], dtype=np.int64),
    })
    assert result == {"valid_samples": 0}
