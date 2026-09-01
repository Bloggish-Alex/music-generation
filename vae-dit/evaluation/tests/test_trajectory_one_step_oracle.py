from __future__ import annotations

import numpy as np

from evaluation_framework.evaluation_trajectory_one_step_oracle import _measure


def test_one_step_oracle_compares_only_first_future_position() -> None:
    source = np.zeros((2, 1, 1, 5), dtype=np.float32)
    source[..., 2] = 1.0
    prediction = source[:, None].copy()
    prediction[1, 0, 0, 0, 0] = 0.5
    result = _measure({
        "target_source_stream_indices": np.asarray([[0], [1]], dtype=np.int64),
        "source_bar_tensors": source,
        "source_render_base_pitches": np.asarray([60.0, 62.0], dtype=np.float32),
        "predicted_bar_tensors": prediction,
        "predicted_render_base_pitches": np.asarray([[60.0], [62.0]], dtype=np.float32),
        "codec": {"pitch": {"pitch_scale": 24.0}},
    })
    assert result["status"] == "MONITOR"
    assert result["valid_boundaries"] == 2
    assert result["absolute_register_rmse_semitones"] == 0.0
    assert result["matched_free_running_control"]["status"] == "UNAVAILABLE"
