"""Artifact-only tests for physical trajectory objective v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_available(root: Path) -> None:
    n, bars, features, dim = 6, 2, 3, 4
    target = np.arange(n * bars * features, dtype=np.float32).reshape(n, bars, features) / 10
    mask = np.ones_like(target, dtype=bool)
    arrays = {
        "validation_target_normalized": target,
        "validation_valid_mask": mask,
        "validation_clean_reconstruction": target.copy(),
        "validation_denoised_reconstruction": target.copy(),
        "validation_summary_embeddings": np.eye(n, dim, dtype=np.float32),
        "validation_token_embeddings": np.random.default_rng(5).normal(size=(n, bars, dim)).astype(np.float32),
        "probe_clean_summary_embeddings": np.zeros((3, dim), dtype=np.float32),
        "probe_coherent_control_summary_embeddings": np.full((3, dim), .01, dtype=np.float32),
        "probe_octave_displacement_summary_embeddings": np.full((3, dim), 2, dtype=np.float32),
        "probe_track_swap_summary_embeddings": np.full((3, dim), 3, dtype=np.float32),
        "probe_boundary_shuffle_summary_embeddings": np.full((3, dim), 4, dtype=np.float32),
        "equivariance_original_values": np.zeros((3, bars, features), dtype=np.float32),
        "equivariance_translated_values": np.dstack([np.zeros((3, bars), dtype=np.float32), np.full((3, bars), 12, dtype=np.float32), np.full((3, bars), 12, dtype=np.float32)]),
        "equivariance_valid_mask": np.ones((3, bars, features), dtype=bool),
    }
    arrays_path = root / "physical_trajectory_objective__raw_arrays.v2.npz"
    np.savez_compressed(arrays_path, **arrays)
    items = {name: {"dtype": values.dtype.name, "shape": list(values.shape)} for name, values in arrays.items()}
    run = {"identity": "fixture", "dataset_identity": "fixture", "dataset_identity_kind": "test", "seed": 1, "checkpoint_sha256": "sha256:" + "1" * 64}
    observation = {
        "schema_version": "physical_trajectory_objective_raw_observation.v2", "run": run,
        "trajectory_schema": {"feature_names": ["chroma", "pitch", "density"], "feature_groups": {"all": [0, 1, 2]}, "plan_bars": bars, "normalizer": {"mean": [0, 0, 0], "std": [1, 1, 1], "degenerate": [False, False, False]}},
        "coverage": {"encoded_bars": {"scope": "all_encoded_bar_index_rows_before_window_selection", "count": 20, "missing_base_pitch_count": 0}, "window_limit": {"configured_max_windows": None, "per_song_max_windows": None, "global_selection": "none"}, "candidate_windows_before_limit": {"scope": "eligible_contiguous_windows_before_per_song_or_global_limit", "count": n, "base_song_window_counts": [{"base_song_id": "a", "window_count": n}], "form_window_counts": [{"form": "A", "window_count": n}]}, "selected_windows_after_limit": {"scope": "training_windows_after_per_song_and_global_limit", "count": n, "base_song_window_counts": [{"base_song_id": "a", "window_count": n}], "form_window_counts": [{"form": "A", "window_count": n}]}},
        "validation": {"window_count": n, "train_window_count": 3, "validation_window_count": n, "rows": [{"row": row, "base_song_id": f"song_{row}", "form": "A"} for row in range(n)], "corruption": {"seed": 2, "mask_probability": .1, "noise_std": .1}},
        "probe": {"sample_count": 3, "variants": ["clean", "coherent_control", "octave_displacement", "track_swap", "boundary_shuffle"]},
        "equivariance": {"octave_shift_semitones": 12.0, "chroma_feature_indices": [0], "density_feature_indices": [0], "pitch_translation_feature_indices": [1, 2]},
        "arrays": {"path": arrays_path.name, "sha256": _sha(arrays_path), "items": items},
        "availability": {"coverage": True, "validation_reconstruction": True, "embedding_health": True, "probe_embeddings": True, "equivariance": True}, "unavailable_reasons": [],
    }
    observation_path = root / "physical_trajectory_objective__raw_observation.v2.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    status = {"schema_version": "physical_trajectory_objective_raw_status.v2", "run": run, "status": "AVAILABLE", "availability": {"observation": True}, "artifacts": {"observation": {"path": observation_path.name, "sha256": _sha(observation_path)}}, "unavailable_reasons": []}
    (root / "physical_trajectory_objective__raw_status.v2.json").write_text(json.dumps(status), encoding="utf-8")


def test_objective_v2_evaluates_raw_arrays_and_writes_marker(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir(); _write_available(source)
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(input_root=source, output_root=tmp_path / "runs", run_id="stage1", modules=("physical_trajectory_objective",), mode=EvaluationMode.ALL))
    report = json.loads((store.run_dir / "stage1__physical_trajectory_objective__report.v1.json").read_text(encoding="utf-8"))
    marker = json.loads((store.run_dir / "stage1__physical_trajectory_objective__freezing_marker.v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "WARN"
    assert report["metrics"]["reconstruction"]["clean"]["overall"]["r2"] == 1.0
    assert marker["validated_for_freezing"] is True


def test_objective_unavailable_status_never_loads_arrays(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    status = {"schema_version": "physical_trajectory_objective_raw_status.v2", "run": {"identity": "fixture", "dataset_identity": "fixture", "dataset_identity_kind": "test", "seed": 1, "checkpoint_sha256": "sha256:" + "1" * 64}, "status": "UNAVAILABLE", "availability": {"observation": False}, "artifacts": {}, "unavailable_reasons": [{"field": "capture", "reason": "fixture"}]}
    (source / "physical_trajectory_objective__raw_status.v2.json").write_text(json.dumps(status), encoding="utf-8")
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(input_root=source, output_root=tmp_path / "runs", run_id="unavailable", modules=("physical_trajectory_objective",), mode=EvaluationMode.ALL))
    report = json.loads((store.run_dir / "unavailable__physical_trajectory_objective__report.v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "UNAVAILABLE"


def test_objective_registry_is_available() -> None:
    assert "physical_trajectory_objective" in DEFAULT_MODULE_REGISTRY.names()
