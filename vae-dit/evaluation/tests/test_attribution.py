from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.evaluation_attribution import ATTRIBUTION_MODULE
from evaluation_framework.evaluation_registry import EvaluationModuleRegistry
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner


def test_attribution_reports_semantic_metrics_from_manifest_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_public_arm(run_dir, "baseline", 0.0)
    _write_public_arm(run_dir, "ablated", 0.5)
    observation = {
        "experiment": {"stream": "theme_memory", "intervention": "masked"},
        "arms": [{"arm_id": "baseline"}, {"arm_id": "ablated"}],
    }
    observation_path = run_dir / "attribution__raw_observation.v3.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    status = {
        "schema_version": "attribution_raw_status.v3",
        "status": "AVAILABLE",
        "artifacts": {"observation": _reference(observation_path)},
    }
    (run_dir / "attribution__raw_status.v3.json").write_text(json.dumps(status), encoding="utf-8")

    store = EvaluationRunner(_registry()).run(EvaluationRunRequest(
        input_root=run_dir,
        output_root=tmp_path,
        run_dir=run_dir,
        run_id="ignored",
        modules=("attribution",),
        mode=EvaluationMode.ALL,
    ))

    report = json.loads((store.run_dir / "attribution__report.v1.json").read_text(encoding="utf-8"))
    semantic = report["metrics"]["semantic_pitch_density"]
    assert semantic["status"] == "AVAILABLE"
    assert semantic["chroma_cosine_mean"] == 1.0
    assert semantic["density_slot_difference_mean"] == 0.0


def test_attribution_keeps_nonsemantic_metrics_when_schema_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_public_arm(run_dir, "baseline", 0.0, include_schema=False)
    _write_public_arm(run_dir, "ablated", 0.5, include_schema=False)
    observation_path = run_dir / "attribution__raw_observation.v3.json"
    observation_path.write_text(json.dumps({"experiment": {}, "arms": [{"arm_id": "baseline"}, {"arm_id": "ablated"}]}), encoding="utf-8")
    (run_dir / "attribution__raw_status.v3.json").write_text(json.dumps({
        "schema_version": "attribution_raw_status.v3",
        "status": "AVAILABLE",
        "artifacts": {"observation": _reference(observation_path)},
    }), encoding="utf-8")

    store = EvaluationRunner(_registry()).run(EvaluationRunRequest(
        input_root=run_dir,
        output_root=tmp_path,
        run_dir=run_dir,
        run_id="ignored",
        modules=("attribution",),
        mode=EvaluationMode.ALL,
    ))

    report = json.loads((store.run_dir / "attribution__report.v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "MONITOR"
    assert report["metrics"]["tensor_mse"] is not None
    assert report["metrics"]["semantic_pitch_density"]["status"] == "UNAVAILABLE"


def _registry() -> EvaluationModuleRegistry:
    registry = EvaluationModuleRegistry()
    registry.register(ATTRIBUTION_MODULE)
    return registry


def _write_public_arm(root: Path, arm_id: str, relative_pitch: float, *, include_schema: bool = True) -> None:
    arrays_path = root / f"{arm_id}__bar_tensors.npz"
    bars = np.zeros((2, 1, 1, 5), dtype=np.float32)
    bars[..., 0] = relative_pitch
    bars[..., 2] = 1.0
    np.savez_compressed(arrays_path, bars=bars, render_base_pitches=np.asarray([60.0, 62.0], dtype=np.float32))
    schema = None
    if include_schema:
        schema = {
            "schema_version": "bar_tensor_schema.v1",
            "axis_order": ["bar", "track", "step", "feature"],
            "feature_names": ["relative_pitch", "is_rest", "is_note_on", "is_hold", "velocity"],
            "track_names": ["melody"],
            "pitch_scale_semitones": 24.0,
        }
    manifest = {
        "artifacts": {"bar_tensors": _reference(arrays_path)},
        "tensor_schema": schema,
    }
    (root / f"attribution_{arm_id}__evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _reference(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
