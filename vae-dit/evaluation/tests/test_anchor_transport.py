"""End-to-end artifact-only tests for deterministic anchor transport."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner
from evaluation_framework.evaluation_anchor_transport import _summary_png


def _anchor(value: int | None) -> dict[str, object]:
    return {"value": value, "coordinate_frame": "transposed_model", "transpose_applied": True, "clipping_stage": "none"}


def _raw(split: str, *, exported: int | None = 48, runtime: int | None = 48) -> dict[str, object]:
    values = {"source": _anchor(48), "encoding": _anchor(48), "exported": _anchor(exported), "trajectory_input": _anchor(runtime), "renderer_input": _anchor(runtime)}
    availability = {
        "source": True, "encoding": True, "exported": exported is not None,
        "trajectory_input": runtime is not None, "renderer_input": runtime is not None,
        "reasons": [],
    }
    return {
        "schema_version": "anchor_transport_raw_observation.v1",
        "dataset": {"identity": "fixture", "identity_kind": "stage_label_unverified", "content_sha256": None, "split": split, "split_unit": "base_song_id"},
        "anchor_spec": {
            "policy_id": "canonical_bar_anchor_v1",
            "definition": "lowest sounding MIDI pitch in a nonempty bar after source quantization and declared transpose normalization",
            "unit": "midi_semitone", "integer_policy": "integer_exact", "missing_behavior": "null",
        },
        "boundaries": [
            {"name": "source_to_encoding", "from_stage": "source", "to_stage": "encoding", "expected_relation": "exact_identity"},
            {"name": "encoding_to_export", "from_stage": "encoding", "to_stage": "exported", "expected_relation": "exact_identity"},
            {"name": "export_to_trajectory_input", "from_stage": "exported", "to_stage": "trajectory_input", "expected_relation": "exact_identity"},
            {"name": "trajectory_input_to_renderer_input", "from_stage": "trajectory_input", "to_stage": "renderer_input", "expected_relation": "exact_identity"},
        ],
        "observations": [
            {"song_id": "song", "base_song_id": "song", "source_bar_index": 0, "applied_transpose_semitones": 2, "anchors": values},
            {"song_id": "song", "base_song_id": "song", "source_bar_index": 1, "applied_transpose_semitones": 2, "anchors": {name: _anchor(None) for name in values}},
        ],
        "availability": availability,
    }


def _run(tmp_path: Path, payload: dict[str, object]) -> dict[str, object]:
    input_root = tmp_path / "input"; input_root.mkdir()
    (input_root / "anchor_transport__raw_observation__validation.v1.json").write_text(json.dumps(payload), encoding="utf-8")
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(
        input_root=input_root, output_root=tmp_path / "runs", run_id="anchor_1", modules=("anchor_transport",), mode=EvaluationMode.ALL,
    ))
    return json.loads((store.run_dir / "anchor_1__anchor_transport__report.v1.json").read_text(encoding="utf-8"))


def test_anchor_transport_reports_pass_for_all_observed_identity_boundaries(tmp_path: Path) -> None:
    report = _run(tmp_path, _raw("validation"))
    boundaries = report["metrics"]["splits"]["validation"]["boundaries"]
    assert report["status"] == "PASS"
    assert all(value["status"] == "PASS" for value in boundaries.values())
    assert boundaries["encoding_to_export"]["exact_match_ratio"] == 1.0
    assert boundaries["encoding_to_export"]["mae_semitones"] == 0.0


def test_anchor_transport_fails_only_the_mismatched_deterministic_boundary(tmp_path: Path) -> None:
    report = _run(tmp_path, _raw("validation", exported=50))
    boundaries = report["metrics"]["splits"]["validation"]["boundaries"]
    assert report["status"] == "FAIL"
    assert boundaries["source_to_encoding"]["status"] == "PASS"
    assert boundaries["encoding_to_export"]["status"] == "FAIL"
    assert boundaries["encoding_to_export"]["signed_error_semitones"]["median"] == 2.0


def test_anchor_transport_preserves_earlier_passes_when_runtime_facts_are_unavailable(tmp_path: Path) -> None:
    report = _run(tmp_path, _raw("validation", runtime=None))
    boundaries = report["metrics"]["splits"]["validation"]["boundaries"]
    assert report["status"] == "UNAVAILABLE"
    assert boundaries["source_to_encoding"]["status"] == "PASS"
    assert boundaries["encoding_to_export"]["status"] == "PASS"
    assert boundaries["export_to_trajectory_input"]["status"] == "UNAVAILABLE"


def test_anchor_transport_summary_uses_a_real_figure_when_runtime_facts_are_unavailable(tmp_path: Path) -> None:
    report = _run(tmp_path, _raw("validation", runtime=None))

    png = _summary_png(report["metrics"]["splits"])

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1_000


def test_default_registry_exposes_anchor_transport() -> None:
    assert "anchor_transport" in DEFAULT_MODULE_REGISTRY.names()
