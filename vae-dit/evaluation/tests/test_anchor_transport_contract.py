"""Contract tests for deterministic absolute-anchor observations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "contracts" / "evaluation" / "v1" / "anchor_transport__raw_observation.v1.schema.json"


def _anchor(value: int | None, frame: str = "transposed_model") -> dict[str, object]:
    return {
        "value": value,
        "coordinate_frame": frame,
        "transpose_applied": frame == "transposed_model",
        "clipping_stage": "none",
    }


def _observation() -> dict[str, object]:
    anchor = _anchor(48)
    return {
        "schema_version": "anchor_transport_raw_observation.v1",
        "dataset": {"identity": "stage3", "identity_kind": "stage_label_unverified", "split": "validation", "split_unit": "base_song_id"},
        "anchor_spec": {
            "policy_id": "canonical_bar_anchor_v1",
            "definition": "lowest sounding MIDI pitch in a nonempty bar after source quantization and declared transpose normalization",
            "unit": "midi_semitone",
            "integer_policy": "integer_exact",
            "missing_behavior": "null",
        },
        "boundaries": [
            {"name": "source_to_encoding", "from_stage": "source", "to_stage": "encoding", "expected_relation": "exact_identity"},
            {"name": "encoding_to_export", "from_stage": "encoding", "to_stage": "export", "expected_relation": "exact_identity"},
        ],
        "observations": [{
            "song_id": "song-a", "base_song_id": "song-a", "source_bar_index": 0, "applied_transpose_semitones": 2,
            "anchors": {"source": anchor, "encoding": anchor, "exported": anchor, "trajectory_input": anchor, "renderer_input": anchor},
        }],
        "availability": {"source": True, "encoding": True, "exported": True, "trajectory_input": True, "renderer_input": True, "reasons": []},
    }


def test_raw_anchor_observation_contract_accepts_versioned_deterministic_facts() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(_observation())


def test_raw_anchor_observation_contract_rejects_learned_prediction_fields() -> None:
    observation = _observation()
    observation["observations"][0]["predicted_future_anchor"] = 60  # type: ignore[index]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(observation)


def test_raw_anchor_observation_contract_rejects_mixed_coordinate_frames() -> None:
    observation = _observation()
    observation["observations"][0]["anchors"]["encoding"] = _anchor(48, "original_source")  # type: ignore[index]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(observation)
