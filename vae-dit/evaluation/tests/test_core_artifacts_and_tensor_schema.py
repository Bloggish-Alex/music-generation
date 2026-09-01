from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.core import SemanticTensorDecoder, VerifiedArtifactResolver
from evaluation_framework.core.policy import maximum_gate, minimum_gate, overall_status


def test_verified_resolver_rejects_path_escape_and_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    resolver = VerifiedArtifactResolver(tmp_path)
    assert resolver.json({"path": artifact.name, "sha256": digest}) == {"ok": True}
    with pytest.raises(ValueError):
        resolver.path({"path": "../outside.json", "sha256": digest})
    with pytest.raises(ValueError):
        resolver.path({"path": artifact.name, "sha256": "sha256:" + "0" * 64})


def test_semantic_decoder_uses_named_features_not_fixed_positions() -> None:
    schema = {
        "schema_version": "bar_tensor_schema.v1",
        "feature_names": ["is_hold", "relative_pitch", "is_note_on"],
        "track_names": ["melody"],
        "pitch_scale_semitones": 12.0,
    }
    bars = np.asarray([[[[0.0, 0.5, 1.0]]]], dtype=np.float32)
    decoder = SemanticTensorDecoder.from_schema(schema)
    assert decoder.active_mask(bars).item() is True
    assert decoder.absolute_pitch(bars, np.asarray([60.0])).item() == 66.0


def test_semantic_decoder_accepts_v1_schema_inferred_from_raw_observation() -> None:
    schema = {
        "feature_names": ["relative_pitch", "is_note_on", "is_hold"],
        "track_names": ["melody"],
        "pitch_scale_semitones": 24.0,
    }

    decoder = SemanticTensorDecoder.from_schema(schema)

    assert decoder.pitch_scale_semitones == 24.0


def test_semantic_decoder_rejects_explicit_unknown_schema_version() -> None:
    schema = {
        "schema_version": "bar_tensor_schema.v2",
        "feature_names": ["relative_pitch", "is_note_on", "is_hold"],
        "track_names": ["melody"],
        "pitch_scale_semitones": 24.0,
    }

    with pytest.raises(ValueError, match="Unsupported bar tensor schema"):
        SemanticTensorDecoder.from_schema(schema)


def test_policy_gates_keep_threshold_boundaries_explicit() -> None:
    maximum = maximum_gate("maximum", "Maximum", 2.0, 2.0)
    minimum = minimum_gate("minimum", "Minimum", 2.0, 2.0)

    assert maximum["status"] == "PASS"
    assert minimum["status"] == "PASS"
    assert overall_status([maximum, minimum]) == "PASS"
    assert overall_status([maximum_gate("failure", "Failure", 2.1, 2.0)]) == "FAIL"
