"""Contract tests for strict cross-run trajectory anchor lineage."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = ROOT / "contracts" / "evaluation" / "v1"


def _schema(name: str) -> dict[str, object]:
    return json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))


def test_trajectory_anchor_context_schemas_are_valid_draft_2020_12() -> None:
    names = (
        "trajectory_anchor_context__encoded_input_manifest.v1.schema.json",
        "trajectory_anchor_context__training_lineage_raw.v1.schema.json",
        "trajectory_anchor_context__checkpoint_lineage.v1.schema.json",
    )

    for name in names:
        Draft202012Validator.check_schema(_schema(name))
