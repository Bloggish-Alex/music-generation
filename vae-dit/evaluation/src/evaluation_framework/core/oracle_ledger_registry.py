"""Versioned Oracle Ladder boundary declarations shared by export and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class OracleBoundary:
    boundary_id: str
    module: str
    label: str


def load_oracle_boundaries() -> tuple[OracleBoundary, ...]:
    registry_path = Path(__file__).resolve().parents[4] / "contracts" / "evaluation" / "registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    boundaries = payload.get("oracle_ladder", {}).get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("Contract registry has no Oracle Ladder boundaries.")

    parsed = tuple(_boundary(value) for value in boundaries)
    if len({item.boundary_id for item in parsed}) != len(parsed):
        raise ValueError("Oracle Ladder boundary IDs must be unique.")
    if len({item.module for item in parsed}) != len(parsed):
        raise ValueError("Oracle Ladder boundary modules must be unique.")
    return parsed


def _boundary(value: Any) -> OracleBoundary:
    if not isinstance(value, Mapping):
        raise ValueError("Oracle Ladder boundary must be an object.")
    boundary_id, module, label = value.get("id"), value.get("module"), value.get("label")
    if not all(isinstance(item, str) and item for item in (boundary_id, module, label)):
        raise ValueError("Oracle Ladder boundary requires non-empty id, module and label.")
    return OracleBoundary(boundary_id, module, label)
