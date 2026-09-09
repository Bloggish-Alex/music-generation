"""Deterministic provenance artifact for frozen raw-pairing normalization."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

NORMALIZATION_POLICY_VERSION = "raw_pairing_normalization.v1"
SCHEMA_VERSION = "raw_pairing_repairs.v1"


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_raw_pairing_repairs(output_dir: Path, songs: Iterable[Any]) -> dict[str, Any]:
    repairs = [repair for song in songs for repair in song.metadata.get("raw_pairing_repairs", [])]
    repairs.sort(key=lambda item: (item["dataset_relative_posix_path"], item["tune_index"], item["physical_track_index"], item["off_tick"], item["off_event_ordinal"], item["repair_kind"]))
    payload = {"schema_version": SCHEMA_VERSION, "normalization_policy_version": NORMALIZATION_POLICY_VERSION, "repairs": repairs}
    path = output_dir / "raw_pairing_repairs.v1.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = Counter(item["repair_kind"] for item in repairs)
    return {"normalization_policy_version": NORMALIZATION_POLICY_VERSION, "repair_artifact": {"path": path.name, "sha256": sha256(path)}, "repair_count": len(repairs), "repair_counts_by_kind": {"same_tick_zero_duration_pair": counts["same_tick_zero_duration_pair"], "redundant_orphan_note_off": counts["redundant_orphan_note_off"]}, "repair_affected_file_count": len({item["source_file_identity"] for item in repairs})}
