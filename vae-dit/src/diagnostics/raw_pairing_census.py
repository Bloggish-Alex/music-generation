"""Read-only raw-SMF FIFO note-pairing census.

This diagnostic deliberately operates on mido events instead of parser records.
It reports the same pairing failures that the canonical parser rejects, but it
never repairs a stream or changes encoder admission policy.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "raw_pairing_census.v1"
POLICY_VERSION = "smf_fifo_track_channel_pitch_v1"
MIDI_SUFFIXES = frozenset({".mid", ".midi"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return "sha256:" + _sha256_bytes(path.read_bytes())


def _relative_path(path: Path, root: Path) -> str:
    return unicodedata.normalize("NFC", path.resolve().relative_to(root.resolve()).as_posix())


def _file_identity(relative: str, content_digest: str) -> str:
    """Match the frozen source identity byte representation."""
    return _sha256_bytes(f"{relative}\0{content_digest}".encode("utf-8"))


def _input_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MIDI_SUFFIXES),
        key=lambda path: _relative_path(path, root),
    )


def _input_manifest_entries(files: Iterable[Path], root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in files:
        relative = _relative_path(path, root)
        content_digest = _sha256_bytes(path.read_bytes())
        entries.append({
            "dataset_relative_posix_path": relative,
            "content_sha256": "sha256:" + content_digest,
            "source_file_identity": _file_identity(relative, content_digest),
        })
    return entries


def _dataset_content_sha256(entries: Iterable[Mapping[str, str]]) -> str:
    lines = [f"{item['dataset_relative_posix_path']}\t{item['content_sha256'].removeprefix('sha256:')}\n" for item in entries]
    return "sha256:" + _sha256_bytes("".join(lines).encode("utf-8"))


def _event_kind(event: Any) -> str:
    return "note_on" if event.type == "note_on" and int(event.velocity) > 0 else "note_off"


def _event_view(event: Any, absolute_tick: int, event_ordinal: int) -> dict[str, int | str]:
    return {
        "event_kind": _event_kind(event),
        "absolute_tick": absolute_tick,
        "event_ordinal": event_ordinal,
        "velocity": int(event.velocity),
    }


def _pending_view(queue: Iterable[tuple[int, int, int]]) -> list[dict[str, int]]:
    return [
        {"absolute_tick": tick, "event_ordinal": ordinal, "velocity": velocity}
        for tick, ordinal, velocity in queue
    ]


def _finding(
    *, identity: str, relative: str, midi: Any, track: int, event: Any,
    absolute_tick: int, event_ordinal: int, queue_depth_before: int,
    code: str, pending: Iterable[tuple[int, int, int]],
    same_tick_events: list[dict[str, int | str]],
    matched: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    return {
        "source_file_identity": identity,
        "dataset_relative_posix_path": relative,
        "tune_index": 0,
        "physical_track_index": track,
        "channel": int(event.channel),
        "pitch": int(event.note),
        "failure_code": code,
        "absolute_tick": absolute_tick,
        "event_ordinal": event_ordinal,
        "queue_depth_before": queue_depth_before,
        "matched_on_tick": matched[0] if matched is not None else None,
        "matched_on_event_ordinal": matched[1] if matched is not None else None,
        "pending_on_queue": _pending_view(pending),
        "same_tick_events": same_tick_events,
        "tick_relation": "unmatched" if matched is None else ("same_tick" if matched[0] == absolute_tick else "different_tick"),
        "ppqn": int(midi.ticks_per_beat),
        "smf_format": int(midi.type),
    }


def census_file(path: Path, root: Path) -> list[dict[str, Any]]:
    """Report raw pairing anomalies for one SMF without changing it."""
    import mido

    midi = mido.MidiFile(str(path))
    if int(midi.type) not in {0, 1}:
        raise ValueError("smf_format_unsupported")
    if int(midi.ticks_per_beat) <= 0:
        raise ValueError("smpte_division_unsupported")
    relative = _relative_path(path, root)
    digest = _sha256_bytes(path.read_bytes())
    identity = _file_identity(relative, digest)
    findings: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        pending: dict[tuple[int, int], deque[tuple[int, int, int]]] = defaultdict(deque)
        events: list[tuple[int, int, Any]] = []
        related: dict[tuple[int, int, int], list[dict[str, int | str]]] = defaultdict(list)
        for ordinal, event in enumerate(track):
            absolute_tick += int(event.time)
            if event.type not in {"note_on", "note_off"}:
                continue
            events.append((absolute_tick, ordinal, event))
            related[(int(event.channel), int(event.note), absolute_tick)].append(_event_view(event, absolute_tick, ordinal))
        for absolute_tick, ordinal, event in events:
            key = (int(event.channel), int(event.note))
            queue = pending[key]
            depth = len(queue)
            same_tick = related[(key[0], key[1], absolute_tick)]
            if _event_kind(event) == "note_on":
                queue.append((absolute_tick, ordinal, int(event.velocity)))
                continue
            if not queue:
                findings.append(_finding(identity=identity, relative=relative, midi=midi, track=track_index, event=event, absolute_tick=absolute_tick, event_ordinal=ordinal, queue_depth_before=depth, code="orphan_note_off", pending=queue, same_tick_events=same_tick))
                continue
            onset = queue.popleft()
            if absolute_tick <= onset[0]:
                findings.append(_finding(identity=identity, relative=relative, midi=midi, track=track_index, event=event, absolute_tick=absolute_tick, event_ordinal=ordinal, queue_depth_before=depth, code="note_nonpositive_duration", pending=(onset, *queue), same_tick_events=same_tick, matched=onset))
        for (channel, pitch), queue in pending.items():
            for onset in queue:
                onset_tick, onset_ordinal, _ = onset
                event = next(event for tick, ordinal, event in events if tick == onset_tick and ordinal == onset_ordinal)
                findings.append(_finding(identity=identity, relative=relative, midi=midi, track=track_index, event=event, absolute_tick=onset_tick, event_ordinal=onset_ordinal, queue_depth_before=len(queue), code="unterminated_note_on", pending=queue, same_tick_events=related[(channel, pitch, onset_tick)], matched=onset))
    return findings


def _aggregate(findings: Iterable[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    counts = Counter(str(item[field]) for item in findings)
    return [{field: key, "count": counts[key]} for key in sorted(counts)]


def run_census(dataset_root: Path, output_dir: Path, code_revision: str) -> Path:
    """Write a deterministic diagnostic artifact and its provenance manifest."""
    root = dataset_root.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root does not exist: {root}")
    files = _input_files(root)
    inputs = _input_manifest_entries(files, root)
    findings = [finding for path in files for finding in census_file(path, root)]
    findings.sort(key=lambda item: (item["dataset_relative_posix_path"], item["physical_track_index"], item["absolute_tick"], item["event_ordinal"], item["failure_code"]))
    artifact_payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "census_pairing_policy_version": POLICY_VERSION,
        "normalization_policy_version": "raw_pairing_normalization.v1",
        "findings": findings,
        "aggregates": {
            "by_failure_code": _aggregate(findings, "failure_code"),
            "by_source_file": _aggregate(findings, "dataset_relative_posix_path"),
            "by_physical_track": _aggregate(findings, "physical_track_index"),
            "by_channel": _aggregate(findings, "channel"),
            "by_pitch": _aggregate(findings, "pitch"),
            "by_tick_relation": _aggregate(findings, "tick_relation"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "raw_pairing_census.v1.json"
    artifact.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_hash = _dataset_content_sha256(inputs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "code_revision": code_revision,
        "input_sort": "dataset_relative_posix_path_nfc_ascending",
        "input_files": inputs,
        "dataset": {"identity": f"content-manifest:{dataset_hash.removeprefix('sha256:')}", "content_sha256": dataset_hash},
        "artifact": {"path": artifact.name, "sha256": _sha256(artifact)},
    }
    manifest_path = output_dir / "raw_pairing_census.manifest.v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
