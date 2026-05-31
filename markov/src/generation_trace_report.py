#!/usr/bin/env python3
"""Render generation trace JSON as a readable data-flow report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _notes_summary(value: Mapping[str, Any]) -> str:
    notes = value.get("notes")
    if not isinstance(notes, Mapping):
        return "notes=n/a"
    pitches = notes.get("pitches") or []
    pitch_text = ",".join(str(x) for x in pitches[:10])
    if len(pitches) > 10:
        pitch_text += ",..."
    return (
        f"notes={_fmt(notes.get('count'))}, "
        f"melody={_fmt(notes.get('melody_count'))}, "
        f"bass={_fmt(notes.get('bass_count'))}, "
        f"pitches=[{pitch_text}]"
    )


def _stage_line(stage: Mapping[str, Any]) -> str:
    before = stage.get("before") if isinstance(stage.get("before"), Mapping) else {}
    after = stage.get("after") if isinstance(stage.get("after"), Mapping) else {}
    marker = "changed" if stage.get("changed") else "same"
    extra = []
    for key in ("context_ready", "bar_target_ready", "rhythm_target_ready", "reused_motif", "generated_new_theme"):
        if before.get(key) != after.get(key):
            extra.append(f"{key}: {_fmt(before.get(key))} -> {_fmt(after.get(key))}")
    if before.get("target_pitch") != after.get("target_pitch"):
        extra.append(f"target_pitch: {_fmt(before.get('target_pitch'))} -> {_fmt(after.get('target_pitch'))}")
    extra_text = f" ({'; '.join(extra)})" if extra else ""
    return (
        f"    - `{stage.get('stage')}`: {marker}; "
        f"{_notes_summary(before)} -> {_notes_summary(after)}{extra_text}"
    )


def render_markdown(trace: Mapping[str, Any]) -> str:
    requested = trace.get("requested") if isinstance(trace.get("requested"), Mapping) else {}
    timeline = trace.get("timeline") if isinstance(trace.get("timeline"), Mapping) else {}
    labels = timeline.get("labels") if isinstance(timeline.get("labels"), Sequence) else []
    events = timeline.get("events") if isinstance(timeline.get("events"), Sequence) else []
    module_trace = trace.get("module_trace") if isinstance(trace.get("module_trace"), Sequence) else []
    measure_context = trace.get("measure_context") if isinstance(trace.get("measure_context"), Sequence) else []
    note_root = trace.get("notes") if isinstance(trace.get("notes"), Mapping) else {}
    note_bars = note_root.get("bars") if isinstance(note_root.get("bars"), Sequence) else []
    final_notes = note_root.get("final") if isinstance(note_root.get("final"), Sequence) else []

    lines: list[str] = []
    lines.append("# Generation Trace Report")
    lines.append("")
    lines.append(f"- output: `{trace.get('output_path', 'unknown')}`")
    lines.append(f"- target measures: {_fmt(requested.get('target_measures'))}")
    lines.append(f"- seed: {_fmt(requested.get('seed'))}")
    lines.append(f"- executed modules: {', '.join(f'`{m}`' for m in trace.get('executed_modules', []))}")
    lines.append(f"- executed note modules: {', '.join(f'`{m}`' for m in trace.get('executed_note_modules', []))}")
    lines.append("")

    lines.append("## Module Flow")
    lines.append("")
    for item in module_trace:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"- `{item.get('module')}`: labels={_fmt(item.get('labels'))}, "
            f"events={_fmt(item.get('events'))}, context={_fmt(item.get('measure_context'))}, "
            f"note_bars={_fmt(item.get('notes_bars'))}, note_events={_fmt(item.get('notes_total'))}"
        )
    lines.append("")

    lines.append("## Timeline")
    lines.append("")
    lines.append(f"- labels ({len(labels)}): `{labels}`")
    for event in events:
        if not isinstance(event, Mapping):
            continue
        label = event.get("label", event.get("kind"))
        lines.append(
            f"- {event.get('kind')}: label={label}, role={event.get('role', 'n/a')}, "
            f"length={event.get('length')}, labels={event.get('labels')}"
        )
    lines.append("")

    lines.append("## Measure Context")
    lines.append("")
    lines.append("| Bar | Cluster | Section | Role | Local | Occurrence | Length |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: |")
    for row in measure_context:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('bar')} | {row.get('cluster')} | {row.get('section_label')} | "
            f"{row.get('role')} | {row.get('local_bar')} | {row.get('occurrence_id')} | {row.get('section_len')} |"
        )
    lines.append("")

    lines.append("## Note Flow")
    lines.append("")
    for bar in note_bars:
        if not isinstance(bar, Mapping):
            continue
        bar_index = int(bar.get("bar", 0) or 0)
        final = final_notes[bar_index - 1] if 0 < bar_index <= len(final_notes) else {}
        lines.append(
            f"### Bar {bar.get('bar')} - cluster {bar.get('cluster')}, "
            f"{bar.get('section_label')} {bar.get('role')} local {bar.get('local_bar')}"
        )
        lines.append("")
        for stage in bar.get("stages", []):
            if isinstance(stage, Mapping):
                lines.append(_stage_line(stage))
        if isinstance(final, Mapping):
            lines.append(f"    - final: {_notes_summary({'notes': final})}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a <output>.generation.json trace as Markdown.",
    )
    parser.add_argument("json_file", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the loaded JSON instead of Markdown.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    with open(args.json_file) as f:
        trace = json.load(f)
    text = json.dumps(trace, indent=2) if args.json else render_markdown(trace)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
