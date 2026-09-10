#!/usr/bin/env python3
"""Read-only census for the Codec V2 initial time-signature policy decision."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mido
from data.canonical_timeline import collect_raw_smf_facts
from data.music_parser import MusicDirectoryParser, MusicParserConfig


def census(root: Path, policy: str) -> dict[str, object]:
    parser = MusicDirectoryParser(replace(MusicParserConfig(), initial_time_signature_policy=policy))
    files = parser.discover_files(root)
    late_initial, no_events = [], []
    successes = 0
    failures = []
    for path in files:
        try:
            facts, _ = collect_raw_smf_facts(mido.MidiFile(path), repairs=[])
            first = min((fact.absolute_tick for fact in facts), default=None)
            item = {"file": path.relative_to(root).as_posix(), "first_real_ts_tick": first}
            if first is None:
                no_events.append(item)
            elif first != 0:
                late_initial.append(item)
            parser.parse_file(path, {}, dataset_root=root)
            successes += 1
        except Exception as error:
            failures.append({"file": path.relative_to(root).as_posix(), "error": f"{type(error).__name__}: {error}"})
    return {"initial_time_signature_policy": policy, "file_count": len(files), "success_count": successes, "success_rate": successes / len(files) if files else 0.0, "missing_tick_zero_with_later_real_ts_count": len(late_initial), "missing_tick_zero_with_later_real_ts_files": late_initial, "no_time_signature_events_count": len(no_events), "no_time_signature_events_files": no_events, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--policy", choices=("error", "smf_default_4_4"), required=True)
    options = parser.parse_args()
    print(json.dumps(census(options.dataset_root, options.policy), indent=2))


if __name__ == "__main__":
    main()
