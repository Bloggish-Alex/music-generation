#!/usr/bin/env python3
"""CLI entrypoint for MidiTok-style bar event encoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codec.miditok_style_bar_encoder import MidiTokStyleBarEventStore


class MidiTokStyleBarEncodingCLI:
    """Command-line adapter for MidiTok-style bar event encoding."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build parser."""
        parser = argparse.ArgumentParser(description="Encode bar_tensors.npz into MidiTok-style event features and JSONL events.")
        parser.add_argument("--model-dir", type=Path, required=True, help="Model directory containing encoded/ artifacts.")
        parser.add_argument("--encoded-dir", type=Path, default=None, help="Defaults to --model-dir/encoded.")
        parser.add_argument("--index-path", type=Path, default=None, help="Defaults to --encoded-dir/bar_tensor_index.json, then latent/latent_index.json.")
        parser.add_argument("--tensor-path", type=Path, default=None, help="Defaults to --encoded-dir/bar_tensors.npz.")
        parser.add_argument("--max-rows", type=int, default=None, help="Optional quick inspection row limit.")
        parser.add_argument("--no-events", action="store_true", help="Do not write miditok_style_bar_events.jsonl.")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run encoder."""
        args = self.build_parser().parse_args(argv)
        model_dir = Path(args.model_dir)
        encoded_dir = Path(args.encoded_dir) if args.encoded_dir else model_dir / "encoded"
        tensor_path = Path(args.tensor_path) if args.tensor_path else encoded_dir / "bar_tensors.npz"
        index_path = Path(args.index_path) if args.index_path else self._default_index_path(model_dir, encoded_dir)
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        if args.max_rows is not None:
            rows = rows[: max(1, int(args.max_rows))]
        summary = MidiTokStyleBarEventStore(encoded_dir).write_from_tensor_archive(
            tensor_path=tensor_path,
            rows=rows,
            write_events=not bool(args.no_events),
        )
        print(f"MidiTok-style bar feature cache -> {encoded_dir / MidiTokStyleBarEventStore.FEATURE_FILE}")
        if not bool(args.no_events):
            print(f"MidiTok-style bar events -> {encoded_dir / MidiTokStyleBarEventStore.EVENT_FILE}")
        print(f"MidiTok-style summary -> {encoded_dir / MidiTokStyleBarEventStore.SUMMARY_FILE}")
        print(f"rows={summary['row_count']} features={summary['feature_count']}")

    def _default_index_path(self, model_dir: Path, encoded_dir: Path) -> Path:
        """Choose index artifact."""
        encoded_index = encoded_dir / "bar_tensor_index.json"
        if encoded_index.exists():
            return encoded_index
        latent_index = model_dir / "latent" / "latent_index.json"
        if latent_index.exists():
            return latent_index
        raise FileNotFoundError(f"Missing index file: {encoded_index} or {latent_index}")


def main() -> None:
    """Run CLI."""
    MidiTokStyleBarEncodingCLI().run()


if __name__ == "__main__":
    main()
