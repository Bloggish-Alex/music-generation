#!/usr/bin/env python3
"""Verify and optionally repair generated MIDI artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from pipeline.latent_generation_pipeline import SequenceTensorMidiRenderer


class GenerationMidiVerifierCLI:
    """Verify a generated MIDI against its diagnostics JSON."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Verify or repair generated MIDI tempo/hash.")
        parser.add_argument("--generation-dir", type=Path, default=None)
        parser.add_argument("--json", type=Path, default=None)
        parser.add_argument("--repair", action="store_true", help="Re-render MIDI from saved bar tensors when it differs.")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run verification."""
        args = self.build_parser().parse_args(argv)
        json_path = self._resolve_json(args.generation_dir, args.json)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        status = self._status(data)
        if status["ok"]:
            self._print_status(status)
            return
        if not args.repair:
            self._print_status(status)
            raise SystemExit(1)
        self._repair(data)
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        repaired_status = self._status(data)
        self._print_status(repaired_status)
        if not repaired_status["ok"]:
            raise SystemExit(1)

    def _resolve_json(self, generation_dir: Optional[Path], json_path: Optional[Path]) -> Path:
        """Resolve diagnostics JSON path."""
        if json_path is not None:
            return Path(json_path)
        if generation_dir is None:
            raise ValueError("Pass --generation-dir or --json.")
        candidates = sorted(
            path for path in Path(generation_dir).glob("*.json")
            if not path.name.endswith(".audio_quality.json")
        )
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one generation JSON under {generation_dir}, found {len(candidates)}.")
        return candidates[0]

    def _status(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return current MIDI consistency status."""
        midi_path = Path(str(data.get("midi_path") or data.get("midi", {}).get("output_path", "")))
        expected_tempo = int(data.get("config", {}).get("tempo_bpm", data.get("midi", {}).get("tempo_bpm", 100)))
        expected_hash = data.get("midi", {}).get("midi_sha256")
        actual_tempo = self._midi_tempo(midi_path)
        actual_hash = self._sha256(midi_path) if midi_path.exists() else None
        tempo_ok = actual_tempo is not None and int(round(float(actual_tempo))) == int(expected_tempo)
        hash_ok = expected_hash is None or actual_hash == expected_hash
        return {
            "ok": bool(tempo_ok and hash_ok),
            "midi_path": str(midi_path),
            "expected_tempo": int(expected_tempo),
            "actual_tempo": actual_tempo,
            "tempo_ok": bool(tempo_ok),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_ok": bool(hash_ok),
        }

    def _repair(self, data: dict[str, Any]) -> None:
        """Re-render MIDI from the saved generation tensors."""
        tensor_path = Path(str(data.get("tensor_path")))
        midi_path = Path(str(data.get("midi_path") or data.get("midi", {}).get("output_path", "")))
        archive = np.load(tensor_path)
        try:
            bars = np.asarray(archive["bars"], dtype=np.float32)
        finally:
            archive.close()
        config = data.get("config", {})
        tempo = int(config.get("tempo_bpm", 100))
        base_pitch = int(config.get("base_pitch", 60))
        base_pitches = data.get("render_base_pitches")
        diag = SequenceTensorMidiRenderer(DVAEMidiRenderConfig(
            tempo_bpm=tempo,
            default_base_pitch=base_pitch,
        )).render(
            bars,
            midi_path,
            base_pitch=base_pitch,
            base_pitches=base_pitches,
        )
        data["midi"] = diag

    def _midi_tempo(self, midi_path: Path) -> Optional[float]:
        """Read the first MIDI tempo as BPM."""
        if not midi_path.exists():
            return None
        import mido

        midi = mido.MidiFile(str(midi_path))
        for track in midi.tracks:
            for message in track:
                if message.type == "set_tempo":
                    return float(mido.tempo2bpm(message.tempo))
        return None

    def _sha256(self, path: Path) -> str:
        """Return file SHA256."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _print_status(self, status: dict[str, Any]) -> None:
        """Print compact verification status."""
        print(json.dumps(status, indent=2))


def main() -> None:
    """Run CLI."""
    GenerationMidiVerifierCLI().run()


if __name__ == "__main__":
    main()
