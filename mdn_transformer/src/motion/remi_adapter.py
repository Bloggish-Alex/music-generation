#!/usr/bin/env python3
"""Adapters between project bar records/tensors and MidiTok REMI tokens."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from diagnostics.dvae_midi_render import DVAEMidiRenderConfig, TensorMidiRenderer


@dataclass(frozen=True)
class RemiTokenizerSettings:
    """Stable MidiTok REMI tokenizer settings for the motion experiment."""

    vocab_size: int = 30000
    pitch_min: int = 21
    pitch_max: int = 109
    num_velocities: int = 24
    num_tempos: int = 32
    tempo_min: int = 50
    tempo_max: int = 200
    use_chords: bool = True
    use_rests: bool = True
    use_tempos: bool = True
    use_time_signatures: bool = True
    use_programs: bool = False


class RemiTokenizerFactory:
    """Create or load MidiTok REMI tokenizers without leaking MidiTok details."""

    BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}

    def __init__(self, settings: RemiTokenizerSettings) -> None:
        self.settings = settings

    def create(self) -> Any:
        """Create an untrained REMI tokenizer."""
        from miditok import REMI, TokenizerConfig

        config = TokenizerConfig(
            pitch_range=(int(self.settings.pitch_min), int(self.settings.pitch_max)),
            beat_res=self.BEAT_RES,
            num_velocities=int(self.settings.num_velocities),
            special_tokens=["PAD", "BOS", "EOS"],
            use_chords=bool(self.settings.use_chords),
            use_rests=bool(self.settings.use_rests),
            use_tempos=bool(self.settings.use_tempos),
            use_time_signatures=bool(self.settings.use_time_signatures),
            use_programs=bool(self.settings.use_programs),
            num_tempos=int(self.settings.num_tempos),
            tempo_range=(int(self.settings.tempo_min), int(self.settings.tempo_max)),
        )
        return REMI(config)

    def load(self, path: str | Path) -> Any:
        """Load a saved REMI tokenizer."""
        from miditok import REMI

        return REMI(params=Path(path))


class MidiBuilder:
    """Build simple MIDI files from encoded song/bar dictionaries or bar tensors."""

    def __init__(self, config: DVAEMidiRenderConfig) -> None:
        self.config = config
        self.tensor_renderer = TensorMidiRenderer(config)

    def write_song(self, song: Dict[str, Any], output_path: str | Path) -> Path:
        """Write one encoded song dictionary to a MIDI file."""
        bars = list(song.get("bars", []))
        return self.write_bars(bars, output_path)

    def write_bars(
        self,
        bars: Sequence[Dict[str, Any]],
        output_path: str | Path,
        rebase_bar_indices: bool = False,
    ) -> Path:
        """Write encoded bars, optionally rebasing their first bar to time zero.

        Whole songs retain their physical bar positions. A standalone bar used as
        a token-model input must instead start at time zero; otherwise MidiTok
        encodes all preceding empty bars and the bar's real events may be clipped.
        """
        import mido

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mid = mido.MidiFile(ticks_per_beat=int(self.config.ticks_per_beat))
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(int(self.config.tempo_bpm)), time=0))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        mid.tracks.append(meta)

        events_by_track: Dict[int, List[Dict[str, int]]] = {}
        first_bar_index = int(bars[0].get("bar_index", 0)) if bars else 0
        for fallback_bar_index, bar in enumerate(bars):
            original_bar_index = int(bar.get("bar_index", fallback_bar_index))
            bar_index = original_bar_index - first_bar_index if rebase_bar_indices else original_bar_index
            bar_start_tick = int(round(bar_index * float(self.config.bar_length_ql) * int(self.config.ticks_per_beat)))
            for track in bar.get("tracks", []):
                track_index = int(track.get("track_index", 0))
                for note in track.get("notes", []):
                    onset = float(note.get("onset_ql", 0.0))
                    duration = max(0.0, float(note.get("duration_ql", 0.0)))
                    start = int(round(bar_start_tick + onset * int(self.config.ticks_per_beat)))
                    end = int(round(start + duration * int(self.config.ticks_per_beat)))
                    if end <= start:
                        continue
                    pitch = self._clip_pitch(int(round(float(note.get("pitch", self.config.default_base_pitch)))))
                    velocity = self._clip_velocity(int(round(float(note.get("velocity", self.config.default_velocity)))))
                    events_by_track.setdefault(track_index, []).extend([
                        {"tick": start, "type": "on", "pitch": pitch, "velocity": velocity},
                        {"tick": end, "type": "off", "pitch": pitch, "velocity": 0},
                    ])

        for track_index in sorted(events_by_track):
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name=f"track_{track_index}", time=0))
            self._append_mido_events(track, events_by_track[track_index])
            mid.tracks.append(track)
        if len(mid.tracks) == 1:
            mid.tracks.append(mido.MidiTrack())
        mid.save(str(output))
        return output

    def write_tensor_bar(self, tensor: np.ndarray, output_path: str | Path, base_pitch: int) -> Path:
        """Write one decoded DVAE tensor bar to MIDI."""
        import mido

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mid = mido.MidiFile(ticks_per_beat=int(self.config.ticks_per_beat))
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(int(self.config.tempo_bpm)), time=0))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        mid.tracks.append(meta)
        events = self.tensor_renderer._tensor_events(np.asarray(tensor, dtype=np.float32), int(base_pitch), start_tick=0)
        for track_index in range(min(3, int(np.asarray(tensor).shape[0]))):
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name=f"track_{track_index}", time=0))
            self.tensor_renderer._append_events(track, [event for event in events if int(event["track"]) == track_index])
            mid.tracks.append(track)
        mid.save(str(output))
        return output

    def _append_mido_events(self, track: Any, events: Sequence[Dict[str, int]]) -> None:
        """Append absolute-tick note events to a mido track."""
        import mido

        previous_tick = 0
        for event in sorted(events, key=lambda item: (int(item["tick"]), 0 if item["type"] == "off" else 1, int(item["pitch"]))):
            tick = int(event["tick"])
            delta = max(0, tick - previous_tick)
            previous_tick = tick
            track.append(mido.Message(
                "note_on" if event["type"] == "on" else "note_off",
                note=int(event["pitch"]),
                velocity=int(event["velocity"]),
                time=delta,
            ))

    def _clip_pitch(self, value: int) -> int:
        return max(int(self.config.min_pitch), min(int(self.config.max_pitch), int(value)))

    def _clip_velocity(self, value: int) -> int:
        return max(1, min(127, int(value)))


class RemiTokenExtractor:
    """Extract integer token ids from MidiTok token sequences."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode_midi(self, midi_path: str | Path) -> List[int]:
        """Encode one MIDI file into a flat REMI id sequence."""
        sequence = self.tokenizer.encode(Path(midi_path))
        return self._flatten_sequence(sequence)

    def encode_score(self, score: Any) -> List[int]:
        """Encode an in-memory symusic score without a temporary MIDI file."""
        sequence = self.tokenizer.encode(score)
        return self._flatten_sequence(sequence)

    def _flatten_sequence(self, sequence: Any) -> List[int]:
        if isinstance(sequence, list):
            if not sequence:
                return []
            ids: List[int] = []
            for item in sequence:
                ids.extend(self._ids_from_sequence(item))
            return ids
        return self._ids_from_sequence(sequence)

    def _ids_from_sequence(self, sequence: Any) -> List[int]:
        ids = getattr(sequence, "ids", sequence)
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return [int(item) for item in ids]


class RemiBarTokenCache:
    """Build and persist per-bar REMI tokens aligned to latent tensor keys."""

    def __init__(self, cache_dir: str | Path, render_config: DVAEMidiRenderConfig, tokenizer_settings: RemiTokenizerSettings) -> None:
        self.cache_dir = Path(cache_dir)
        self.render_config = render_config
        self.tokenizer_settings = tokenizer_settings
        self.midi_builder = MidiBuilder(render_config)
        self.factory = RemiTokenizerFactory(tokenizer_settings)

    @property
    def tokenizer_path(self) -> Path:
        return self.cache_dir / "tokenizer.json"

    @property
    def tokens_path(self) -> Path:
        return self.cache_dir / "remi_bar_tokens.json"

    def build_or_load(self, encoded_dir: str | Path, force_rebuild: bool = False, max_songs: Optional[int] = None) -> Dict[str, Any]:
        """Build tokenizer and per-bar token cache from encoded songs.json."""
        if self.tokens_path.exists() and self.tokenizer_path.exists() and not force_rebuild:
            return self._load_tokens_payload()

        encoded = Path(encoded_dir)
        songs_path = encoded / "songs.json"
        if not songs_path.exists():
            raise FileNotFoundError(f"Missing songs.json: {songs_path}")
        songs = json.loads(songs_path.read_text(encoding="utf-8"))
        if max_songs is not None:
            songs = songs[: max(1, int(max_songs))]
        midi_dir = self.cache_dir / "midi_songs"
        bar_midi_dir = self.cache_dir / "midi_bars"
        midi_dir.mkdir(parents=True, exist_ok=True)
        bar_midi_dir.mkdir(parents=True, exist_ok=True)
        song_midi_paths = [self.midi_builder.write_song(song, midi_dir / f"{self._safe_name(song.get('song_id', index))}.mid") for index, song in enumerate(songs)]

        tokenizer = self.factory.create()
        tokenizer.train(vocab_size=int(self.tokenizer_settings.vocab_size), files_paths=song_midi_paths)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(self.tokenizer_path)
        extractor = RemiTokenExtractor(tokenizer)

        token_by_key: Dict[str, List[int]] = {}
        token_lengths: List[int] = []
        for song in songs:
            song_id = str(song.get("song_id", "UNKNOWN"))
            for bar in song.get("bars", []):
                bar_index = int(bar.get("bar_index", 0))
                key = f"{song_id}__bar_{bar_index:04d}"
                # Per-bar REMI inputs represent only this bar. Rebase to zero so
                # their tokens match generated feedback bars written at time zero.
                bar_midi = self.midi_builder.write_bars(
                    [bar],
                    bar_midi_dir / f"{self._safe_name(key)}.mid",
                    rebase_bar_indices=True,
                )
                ids = extractor.encode_midi(bar_midi)
                if not ids:
                    continue
                token_by_key[key] = ids
                token_lengths.append(len(ids))

        payload = {
            "tokenizer_path": "tokenizer.json",
            "token_by_key": token_by_key,
            "summary": {
                "song_count": int(len(songs)),
                "bar_token_count": int(len(token_by_key)),
                "vocab_size": int(len(tokenizer)),
                "mean_bar_tokens": float(np.mean(token_lengths)) if token_lengths else 0.0,
                "max_bar_tokens": int(max(token_lengths)) if token_lengths else 0,
                "min_bar_tokens": int(min(token_lengths)) if token_lengths else 0,
            },
        }
        self.tokens_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _load_tokens_payload(self) -> Dict[str, Any]:
        """Load token cache, repairing old non-portable tokenizer_path lines when needed."""
        text = self.tokens_path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = re.sub(
                r'^\s*"tokenizer_path"\s*:\s*.*?,\s*$',
                '  "tokenizer_path": "tokenizer.json",',
                text,
                count=1,
                flags=re.MULTILINE,
            )
            return json.loads(repaired)

    def tokenize_tensor_bar(self, tokenizer: Any, tensor: np.ndarray, output_path: str | Path, base_pitch: int) -> List[int]:
        """Tokenize one generated tensor bar through a temporary MIDI file."""
        midi_path = self.midi_builder.write_tensor_bar(tensor, output_path, base_pitch)
        return RemiTokenExtractor(tokenizer).encode_midi(midi_path)

    def tokenize_tensor_bar_in_memory(self, tokenizer: Any, tensor: np.ndarray, base_pitch: int) -> List[int]:
        """Tokenize one generated tensor bar through an in-memory symusic score."""
        from symusic import Note, Score, Tempo, TimeSignature, Track

        score = Score(int(self.render_config.ticks_per_beat))
        score.tempos.append(Tempo(0, qpm=float(self.render_config.tempo_bpm)))
        score.time_signatures.append(TimeSignature(0, 4, 4))
        events = self.midi_builder.tensor_renderer._tensor_events(
            np.asarray(tensor, dtype=np.float32), int(base_pitch), start_tick=0
        )
        for track_index in range(min(3, int(np.asarray(tensor).shape[0]))):
            active: Dict[int, tuple[int, int]] = {}
            notes: List[Any] = []
            track_events = [event for event in events if int(event["track"]) == track_index]
            for event in track_events:
                pitch = int(event["pitch"])
                tick = int(event["tick"])
                if event["type"] == "on":
                    active[pitch] = (tick, int(event["velocity"]))
                    continue
                onset = active.pop(pitch, None)
                if onset is not None and tick > onset[0]:
                    notes.append(Note(onset[0], tick - onset[0], pitch, onset[1]))
            score.tracks.append(Track(name=f"track_{track_index}", program=0, is_drum=False, notes=notes))
        return RemiTokenExtractor(tokenizer).encode_score(score)

    def _safe_name(self, value: Any) -> str:
        text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))
        return text[:180] or "item"
