#!/usr/bin/env python3
"""Shared input loader for encoder diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.model_store import ModelBundle
from data.core_data import BarRecord, SongRecord


class EncoderInputLoader:
    """Load SongRecord data from model, parsed JSON, or raw music directory."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def load(
        self,
        model_dir: Optional[Path],
        model_bundle: Optional[Path],
        songs_json: Optional[Path],
        music_dir: Optional[Path],
    ) -> List[SongRecord]:
        if model_dir is not None or model_bundle is not None:
            bundle = ModelBundle.load(model_dir if model_dir is not None else model_bundle.parent)
            return self._songs_from_observation_pools(bundle)
        if songs_json is not None:
            payload = json.loads(songs_json.read_text(encoding="utf-8"))
            return [SongRecord.from_dict(item) for item in payload.get("songs", [])]
        if music_dir is not None:
            from data.music_input import InputParser

            return InputParser.from_style_config(self.config).parse_directory(music_dir)
        raise ValueError("One input source is required.")

    def _songs_from_observation_pools(self, bundle: ModelBundle) -> List[SongRecord]:
        bars_by_song: Dict[tuple[str, str], List[BarRecord]] = defaultdict(list)
        for pool in bundle.observation_to_bars.values():
            for bar in pool:
                bars_by_song[(bar.song_id, bar.file_path)].append(bar)
        songs: List[SongRecord] = []
        for (song_id, file_path), bars in sorted(bars_by_song.items(), key=lambda item: item[0]):
            ordered = sorted(bars, key=lambda bar: int(bar.bar_index))
            songs.append(SongRecord(
                song_id=song_id,
                file_path=file_path,
                genre=ordered[0].genre if ordered else None,
                form=ordered[0].form if ordered else None,
                bars=ordered,
            ))
        return songs
