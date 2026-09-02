from __future__ import annotations

from codec.action_labeler import ACTION_RETURN, ActionLabeler, ActionLabelerConfig
from data.core import BarRecord, NoteEvent, SongRecord, TrackRecord


def _bar(index: int, pitch: int) -> BarRecord:
    return BarRecord("fixture", "fixture.mid", index, 4.0, tracks=[TrackRecord(0, "track", [NoteEvent(pitch=pitch, onset_ql=0.0, duration_ql=1.0)])])


def test_return_search_starts_immediately_after_opening_anchor() -> None:
    bars = [_bar(index, 60 if index < 4 or index in {16, 17} else 67) for index in range(40)]
    config = ActionLabelerConfig(theme_anchor_bars=4, return_min_consecutive=2, return_similarity_threshold=.99, repeat_similarity_threshold=1.1)
    ActionLabeler(config).label_song(SongRecord("fixture", "fixture.mid", bars=bars))
    labels = [bar.action for bar in bars]
    assert labels[16:18] == [ACTION_RETURN, ACTION_RETURN]
    assert all(labels[index] != ACTION_RETURN for index in range(4))
