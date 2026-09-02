# Final Codec V2 Encoding

The only supported encoding format is `bar_tensor_schema.v2`. The encoder
parses real music21 measures with a nominal 0.25 QL grid, 48-slot capacity,
and padding represented separately from rests.

Each run contains `voice_tensors.npz`, `bar_tensor_index.json`,
`encoding_manifest.json`, and `songs.json`. The archive contains
`voice_tensors float32[N,18,48,6]`, `slot_valid_mask bool[N,48]`,
`slot_durations_ql float32[N,48]`, 12-bin `bar_contexts`, and base-pitch
arrays. Invalid slots are excluded from all feature and evaluation denominators.

Lane 0 is melody, lanes 1..16 are slot-local harmony, and lane 17 is bass.
Cross-bar melody continuity uses stable `source_note_id`; a clipped
continuation is a first-valid-slot hold, not a new onset.

Run `bin/encode.sh --dataset-root <dir> --dataset-name <name> --run-id <id>`.
It writes `output/$MUSICAI_STAGE/models/<name>/encoded/<id>/`. Legacy V1,
three-voice tensors, 11-bin projections, and version-routed output folders
are unsupported and must be re-encoded.
