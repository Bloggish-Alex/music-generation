# Codec V2 Evaluation Design

Evaluation uses the existing framework. `AVAILABLE` requires aligned manifest, index, arrays, slot grid, and source facts. The manifest hash is SHA-256 of exact bytes with a `sha256:` prefix; the config hash is SHA-256 of canonical sorted-key JSON. All slot metrics exclude invalid padding and use `slot_durations_ql`; valid duration sums must equal the actual measure duration. Quantization residuals are per-file/per-meter onset/end absolute deltas to 0.25 QL and are `MONITOR` facts, never claims of source-exact timing. CC64 is raw observed data or explicitly unavailable.
# Follow-up semantics

`form_action_alignment` searches for RETURN immediately after the opening
anchor window. It does not wait for the midpoint of the song, so middle and
late thematic recurrences are both detectable.

Quantization residual samples are stored only in
`quantization_residual_samples.v2.npz`; `songs.json` retains per-song/per-meter
summary counts and p95/max values, never sample lists.
