# Engineering Evaluation Data Export Interface

## Purpose

Engineering code exposes verifiable facts. The independent Evaluation Framework computes metrics, applies policy, and publishes musical analysis.

```text
src/diagnostics -> internal observations
src/export      -> public artifacts
evaluation      -> metrics and reports
```

## `src/diagnostics`

Diagnostics may access model and pipeline internals and record raw observations needed for evaluation, including bar tensors and latent values, target and predicted states, future trajectory predictions, anchor and register information, and alignment indexes for MIDI, WAV, and tensors.

Diagnostics must not publish PASS/WARN/FAIL results, rank root causes, set generation constraints, apply tonal masks or repair penalties, or convert a stage error into a percentage of final quality loss.

## `src/export`

An exporter is the engineering-side contract adapter. It selects public fields, removes absolute paths and internal implementation names, stores arrays as NPZ with keys, shapes, and dtypes, records schema versions and relative SHA-256 references, records run identity, and atomically closes the manifest. It does not recompute quality metrics.

## Public bundle

```text
evaluation_manifest.json
generation_metric_inputs.json
bar_tensor_schema.json
bar_tensors.npz
generation_trace.json
generated.mid
generated.wav              # optional
```

`evaluation_manifest.json` is the entry point. Every other artifact is referenced by a relative path and hash.

## Availability and paired experiments

Missing optional input must be represented explicitly, for example `{"availability":"unavailable","reason":"optional_audio_missing"}`. Never use empty arrays, zero values, or `unknown` as a substitute for unavailable data.

Free-running and teacher-forced paired runs must share dataset hash, split, song, primer, bar count, seed, initialization, sampling steps, noise hash, and checkpoint identity. Only `arm` and `history_source` may differ. The exporter creates the paired manifest and the evaluator verifies it again.

## Stability

- Do not modify a closed artifact bundle.
- Do not publish absolute paths or pickle data.
- Do not let an evaluator load a checkpoint.
- New fields are additive; a semantic breaking change requires a new schema major version.
