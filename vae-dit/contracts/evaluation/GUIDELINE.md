# Evaluation Framework Interface Guideline

This is the authoritative interface specification for the Evaluation Framework.

## 1. Ownership

```text
src/diagnostics -> internal observations
src/export      -> public artifacts following contracts
evaluation      -> metrics, policies and reports
```

`src/diagnostics` may access model and pipeline internals and writes raw
observations. It must not decide PASS/WARN/FAIL or add generation rules.

`src/export` is the engineering-side contract adapter. It selects public fields,
removes private paths and implementation names, records shapes/dtypes/units,
alignment, provenance and hashes, marks optional inputs, and writes artifacts.
It must not calculate quality metrics or modify generated results.

`evaluation` reads public artifacts only. It calculates metrics, applies policy,
produces findings and writes reports. It must not import model, pipeline,
renderer or checkpoint implementation code.

`src/diagnostics` is now part of the same framework boundary, but only on the
engineering side.  A generation pipeline emits typed diagnostic events at
explicit processing boundaries.  Diagnostics collectors consume those events
and write versioned raw observations.  Exporters later turn those raw
observations into public artifacts.  Evaluators still never import
`src/diagnostics`.

## 2. Flat module layout

Test-point modules do not use per-module subdirectories. The test point is a
stable namespace encoded in each filename.

```text
evaluation/src/evaluation_framework/
  evaluation_api.py
  evaluation_context.py
  evaluation_registry.py
  evaluation_runner.py
  evaluation_artifact_store.py
  evaluation_reporting.py
  evaluation_codec_fidelity.py
  evaluation_dvae_fidelity.py
  evaluation_latent_probe.py
  evaluation_trajectory_teacher_forced.py
  evaluation_trajectory_rollout.py
  evaluation_renderer_consistency.py
  evaluation_oracle_ladder.py
  evaluation_trajectory_anchor_context.py

src/export/
  evaluation_export_base.py
  evaluation_codec_fidelity_exporter.py
  dvae_fidelity_artifact_export.py
  latent_probe_artifact_export.py
  evaluation_trajectory_teacher_forced_exporter.py
  evaluation_trajectory_rollout_exporter.py
  evaluation_renderer_consistency_exporter.py
  evaluation_dataset_tonality_exporter.py
  trajectory_anchor_context_artifact_export.py
```

## 3. Naming rules

Test-point names use lowercase `snake_case` and describe an information
boundary, not a model class or implementation branch:

```text
dataset_tonality
codec_fidelity
anchor_transport
dvae_fidelity
latent_probe
trajectory_teacher_forced
trajectory_rollout
renderer_consistency
attribution
oracle_ladder
```

Contract files:

```text
<test_point>__<artifact_role>.v<major>.schema.json
```

Metric descriptions:

```text
<test_point>__metrics.<language>.md
```

Run artifacts and reports:

```text
<run_id>__<test_point>__<role>.v<major>.<extension>
```

Examples:

```text
codec_fidelity__inputs.v1.schema.json
run_001__codec_fidelity__arrays.v1.npz
run_001__codec_fidelity__report.v1.md
```

Names must not contain absolute paths, spaces, checkpoint names or branch
names.

## 4. Contract and artifact rules

All contracts live in the flat directory:

```text
contracts/evaluation/v1/
```

Every artifact records schema version, run identity, dataset/split/hash, sample
identity, seed/primer/length, sampling identity, checkpoint roles and hashes,
relative file references, file hashes, array shapes/dtypes/units, alignment and
optional-input availability. Checkpoint paths and model-internal names are not
public fields. Public NPZ files must not use pickle/object arrays.

### 4.1 Absolute anchor transport

An absolute anchor is a bar-level coordinate-frame value, not a tensor feature
that must be repeated into every slot. `anchor_transport` is a distinct test
point for deterministic transport only:

```text
source -> encoding -> export -> trajectory input -> renderer input
```

Its v1 policy is `canonical_bar_anchor_v1`. Every compared value is an
integer MIDI semitone in the `transposed_model` coordinate frame, with the
declared transpose already applied and before clipping. Original-source and
rendered-frame facts may be retained separately but cannot enter the v1
exact-match assertion. An exact-identity boundary has an expected MAE of zero;
a nonzero difference is a contract failure, never a learned-model metric.

Learned `history -> predicted delta -> future anchor` behavior belongs to a
trajectory module and must not be included in anchor transport reports. A
relative tensor without its anchor is not expected to reconstruct absolute MIDI
pitch; that result is `UNAVAILABLE`, not a fidelity failure.

For trajectory teacher-forced observations, diagnostics writes only
`trajectory_future_position_raw.v1` facts. The module-specific exporter
deterministically projects them to
`<arm>__future_position.npz`, following
`trajectory_teacher_forced__future_position.v1.schema.json`. It may construct
aligned targets, masks and state metadata, but it must not calculate MSE or a
quality conclusion.

## 4.2 Diagnostics runtime rules

Generation diagnostics are attached through three stable concepts:

```text
stage     -> one typed transformation in the generation flow
event     -> facts emitted at a stage boundary
collector -> optional diagnostics module that consumes events and writes raw data
```

Stage implementations transform explicit state objects.  They may emit events,
but they must not know which collectors are active or where raw artifacts are
written.  Collectors may write raw JSON/NPZ references, but they must not
calculate evaluation metrics, make PASS/WARN/FAIL decisions, add repair rules
or change generated music.

New diagnostics modules must be added by introducing a typed event or consuming
an existing event.  The pipeline must not grow direct calls to concrete raw
writers.  If a diagnostic experiment needs counterfactual sampling, that
capability must be passed as a narrow injected function in an explicit context;
the experiment must not mutate the normal generation state.

The same rule applies to model-preparation pipelines. A training pipeline
publishes typed facts for encoding, pair preparation, split resolution,
training completion and serialized artifacts through one injected diagnostics
runtime. Raw collectors assemble their own requests from those facts. Adding a
dataset, codec or representation diagnostic must not add a collector argument,
raw-artifact request, or `record_stage` call to the training pipeline.

### 4.3 Cross-run trajectory anchor lineage

Model preparation, trajectory training and generation execute independently.
Their deterministic anchor facts are therefore connected by artifact lineage,
not by shared process memory or directory-name inference.

`trajectory_anchor_context` has three engineering-side contracts:

```text
encoded_input_manifest.v1
  -> trajectory_training_input_lineage_raw.v2
  -> trajectory_checkpoint_lineage.v1
```

The parent manifest hashes the exact encoded index and tensor archive consumed
by trajectory training. Training verifies that manifest before use and records
the same parent identity in both its raw lineage observation and checkpoint.
Later generation diagnostics can follow the checkpoint parent identity back to
the encoded rows used as primer or history.

Every bar-level join requires the complete identity:

```text
parent encoded-index hash
+ tensor_key
+ song_id
+ source_bar_index
+ applied_transpose_semitones
+ tensor schema version
```

Missing or mismatched identity fields are `UNAVAILABLE` evidence. They must
never be repaired by path guessing or inferred hashes. A codec-side `null`
anchor for an empty bar is not a broken identity: the v2 lineage contract must
record it as `runtime_fallback`, together with the explicitly configured
training fallback anchor. It must never be relabeled as a serialized codec
anchor.
`song_anchor` is a declared derivation (`round(median(base_pitch) within
song_id)`), while renderer clipping is a separate transform; neither is an
exact-identity assertion between independent stages.

### 4.4 Pitch-supervision runtime observations

`dvae_pitch_supervision_audit` and `dvae_pitch_gradient_probe` are engineering
raw diagnostics for a suspected reconstruction-channel failure. They record
runtime wiring and opt-in autograd facts, not quality metrics. The training
pipeline only publishes typed facts; injected collectors may use the existing
pitch loss with `autograd.grad`, but must not write parameter `.grad`, alter an
optimizer step, mutate model state, or reimplement the training loss.

Both diagnostics are status-led. `AVAILABLE` status must reference exactly one
flat raw observation with a hash. `UNAVAILABLE` must remove its stale
observation reference and explain the missing fact. Parameter groups use
stable logical identifiers (`decoder_input`, `decoder_hidden`,
`decoder_pitch_output`) rather than concrete parameter paths or layer names.

### 4.5 Physical trajectory objective observations

`physical_trajectory_objective` replaces the former assessment-time checkpoint
load with one status-led raw observation. The training or assessment composition
root may run the existing Stage 1 model and physical projector, but it must
write facts rather than metrics:

```text
validation normalized target, mask, clean reconstruction, denoised reconstruction
summary and token embeddings
clean/coherent/octave/track-swap/boundary-shuffle probe embeddings
original/one-octave-shifted projector outputs and validity masks
```

The observation also declares feature names/groups, normalizer facts,
validation-row-to-base-song alignment, fixed corruption settings and the
projector feature indices used by the octave check. Numeric arrays are finite
`float32`; validity masks are `bool`; NPZ files never use object dtypes. The evaluator alone derives the
mean baseline, R2, probe separation, embedding-health statistics,
octave-equivariance error, policy status and freezing marker.

The raw writer must not load an arbitrary checkpoint by path, infer an encoded
row from a filename, or decide a gate result. An unavailable capture removes
stale observations and writes a formal reason in its status artifact.

Physical trajectory raw v2 adds coverage facts required by the former
data-contract gate. It records all encoded bar-index rows before any window
selection, then records eligible contiguous-window distributions both before
and after per-song/global window limits. Coverage facts use aggregate counts
and public `base_song_id`/form labels only; tensor keys and encoded paths are
never exposed. The configured limit, derived per-song limit, and global
selection strategy are mandatory provenance, so smoke-test coverage cannot be
mistaken for full-corpus coverage.

## 5. One result directory

There is one result directory per run. Modules must not create module-specific
result directories:

```text
output/evaluation_runs/<run_id>/
  run_manifest.json
  index.json
  run_001__codec_fidelity__inputs.v1.json
  run_001__codec_fidelity__arrays.v1.npz
  run_001__codec_fidelity__report.v1.json
  run_001__codec_fidelity__report.v1.md
  run_001__codec_fidelity__report.v1.png
```

Temporary files must be atomically renamed into this directory before a module
completes.

## 6. Module API

Each test point has one exporter and one evaluator registered under the same
name:

```python
class ArtifactExporter:
    test_point: str
    input_contract: str
    output_contract: str

    def export(self, context) -> ArtifactBundle:
        """Read diagnostics and write public artifacts."""
```

```python
class ArtifactEvaluator:
    test_point: str
    required_artifacts: tuple[str, ...]

    def evaluate(self, artifact) -> AssessmentReport:
        """Read artifacts and produce independent metrics and findings."""
```

## 7. Single entry point

All runs go through:

```text
tools/evaluate.py
```

Supported modes are `export`, `evaluate` and `all`. The runner creates the run
directory, discovers registered modules, executes exporters, validates
contracts, executes evaluators, writes reports and updates `index.json`.

## 8. Adding a module

1. Choose a stable test-point name.
2. Add its schema under `contracts/evaluation/v1/`.
3. Add its music-facing metric description.
4. Add its engineering exporter under `src/export/` when the necessary diagnostics already exist. If the observations do not exist, record the exact missing fields, shape, units, alignment and provenance requirement in `docs/evaluation_framework_tracking.md`. When an experiment is explicitly approved for engineering implementation, `src/diagnostics` may add an injected, versioned raw capture and the pipeline may expose a diagnostic run mode. That mode may select or transform runtime inputs, but must not calculate quality metrics, apply policy or alter the generated result as a repair rule.
5. Add its artifact-only evaluator under `evaluation/src/evaluation_framework/`.
6. Register the module.
7. Add contract, exporter and evaluator tests.
8. Run `tools/evaluate.py` against a fixture bundle.

No new module should require central-runner changes beyond registry registration.
Adding a metric must not create a generation-time rule.

## 9. Versioning and status

Breaking changes to public fields, alignment, units or interpretation require a
new major schema version. Reports use `PASS`, `WARN`, `FAIL`, `UNAVAILABLE` and
`MONITOR`. `UNAVAILABLE` is missing evidence, not a quality judgment.
