# Evaluation Contract Naming And Ownership

For the complete interface specification, see [GUIDELINE.md](GUIDELINE.md).
This README is a short reference; `GUIDELINE.md` is authoritative.

For the music-facing explanation of the full data flow, derived variables and
diagnostic metrics, see [Evaluation Diagnostics Reference](../../docs/evaluation_diagnostics_reference.zh-CN.md).

This directory contains the public data contracts shared by the engineering
exporters and the independent evaluation package.

## Naming rule

Every versioned contract uses:

```text
<test_point>__<artifact_role>.v<major>.schema.json
```

`<test_point>` names the test boundary, not an implementation module. The
allowed test points are:

```text
common
dataset_tonality
codec_fidelity
anchor_transport
trajectory_anchor_context
dvae_fidelity
latent_probe
trajectory_teacher_forced
trajectory_rollout
attribution
renderer_consistency
oracle_ladder
```

`<artifact_role>` is one of `manifest`, `inputs`, `profile`, `pair`, or
`report`. A contract change that breaks readers increments the major version.

## Ownership

```text
src/diagnostics -> internal observations
src/export      -> public artifacts following these contracts
evaluation      -> metrics, interpretation, policy and reports
```

Exporters may read diagnostics, but they do not calculate quality metrics or
write diagnoses. Evaluators may not import model, pipeline, renderer or
checkpoint code.

## Artifact rules

- Paths inside public artifacts are relative to the artifact manifest.
- Files are immutable after the manifest is closed.
- Large arrays use NPZ with an explicit key, dtype and shape declaration.
- JSON stores identity, alignment, provenance and availability; it does not
  duplicate large numeric arrays.
- Checkpoint identity is recorded by role and SHA-256 only. Checkpoint paths
  and model-internal names are private.
- Missing optional inputs are represented as `UNAVAILABLE`, never as zeros.
