# Evaluation Framework

This package consumes versioned artifacts from `contracts/evaluation` and does
not import the engineering model, pipeline, renderer or checkpoint code.

## Local execution

From the repository root, use the single framework entry point:

```powershell
python tools/evaluate.py `
  --input-root path/to/engineering/diagnostics `
  --output-root output/evaluation_runs `
  --run-id run_001 `
  --modules all `
  --mode all
```

The runner creates the only result directory for the run, exports registered
modules, evaluates the resulting artifacts and writes `index.json`. At present,
the shared core is implemented; test-point modules are tracked in
`docs/evaluation_framework_tracking.md`.

## Ownership

Engineering writes observations through `src/export`. This package computes
metrics and interpretations from those observations. No evaluator result is
used as a generation-time rule. The formal interface is
`contracts/evaluation/GUIDELINE.md`.
