# Final Codec V2 Diagnostics

Codec V2 produces a 31D bar diagnostic vector using only valid slots. The
framework captures parser integrity, quantization audit, performance controls,
form/action alignment, and source-to-tensor codec fidelity. These are MONITOR
observations, not model-quality scores.

Quantization residuals are grouped by source-file identity and actual meter.
CC64 is source-only metadata and never a note tensor or generation target.
Every raw observation is bound to canonical manifest and index hashes.
