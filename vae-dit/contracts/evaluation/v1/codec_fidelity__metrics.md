# Bar Codec Fidelity

This assessment compares a bar's source harmony profile with the profile still readable from its codec tensor. It determines whether information was lost before DVAE; it does not assess generated music or require exact bar copying.

`base_pitch` is a bar-level coordinate reference outside the relative tensor. Its deterministic transport is assessed separately by `anchor_transport`; inability to recover absolute pitch from the tensor alone is not a fidelity failure.

Report these observations independently:

- source pitch-class profile, weighted by note duration;
- audible tensor-track profile reconstructed from pitch, activity, onset, and duration fields;
- chroma-condition profile, when the codec exposes such a condition;
- inter-bar change direction and magnitude;
- independent register and density gaps.

The overall median-register gap is only an aggregate monitor. Sustain-slot weighting, semantic voice assignment, and duration representation can affect it, so it does not by itself diagnose anchor transport or register-trajectory failure.

Do not combine these gaps into one score or automatically derive a tonal mask, repair rule, or model-change recommendation. With global-transposition augmentation, comparisons must use `applied_transpose_semitones` to return to the original-work frame and report both original-work and augmented-version counts.
