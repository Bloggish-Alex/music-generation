# Dataset Tonality Profile

This assessment describes the reference dataset rather than judging a model. It reports typical tonal material, pitch-class usage, register, rhythmic density, and variation across works. It is a reference for codec, DVAE, latent, and generation assessments, not a correction rule or a total score.

Source facts are bar-level notes with pitch, onset, duration, velocity, voice index, meter, bar length, tempo when available, and applied global transposition. The profile must be computed from source notes, never inferred from tensors, latents, or generated MIDI. Profiles are grouped by original work after undoing recorded global transposition; dataset splits are complete-work units.

Report key distribution, bar-level tonal trajectory, pitch-class profile, MIDI register statistics, and rhythmic/voice activity. Empty bars, missing splits, invalid meter, and untraceable works are `UNAVAILABLE`, never substituted with defaults.
