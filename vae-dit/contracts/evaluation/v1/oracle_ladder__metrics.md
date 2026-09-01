# Oracle Ladder Metrics

The oracle ladder attributes information boundaries in order: source MIDI, codec tensor, DVAE latent, DVAE reconstruction, teacher-forced trajectory, free-running trajectory, and rendered MIDI. Each boundary reports harmony, register, rhythm, continuity, and structural metrics separately. Metrics from different representation spaces must not be added or presented as a causal percentage of final loss.
