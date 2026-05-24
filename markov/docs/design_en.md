# Music Markov Model — Design Document

## 1. Problem Statement

Given a corpus of MIDI files, we want to generate **new, stylistically coherent** music. The core challenge: music is a sequence where each note's pitch, duration, velocity, and rhythmic placement depend on what came before. A naive random generator produces gibberish — random notes with no melodic contour, no rhythmic sense, and no harmonic logic.

The model must:
- **Learn sequential dependencies**: which pitches tend to follow which, how durations relate to each other, how beat positions constrain note placement.
- **Handle multi-dimensional data**: every event carries pitch, duration, velocity (loudness), beat position, and instrument program.
- **Produce musically plausible output**: the generated sequence should sound like it belongs to the same style as the training corpus, not like uniform noise.

## 2. Why a Markov Chain?

Music is fundamentally a **sequential decision process**. The probability of the next note depends — to a useful approximation — on the preceding few notes. This is exactly what a Markov chain models:

$$P(e_t \mid e_{t-1}, e_{t-2}, \dots, e_{t-k})$$

where $k$ is the **order** of the chain.

**Why not an RNN / Transformer?** Markov chains are:
- **Interpretable**: you can inspect every transition probability directly.
- **Fast to train**: a single pass over the data, count-and-normalize.
- **No mode collapse**: you sample from the exact empirical distribution of the training corpus.
- **Musically adequate**: local context (3–4 previous events) captures melody contour, harmonic rhythm, and phrasing patterns well enough for many styles.

The trade-off: Markov chains have no long-term memory (no global structure like ABA form). But for short-to-medium generations and local coherence, they work surprisingly well.

## 3. MIDI Event Parsing

Before a Markov chain can learn, raw MIDI files must be converted into discrete `MusicEvent` objects. This is the job of `MidiParser`, which performs a **two-pass** extraction: metadata first, then per-part note/rest events.

### 3.1 Two-Pass Parsing

**Pass 1 — Metadata extraction.** The parser flattens the entire `music21.Score` into a single stream and scans for global musical markers:

| music21 class | Extracted information | Stored as |
|---------------|----------------------|-----------|
| `MetronomeMark` | Offset (seconds) + BPM (quarter-note beats per minute) | `MidiMetadata.tempos` |
| `TimeSignature` | Offset + numerator + denominator | `MidiMetadata.time_signatures` |
| `KeySignature` | Offset + number of sharps | `MidiMetadata.key_signatures` |

`flatten()` is used here because tempo, time signature, and key signature objects typically live in conductor/part-0 staves, not individual instrument parts. Flattening ensures they are found regardless of where they reside in the score hierarchy.

**Pass 2 — Per-part event extraction.** The parser iterates over each `Part` in `score.parts` and calls `part.flatten().notesAndRests`, which yields every `Note`, `Rest`, and `Chord` in chronological order (by offset). For each element, it produces one or more `MusicEvent` objects.

### 3.2 Time Signature Tracking

Time signature affects **beat position** — where a note falls within the bar — which is critical rhythmic information. The parser handles time signature changes **dynamically** as it walks through the note stream:

1. Metadata time signatures are sorted by offset into a chronological list `ts_events`.
2. A pointer `ts_idx` walks through this list. Before processing each note/rest, the parser checks: *has a time signature change occurred at or before this note's offset?*
3. If yes, `current_ts` updates to the new `(numerator, denominator)`, and `ts_idx` advances.
4. A **default of `(4, 4)`** is used before the first time signature is encountered.

```
current_ts = (4, 4)          # default
ts_events = sorted(metadata.time_signatures, key=offset)
ts_idx = 0

for each note/rest:
    while ts_idx < len(ts_events) and ts_events[ts_idx].offset <= note.offset:
        current_ts = (ts_events[ts_idx].numerator, ts_events[ts_idx].denominator)
        ts_idx += 1
    beat_pos = _beat_position(note.offset, current_ts.num, current_ts.den)
```

### 3.3 Beat Position Calculation

`_beat_position(offset, ts_num, ts_den)` computes where a note sits within a bar on a 16-division grid (0–15):

```python
beat_length = 4.0 / ts_den           # e.g., 4/4 → 1.0 quarter, 6/8 → 0.5 quarter
bar_length  = ts_num * beat_length   # e.g., 4/4 → 4.0 quarters, 6/8 → 3.0 quarters
pos_in_bar  = offset % bar_length    # remainder within the current bar
# map to 0..15 grid
return int(pos_in_bar / bar_length * BEAT_DIVISIONS) % BEAT_DIVISIONS
```

Key properties:
- `offset` is in music21's **quarter-length units** (1.0 = one quarter note), measured from the start of the piece.
- The modulo `offset % bar_length` removes the bar count, leaving only the position within the current bar.
- Scaling by `BEAT_DIVISIONS` (16) maps to a 16th-note grid — position 0 is the downbeat, 4 is the second beat (in 4/4), 8 is the third beat, etc.
- The final `% BEAT_DIVISIONS` is a safety clamp (prevents wrap-around from floating-point edge cases).

### 3.4 Duration Calculation

Durations come from music21's `el.quarterLength`, which represents note length in quarter-note units (e.g., a half note = 2.0, an eighth note = 0.5, a dotted quarter = 1.5).

Raw `quarterLength` values are continuous floats. The system **quantizes** them to the closest entry in `DURATION_CATEGORIES`:

```python
def _quantize_duration(ql):
    candidates = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 3.0, 1.5, 0.75, 0.667, 0.333]
    idx = argmin(|ql - v| for v in candidates)
    return idx
```

This is nearest-neighbor quantization: the category whose numeric quarter-length value is closest to the actual `quarterLength` is chosen. The 11 categories cover common Western notation values, including dotted rhythms and triplet subdivisions.

The returned value is an **index** (0–10) into `DURATION_CATEGORIES`, not the quarter-length itself. This index becomes the `duration_idx` field in `MusicEvent`.

### 3.5 Handling Notes, Rests, and Chords

The parser handles three element types from `notesAndRests`:

**Rest** (`el.isRest`):
- `pitch = -1` (sentinel for silence)
- `velocity_idx = 0` (no velocity for rests)
- Duration and beat position are computed normally.
- Program is inherited from the part's instrument.

**Note** (`el.isNote`):
- Pitch: `el.pitch.midi` (0–127). Defaults to 60 (C4) if pitch is somehow missing.
- Velocity: `el.volume.velocity`, defaulting to 80 if not specified.
- Duration and beat position computed normally.

**Chord** (`el.isChord`):
- **Expanded**: each pitch in the chord becomes its own `MusicEvent` with the same offset, duration, velocity, and beat position.
- For example, a C major triad at offset 4.0 becomes three consecutive events: C4, E4, G4 — all with identical offset, duration, beat position, and velocity.
- This is a deliberate simplification: the Markov chain sees chord tones as sequential events, not simultaneous ones. The sequential order follows music21's pitch ordering (lowest to highest).

### 3.6 Program (Instrument) Detection

For each part, the parser scans for an `Instrument` object:

```python
for inst_obj in part.recurse().getElementsByClass(instrument.Instrument):
    prog = inst_obj.midiProgram
    break
```

`recurse()` is used instead of `flatten()` because instruments are often nested inside measures within parts. If no instrument is found, `prog` defaults to 0 (Acoustic Grand Piano).

All events from a given part share the same `program` value. Multi-instrument MIDI files produce events with different `program` fields corresponding to each part.

### 3.7 Parsing Summary

```
MIDI file
    │
    ▼
music21 converter.parse() → Score
    │
    ├── flatten() → extract MetronomeMark, TimeSignature, KeySignature → MidiMetadata
    │
    └── for each Part:
            ├── recurse() → find Instrument → program number
            ├── track current time signature (sorted ts_events + walking pointer)
            └── flatten().notesAndRests:
                    ├── Rest  → pitch=-1, velocity=0
                    ├── Note  → one MusicEvent
                    └── Chord → N MusicEvents (one per pitch)
                                    │
                                    ▼
                            List[MusicEvent] + MidiMetadata
```

## 4. Core Architecture: Dual-Chain Design

The system uses **two separate Markov chains**:

| Chain | Models | Order (default) |
|-------|--------|-----------------|
| **Main chain** (`HigherOrderMarkovChain`) | Full token: pitch + duration + beat_position + velocity + program | 3 |
| **Duration chain** (`DurationMarkovChain`) | Duration indices only | 4 |

### Why two chains?

The main chain models all dimensions **jointly** in a single token. This means the chain learns that "C4 (MIDI 60) followed by E4 (64)" is common, and also that certain pitch transitions tend to occur with certain duration patterns. Joint modeling captures cross-dimension correlations.

However, joint tokens make the state space enormous — each of the 5 dimensions multiplies the vocabulary size. This makes **rhythmic patterns** harder to learn at higher orders, because the chain sees each duration in the context of specific pitches, diluting the purely rhythmic signal.

The **duration chain** solves this by modeling duration sequences **in isolation**, at a higher order (4 vs. 3). It sees only the rhythm — "eighth, eighth, quarter, quarter" — stripped of pitch and velocity. After the main chain generates a sequence, the duration chain **replaces** the durations with ones sampled from its own model, while keeping pitches, velocities, and beat positions intact.

This separation gives rhythmic patterns more statistical power (higher effective sample count per duration context) and a higher-order model (4 vs. 3) for longer rhythmic dependencies.

## 5. Token Design

Each musical event is encoded as a string token:

```
p{pitch}_d{duration_idx}_b{beat_position}_v{velocity_idx}_pg{program}
```

Example: `p60_d2_b0_v3_pg0`
- `p60`: MIDI pitch 60 (C4)
- `d2`: duration index 2 (quarter note, from `DURATION_CATEGORIES`)
- `b0`: beat position 0 (downbeat)
- `v3`: velocity index 3 (mezzo-piano, ~64)
- `pg0`: program 0 (Acoustic Grand Piano)

**Why encode as strings?** The Markov chain uses dictionary lookups on tuples of tokens. String tokens are hashable, human-readable, and trivially reversible (parse with `split("_")`). No embedding layer needed.

### Discretization (Quantization)

Music is continuous (pitch in Hz, duration in seconds, velocity 0–127). Markov chains need **discrete states**. The system quantizes:

| Dimension | Discretization | Categories |
|-----------|---------------|------------|
| Pitch | Raw MIDI number (already discrete 0–127) | 128 |
| Duration | Nearest-neighbor to 11 named categories | whole, half, quarter, eighth, sixteenth, thirtysecond, dotted_half, dotted_quarter, dotted_eighth, triplet_quarter, triplet_eighth |
| Velocity | Nearest-neighbor to 8 dynamic markings | ppp through fff |
| Beat position | 16th-note grid within a bar (`offset % bar_length`) | 16 divisions |
| Program | Raw MIDI program number (0–127) | 128 |

Rests are encoded as `pitch=-1` with `velocity_idx=0`. This makes silence a learnable event in the chain.

## 6. Higher-Order Chain with Back-Off Smoothing

### Training (`.fit()`)

For each token sequence and each order $o \in [1, k]$:

1. Slide a window of size $o+1$ over the sequence.
2. The first $o$ tokens form the **history** (key), the $(o+1)$-th is the **next token**.
3. Increment the count: `counts[o][(t_i, ..., t_{i+o-1})][t_{i+o}] += 1`
4. After all sequences are processed, convert counts to probabilities:

$$P(\text{next} \mid \text{history}) = \frac{\text{count}(\text{history}, \text{next})}{\sum_{n} \text{count}(\text{history}, n)}$$

### Generation with Back-Off (`.sample_next()`)

Back-off smoothing handles **unseen contexts** at test time:

1. Try to sample from the **highest order** $k$ using the last $k$ tokens as history.
2. If that history was never seen in training, **back off** to order $k-1$ (use the last $k-1$ tokens).
3. Repeat until order 1. If nothing matches, return `None` (generation stops).

This is a simple but effective form of smoothing — it guarantees the model can always produce *something* as long as any 1-gram was seen.

### Temperature Sampling

Before sampling, probabilities are reweighted:

$$p_i' = \frac{p_i^{1/T}}{\sum_j p_j^{1/T}}$$

- $T = 1.0$: original distribution (default)
- $T > 1.0$: flattens distribution → more variety, less faithful
- $T < 1.0$: sharpens distribution → more deterministic, more faithful to training peaks

## 7. Duration Chain Design

The duration chain is architecturally simpler than the main chain:
- **No back-off**: fixed order (default 4), no fallback to lower orders.
- **Fixed order only**: trains and samples at exactly one order.
- **No temperature**: always samples from raw empirical probabilities (temperature is handled at the main-chain level).
- **Fallback on miss**: if a duration context is unseen, pick a random duration category.

### Why no back-off?

The duration vocabulary is small (11 categories). At order 4, the maximum number of possible contexts is $11^4 = 14,641$. With enough MIDI data, most common rhythmic patterns are well-covered. When a context is truly unseen, random fallback is acceptable because durations are more "forgiving" than pitches — a random quarter note is less jarring than a random pitch.

## 8. Generation Pipeline

The full generation flow:

```
1. Pick random seed tokens from training distribution
                    ↓
2. Main Markov chain generates N full tokens (pitch+duration+beat+velocity+program)
   using back-off sampling with temperature
                    ↓
3. Parse tokens → MusicEvent objects
                    ↓
4. Duration chain overrides each event's duration_idx:
   - Extract duration sequence from generated events
   - Feed first (duration_order) durations as seed
   - Generate replacement duration sequence
   - Reconstruct events with new durations, keeping pitch/velocity/beat/program
                    ↓
5. Humanizer adds Gaussian noise to velocities and timings
                    ↓
6. MidiGenerator converts events → music21 Score → MIDI file
```

### Why override durations after the main chain?

This is a **post-processing** approach. The main chain generates a sequence with all dimensions jointly modeled. The duration chain then "corrects" the rhythm to be more coherent, based on its higher-order, pitch-independent model. This works because:

- The main chain captures the joint distribution (e.g., "high notes tend to be short").
- The duration chain captures pure rhythmic grammar (e.g., "eighth-eighth-quarter is a common pattern").
- Overriding durations with the duration chain's output preserves the joint structure for pitch/velocity while improving rhythmic flow.

The first `duration_order` durations from the main chain are used as the **seed** for the duration chain, so the rhythm starts from the main chain's output and then transitions into the duration chain's model.

## 9. Humanizer

After generation, small perturbations make the output sound less mechanical:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `velocity_jitter` | 8.0 | Gaussian noise added to MIDI velocity (std dev = 8, clamped to 1–127) |
| `timing_jitter_sec` | 0.015 | Gaussian noise added to note onset times (std dev = 15ms) |

Human players never hit exact velocities or microsecond-precise timing. These perturbations introduce realistic imperfection without changing the musical content.

## 11. Enhanced Features (from markov_abc.py Integration)

Analysis of a second Markov-chain implementation (`markov_abc.py`) — a first-order, factorized, chord-constrained system for ABC notation — inspired four enhancements that are now integrated as opt-in features. All default to off for full backward compatibility.

### 11.1 Additive Smoothing (replacing pure back-off)

**The problem with pure back-off**: When a high-order context is unseen, back-off falls to progressively lower orders. At order 0 or when the context is completely unseen, the sampler returns `None` (generation stops) or degrades to a 1-gram distribution with no context awareness.

**Additive smoothing** blends the empirical transition distribution with a **unigram prior**:

$$P_{\text{blended}}(\text{next} \mid \text{history}) = (1 - \alpha_{\text{eff}}) \cdot P_{\text{empirical}} + \alpha_{\text{eff}} \cdot P_{\text{unigram}}$$

where the effective smoothing strength is:

$$\alpha_{\text{eff}} = \frac{\alpha}{\alpha + \text{count}(\text{history})}$$

**How it works**:

1. During `fit()`, build a unigram prior distribution from all 1-gram counts, filtering to tokens with count ≥ `min_unigram_count` (default 3).
2. During `sample_next()`, for each context that exists in the chain, blend the empirical distribution with the unigram prior:
   - High-frequency contexts (`count(history)` large) → `alpha_eff` ≈ 0 → empirical dominates.
   - Low-frequency contexts (`count(history)` small) → `alpha_eff` ≈ 1 → prior dominates.
3. The unigram prior reflects global token frequency: common pitches get more smoothing mass than rare ones.

**Parameters**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.0 (off) | Pseudo-count strength. 0.05 is a typical music value. |
| `min_unigram_count` | 3 | Minimum occurrence for a token to enter the unigram prior |

**Key design**: `alpha_eff` is adaptive — it depends on the sample size backing each context row. Large counts → little smoothing; small counts → strong smoothing. This is equivalent to adding `alpha` pseudo-counts drawn from the unigram distribution to every transition row before normalization.

### 11.2 Chord-Aware Generation

The most impactful enhancement: explicitly model chord progressions and constrain pitch generation to chord tones.

**Chord Extraction in MidiParser**:

Two sources of chord information:

| Source | Detection | Example |
|--------|-----------|---------|
| Explicit labels (ABC `"..."` syntax, lead sheets) | `isinstance(el, harmony.ChordSymbol)` → `el.figure` | `"C"`, `"G7"`, `"Am"` |
| Implicit (MIDI chords, no labels) | `harmony.chordSymbolFromChord(chord_obj).figure` | inferred from sounding pitches |

Extracted chord symbols are stored in `MidiMetadata.chord_symbols` as `(offset, chord_name)` tuples.

**Chord-to-Pitch-Class Mapping**:

A lookup table `CHORD_QUALITY_INTERVALS` maps chord quality strings to semitone intervals above the root:

```python
"C"    → root=C(0) + [0,4,7]        → {0, 4, 7}      (C, E, G)
"G7"   → root=G(7) + [0,4,7,10]     → {7, 11, 2, 5}   (G, B, D, F)
"Am"   → root=A(9) + [0,3,7]        → {9, 0, 4}       (A, C, E)
"Dm7"  → root=D(2) + [0,3,7,10]     → {2, 5, 9, 0}    (D, F, A, C)
```

13 chord qualities are recognized: major, minor, dim, aug, 7, maj7, m7, dim7, m7b5, sus4, sus2, 6, m6.

**ChordMarkovChain Class**:

A fixed-order Markov chain (default order 2) trained on **bar-level, deduplicated** chord sequences. Deduplication is critical: `["C","C","C","G7","G7","C"]` → `["C","G7","C"]`. This captures harmonic rhythm (when chords change) rather than bar counts.

The chain is structurally identical to `DurationMarkovChain` but operates on string chord symbols instead of integer duration indices.

**Chord-Constrained Pitch Sampling** (in PitchMarkovChain, see 11.4):

When generating pitches for a bar with active chord `"Am"`:
1. Compute the allowed pitch class set: `{9, 0, 4}`.
2. Filter the Markov chain's candidate next-pitch list to only those whose MIDI pitch mod 12 falls in the allowed set.
3. Sample from the filtered distribution.
4. For the **first note of each bar**, the constraint is strict (must be a chord tone). Subsequent notes in the same bar are unconstrained (allowing passing tones and neighbor tones).

### 11.3 Bar-Constrained Rhythm Generation

**The problem**: The original `DurationMarkovChain.generate()` produces a flat sequence of N durations with no regard for bar boundaries. A 4/4 bar should contain durations summing to exactly 4.0 quarterLength units; free generation may produce bars of any length.

**New method `generate_bars()`**:

```
For each bar:
    remain = bar_length_ql  (e.g. 4.0 for 4/4, 3.0 for 3/4)
    while remain > tolerance:
        candidates = [idx for idx, ql in dur_values if ql <= remain + tolerance]
        sample from filtered Markov distribution
        remain -= sampled_ql
```

**Constraint sampling** via candidate filtering: at each step, only duration indices whose quarterLength value fits within the remaining bar space are eligible. The Markov transition probabilities are re-normalized over the eligible subset.

**Floating-point tolerance**: Triplet values (2/3 ≈ 0.666..., 1/3 ≈ 0.333...) cannot sum exactly to 3.0 or 4.0 in floating point. A tolerance of 0.005 quarterLength (~1.5 ms at 120 BPM) handles this. If the bar cannot be filled exactly, the remaining space is filled with the smallest available duration (triplet_eighth or sixteenth).

### 11.4 Factorized Pitch Chain (PitchMarkovChain)

**Why a separate pitch chain?** The joint token in the main chain encodes pitch alongside 4 other dimensions. A pure pitch chain (higher order, e.g. 4) can learn melodic contour patterns with greater statistical density:

- Joint token vocabulary: ~128 × 11 × 16 × 8 × 128 ≈ 23 million possible tokens
- Pure pitch vocabulary: 128 MIDI pitches (plus -1 for rest)

**PitchMarkovChain** mirrors `DurationMarkovChain`:
- Fixed order (default 4), no back-off.
- Trained on pure pitch integer sequences (rests excluded).
- `sample_next()` accepts an optional `allowed_pitch_classes: Set[int]` parameter for chord-constrained filtering.
- When `allowed_pitch_classes` is provided, candidates not matching are excluded before sampling.

### 11.5 Updated Architecture: Multi-Chain Design

With all features enabled, the system uses **four chains**:

| Chain | Order | Models | Constraint |
|-------|-------|--------|------------|
| Main chain | 3 | Joint token (pitch+duration+beat+velocity+program) | — |
| Duration chain | 4 | Duration indices | Bar-length (opt-in) |
| Chord chain | 2 | Chord symbols (bar-level) | — |
| Pitch chain | 4 | Pitch integers | Chord pitch-classes (opt-in) |

All auxiliary chains are optional; the system degrades gracefully to the original dual-chain design when they are disabled.

## 12. Updated Generation Pipeline

With all enhancements enabled, the generation flow becomes **hierarchical**:

```
1. Chord chain generates chord progression (N bars)
                    ↓
2. Duration chain generates rhythm bars (bar-constrained)
                    ↓
3. Pitch chain generates pitches (chord-constrained per bar)
                    ↓
4. Main chain generates joint tokens (velocity, program, beat position)
                    ↓
5. Overlay: replace pitches from step 3, durations from step 2
                    ↓
6. Humanizer adds Gaussian noise
                    ↓
7. MidiGenerator → MIDI / ABC file
```

When enhancements are disabled, the flow is unchanged from the original (Section 8).

## 13. Summary of Key Design Decisions (Updated)

| Decision | Rationale |
|----------|-----------|
| String tokens (not embeddings) | Hashable, human-readable, no training needed |
| Full joint token (not factorized) | Captures cross-dimension correlations |
| Separate duration chain | Gives rhythm more statistical power at higher order |
| Back-off + additive smoothing | Hybrid: back-off for unseen contexts, additive smoothing prevents zero-probability tokens |
| Factorized auxiliary chains (pitch, chord) | Opt-in: higher statistical power per dimension, chord-aware generation |
| Chord-constrained pitch sampling | First note of each bar restricted to chord tones; explicit harmonic coherence |
| Bar-constrained rhythm generation | Accumulate durations to exact bar length; metrically correct output |
| Discretize to ~11–16 categories per dimension | Enough resolution for musical sense, small enough for statistical density |
| Post-processing duration override | Preserves joint pitch-velocity structure while improving rhythm |
| Gaussian humanization | Simple, effective, no training needed |
| Pickle-based persistence with versioning | Zero-dependency serialization; backward-compatible with v1 models |
