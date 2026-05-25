# Music Markov Model — Design Document

## Part I: System Overview & Data Flow

### 1.1 What This System Does

The system analyzes a corpus of music files (MIDI, ABC, Humdrum **kern), extracts
per-measure rhythmic features, clusters similar measures into a finite set of
"musical states," and learns a statistical model from which new music-like phrase
structures can be generated.

```
Music Files (.mid/.abc/.krn)
        │
        ▼
  MeasureExtractor       ◄── Part II
  (parse → vectorize)
        │
        ▼
  MeasureClusterer       ◄── Part III
  (KMeans, k clusters)
        │
        ▼
  classify_files()       ◄── file_labels: List[List[int]]
        │
        ├──► TransitionMatrixBuilder   ◄── Part IV
        ├──► PersistenceDurationBuilder ◄── Part V
        └──► StartDistributionBuilder  ◄── Part VI
                │
                ▼
          MusicModel
                │
                ▼
         PhraseGenerator     ◄── Part VII
```

### 1.2 Core Data Abstraction: `file_labels`

The central intermediate representation is `file_labels: List[List[int]]` — one
ordered integer list per music file, where each integer is the cluster label (0 to
k-1) of a measure in that file. File boundaries are preserved. The three model
components (transition, persistence, start) are all built purely from this
representation, without accessing the original music data.

**Example** — `op01n01a.krn` (14 measures, Corelli Op.1 No.1):

```
Labels: [1, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 0]
```

### 1.3 Training Data (Corelli Corpus)

Throughout this document we use a model trained on Arcangelo Corelli's complete
trio sonatas (Opp. 1–6, 250 files, 9,279 measures) with k=5 clusters:

| # Files | # Measures | # Clusters (k) | Total Transitions | Total Runs |
|--------:|-----------:|---------------:|------------------:|-----------:|
|     250 |      9,279 |              5 |             2,672 |      1,610 |

---

## Part II: Measure Vectorization

### 2.1 Overview

Each measure (bar) in a music file is converted into an 8-dimensional real-valued
vector. The eight features capture different aspects of rhythmic character.
All features are computed from the note list of a single measure — chord notes
are expanded into individual note entries sharing the same onset and duration;
rests contribute only to the silence ratio, not to other features.

### 2.2 Feature Definitions

For a measure with *n* notes, durations `d₁, d₂, ..., dₙ` (in quarterLength),
onsets `o₁, o₂, ..., oₙ` (in quarterLength from bar start), and bar length *B*
(in quarterLength, e.g., B=4.0 for 4/4):

#### 2.2.1 note_density — Notes per Beat

```
note_density = n / B
```

- **Musical meaning**: How "busy" the measure is, normalized by time signature.
- **Interpretation**: Low (< 2) = sparse / chorale style; medium (2–4) = moderate
  activity; high (> 5) = florid figuration with many rapid notes.
- **Example** (Measure 0 of op01n01a.krn): 18 notes / 4.0 beats = **4.50** — busy but not extreme.

#### 2.2.2 mean_duration — Average Note Length

```
mean_duration = (1/n) · Σ dᵢ
```

- **Musical meaning**: Are the notes predominantly long or short? Inversely related
  to note_density.
- **Interpretation**: Low (< 0.5) = sixteenth-note motion; medium (0.5–1.0) =
  eighth-note motion; high (> 1.5) = quarter-note or longer, sustained style.
- **Example** (Measure 0): (1.0 + 1.0 + 1.5 + 0.5 + ...) / 18 = **0.889** — close to quarter-note pulse.

#### 2.2.3 duration_variance — Variability of Note Lengths

```
duration_variance = (1/n) · Σ (dᵢ - mean_duration)²
```

- **Musical meaning**: How much the note durations "spread out" — a measure of
  rhythmic uniformity vs. variety.
- **Interpretation**: Low (< 0.15) = uniform durations (e.g., all eighth notes);
  high (> 0.4) = mixed note values in the same measure.
- **Example** (Measure 1): With durations ranging from 0.25 to 2.0, variance = **0.378** — highly varied.

#### 2.2.4 short_note_ratio — Proportion of Ornamental Notes

```
short_note_ratio = count(dᵢ < 0.5) / n
```

The threshold 0.5 quarterLength = one eighth note. Notes shorter than this are
typically sixteenth notes (0.25), dotted sixteenths (0.375), or triplet eighths (0.333).

- **Musical meaning**: What fraction of notes are rapid/ornamental?
- **Interpretation**: 0.0 = no notes shorter than an eighth; > 0.5 = heavily
  ornamented passage with many sixteenth notes.
- **Example** (Measure 3): 16 out of 26 notes are shorter than 0.5 → **0.615** — heavily ornamented.

#### 2.2.5 silence_ratio — Proportion of "Empty" Time

```
total_sounding = Σ dᵢ
silence_ratio = max(0, min(1, 1.0 - total_sounding / B))
```

Clamped to [0, 1] since overlapping voices can make `total_sounding > B`.

- **Musical meaning**: How much "breathing room" the measure has. A rest-heavy
  measure has high silence_ratio.
- **Interpretation**: 0.0 = every beat contains sounding notes; > 0.3 =
  significant rests; > 0.5 = very sparse.
- **Example** (Measure 0): total_sounding = 16.0, B = 4.0, silence = 1 − 16/4 = 0? No —
  actually the notes have overlapping onsets (chordal texture from four parts),
  so many note durations overlap. Let's compute: total_sounding = sum of
  durations = 16.0, B = 4.0. silence = 1 − 16/4 = **0.0** after clamping —
  continuous sound.

#### 2.2.6 offbeat_ratio — Rhythmic Displacement from the Beat

```
offbeat_ratio = count(oᵢ % 1.0 ≠ 0) / n
```

An onset is "offbeat" if it does not fall on a quarter-beat grid boundary
(0.0, 1.0, 2.0, 3.0 in 4/4).

- **Musical meaning**: Degree of rhythmic displacement / syncopation at the onset
  level. Measures where notes consistently start between beats have high offbeat_ratio.
- **Interpretation**: 0.0 = every note lands squarely on a beat; > 0.4 =
  significant off-beat activity.
- **Example** (Measure 0): onsets [0.0, 1.0, 2.0, 3.5, 0.0, 1.0, 2.0, 3.5, 0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 0.0, 1.0, 2.0, 3.5].
  5 onsets at 3.5 and 2.5 are off-beat → 5/18 = **0.278**.

#### 2.2.7 syncopation_score — Emphasized Off-beat Notes

A syncopation is counted when an **offbeat** note has a duration **longer than**
the most recent on-beat note. This captures the classic "syncopation" pattern
where an off-beat note is stressed (held longer) relative to the preceding beat.

```
For each note i in onset order:
  if is_onbeat(oᵢ):
      prev_onbeat_dur = dᵢ
  else:  # offbeat
      if dᵢ > prev_onbeat_dur AND prev_onbeat_dur > 0:
          count += 1

syncopation_score = count / n
```

- **Musical meaning**: True syncopation — not merely being off the beat, but
  being rhythmically *emphasized* off the beat (longer than the preceding
  on-beat note). This is the "tied-across-the-barline" or "stressed upbeat" pattern.
- **Interpretation**: 0.0 = no syncopated emphasis; > 0.1 = noticeable syncopation;
  > 0.2 = strongly syncopated passage.
- **Example** (Measure 0): All offbeat notes at 3.5 (quarters) follow an onbeat
  at 3.0 (quarters). Since 0.5 ≤ 0.5, no syncopation counted → **0.000**.

#### 2.2.8 entropy — Rhythmic Diversity

Duration values are binned into 11 categories:
`[0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, ∞)`.

```
histogram = count(dᵢ in each bin)
prob[b] = histogram[b] / total_notes
entropy = -Σ prob[b] · log₂(prob[b])    for bins with prob[b] > 0
```

The maximum entropy with 11 bins is log₂(11) ≈ 3.46 bits.

- **Musical meaning**: Shannon entropy of the duration distribution. A measure
  using only one duration value has entropy 0; one using many different values
  has higher entropy.
- **Interpretation**: < 0.5 = very simple (one duration type); 1.0–1.5 = typical
  baroque texture; > 2.0 = highly diverse rhythmic vocabulary.
- **Example** (Measure 0): Mainly quarters (1.0), dotted quarters (1.5), and
  eighths (0.5) → **1.481** bits.

### 2.3 Feature Summary — Corelli Example

| Feature | Meas.0 | Meas.1 | Meas.2 | Meas.3 | Meas.4 |
|---------|-------:|-------:|-------:|-------:|-------:|
| note_density | 4.50 | 5.25 | 6.25 | 6.50 | 3.75 |
| mean_duration | 0.889 | 0.762 | 0.640 | 0.615 | 1.067 |
| duration_variance | 0.127 | 0.378 | 0.210 | 0.275 | 0.162 |
| short_note_ratio | 0.000 | 0.190 | 0.320 | 0.615 | 0.000 |
| silence_ratio | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| offbeat_ratio | 0.278 | 0.429 | 0.560 | 0.615 | 0.067 |
| syncopation_score | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| entropy | 1.481 | 1.723 | 1.697 | 1.489 | 1.103 |

---

## Part III: Measure Clustering

### 3.1 Algorithm: StandardScaler + KMeans

The 9,279 measure vectors are clustered into k=5 groups using scikit-learn's
KMeans, with a StandardScaler normalization step:

```
Step 1: X = stack all 8-d vectors into (9279, 8) matrix
Step 2: X_scaled = StandardScaler.fit_transform(X)
         → each feature column has μ=0, σ=1
Step 3: kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
         labels = kmeans.fit_predict(X_scaled)
Step 4: centroids_raw = scaler.inverse_transform(kmeans.cluster_centers_)
         → centroids back in original feature space for interpretability
```

**Why StandardScaler?** The 8 features have different scales: note_density ranges
0–15, entropy ranges 0–3.5, silence_ratio is bounded 0–1. Without normalization,
larger-magnitude features would dominate the Euclidean distance metric in KMeans.

### 3.2 KMeans Inertia

Inertia = sum of squared distances from each point to its assigned centroid.
For this model: **35,005**.

### 3.3 Classification Pipeline

To classify a new measure:

```
extract(file) → MeasureInfo → vectorize(info) → MeasureVector.as_array()
    → scaler.transform([array]) → kmeans.predict([scaled]) → cluster label
```

The function `classify_files()` wraps this for all files in a directory,
preserving file boundaries. Files with fewer than 2 measures are skipped
(to ensure meaningful transitions).

### 3.4 Cluster Centroids (Corelli k=5)

| Cluster | note_density | mean_dur | dur_var | short_ratio | silence | offbeat | syncopation | entropy |
|--------:|-------------:|---------:|--------:|------------:|--------:|--------:|------------:|--------:|
| 0 | 2.477 | 1.823 | 0.414 | 0.000 | 0.001 | 0.013 | 0.000 | 0.701 |
| 1 | 4.582 | 0.828 | 0.216 | 0.035 | 0.003 | 0.336 | 0.004 | 1.174 |
| 2 | 7.585 | 0.512 | 0.139 | 0.555 | 0.000 | 0.537 | 0.015 | 1.190 |
| 3 | 0.596 | 0.619 | 0.008 | 0.086 | 0.644 | 0.716 | 0.000 | 0.023 |
| 4 | 4.909 | 0.767 | 0.172 | 0.098 | 0.000 | 0.525 | 0.233 | 1.271 |

### 3.5 Musical Interpretation of Each Cluster

**Cluster 0 — Sparse, Sustained, On-beat** ("Chorale / Long-note")
- Lowest note_density (2.48), highest mean_duration (1.82), near-zero offbeat and
  syncopation. Characteristic of slow-moving, chordal, square-rhythm passages.

**Cluster 1 — Moderate Activity, Some Off-beat** ("Standard Figuration")
- Medium density (4.58), medium durations (0.83), moderate offbeat (0.34).
  The "default" texture for Corelli's trio sonata writing.

**Cluster 2 — Dense, Short-note, Off-beat** ("Florid Runs")
- Highest density (7.59), shortest average duration (0.51), highest short-note
  ratio (0.55). Busy sixteenth-note figuration with strong off-beat presence.

**Cluster 3 — Sparse, High Silence** ("Rest Transition")
- Very low density (0.60), high silence (0.64), low entropy (0.02). Measures
  dominated by rests — typically cadence endings or phrase boundaries.

**Cluster 4 — Off-beat with Strong Syncopation** ("Syncopated Passage")
- Highest syncopation score (0.233), high offbeat (0.53). Passages where
  off-beat notes are longer/stressed — the classic hemiola or tied-across-barline
  pattern. A distinctive feature of Corelli's style.

### 3.6 Cluster Assignment — Complete Example

`op01n01a.krn` (14 measures) → labels: `[1, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 0]`

This piece alternates between moderate figuration (cluster 1), florid runs
(cluster 2), and sustained passages (cluster 0). Note that cluster 3 (rest-heavy)
and cluster 4 (syncopated) do not appear in this particular movement.

---

## Part IV: Transition Matrix

### 4.1 Core Concept

The transition matrix captures **how musical states follow each other** across
the corpus. It is a k×k row-stochastic matrix where `P[i, j]` = probability that
a measure in cluster i is immediately followed by a measure in cluster j.

### 4.2 Counting Algorithm

```
For each file's label sequence [L₀, L₁, L₂, ..., Lₘ₋₁]:
    For each adjacent pair (Lₜ, Lₜ₊₁):
        IF skip_self_transitions AND Lₜ == Lₜ₊₁:
            skip  (persistence is captured by PersistenceDuration)
        ELSE:
            count_matrix[Lₜ, Lₜ₊₁] += 1
```

**Why skip self-transitions?** A→A transitions are redundant with the persistence
duration model. If the transition matrix also modeled self-stay probabilities,
the generator would "double count" persistence: once through the matrix staying
on the same state, and once through run-length sampling. By excluding A→A, the
transition matrix answers only "after this state ends, where do we go next?"

**No cross-file leakage:** The last measure of file N never transitions to the
first measure of file N+1. Each file is an independent observation.

### 4.3 Normalization

```
For each row r:
    row_sum = Σⱼ count_matrix[r, j]
    IF row_sum > 0:
        prob_matrix[r, :] = count_matrix[r, :] / row_sum
    ELSE:
        prob_matrix[r, :] = zeros  (zero-outgoing state)
```

Rows sum to 1.0 (or 0.0 for zero-outgoing states). Zero-outgoing states can
exist when a cluster only ever appears as the last measure of a file — in
generation, these are handled by sampling uniformly from all states as a fallback.

### 4.4 Corelli k=5 — Transition Probability Matrix

| From ↓ / To → | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------------|----------:|----------:|----------:|----------:|----------:|
| **Cluster 0** | 0.0000 | **0.8305** | 0.1191 | 0.0151 | 0.0352 |
| **Cluster 1** | **0.5025** | 0.0000 | 0.3067 | 0.0059 | 0.1849 |
| **Cluster 2** | 0.1147 | **0.7707** | 0.0000 | 0.0000 | 0.1147 |
| **Cluster 3** | 0.0682 | **0.6136** | 0.1136 | 0.0000 | 0.2045 |
| **Cluster 4** | 0.1613 | **0.6387** | 0.1935 | 0.0065 | 0.0000 |

### 4.5 Key Observations

- **Cluster 1 is a universal predecessor**: Every other cluster most frequently
  transitions to cluster 1 (0.50–0.83 probability). Cluster 1 is the "hub" of
  Corelli's style — the default texture other gestures return to.
- **Cluster 0 → 1 is the strongest single transition** (0.83): sustained passages
  almost always lead into the standard figuration.
- **Cluster 2 → 1 is second strongest** (0.77): florid runs resolve back to
  standard figuration, not straight to rest.
- **Cluster 3 has only 44 outgoing transitions** in the entire corpus — it's a
  rare endpoint/transition state.
- **27 files skipped all-self**: 27 out of 250 files consisted entirely of
  self-transitions (runs staying in one cluster) — these contributed nothing
  to the transition matrix but did contribute to persistence duration.

### 4.6 Next-State Sampling

```python
def sample_next(current: int, seed: Optional[int] = None) -> int:
    row = prob_matrix[current]
    rng = np.random.RandomState(seed)
    if row.sum() == 0:
        return rng.choice(n_clusters)       # uniform fallback
    return rng.choice(n_clusters, p=row)    # weighted by probabilities
```

**Example**: Starting from cluster 1 with seed=42:
Row = [0.5025, 0.0000, 0.3067, 0.0059, 0.1849]
→ ~50.3% chance of cluster 0, ~30.7% chance of cluster 2, ~18.5% chance of cluster 4.

---

## Part V: Persistence Duration

### 5.1 Core Concept

While the transition matrix answers "where to next?", persistence duration answers
"how long do we stay here?" For each cluster, it models the distribution of
consecutive-run lengths observed in the corpus — how many measures in a row a
given cluster tends to persist.

### 5.2 Run-Length Definition

A **run** is a maximal sequence of consecutive identical cluster labels within a
single file. For the label sequence:

```
[1, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 0]
```

The runs are:

| Run # | Label | Length | Explanation |
|------:|------:|-------:|-------------|
| 1 | 1 | 2 | Two 1's at start |
| 2 | 2 | 2 | Then two 2's |
| 3 | 1 | 1 | Isolated single 1 (ABA pattern, smoothed) |
| 4 | 0 | 2 | Two 0's |
| 5 | 1 | 1 | Isolated single 1 |
| 6 | 2 | 5 | Five consecutive 2's |
| 7 | 0 | 1 | Final 0 |

### 5.3 Preprocessing Chain

Before counting run-lengths, two optional filters can smooth noise:

#### ABA Smoother (always active)

Isolated single-measure patterns of the form **A → B → A** (where B is different
from A and has length 1) are smoothed to **A → A → A**. This removes single-measure
"blips" caused by classification noise at cluster boundaries.

```
Before: [1, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 0]
                            ↑ isolated 1 (run #3, 1 measure)
                            between runs of cluster 2 (left) and 0 (right)?
No: left=2, right=0, they differ → not ABA, kept as-is.

But run #5 (single 1): [..., 0, 0, 1, 2, ...] → left=0, right=2, differ → not ABA.
In our example, neither single 1 is sandwiched by the same cluster,
so no smoothing occurs.
```

A classic ABA example would be: `[2, 2, 1, 2, 2]` → `[2, 2, 2, 2, 2]` (the
isolated "1" is absorbed by the surrounding "2"s).

#### Short-Run Merger (optional, `min_run_length` > 1)

Runs shorter than `min_run_length` are merged into the longer of their two
neighbors (right wins ties). This is useful for removing very short
gestures that may be classification artifacts.

### 5.4 Distribution Building

```
For each file's (possibly preprocessed) labels:
    Scan linearly to identify runs → (cluster, length) pairs
    Append each length to run_lengths[cluster]

Result: run_lengths = {0: [2, 1, ...], 1: [2, 1, ...], 2: [2, 5, ...], ...}
```

### 5.5 Corelli k=5 — Persistence Duration Summary

```
  Cluster    Runs      Mean       Std   W.Avg  Run-length histogram (length: count)
----------------------------------------------------------------------------------------------
        0     460      4.73      8.12       5  1:205 2:82 3:31 4:28 5:22 6:17 9:9 7:9 10:7 12:6 (+27 more)
        1     646      6.81      9.70       7  1:176 2:106 3:68 4:44 5:36 6:28 8:23 7:19 11:15 10:14 (+35 more)
        2     309      6.27      7.51       6  1:72 2:66 4:26 3:24 5:17 6:14 8:10 10:10 7:8 9:7 (+23 more)
        3      36      1.61      1.85       2  1:31 2:2 8:2 7:1
        4     159      4.48      5.65       4  1:54 2:33 3:24 4:6 5:6 7:4 16:4 6:4 11:3 8:3 (+12 more)
```

### 5.6 Interpreting the Histogram

**Cluster 0** (460 runs): The most common run-length is 1 (205 runs, 44.6%),
meaning 0 often appears as a single-measure gesture. But the mean is 4.73, pulled
up by a long tail — the longest runs stretch to ~84 measures (continuous sustained
passages). The standard deviation (8.12) confirms the right-skewed distribution.

**Cluster 1** (646 runs): The most frequent cluster overall. Run-length 1 is also
most common (176 runs, 27.2%), but longer runs up to 11 measures appear regularly.
Mean 6.81 with std 9.70 — highly variable.

**Cluster 3** (36 runs): Very few runs, mostly of length 1 (31/36 = 86.1%).
This cluster rarely persists beyond a single measure — consistent with its role
as a "cadence/rest transition" state.

### 5.7 Weighted Average (Expected Duration)

When generating a phrase, the **weighted average** (W.Avg) is used as the
deterministic expected run-length for each cluster:

```
W.Avg(cluster) = int(round( Σ(length × frequency) / total_runs ))
```

For example, cluster 0: W.Avg = 5 means the generator will typically emit ~5
measures of cluster 0 before transitioning. This is deterministic (no random
jitter) so the same seed always produces the same result.

---

## Part VI: Start Distribution

### 6.1 Core Concept

The start distribution models **which cluster a piece of music is most likely
to begin with**. It is a simple categorical distribution over the k clusters.

### 6.2 Algorithm

```
For each file in the corpus:
    IF file has at least 1 label:
        start_counts[labels[0]] += 1

start_probs = start_counts / total_files
```

Only the first label matters — the rest of the file is ignored for this
computation.

### 6.3 Corelli k=5 — Start Distribution

| Cluster | Count | Probability | Interpretation |
|--------:|------:|------------:|---------------|
| 0 | 74 | 0.296 | Sparse/sustained openings |
| 1 | 102 | 0.408 | Standard figuration openings |
| 2 | 41 | 0.164 | Dense florid openings |
| 3 | 26 | 0.104 | Rest-transition openings |
| 4 | 7 | 0.028 | Syncopated openings |

**Key observation**: Cluster 1 (standard figuration) is the most common opening
(40.8%), followed by cluster 0 (sustained, 29.6%). Cluster 4 (syncopated passages)
almost never starts a piece — it appears mid-phrase. This matches musical
intuition: Corelli's sonatas typically begin with either a sustained chord or
moderate figuration, not with strong syncopation.

### 6.4 Sampling

```python
def sample(seed: Optional[int] = None) -> int:
    rng = np.random.RandomState(seed)
    return rng.choice(states, p=start_probs)
```

With seed=42, corelli_k5 starts with cluster 1 (~40.8% chance). With seed=99,
it might start with cluster 0 (~29.6% chance).

---

## Part VII: Phrase Generation

### 7.1 Core Concept

The phrase generator produces a sequence of cluster labels of a target length
by repeatedly: (1) sampling how long to stay in the current state from the
persistence model, (2) emitting that many copies of the state, and (3) sampling
the next state from the transition matrix.

### 7.2 Algorithm (Single Phrase)

```
Input: num_measures (target length), seed (optional)
Output: labels (list of integers, length = num_measures)

1. If seed is None, pick a random seed from [0, 2³¹-1]
2. Initialize seed counter n = seed
3. Sample start state: state = start_distribution.sample(seed=n); n += 1
4. While len(labels) < num_measures:
    a. run_len = persistence_duration.sample_duration(state)
    b. run_len = min(run_len, num_measures - len(labels))  # clamp at target
    c. labels.extend([state] * run_len)                   # emit run
    d. If len(labels) >= num_measures: break
    e. state = transition_matrix.sample_next(state, seed=n); n += 1
5. Return labels
```

### 7.3 Seed Counter Determinism

Each sampling step consumes one distinct seed value (`seed`, `seed+1`, `seed+2`,
...). Since each model component's `sample()` method uses `RandomState(seed)`,
the entire generation is fully deterministic given one base seed:

```
seed=42 → start: seed 42, first duration: seed 42 (deterministic), next state: seed 43, ...
seed=42 → always produces the same output
seed=99 → always produces a different, but equally deterministic, output
```

### 7.4 Concrete Example (Corelli k=5, seed=42, 30 measures)

```
Generated: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 0]
```

Step-by-step trace:

| Step | Action | State | Run-length | Notes |
|-----:|--------|------:|-----------:|-------|
| 1 | start_dist.sample(seed=42) | 1 | — | Start in cluster 1 |
| 2 | persist.sample_duration(1) | — | 7 | W.Avg for cluster 1 = 7 |
| 3 | emit | 1 | 7 | `[1,1,1,1,1,1,1]` |
| 4 | trans.sample_next(1, seed=43) | 0 | — | Row 1 → 0 (50.3% chance) |
| 5 | persist.sample_duration(0) | — | 5 | W.Avg for cluster 0 = 5 |
| 6 | emit | 0 | 5 | `[1×7, 0,0,0,0,0]` |
| 7 | trans.sample_next(0, seed=44) | 2 | — | Row 0 → 2 (11.9% chance) |
| 8 | persist.sample_duration(2) | — | 6 | W.Avg for cluster 2 = 6 |
| 9 | emit | 2 | 6 | `[1×7, 0×5, 2,2,2,2,2,2]` |
| 10 | trans.sample_next(2, seed=45) | 4 | — | Row 2 → 4 (11.5% chance) |
| 11 | persist.sample_duration(4) | — | 4 | W.Avg for cluster 4 = 4 |
| 12 | emit | 4 | 4 | `[1×7,0×5,2×6, 4,4,4,4]` |
| 13 | trans.sample_next(4, seed=46) | 1 | — | Row 4 → 1 (63.9% chance) |
| 14 | persist.sample_duration(1) | — | 7 | W.Avg for cluster 1 = 7 |
| 15 | emit | 1 | 7 | `[1×7,0×5,2×6,4×4, 1×7]` |
| 16 | trans.sample_next(1, seed=47) | 0 | — | Row 1 → 0 (50.3% chance) |
| 17 | persist.sample_duration(0) | — | 1 | Remaining: 1 measure, clamped |
| 18 | emit | 0 | 1 | Done: 30 measures total |

### 7.5 Multi-Phrase Generation

Each phrase is generated independently with its own seed (`base_seed + phrase_index`),
restarting from a fresh start state. This honors the "file boundary" principle:
no cross-phrase transitions.

```
generate_phrases([20, 15, 25], seed=100) produces:

Phrase 0 (20): [1,1,1,1,1,1,1, 2,2,2,2,2,2, 1,1,1,1,1,1,1]
Phrase 1 (15): [1,1,1,1,1,1,1, 2,2,2,2,2,2, 1,1]
Phrase 2 (25): [1,1,1,1,1,1,1, 0,0,0,0,0, 1,1,1,1,1,1,1, 0,0,0,0,0, 1]
```

Each phrase gets its own start state and independent transitions.

### 7.6 Why Run-length then Transition (Not Measure-by-Measure)

A naive Markov chain would sample state-by-state: at each measure, consult
P[current, :] to get the next state. The problem is that self-transitions are
excluded from the matrix, so the chain would change state at every single measure
— producing jittery, unrealistic output where no cluster persists.

The two-phase approach (duration first, then transition) decouples two distinct
musical decisions: "how long does this texture last?" (answered by persistence
duration) and "which texture comes next?" (answered by the transition matrix).

### 7.7 Timeline Visualization

The `plot_timeline()` function renders a label sequence as a grid of colored
squares. Each run becomes one row, with one square per measure. When the cluster
changes, the next run starts a new row below.

For the generated 30-measure phrase:

```
[1×7]  🟩🟩🟩🟩🟩🟩🟩  ×7
[0×5]  🟦🟦🟦🟦🟦      ×5
[2×6]  🟧🟧🟧🟧🟧🟧    ×6
[4×4]  🟪🟪🟪🟪        ×4
[1×7]  🟩🟩🟩🟩🟩🟩🟩  ×7
[0×1]  🟦              ×1
```

Each row contains only one cluster; the visual groupings immediately show phrase
structure — sustained textures (long rows) vs. brief gestures (single squares).

---

## Appendix: Quick Reference

### A.1 Feature Vector (8 dimensions)

| # | Feature | Formula | Range |
|---|---------|---------|-------|
| 1 | note_density | n / B | 0 – ~15 |
| 2 | mean_duration | mean(dᵢ) | 0.1 – 4.0 |
| 3 | duration_variance | var(dᵢ) | 0 – ~2 |
| 4 | short_note_ratio | count(dᵢ < 0.5) / n | 0.0 – 1.0 |
| 5 | silence_ratio | 1 − Σdᵢ / B | 0.0 – 1.0 |
| 6 | offbeat_ratio | count(oᵢ % 1 ≠ 0) / n | 0.0 – 1.0 |
| 7 | syncopation_score | (see §2.2.7) / n | 0.0 – ~0.5 |
| 8 | entropy | −Σ p·log₂(p) | 0.0 – 3.46 |

### A.2 Model Components

| Component | Answers the Question | Data Structure |
|-----------|---------------------|---------------|
| TransitionMatrix | After state i ends, where do we go? | (k×k) row-stochastic matrix |
| PersistenceDuration | How many measures does state i last? | Per-cluster run-length lists + stats |
| StartDistribution | Which state starts a piece? | k-element probability distribution |

### A.3 Generation Algorithm

```
start → [duration] → emit run → [transition] → next state → [duration] → ...
  ↑                                                        ↓
  └────────────────── loop until target length ────────────┘
```
