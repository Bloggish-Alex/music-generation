# Encoding Analysis Design

## Purpose

The encoding analysis path measures how much source harmony survives in the semantic bar tensor. It is a read-only diagnostic and does not participate in training or generation. Its main purpose is fault isolation: determine whether harmonic information is already lost by the codec, or whether a later model such as the DVAE is the more likely bottleneck.

The current thresholds are engineering heuristics used to route investigation. They are not yet stable musical-quality acceptance standards. Metrics without an explicit threshold below must be compared across representations, datasets, or iterations under the same evaluation setup.

## Entrypoint

```text
bash bin/analyze_bar_codec_harmony_oracle.sh <dataset-name>
```

The Bash wrapper sources `bin/init_env.sh`, resolves the model directory as `output/${MUSICAI_STAGE}/models/<dataset-name>`, verifies that its `encoded/` directory exists, and invokes the Python CLI through `PYTHON_BIN`. The lower-level Python interface is:

```text
python bin/analyze_bar_codec_harmony_oracle.py --model-dir <model-dir>
```

## Shared Relative Chroma

Both the semantic codec and harmony oracle call `codec.relative_chroma.relative_chromagram`. A shared implementation prevents the analyzer from evaluating a different target from the one used by the encoder.

### Input

- Notes: a sequence of note objects or mappings.
- Required fields: integer MIDI `pitch`, quarter-length `duration_ql`, and MIDI `velocity`.
- Optional `base_pitch`: an integer MIDI pitch. The minimum physical pitch is used when it is omitted.
- `velocity_scale`: a configurable positive normalization scale. The current configuration uses `127`.

### Processing

For note `i`, the relative pitch-class bin and weight are:

```text
bin_i = (pitch_i - (base_pitch mod 12)) mod 12
weight_i = max(duration_ql_i, 0) * clamp(velocity_i, 0, velocity_scale) / velocity_scale
```

The twelve accumulated bins are normalized only when their total is positive. An empty or zero-weight note sequence returns an all-zero vector.

`FixedChromagramProjector` converts the resulting `relative_chromagram` from `float32[12]` to the tensor's `relative_chroma_embedding` in `float32[11]`. It copies bins 0 through 9 and stores `(bin10 + bin11) / 2` in the last dimension. This deterministic compression cannot preserve the distinction between source bins 10 and 11.

### Output

- `relative_chromagram`: NumPy `float32[12]`.
- `relative_chroma_embedding`: NumPy `float32[11]`.
- Chromagram values are non-negative and sum to 1 when total note weight is positive.

### Simple Example

C3 with duration `1.0` and velocity `127`, and G3 with duration `1.0` and velocity `64`, use C3 as `base_pitch`. Their raw weights are `1.0` and `64/127`. After normalization, relative chroma bins 0 and 7 are approximately `0.664921` and `0.335079`. Different velocities therefore produce different contributions even when durations are equal.

## BarCodec Harmony Oracle

### Inputs

The analyzer reads:

- `encoded/songs.json`: source note records grouped by song and bar.
- `encoded/bar_tensors.npz`: keyed `float32[3, 16, 18]` semantic tensors.
- `encoded/bar_tensor_index.json`: ordered metadata rows containing `song_id`, `bar_index`, and `tensor_key`.

`sample_count` is the number of index rows evaluated. `active_bar_count` is the number whose source chromagram has positive mass. Per-bar metrics exclude inactive bars.

### Compared Representations

For every indexed bar, the analyzer builds one target and two candidates, all represented as normalized `float32[12]` vectors.

#### Source Chroma Target

The target is recomputed from the raw notes in `songs.json` using the shared duration-and-velocity-weighted relative chromagram. This is the reference distribution that the tensor representations are expected to retain.

#### Semantic Physical Chroma

This candidate approximates the harmony reconstructible from the tensor's Melody, Harmony, and Bass pitch/state tracks:

1. Tensor feature 0 is multiplied by `pitch_scale` to recover relative pitch. The current default is `24.0`.
2. A slot is active when feature 2 (`note_on`) or feature 3 (`hold`) is greater than `0.5`.
3. Every active relative pitch is mapped softly to 12 pitch classes. Circular semitone distance is converted to a Gaussian logit using `pitch_class_sigma`, currently `0.35`, and normalized with softmax.
4. Membership is summed across three voices and sixteen time slots, then normalized to total mass 1.

Each active voice-slot contributes one unit before final normalization. This approximates duration through repeated active slots, but it does not restore source velocity weighting. The Harmony track can also represent several simultaneous inner notes by one mean pitch. A gap from the source target can therefore reveal information lost during semantic voice assignment or physical tensor construction.

#### Tensor Chroma Condition

Tensor features 7 through 17 repeat the same `float32[11]` relative-chroma condition across voices and slots. The analyzer averages those copies, copies dimensions 0 through 9 into output bins 0 through 9, duplicates the final merged dimension into output bins 10 and 11, and normalizes the result.

The recovered bins 10 and 11 each receive half of their original combined normalized mass. Their individual source values cannot be reconstructed. This candidate measures information retained by the condition channel, not information recoverable from the physical pitch tracks.

## Per-Bar Metrics

Let `t_b` be the source vector and `c_b` a candidate vector for active bar `b`. Let `N` be the number of active bars and `D=12`.

### MSE

```text
mse = sum_b sum_d (c_b[d] - t_b[d])^2 / (N * D)
```

MSE measures average bin-level reconstruction error. `0` is exact, and lower is better. Because all compared vectors are normalized non-negative distributions, the theoretical range is `0` to `1/6`, but there is no code-level pass threshold. Use it to compare the physical representation with the condition representation, or to compare iterations on the same dataset. A high physical MSE with a low condition MSE points to loss in semantic pitch/state tracks rather than loss in the stored condition.

### Cosine Similarity

```text
cosine_b = dot(t_b, c_b) / max(norm(t_b) * norm(c_b), 1e-8)
```

For these non-negative chroma vectors, values normally lie between `0` and `1`. `1` means the pitch-class emphasis has the same direction, while a value near `0` means little overlap. Cosine is less sensitive than MSE to the exact concentration of the distribution.

- `cosine_mean`: average over active bars. The current decision logic requires at least `0.95` for the tensor condition and `0.85` for semantic physical chroma.
- `cosine_p10`: 10th percentile. Ten percent of active bars score at or below this value. It exposes a weak tail that a strong average can hide; it has no current pass threshold.
- `cosine_p50`: median. Half of active bars score at or below this value. It describes a typical bar and has no current pass threshold.

A high median with a much lower p10 usually means most bars are represented well but a minority need targeted diagnosis. Do not respond by adding per-bar repair rules first; inspect whether those bars share a systematic representation, parsing, or data pattern.

### Distribution Spread

```text
target_std  = std(flatten(all t_b))
decoded_std = std(flatten(all c_b))
std_ratio   = decoded_std / max(target_std, 1e-8)
```

These metrics measure contrast across all bar/bin values, not musical pitch variance inside one bar.

- `target_std` is the reference spread and is not a quality score by itself.
- `decoded_std` is candidate spread.
- `std_ratio` near `1` means the candidate retains approximately the same concentration and contrast as the target. A value below `1` indicates flattening or collapse; above `1` indicates exaggerated concentration. The current physical-harmony decision requires `std_ratio >= 0.70`; there is no upper threshold and no condition-channel spread threshold.

Spread cannot establish correctness alone. A candidate can have `std_ratio` near `1` while placing mass in the wrong bins, so it must be read with cosine and error metrics.

### Mean L2 Distance

```text
mean_l2 = mean_b(norm(c_b - t_b))
```

`0` is exact and lower is better. For normalized non-negative vectors, the theoretical range is `0` to `sqrt(2)`. Unlike MSE, L2 reports one distance per bar before averaging and is easier to compare with individual worst-bar distances. It has no current pass threshold.

## Transition Metrics

The analyzer groups rows by `song_id`, sorts by `bar_index`, and only uses pairs where the right index is exactly the left index plus one and both bars are active. For each accepted pair:

```text
target_delta    = target[right] - target[left]
candidate_delta = candidate[right] - candidate[left]
```

- `pair_count`: number of valid adjacent pairs. It measures coverage, not quality. Unexpectedly low coverage indicates inactive bars, missing indices, or discontinuous song/bar indexing.
- `mse`: mean squared error between target and candidate delta components. `0` is exact; lower is better. There is no current threshold.
- `cosine_mean`: mean directional alignment of harmonic changes. Values can range from `-1` to `1` because deltas contain signed values. `1` means movement in the same direction, `0` means unrelated movement or a zero-norm delta, and `-1` means opposite movement. The current physical-transition decision requires at least `0.50`.
- `target_delta_norm`: mean magnitude of source harmonic movement; it is a reference, not a score.
- `decoded_delta_norm`: mean magnitude of candidate movement. The ratio `decoded_delta_norm / target_delta_norm` near `1` means movement magnitude is retained; below `1` means transitions are attenuated; above `1` means they are exaggerated. This ratio is interpretive only and has no current threshold.

The condition-transition metrics are reported for comparison but are not used by the current final decision. Transition scores also do not measure functional harmony, cadence correctness, or perceptual quality; they only compare changes in these relative-chroma vectors.

## Current Decision Rules

| Decision | Current condition | Meaning |
| --- | --- | --- |
| `tensor_chroma_condition_retained` | condition `cosine_mean >= 0.95` | The deterministic condition channel retains enough source chroma for the current routing heuristic. |
| `semantic_physical_harmony_retained` | physical `cosine_mean >= 0.85` and physical `std_ratio >= 0.70` | The three physical tracks retain enough per-bar pitch-class alignment and contrast. |
| `semantic_physical_harmony_movement_retained` | physical transition `cosine_mean >= 0.50` | Adjacent-bar harmonic movement is directionally retained. |

No current decision uses MSE, p10, p50, mean L2, transition MSE, delta norms, or condition-transition metrics. Those fields provide supporting evidence and regression sensitivity. The thresholds above must be revalidated through repeated experiments before becoming stable acceptance criteria.

## Diagnosis And Response

### Condition Channel Fails

Signal: condition `cosine_mean < 0.95`.

Interpretation: the deterministic 12D-to-11D condition representation or its target alignment loses substantial source harmony before model training.

Investigation and possible response:

- Verify that encoder and analyzer use the same relative-chroma implementation and bar alignment.
- Inspect whether merging pitch classes 10 and 11 causes the observed errors.
- Consider a 12D or otherwise invertible condition representation if the merge is material.
- Fix the codec data contract before increasing DVAE capacity.

### Condition Passes But Physical Harmony Fails

Signal: condition passes, but physical `cosine_mean < 0.85` or physical `std_ratio < 0.70`.

Interpretation: source harmony survives in conditioning features but is not fully expressible through Melody, Harmony, and Bass pitch/state tracks. Likely causes include polyphony collapse, mean-pitch representation of inner voices, voice assignment, active-state masks, or pitch scaling.

Investigation and possible response:

- Compare condition and physical vectors for the same worst bars.
- Inspect semantic voice assignment and slots with more than three simultaneous notes.
- Check `note_on`/`hold` masks and `pitch_scale` reconstruction.
- Consider a richer physical representation, additional voices, multi-hot harmony, or a decoder that explicitly consumes the retained chroma condition.
- Do not treat DVAE capacity as the first fix because the loss already exists before DVAE training.

### Per-Bar Harmony Passes But Movement Fails

Signal: physical per-bar metrics pass, but physical transition `cosine_mean < 0.50`.

Interpretation: individual bars are represented adequately, but their changes do not follow source harmonic motion.

Investigation and possible response:

- Check for discontinuous bar indices and unexpected inactive bars.
- Compare target and physical delta vectors around low-scoring transitions.
- Consider an explicit harmonic trajectory or delta target, transition-aware loss, or temporal conditioning in the downstream model.

### All Three Decisions Pass

Interpretation: the codec retains physical harmony and movement under the current heuristic, so DVAE reconstruction becomes the next investigation target. This is a routing inference, not a direct measurement of DVAE quality. Confirm it with DVAE reconstruction diagnostics before changing the model.

Prefer systematic representation, objective, or learned-selection changes when failures repeat across bars. Avoid accumulating isolated repair rules that improve one bar while moving the same failure elsewhere.

## Worst-Bar Analysis

The Markdown report summarizes aggregate metrics, but the JSON diagnostics contains `worst_semantic_physical_bars`: up to 20 active bars sorted by ascending physical cosine. Each row includes `song_id`, `bar_index`, `tensor_key`, `cosine`, `target_chroma`, and `semantic_physical_chroma`.

Use the list as follows:

1. Confirm the `tensor_key` resolves to the expected song and bar.
2. Compare the two 12D vectors to identify missing, excessive, or shifted pitch-class mass.
3. Inspect source notes, quantization, bar clipping, semantic voice assignment, and active masks for that bar.
4. Look for the same failure pattern across many songs. Repeated patterns suggest a representation or algorithm issue; isolated failures more often suggest input, parsing, or boundary conditions.
5. Detect duplicate source pieces before counting repeated low-scoring rows as independent evidence.

## Outputs

- `bar_codec_harmony_oracle/bar_codec_harmony_oracle_diagnostics.json`: complete metrics, configuration, conclusions, and worst-bar vectors.
- `bar_codec_harmony_oracle/bar_codec_harmony_oracle_report.md`: compact aggregate report intended to be interpreted with this design document.

The analyzer does not modify encoded artifacts.

## Worked Example: Mozart Report

The report at `output/stage3/models/mozart/bar_codec_harmony_oracle/bar_codec_harmony_oracle_report.md` can be read as follows.

### Coverage

- `sample_count = 7581` and `active_bar_count = 7568`: 13 indexed bars have zero source chroma and are excluded from per-bar metrics.
- `pair_count = 7520`: 7,520 exact, active, within-song adjacent pairs contribute to each transition section.

### Tensor Chroma Condition

- `cosine_mean = 0.984876` exceeds the `0.95` routing threshold.
- `cosine_p10 = 0.953831` and `cosine_p50 = 0.997153` show that both the lower tail and typical bars retain strong alignment.
- `std_ratio = 0.982528` is close to `1`, so condition contrast is nearly preserved.
- `mse = 0.000532` and `mean_l2 = 0.052551` form the lower-error reference for the physical representation.

Conclusion: the stored condition is not the primary codec bottleneck for this dataset.

### Semantic Physical Chroma

- `cosine_mean = 0.953491` exceeds `0.85` and `std_ratio = 0.943618` exceeds `0.70`, so the physical-harmony decision passes.
- `cosine_p50 = 0.993832` shows excellent alignment for a typical bar, while `cosine_p10 = 0.853039` reveals a materially weaker tail.
- Physical MSE is about `3.61` times condition MSE (`0.001922 / 0.000532`), and physical mean L2 is about `2.03` times condition mean L2 (`0.106571 / 0.052551`). The physical tracks therefore lose more source information than the condition even though they pass the current threshold.

Conclusion: physical encoding is adequate on average but still has local failures worth inspecting.

### Transitions

- Physical transition `cosine_mean = 0.869913` exceeds `0.50`, so movement retention passes.
- Physical movement magnitude ratio is `0.417992 / 0.443785 = 0.941880`; physical transitions are about `5.8%` weaker in magnitude than the source on average.
- Condition transition `cosine_mean = 0.945917` and magnitude ratio `0.429369 / 0.443785 = 0.967516` show that the condition preserves movement more closely. These condition-transition values are informative but do not affect the three booleans.

### Tail Inspection And Final Routing

All three booleans are `True`, so the report routes the next investigation to DVAE reconstruction. However, the diagnostics JSON identifies `wamozart-symphony-no40-in-gm-k550-1st-mvt`, bar `200`, as the worst physical bar with cosine `0.392532`. Its source chroma has substantial mass in bins 5, 6, and 9, while the physical vector concentrates mainly in bins 0 through 3. That local mismatch should be inspected even though aggregate thresholds pass.

The correct overall reading is: the condition channel is strong; physical tracks are somewhat less faithful and have a weak tail; aggregate physical harmony and movement still pass; therefore inspect local codec outliers and then use DVAE diagnostics to test whether reconstruction is the dominant remaining bottleneck.

## Limitations

- Relative chroma removes absolute key and register information.
- The condition representation merges source pitch classes 10 and 11.
- Physical reconstruction weights active slots, not original note velocities.
- Aggregate scores can hide rare severe failures; always inspect p10 and worst rows.
- Transition deltas are vector differences, not harmonic-function labels.
- Passing the current heuristic does not prove perceptual quality or establish a stable production threshold.
