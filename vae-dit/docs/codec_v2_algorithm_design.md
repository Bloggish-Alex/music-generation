# Codec V2 Algorithm Design: Current Tensor Contract and Canonical Raw-SMF Parser

## 1. Purpose and status

### Acronyms used in this document

| Abbreviation | Expansion | Meaning here |
|---|---|---|
| MIDI | Musical Instrument Digital Interface | The musical-event protocol/file family being encoded. |
| SMF | Standard MIDI File | The raw file format whose tracks and ticks are parser facts. |
| QL | Quarter Length | Musical time unit; `1.0 QL` is one quarter note. |
| TS | Time Signature | Meta event defining a canonical bar length. |
| PPQN | Pulses Per Quarter Note | SMF tick timing division. |
| FIFO | First In, First Out | Pairing order for overlapping same-pitch notes. |
| SMPTE | Society of Motion Picture and Television Engineers | Alternate MIDI timing division; unsupported in the first parser. |
| JSON | JavaScript Object Notation | Encoding of form metadata. |
| NPZ | NumPy Zipped Archive | Archive format of `voice_tensors.npz`. |
| CC64 | MIDI Continuous Controller 64 | Sustain-pedal control collected/evaluated outside the V2 tensor. |
| SHA-256 | Secure Hash Algorithm, 256-bit | Hash algorithm for artifact/content integrity. |

This document is the standalone, implementation-level description of the final
runtime codec, `semantic_harmony_set_v2`, and of the canonical artifact it
produces, `bar_tensor_schema.v2`.

It describes code currently implemented in `src/codec/`, `src/data/`, and
`src/pipeline/`. It is deliberately more detailed than the schema contract:
the contract is normative for public artifact compatibility, while this document
explains how source notes become tensor values.

Codec V2 has one job: encode every parsed, bar-local MIDI (Musical Instrument Digital Interface) note into a fixed-shape
semantic tensor without silently dropping notes. It does **not** encode pedal,
tempo, key signature, or a separate phrase-level dynamics target. Those are
source facts captured by the evaluation framework.

> Important parser boundary: `raw_smf_v1` is the implemented parser authority.
> It builds validated `CanonicalBarSpan` records from raw PPQN SMF ticks and
> emits `BarRecord` values only from that timeline. No music21 Part/Measure map
> is a production fallback or bar authority.

### Configurable slot capacity: frozen per encoding and training run

`0.25 QL` is the temporal resolution; `slot_capacity` is the maximum number of
such intervals a bar may occupy. They are separate concepts. The old `48` was
an unpublished experimental capacity (`12.0 QL`), not an inherent V2 semantic
rule. Valid observed bars such as `20/4 = 20.0 QL` and `41/8 = 20.5 QL` require
80 and 82 slots respectively. They must not be truncated, split into invented
bars, or assigned a rewritten meter merely because an old capacity is too small.

The frozen policy is: **the quantum remains `0.25 QL`; capacity is an
encoding-run configuration; every tensor consumed by one training run has the
same configured capacity `C`.** Every production invocation declares it
explicitly:

```yaml
bar_tensor:
  schema_version: bar_tensor_schema.v2
  slot_grid:
    quantum_ql: 0.25
    capacity: 92
    epsilon_ql: 0.000001
```

`92 × 0.25 = 23.0 QL`. `92` is the frozen current Beethoven profile: it covers
the observed 82-slot maximum with ten slots of margin. It is not a universal
capacity for all future datasets. A different dataset may choose a different
positive integer in a **separate encoding run, artifact set, and model run**;
artifacts with different capacity must never be combined into one training
dataset. Capacity need not be a multiple of 16, 32, or 48.

For bar length `L`, the required count is:

```text
required_slot_count = max(1, ceil((L - epsilon_ql) / quantum_ql))
```

If it exceeds `capacity`, raise structured `CodecCapacityError` with reason
`slot_capacity_exceeded`. It records at least song ID, source-file identity and
path, tune index, canonical bar index, start/end tick, meter, bar length,
quantum, required count, and configured capacity. This is a **codec
representability failure**, not a parser failure: the parser has correctly
recovered a real canonical bar.

Within one run, artifact shapes are always:

```text
voice_tensors:     float32 [N, 18, C, 6]
slot_valid_mask:   bool    [N, C]
slot_durations_ql: float32 [N, C]
```

For example, at `C=92`, 4/4 uses real musical slots `0..15` and padding
`16..91`; 20/4 uses `0..79`; 41/8 uses `0..81`. Padding is never a musical
rest.

A Discrete Variational Autoencoder (DVAE) training entry point may read and
validate `C` from the manifest or first batch, but cannot change it from batch
to batch. Its input projection, positional embedding, Diffusion Transformer
(DiT) token layout, attention mask, and loss mask depend on this shape at model
construction time. The loader must fail fast unless every artifact agrees on
quantum, capacity, lane/feature definitions, and configuration hash; it records
capacity and the encoding-manifest SHA-256 in the training log.

Implementation propagation rules:

1. `SlotGrid.for_bar()` takes quantum, capacity, and epsilon from configuration;
   no slot-capacity literal `48` remains in runtime code. The independent
   48-track safety limit may remain.
2. Codec allocation, empty-bar initialization, padding, array descriptors, and
   index `voice_tensor_shape` derive from `SlotGrid.capacity`.
3. `encoding_manifest.json.slot_grid_policy` records quantum, capacity, and
   epsilon; all must participate in `configuration_sha256`.
4. Every capture, exporter, and evaluator reads capacity from the manifest and
   verifies tensor/mask/duration axes; none may assume 48.
5. A training loader constructs the model from manifest capacity and validates
   every batch as `[18,C,6]`. It rejects mixed capacity, quantum, or lane/feature
   contracts.
6. A capacity change uses a new run ID and a full re-encode. Unpublished
   `[18,48,6]` artifacts cannot mix with `[18,92,6]` artifacts.
7. Required regressions cover configuration propagation, 92-slot 20/4, 92-slot
   41/8, ordinary 4/4 padding, capacity overflow with structured context, and
   manifest/index/NPZ/evaluation/training-loader agreement.

### 1.1 Implemented canonical raw-SMF parser specification

#### Status and non-negotiable boundary

This document deliberately describes two layers:

| Layer | Status | Contract |
|---|---|---|
| `[18,C,6]` V2 tensor, `SlotGrid`, lane and feature semantics | Implemented | `bar_tensor_schema.v2` semantics remain; `C` is the explicit run configuration above. |
| Raw SMF to `CanonicalBarSpan[]` | Implemented as `raw_smf_v1` | The sole production bar authority. |
| Controls, form metadata, diagnostics, CLI acceptance | Implemented boundary | Controls remain evaluation-only; legacy form coordinates fail closed. |

Standard MIDI has no physical Part or Measure object.  A music21 Measure list
is therefore diagnostic evidence only, never an authority for accepting,
rejecting, or defining a V2 bar.  Raw SMF events are facts; the resolved global
timeline is the sole bar coordinate system.

```text
raw SMF (Standard MIDI File) header/tracks
→ raw TS (Time Signature)/note/control facts (raw tick authority)
→ resolved TS chain → CanonicalBarSpan[] → absolute quantization and clipping
→ unchanged V2 codec → canonical artifacts and framework diagnostics
```

The rules below are the implemented frozen behavior. Production code must not
replace them with convenience defaults.

#### Supported input and exact time

Production encoding accepts PPQN (Pulses Per Quarter Note) SMF format 0 and 1
only. Format 2
fails with `smf_format_2_unsupported`; SMPTE (Society of Motion Picture and Television Engineers) division fails with
`smpte_division_unsupported`. Format-2 tracks are independent patterns, not
parallel parts of one song. The current production parser emits exactly one
`SongRecord` for each Type 0/1 source file and fixes `tune_index=0`. SMF Type 2
and music21 `Opus` are outside its support boundary. A same-file multi-tune
identity requires a separately designed parser/API before it can be enabled.

ABC, KRN, MusicXML, and other symbolic score formats are outside the V2
canonical parser boundary, not failed MIDI inputs. The encoding manifest
records `supported_source_formats=["smf_ppqn_type_0", "smf_ppqn_type_1"]` and
`canonical_parser_version="raw_smf_v1"`.

`physical_track_index` is always the zero-based raw SMF track index.  It is
never renumbered when a conductor track is omitted from notes.  The 48-track
safety limit counts only tracks with at least one successfully paired note;
conductor-only tracks still contribute metadata facts.

For PPQN `P`, all parser decisions use exact rational time:

```text
raw_ql = Fraction(absolute_tick, P)
```

Float QL is display/artifact data only.  `EPSILON_QL` must never make a TS
change legal.  `RawTimeSignatureFact` retains track, event ordinal, tick, `nn`,
raw denominator exponent `dd`, and denominator `2^dd`; `RawSourceNote`
retains track/channel/ordinal, exact start/end tick and QL, pitch and velocity.
Proposed validation is `1 <= nn <= 255` and `0 <= dd <= 7`; otherwise raise
`time_signature_invalid`.

#### TS chain and exact bar construction

1. Gather every raw TS meta event, then order facts by
   `(absolute_tick, physical_track_index, event_ordinal)`.
2. At one tick, identical `(nn, dd)` facts merge.  Preserve all contributing
   `(track, ordinal)` entries and increase `duplicate_merge_count` by `n-1`.
3. Different meters at one tick cause `time_signature_conflict`; the diagnostic
   includes every conflicting fact, regardless of whether they share a track.
4. The default training policy is `initial_time_signature_policy=error`:
   exactly one resolved TS must exist at tick 0, otherwise fail
   `time_signature_initial_missing`.
5. The explicit compatibility policy is `smf_default_4_4`.  Only when tick 0
   has no TS (including an SMF with no TS events at all), inject a synthetic
   4/4 declaration at tick 0.  It is never silent: parser provenance records
   `origin="smf_default"`, `injected_at_tick=0`, `meter="4/4"`, and the first
   real TS tick (or `null` when no real TS exists).  Parser-integrity reports
   the policy, origin counts, and every injected source.  This compatibility
   mode does not alter the default training boundary.

For a resolved meter `(nn,dd)` and PPQN `P`:

```text
nominal_bar_length_tick = Fraction(nn * 4 * P, 2^dd)
nominal_bar_length_ql   = Fraction(nn * 4, 2^dd)
```

Every resolved TS event's exact tick is the next canonical-timeline start.  It
is never moved, snapped with a float epsilon, or rejected merely because it
lands inside the prior meter's nominal bar.  Given canonical cursor `C`, old
meter nominal length `B`, and next event `T`, use exact `Fraction` arithmetic:

```text
while C + B <= T:
    emit full CanonicalBarSpan [C, C + B), meter = old_meter
    C = C + B

C == T  -> ordinary full-bar boundary; emit no partial span.
C < T   -> emit partial CanonicalBarSpan [C, T), meter = old_meter.
C > T   -> impossible; the loop must never step beyond T.
```

The second case is `canonical_partial_bar_before_ts_change`.  It has
`is_partial=true`, `partial_reason="time_signature_change"`, and
`nominal_meter=old_meter`; the new TS begins its full-bar sequence exactly at
`T`.  Its end is a raw TS fact, not a guessed measure, pickup, or snap.

For example, if a 3/4 bar starts at QL 1380 and a raw 4/4 TS occurs at QL 1382:

```text
[1380, 1382) -> partial span, 3/4 meter context, actual length 2.0 QL
[1382, 1386) -> new full 4/4 span
```

`time_signature_mid_bar_change` is therefore no longer a failure. Under the
default `error` policy missing initial TS, same-tick conflicts, and invalid TS
values remain fail-fast. Tempo
at the same tick retains its own ordinal but cannot change the QL grid. TS facts
after the terminal bar remain provenance facts; they do not create a trailing bar.

#### Raw note pairing and stable identity

For each raw track, consume events in `(absolute_tick,event_ordinal)` order.
Maintain a FIFO (First In, First Out) queue for each `(physical_track_index, channel, pitch)`:

```text
NoteOn velocity > 0: enqueue; retain its raw event identity for file-scoped
source_note_ordinal assignment.
NoteOff or NoteOn velocity = 0: dequeue earliest pending onset and pair it.
empty queue on off: apply `redundant_orphan_note_off` in §3.1.
pending onset at EOF: unterminated_note_on.
paired end > start: produce a candidate RawSourceNote.
paired end == start: apply `same_tick_zero_duration_pair` in §3.1.
paired end < start: note_nonpositive_duration.
```

Each listed error is a song parser failure.  FIFO is intentional for overlapping
same-pitch notes.  A chord is stable because same-tick order is the raw track
event ordinal and cross-track notes have distinct physical track indexes. After
collecting successful non-zero Note Ons, assign one monotonically increasing
raw-file ordinal using `(absolute_tick,physical_track_index,event_ordinal)`.
The current production parser has `tune_index=0`, so the three-part ID remains
unique within its one-file/one-song boundary.
The identity is created before quantization, sorting or clipping:

```text
source_note_id = source_file_identity : physical_track_index : source_note_ordinal
```

#### Raw pairing normalization v1: discard only zero-time pairs and redundant offs

This happens before `RawSourceNote` creation and is distinct from 0.25-QL
quantization projection.  Three read-only censuses established the boundary:
Mozart contained 331 and the Beethoven/mixed corpus 890 nonpositive durations;
all were same-tick FIFO pairs with `queue_depth_before == 1`.  Neither corpus
contained an unterminated note-on, while the MAESTRO sample had no pairing
findings.  This evidence supports only the two exact rules below, not generic
MIDI error tolerance.

```text
policy_version = "raw_pairing_normalization.v1"
key            = (physical_track_index, channel, pitch)
```

**Rule A: `same_tick_zero_duration_pair`**

When a Note Off (including velocity-zero Note On) dequeues the earliest pending
Note On for the same key with `end_tick == start_tick` and
`queue_depth_before == 1`, remove that pending onset, discard that On/Off pair,
and continue with same-tick and later events.  It
produces no `RawSourceNote`: it has no positive physical interval, so it is not
a trainable note or a quantization fragment.  It receives no
`source_note_ordinal` or `source_note_id` and never reaches clipping, tensors,
or quantization audit.

**Rule B: `redundant_orphan_note_off`**

When a Note Off (including velocity-zero Note On) arrives for an empty queue of
the same key, discard that Off, retain the empty queue, and continue.  It ends
no known active note under the deterministic parser state and therefore changes
no positive-duration note interval.

Every Rule A/B application must publish a separate `raw_pairing_repair`
diagnostic containing at least:

```text
repair_kind, source_file_identity, dataset-relative POSIX path, tune_index,
physical_track_index, channel, pitch,
on_tick/on_event_ordinal for Rule A, off_tick/off_event_ordinal,
on_velocity for Rule A, queue_depth_before, same-tick events, SMF format, PPQN
```

The run manifest binds policy version, repair-artifact hash, total repair count,
counts by repair kind, and affected-file count; `parser_integrity` reports the
same summaries.  The repair artifact and raw-pairing census are diagnostics,
not training data.

The following remain strict parser failures:

```text
unterminated_note_on
end_tick < start_tick
any attempted cross physical-track / channel / pitch pairing
any policy changing FIFO order, such as LIFO
all TS, SMF format, SMPTE, track-limit and slot-capacity failures
```

A repair must never lengthen, shorten, move, or merge a positive-duration note.
If a future census finds queue depth above one, a different-tick nonpositive
duration, or a dangling On, it requires a new design decision.

`source_note_ordinal` remains raw-file scoped but now numbers only Note Ons
that successfully form positive-duration `RawSourceNote`s after normalization.
Collect those candidate ons and sort them by
`(absolute_tick, physical_track_index, event_ordinal)` into `0..K-1`.
Discarded pairs consume no ordinal and cannot affect source identity or
cross-bar continuity.

#### Raw pairing census: ongoing monitor, not broad authorization

`raw_pairing_census` reads SMF bytes with exactly the parser's `(track, channel,
pitch)` FIFO state machine, but creates no `SongRecord`, tensor, or modified
event stream.  It reports each pre-normalization anomaly with:

```text
source_file_identity, dataset-relative POSIX path, tune_index,
physical_track_index, channel, pitch, failure_code,
absolute_tick, event_ordinal, queue_depth_before,
matched_on_tick/event_ordinal when present, pending_on_queue snapshot,
same-key same-tick events, PPQN, SMF format
```

It also writes counts grouped by failure code, file, track, channel, pitch, and
same-tick versus different-tick cases.  The corpus manifest must bind the input
file list, content hash, and code revision, plus two non-interchangeable
versions: `census_pairing_policy_version="smf_fifo_track_channel_pitch_v1"`
for the observed raw FIFO, and
`normalization_policy_version="raw_pairing_normalization.v1"` for the encoder
repair rules.  The census is diagnostic data, not trainable data: it explains
when Rules A/B trigger, but cannot relax any other strict failure boundary.

#### Quantization, clipping, and continuation

Absolute 0.25-QL quantization is incompatible with a bar whose length is not a
multiple of 0.25 QL (for example 5/32 = 0.625 QL).  Therefore there is no
global `canonical_quantized_onset/end_ql` tensor-time layer.  It must not remain
in code, artifacts, or audits.

One `RawSourceNote` may produce zero or more fragments, identified by
`(source_note_id, canonical_bar_index)`.  Each fragment retains:

```text
raw_source_onset/end_ql              # exact global source facts
raw_local_start/end_ql               # raw source interval clipped to one span
quantized_local_start/end_ql         # boundaries quantized in that bar only
bar_local_quantized_duration_ql      # quantized_local_end - quantized_local_start
```

For a bar of local length `L`, construct separate representable-boundary sets:

```text
S = every valid slot start                  # legal onset values
E = every valid slot boundary, including L  # legal end values
```

For 5/32 (`L=0.625`), `S=[0,.25,.50]` and `E=[0,.25,.50,.625]`.  The terminal
bar boundary cannot be an onset, but it is a valid end for a partial final slot.

For each non-empty raw intersection with `[bar_start,bar_end)`:

```text
raw_local_start = max(0, raw_source_start - bar_start)
raw_local_end   = min(L, raw_source_end - bar_start)
q_start = nearest(raw_local_start, S)  # half ties choose the larger boundary
q_end   = nearest(raw_local_end, E)    # half ties choose the larger boundary
```

If `q_end > q_start`, use those boundaries directly.  If `q_end <= q_start`,
apply the frozen **minimum-representable-slot projection**.  It is a recovery
only for a collapsed fragment; it never changes a normally quantized fragment:

```text
candidates[i]    = each valid interval [S[i], E[i+1]) in this bar
overlap(i)        = length(raw_local_interval ∩ candidates[i])
endpoint_error(i) = abs(S[i] - raw_local_start) + abs(E[i+1] - raw_local_end)

select greatest overlap;
on a tie select least endpoint_error;
on a further tie select the later slot (greatest i).

q_start = S[i]
q_end   = E[i+1]
quantization_repair_kind = "minimum_representable_slot_projection"
```

The result is always a strictly positive interval inside the current canonical
bar.  For example, a collapsed raw-local interval `[0.11,0.19)` projects to
`[0.00,0.25)`.  This is not a blanket short-note extension: it applies only
after endpoint quantization has produced a non-positive duration.  The audit
must retain the raw boundaries, ordinary candidate boundaries, final boundaries,
repair kind, selected slot, overlap and endpoint error.  Its run summary must
report projection count/rate, including aggregations by source file, meter and
canonical bar.  Those repairs are known timing distortion, never zero-error
fidelity.

The resulting local fragment is the only timing input to V2 slot onset/hold
encoding.

Raw tick-derived QL is exact and receives no epsilon snap.  Continuation is
physical source semantics, not a local quantization result:

```text
continues_from_previous_bar = raw_source_start < bar_start
continues_into_next_bar     = raw_source_end > bar_end
```

Thus local start zero does not imply hold.  A note starting inside the current
bar but quantized to local zero is an onset; only a raw start before `bar_start`
is a hold.

`quantization_audit` is one sample per `(source_note_id, canonical_bar_index)`
fragment.  Its meter is the unique meter of that fragment's span and its
residuals are `abs(q_start-raw_local_start)` and `abs(q_end-raw_local_end)`.
A cross-bar source note therefore yields multiple audit samples, accurately
reflecting its independent local representations.

#### Pickup, terminal bar, silence, and partial slots

The implemented parser never infers pickup from the first note and currently
accepts no pickup metadata. It starts the initial nominal bar at tick/QL zero;
leading silence is valid rest and `is_pickup=false`.  For initial 4/4 and a
first note at QL 1.5, bar 0 remains `[0,4)` and slots 0--5 are rests.  A later,
separately versioned metadata mode may use an exact `pickup_length_ql` bound to
`source_file_identity+tune_index`; it would make `[0,pickup_length)` a short
first span and change every later bar phase.  It must not be mixed with this
first-release mode.

No successfully paired notes is `no_note_events` parser failure: do not emit a
zero-row run or invent an all-rest bar.  Otherwise let `last_end` be the maximum
raw source note end, a physical fact independent of bar-local quantization. The terminal bar is the span whose half-open interval
contains that end; if the end equals a span end, it is the preceding span.
Materialize bars 0 through it, inclusive.  The remaining
valid slots after the final note are true Rest vectors, but slots outside a
bar's valid duration are zero padding.  Do not append a trailing empty bar.

All TS-defined spans from zero to terminal end are materialized.  A span with
no successfully quantized fragment is an Empty Bar: valid cells are Rest,
`base_pitch_valid=false`, stored base pitch is 0, and 12D context is zero.
It clears melody sequence state.  Continuity may cross only contiguous
`canonical_bar_index` values in one song/tune/transpose variant, without an
empty bar; song, tune, transpose, and split boundaries reset it.

`SlotGrid` may represent a final short valid slot. Full raw-MIDI spans still use
their current meter's nominal length, but the parser permits exactly one
raw-derived partial-bar source: a
`canonical_partial_bar_before_ts_change` cut by an exact TS event under the
rule above.

Its actual `bar_length_ql` is
`ts_event_ql - previous_canonical_boundary_ql`, while its meter context remains
the old meter. `SlotGrid.for_bar(bar_length_ql)` creates the valid mask and
durations. A 3/4 span cut after 2.0 QL has eight 0.25-QL valid slots; if its
length is not a 0.25 multiple, its last valid slot holds the real remainder.
It participates normally in raw clipping, bar-local fragment quantization,
chroma, form full-coverage, codec fidelity, and quantization audit.

No other partial source is legal: never infer one from note-off, terminal
shortening, silence, music21 measure heuristics, or guessed pickup. Terminal
time still follows the complete Rest-tail-bar policy; only an exact raw TS event
can emit a partial canonical span.

#### Form mapping, diagnostics, and implementation scope

The implemented parser accepts only **bound canonical-bar metadata**. Its
coordinate system is `canonical_bar_index.v1`; a valid payload binds the exact
source identity, `raw_smf_v1` parser version, and canonical-timeline hash:

```json
{
  "coordinate_system": "canonical_bar_index.v1",
  "source_file_identity": "...",
  "canonical_parser_version": "raw_smf_v1",
  "canonical_timeline_sha256": "sha256:...",
  "form": "binary",
  "sections": [{"name": "A", "canonical_start_bar_index": 0, "canonical_end_bar_index": 8}]
}
```

Each section must be a non-empty, non-overlapping range of existing canonical
bar indexes. Only after the full payload validates does the parser set
`song.form` to the global template and `bar.form`/`section_label` to the local
section label. The complete status vocabulary is `absent`, `mapped`,
`unavailable_legacy_measure_index`, `unavailable_timeline_mismatch`, and
`unavailable_empty_canonical_sections`.

The current offline form generator produces only
`coordinate_system=legacy_measure_index`. It cannot produce the bound
canonical contract above, so Codec V2 deliberately fails closed: it applies no
form labels and `form_action_alignment` is `UNAVAILABLE`. A canonical form
classifier/adapter is future independent work. A QL-interval form contract is
also only a future proposal; it is not accepted by the current parser.

`ActionLabeler` remains a legacy 16-bin diagnostic heuristic: it normalizes
each bar into `bar_length_ql / 16` rhythm bins. It neither determines the V2
tensor nor forms training input. Until a canonical form classifier/adapter is
enabled, it must migrate to `SlotGrid` valid-slot semantics or
`form_action_alignment` must remain `UNAVAILABLE`; it must not be represented
as an implemented SlotGrid consumer.

### Parser-integrity publication status

The current `parser_integrity_raw_observation.v2` publishes only the fields
that the runtime capture actually materializes:

| Current published group | Fields |
|---|---|
| Measure map | `measure_map` (song/measure counts, meter distribution, Opus-tune count, over-capacity count) |
| Initial TS | `initial_time_signature_policy`, `initial_time_signature_origin_counts`, `initial_time_signature_injections` |
| Partial spans | `partial_span_count`, `partial_reason_counts`, `partial_spans` |
| Track retention | `track_retention` |
| Pairing normalization | `normalization_policy_version`, `repair_artifact`, `repair_count`, `repair_counts_by_kind`, `raw_pairing_repair_counts`, pairing-loss counts/ratio, affected-file count |
| Failures | `parser_failures` |

The quantization-audit observation currently declares
`audit_unit="source_note_fragment"`, fragment/projection counts, pairing-loss
summary, grid policy, and per-file/meter residuals. Every sample key is
`(source_note_id, canonical_bar_index)` and carries meter, raw-local and
quantized-local start/end, and onset/end residuals. It does not publish or
consume global quantized onset/end fields.

The following are **future parser-integrity extensions**, not current raw
observation fields: `canonical_parser_version`, TS fact/duplicate/conflict
counts, `initial_ts_source`, `canonical_span_count`, `empty_bar_count`,
terminal/pickup policy and status, note-pairing/quantization policy, and
retained-note-track count. They require a schema and capture implementation
before evaluators may consume them.

Parser failure means no apparently AVAILABLE partial encoding. `UNAVAILABLE`
is reserved for evaluation capture with missing or unaligned materialized
inputs. `time_signature_change` is the only currently published partial reason.
`source_measure_index` remains only a same-value serialized compatibility alias
of `canonical_bar_index`; no codec algorithm may use it as music21 Measure
authority. V2 tensor semantics remain unchanged.

## 2. Terms and units

| Term | Meaning |
|---|---|
| QL | Quarter length. `1.0 QL` is one quarter note in music21 time. |
| source note | A note before bar clipping, identified by its source file, physical track, and ordinal. |
| bar-local note | The part of a source note inside one validated bar. A source note crossing a bar boundary creates one bar-local note in each intersected bar. |
| slot | A time interval inside one bar. Slots have nominal duration `0.25 QL`. |
| lane | One semantic note role at one slot: melody, one harmony-set member, or bass. |
| valid slot | A slot containing real musical time for the current bar. |
| padding slot | One of the fixed `C` positions not used by a shorter bar. Padding is not a musical rest. |

All pitches below are absolute MIDI pitches. All time comparisons use the frozen
time epsilon `1e-6 QL` unless stated otherwise.

## 3. Input to the codec

### 3.1 Required bar-local fields

The codec receives a `BarRecord` with zero or more `TrackRecord`s. Each
`NoteEvent` must already be clipped to the bar and carries at least the following
information:

| Field | Source / meaning |
|---|---|
| `pitch` | Absolute MIDI pitch. |
| `onset_ql` | `quantized_local_start`; a legal valid-slot start in `[0, bar_length_ql)`. |
| `duration_ql` | `quantized_local_end - quantized_local_start`, strictly positive. |
| `velocity` | Source MIDI velocity; codec clamps it to `[0, 127]`. |
| `physical_track_index` | Source physical part/track index, retained for deterministic harmony ordering. |
| `source_note_ordinal` | Raw-file-scoped stable ordinal attached before quantization and bar clipping; it never resets for a tune or physical track. |
| `source_note_id` | `<source_file_identity>:<physical_track_index>:<source_note_ordinal>`. The same source note retains this ID in every bar it intersects. |
| `source_onset_ql` | Raw absolute source onset, used as a deterministic harmony tie-break. |
| `continues_from_previous_bar` | True when the source note began before this bar. |
| `continues_into_next_bar` | True when the source note ends after this bar. |

The source file identity is derived from the normalized dataset-relative POSIX
path and raw file bytes. It is stable across process lifetimes; Python object
identity is never used.

### 3.2 Bar clipping and continuation

For a raw source interval `[source_start, source_end)` and a bar interval
`[bar_start, bar_end)`, the parser emits a fragment only when the two overlap.
It first clips raw-local boundaries, then uses the `S`/`E` local quantizer in
§1.1. Its codec-facing values are:

```text
raw_local_start = max(0, source_start - bar_start)
raw_local_end   = min(bar_end - bar_start, source_end - bar_start)
onset_ql        = quantized_local_start
duration_ql     = quantized_local_end - quantized_local_start

continues_from_previous_bar = source_start < bar_start
continues_into_next_bar     = source_end > bar_end
```

Thus a source note is not lost at a bar boundary. The codec writes a continuing
fragment as `hold` only when `source_start < bar_start`; local onset zero alone
never changes an onset into a hold.

## 4. Canonical artifact layout

One encoding run writes `voice_tensors.npz` and row-aligned metadata. Let `N`
be the number of encoded bars after the pipeline applies its deterministic row
sort. The archive contains:

| Array | dtype | shape | Row meaning |
|---|---|---|---|
| `voice_tensors` | `float32` | `[N, 18, C, 6]` | Main note representation; `C` is manifest capacity. |
| `slot_valid_mask` | `bool` | `[N, C]` | Which slot positions are real time. |
| `slot_durations_ql` | `float32` | `[N, C]` | Real duration of each slot; zero for padding. |
| `bar_contexts` | `float32` | `[N, 12]` | Full 12-bin duration/velocity-weighted relative chroma. |
| `base_pitches` | `int16` | `[N]` | Bar bass anchor; stored as `0` if invalid. |
| `base_pitch_valid` | `bool` | `[N]` | Whether `base_pitches[row]` has musical meaning. |

`bar_tensor_index.json[row]` is the authority linking a row to its song and
canonical bar. It includes `row`, `tensor_key`, `song_id`, `base_song_id`,
`canonical_bar_index`, transpose amount, schema version, and slot-valid count.
During migration, `source_bar_index`/`source_measure_index` may be serialized
only as aliases whose values equal `canonical_bar_index`; they are not Measure
authority. `encoding_manifest.json` binds exact archive and index
bytes using SHA-256 and declares all dtypes/shapes.

### 4.1 Meaning of each main-tensor axis

For `V = voice_tensors[row, lane, slot, feature]`:

| Axis | Range | Meaning |
|---|---|---|
| `row` | `0..N-1` | One canonical encoded bar. |
| `lane` | `0..17` | Semantic musical role, defined below. |
| `slot` | `0..C-1` | Bar-local nominal 0.25-QL position. |
| `feature` | `0..5` | Numeric representation of the role at the slot. |

Lane indexes are fixed:

```text
0       melody
1..16   harmony_00 .. harmony_15
17      bass
```

Feature indexes are fixed:

```text
0  relative_pitch
1  is_rest
2  is_note_on
3  is_hold
4  normalized_velocity
5  velocity_ratio
```

No lane is a physical track. In particular, harmony lanes are a slot-local set:
the same sounding source note may occupy different harmony lane numbers in
neighbouring slots if the sorted active set changes. Stable continuity is only a
melody selection rule, not a harmony voice-tracking rule.

## 5. Slot grid and padding

### 5.1 Grid construction

The `SlotGrid` values are supplied by the frozen run configuration:

```text
QUANTUM_QL = 0.25
CAPACITY   = C  # Beethoven profile: 92
EPSILON_QL = 0.000001
```

For one validated bar length `L`:

1. Compute the nearest multiple of `0.25 QL`.
2. If that multiple differs from `L` by no more than `EPSILON_QL`, use the
   snapped value. This removes floating-point boundary noise.
3. Otherwise retain the actual positive length. The final valid slot may be
   shorter than `0.25 QL`.
4. Allocate `ceil((L - EPSILON_QL) / 0.25)`, with a minimum of one valid slot.
5. Reject the bar with `slot_capacity_exceeded` if the count exceeds configured
   `C`; include the structured representability context defined above.

For valid slot `s`, zero based:

```text
slot_start(s) = s * 0.25
slot_end(s)   = slot_start(s) + slot_durations_ql[s]
```

Examples:

| Meter / measured length | Valid slots | Slot durations |
|---|---:|---|
| 2/4 = 2.0 QL | 8 | eight × `0.25` |
| 3/4 = 3.0 QL | 12 | twelve × `0.25` |
| 4/4 or 2/2 = 4.0 QL | 16 | sixteen × `0.25` |
| 6/8 = 3.0 QL | 12 | twelve × `0.25` |
| 12/8 = 6.0 QL | 24 | twenty-four × `0.25` |
| measured 6.375 QL | 26 | 25 × `0.25`, then `0.125` |

### 5.2 Rest is not padding

For every valid slot, every lane begins as a true musical rest:

```text
[0, 1, 0, 0, 0, 0]
```

For padding slots, all features remain zero:

```text
[0, 0, 0, 0, 0, 0]
```

The distinction is essential:

```text
valid silent time      mask=true,  duration>0, is_rest=1
invalid fixed padding  mask=false, duration=0, is_rest=0
```

All codec and evaluation denominators must exclude invalid padding.

## 6. Base pitch and bar context

### 6.1 Base pitch

For each bar, the base pitch is the lowest absolute pitch among every
bar-local note, irrespective of its assigned semantic lane:

```text
base_pitch = min(note.pitch for note in bar.all_notes)
```

If the bar has no notes:

```text
base_pitch_valid = false
base_pitches[row] = 0          # storage sentinel only
bar_contexts[row] = [0] * 12
```

If the bar has notes and the highest pitch is more than 96 semitones above the
base pitch, encoding raises `relative_pitch_range_overflow`. It must not write
an unbounded relative-pitch value.

### 6.2 12D relative chroma

`bar_context` is computed from **all bar-local source notes**, before semantic
role assignment. It does not use slot sampling, so it retains clipped duration
information independently of onset/hold representation.

For note `n`:

```text
bin(n)    = (n.pitch - (base_pitch mod 12)) mod 12
weight(n) = max(n.duration_ql, 0) * clamp(n.velocity, 0, 127) / 127
```

Accumulate `weight(n)` in its bin, then L1-normalize the 12 values. If the
total weight is zero, return all zeros. All 12 pitch classes are preserved;
there is no legacy 11-bin projection.

## 7. Active-note decision per slot

For each valid slot `[slot_start, slot_end)`, a bar-local note is active when:

```text
note.onset_ql < slot_end - EPSILON_QL
and
note.onset_ql + note.duration_ql > slot_start + EPSILON_QL
```

This is a left-closed, right-open interval policy with epsilon protection:

- A note ending exactly at a slot boundary is not active in the next slot.
- A note starting exactly at a slot boundary is active in that slot.
- A note continuing from a previous bar is active at bar slot 0 if it overlaps
  that slot, but is written as `hold`.

If there are no active notes, the slot keeps its initialized rest vectors.

## 8. Semantic role assignment

For each active-note set, `assign(active, previous_melody, tolerance=7)` returns
exactly one melody, zero or one bass, and zero to sixteen harmony notes.

### 8.1 Melody

First select the highest active candidate using this deterministic ordering:

```text
pitch descending
velocity descending
source_note_id ascending
```

Then apply continuity. If the preceding melody is still active and its pitch is
no more than seven semitones below the current highest candidate, retain that
preceding melody:

```text
previous.pitch >= highest.pitch - 7
```

This prevents small temporary upper-note movements from needlessly swapping the
melody lane.

Within a bar, the melody selected in one active slot is the candidate for the
next slot. Across bars, continuity is available only through `encode_song()`:

```text
state may continue only when
current.canonical_bar_index == previous.canonical_bar_index + 1
and no Empty Bar lies between them
```

The carried melody is found by exact `source_note_id`, never object identity.
`encode(bar)` without an explicit state deliberately encodes an independent bar;
callers must use `encode_song(song)` for a multi-bar sequence.

### 8.2 Bass

Remove melody from the active set. If at least one note remains, choose bass by:

```text
pitch ascending
velocity descending
source_note_id ascending
```

If melody is the only active note, bass is rest. Melody and bass are therefore
never the same source-note occurrence.

### 8.3 Harmony set

Remove melody and bass. The remaining notes form the harmony set. Sort them by:

```text
pitch ascending,
physical_track_index ascending,
bar-local clipped duration_ql ascending,
source_onset_ql ascending,
velocity ascending,
source_note_id ascending
```

Write the first note to lane 1, the second to lane 2, and so on. If more than
16 harmony notes are active in one slot, raise `harmony_lane_overflow`; do not
truncate the set.

## 9. Writing a six-feature lane vector

For each assigned `(lane, note)` in a valid slot:

```text
clamped_velocity = clamp(note.velocity, 0, 127)
relative_pitch   = (note.pitch - base_pitch) / 24
normalized_vel   = clamped_velocity / 127
velocity_ratio   = clamped_velocity / sum(clamped_velocity of all assigned lanes)
```

`velocity_ratio` is zero if the denominator is zero. The denominator includes
melody, every active harmony lane, and bass, but never a rest lane or padding.

State flags are mutually exclusive for a valid lane-slot:

```text
rest:  [0, 1, 0, 0, 0, 0]
onset: [relative_pitch, 0, 1, 0, normalized_vel, velocity_ratio]
hold:  [relative_pitch, 0, 0, 1, normalized_vel, velocity_ratio]
```

An assigned note is onset only if both conditions hold:

```text
note is not a continuation from a previous bar at slot 0
and abs(note.onset_ql - slot_start) <= EPSILON_QL
```

Every other active assigned note is a hold. Consequently, a note that starts at
slot 0 of the current bar but began before the bar is a hold, not a new onset.

## 10. Worked example: one 4/4 bar

This is a fully reproducible source-bar example. It uses the same fields the
parser supplies; it is intentionally small enough to calculate by hand.

### 10.1 Source bar

Bar length is `4.0 QL` (4/4), so slots `0..15` are valid and `16..C-1` are
padding (`16..91` for the current Beethoven profile). All notes start at `0.0 QL`, last `0.75 QL`, and do not continue from a
previous bar.

| Source note | Track | ID suffix | Pitch | Velocity | Local interval |
|---|---:|---:|---:|---:|---|
| C4 | 0 | `:0:0` | 60 | 80 | `[0.00, 0.75)` |
| E4 | 1 | `:1:0` | 64 | 70 | `[0.00, 0.75)` |
| G4 | 2 | `:2:0` | 67 | 60 | `[0.00, 0.75)` |
| C5 | 0 | `:0:1` | 72 | 100 | `[0.00, 0.75)` |

The lowest pitch is C4, therefore:

```text
base_pitch = 60
base_pitch_valid = true
```

### 10.2 Source notes mapped to roles

For slots 0, 1, and 2, all four notes are active. Their role mapping is:

```text
highest pitch: C5 (72)  → melody lane 0
lowest remaining: C4    → bass lane 17
remaining, sorted: E4, G4
                         → harmony_00 lane 1, harmony_01 lane 2
```

The following diagram omits unused harmony lanes for clarity:

```text
slot time       [0.00,.25)  [.25,.50)  [.50,.75)  [.75,1.00)
                 slot 0      slot 1      slot 2      slot 3
                 ------------------------------------------------
melody  lane 0  C5 onset    C5 hold     C5 hold     rest
harmony lane 1  E4 onset    E4 hold     E4 hold     rest
harmony lane 2  G4 onset    G4 hold     G4 hold     rest
bass    lane17  C4 onset    C4 hold     C4 hold     rest
```

At slot 3 the notes end exactly at the left boundary (`0.75 QL`), so the
right-open active interval rule correctly makes every lane rest.

### 10.3 Tensor values at slot 0

The assigned velocity denominator is:

```text
80 + 70 + 60 + 100 = 310
```

The non-rest vectors at `voice_tensors[row, :, 0, :]` are approximately:

| Lane | Source note | Vector `[relative_pitch, rest, onset, hold, normalized_velocity, velocity_ratio]` |
|---|---|---|
| 0 | C5 | `[0.500000, 0, 1, 0, 0.787402, 0.322581]` |
| 1 | E4 | `[0.166667, 0, 1, 0, 0.551181, 0.225806]` |
| 2 | G4 | `[0.291667, 0, 1, 0, 0.472441, 0.193548]` |
| 17 | C4 | `[0.000000, 0, 1, 0, 0.629921, 0.258065]` |

All other valid lanes at slot 0 are exactly `[0, 1, 0, 0, 0, 0]`.

At slot 1, the same lanes contain the same pitch and velocity values but have
`onset=0, hold=1`. At valid slot 3, all 18 lanes are rest vectors. At padding
slot 16, all 18 lanes are all-zero padding vectors.

### 10.4 Bar context for the example

Every note has the same duration, so the common factor `0.75 / 127` cancels in
L1 normalization. Relative chroma bins are:

```text
C4 (60) and C5 (72) → bin 0: 80 + 100 = 180
E4 (64)             → bin 4: 70
G4 (67)             → bin 7: 60
total                          310
```

Therefore the 12D context is:

```text
bin index:     0        1  2  3    4        5  6    7        8  9 10 11
bar_context: [0.580645, 0, 0, 0, 0.225806, 0, 0, 0.193548, 0, 0, 0, 0]
```

This context is independent of which lane received E4 or G4.

## 11. Worked example: cross-bar continuation and melody continuity

Consider two contiguous 4/4 bars. The C5 source note has stable ID `fixture:0:0`.

```text
bar 0: C5 starts at 0.0 and continues past the bar end
bar 1: clipped C5 begins at local 0.0 with continues_from_previous_bar=true
bar 1: E5 (76) also starts at 0.0
```

The highest current pitch is E5, but C5 is only four semitones lower. Because
C5 is the prior melody, it remains lane 0 under the seven-semitone rule.

```text
bar 0 lane 0, slot 0: C5 onset
bar 1 lane 0, slot 0: C5 hold
bar 1 E5: assigned to the remaining semantic role according to bass/harmony rules
```

The bar-1 C5 is `hold` even though its bar-local onset is `0.0`, because its
`continues_from_previous_bar` flag is true. If the next bar were not contiguous
in `canonical_bar_index`, or an Empty Bar lay between the two bars, sequence
state would reset and C5 would receive no cross-bar melody preference.

## 12. Failure conditions and non-loss policy

The codec intentionally fails instead of silently truncating in these cases:

| Condition | Error | Reason |
|---|---|---|
| Bar duration needs more than configured `C` slots | `slot_capacity_exceeded` | Fixed run capacity would otherwise hide time. |
| More than 16 harmony notes in one slot | `harmony_lane_overflow` | No harmony source note may be silently dropped. |
| Highest pitch exceeds base pitch by more than 96 semitones | `relative_pitch_range_overflow` | Out-of-contract pitch normalization must not be emitted. |
| Missing, conflicting, or invalid canonical TS chain | parser failure before codec | Do not guess meter; an exact TS event inside a nominal bar emits an explicit partial span instead of failing. |
| Raw pairing outside normalization v1 | `unterminated_note_on` / `note_nonpositive_duration` where `end_tick < start_tick` | Only zero-time pairs and empty-queue redundant offs may be discarded; never fabricate a source note. |
| More than 48 retained physical parts without explicit truncate policy | `track_limit_exceeded` | Source track loss must be explicit. |

The explicit truncate policy is diagnostic-only and records dropped-part counts
and note ratio. Canonical training encoding uses the default error policy.
`quantization_nonpositive_clipped_duration` is intentionally absent: a paired
fragment with positive raw overlap that collapses under endpoint quantization is
encoded through the audited minimum-representable-slot projection, never
silently removed.

## 13. What this tensor intentionally does not preserve

Codec V2 preserves note pitch, slot-level onset/hold state, note-local velocity,
relative vertical balance, full relative chroma, source identity, and semantic
melody/bass/harmony roles. It does not directly encode:

- sub-slot onset or duration precision beyond the 0.25-QL grid;
- exact overlap duration inside a slot (the 12D bar context retains a separate
  duration-weighted aggregate);
- persistent physical harmony voice identity across slots;
- pedal/CC64, tempo, or key as tensor targets;
- a separate bar- or phrase-level dynamics-shape channel.

Quantization residuals, controls, parser integrity, and codec fidelity are
captured by the existing evaluation framework so these limitations are visible
rather than silently treated as reconstruction fidelity.

## 14. Implementation references

| Concern | Source of truth |
|---|---|
| Public artifact contract | `contracts/codec/bar_tensor_schema.v2.md` |
| Slot grid | `src/codec/slot_grid.py` |
| Role assignment and continuity state | `src/codec/semantic_harmony_assignment.py` |
| Tensor writing | `src/codec/semantic_harmony_set_codec.py` |
| Bar clipping and source identity | `src/data/music_parser.py` |
| Canonical NPZ/index/manifest writing | `src/pipeline/encoding_pipeline.py` |
| Codec-fidelity evaluation | `src/diagnostics/codec_fidelity_raw_capture.py` and `evaluation/src/evaluation_framework/evaluation_codec_fidelity.py` |

When this document and the public schema contract differ, the public schema
contract is authoritative for released artifact compatibility; implementation
and tests are authoritative for current runtime behaviour.
