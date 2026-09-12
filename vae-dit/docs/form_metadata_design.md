# Form Metadata Tool Design

## Purpose

The form metadata tool labels symbolic music files before model data processing. It is an offline dataset preparation tool and has no Python import relationship with the model runtime under `src/`.

## Input Data

- Offline classifier input: the preserved form tool may inspect `.mid`, `.midi`,
  `.abc`, `.krn`, `.musicxml`, or `.xml` files.
- Codec V2 production encoding input: **only PPQN SMF Type 0/1 `.mid` or
  `.midi`**. ABC, KRN, and MusicXML are outside the canonical parser support
  boundary; they are not MIDI files that merely failed to parse.
- Logical shape: zero or more recursively discovered music files.
- Optional configuration: YAML mappings for grid tokenization, distance calculation, bar autoencoding, and form templates.
- Constraints: each parseable file must contain at least one note event; an existing `form.json` is replaced only with `--overwrite`.
- Runtime dependencies: NumPy, music21, PyYAML, python-Levenshtein, and PyTorch. PyTorch is required because the preserved default distance backend fits a small CPU token autoencoder.

## Processing Summary

The tool first checks the filename for explicit form hints. When the filename is inconclusive, it parses and quantizes the score, converts bars into relative event tokens, computes a self-similarity matrix from pairwise bar distances, and ranks form templates. Template section lengths are scaled to the observed bar count. A parse or classification failure is recorded and receives a reviewable fallback entry instead of terminating the full directory run.

The tool is isolated in `tools/form_metadata/`. Its legacy classifier output
uses `coordinate_system=legacy_measure_index`; Codec V2 deliberately fails
closed for that coordinate system. Encoding applies form labels only from
metadata bound to the same `source_file_identity`, `raw_smf_v1` parser version,
and canonical-timeline hash using `canonical_bar_index.v1` coordinates.

## Output Data

- `form.json`: JSON object keyed by source file name.
- Each value contains the selected `form`, confidence, classification source,
  candidates, and a `sections` list. The current offline classifier emits
  legacy zero-based measure indexes; it is useful for review but is not
  production Canonical V2 form metadata until regenerated in canonical-bar
  coordinates with the required provenance binding.
- `diagnostics_<timestamp>.json`: lists successful files and failures.
- Logical shape: one top-level entry per supported discovered music file.

## Simple Example

Input directory:

```text
datasets/test/
  example.mid
```

Command:

```bash
bash bin/form.sh test --overwrite
```

Representative output:

```json
{
  "example.mid": {
    "form": "ternary",
    "sections": [
      {"name": "Theme_A", "start": 0, "length": 4}
    ]
  }
}
```
