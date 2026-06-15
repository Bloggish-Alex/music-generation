# HMM — Form-Conditioned Symbolic Music Generation

A three-layer pipeline that learns a finite vocabulary of bar-length musical patterns from MIDI corpora, then generates form-aware classical music through a Hidden Markov Model decoder and a knowledge-based harmonic renderer.

## Core Idea

Classical music follows explicit formal structures (Sonata, Ternary, Rondo). This project decomposes music generation into three independent layers:

```
MIDI corpus
  → Encoder   — "What musical material exists?"
       Quantize bars → Denoising VAE → Latent clustering → Global codebook
  → Decoder   — "When to use which material?"
       Form template → Left-to-right HMM → Observation sequence
  → Renderer  — "How should it sound?"
       Markov bridge harmony → Token harmonization → MIDI output
```

- **Encoder** compresses thousands of physical bars into a compact, reusable codebook (~192 bar types) via a denoising variational autoencoder.
- **Decoder** generates bar-type sequences that respect a musical form template (e.g., Exposition → Development → Recapitulation).
- **Renderer** realizes abstract bar symbols as pitched MIDI notes using style-specific harmonic progressions.

## Project Structure

```
hmm/
├── config/
│   ├── style_defaults.yaml      # All tunable parameters
│   └── harmony_matrix.json      # Harmonic transition matrices by style
├── src/                  # Source modules (Python)
│
```

## Quick Start

### Prerequisites

```bash
pip install music21 mido numpy scipy scikit-learn pyyaml
# Optional: pip install torch matplotlib pandas seaborn
```

### Train a Model

```bash
cd hmm
PYTHONPATH=src python3 -m src.train \
  --music-dir <dir contains your dataset,> \
  --model-dir <dir to save model>
```

Training reads all MIDI/ABC/KRN files under `--music-dir`, extracts per-bar features, trains the VAE encoder, builds the global codebook, trains form-conditioned HMMs, and saves the model bundle to `--model-dir`.

Optional: provide a `form.json` in the music directory to annotate each file with its musical form:

```json
{
    "sonata_no1.mid": {"genre": "Classical", "form": "Sonata"},
    "sonata_no2.mid": {"genre": "Classical", "form": "Sonata"}
}
```

Use `bin/form.sh <dataset>` to auto-generate a `form.json` for manual review.

### Generate Music

```bash
cd hmm
PYTHONPATH=src python3 -m src.generate \
  --model-dir <dir containing model> \
  --form sonata \
  --style classic \
  --key C \
  --mode major \
  --output-json generated/output.json \
  --output-midi generated/output.mid
```

Supported forms: `sonata`, `ternary`, `binary`, `rondo`.

## Configuration

All tunable parameters live in `config/style_defaults.yaml`. Key sections:

| Section | What it controls |
|---|---|
| `encoder` | Backend selection: `vae_latent` (recommended) or `legacy` |
| `vae_encoder` | VAE architecture, training, denoising noise, clustering |
| `global_agglomerative_clustering` | Edit-distance codebook (legacy backend) |
| `observation_vocab` | How bar symbols are composed and positioned |
| `section_hmm` | HMM topology, Baum-Welch params, warm-start |
| `forms` | Form templates: sections, lengths, pitch offsets, cadences |
| `candidate_selector` | Learned bar-to-bar continuity model |
| `harmonic_engine` | Key, mode, harmonic matrix, Markov bridge |
| `harmonic_render` | MIDI tempo, track mode (single / split_by_pitch) |
| `hmm_generation` | Sampling temperatures, repetition caps, source reuse |

Override defaults by passing `--config my_overrides.yaml` — it deep-merges with the base config.

## Key Concepts

### Bar Tokenization

Each bar is quantized into a 16-step grid (sixteenth-note resolution for 4/4). Tokens:

| Token | Meaning |
|---|---|
| ≥ 0 | Note-on at relative pitch (shifted by bar's lowest note) |
| −1 | Rest |
| −2 | Sustain (continuation of previous note) |

### Codebook

The VAE learns an 8-dimensional latent space over these 16-step token vectors. KMeans (k=192) clusters the latent space into a **global codebook** — each cluster is a reusable bar "type" with a pool of concrete physical bars.

At k=192 on Mozart (3,629 bars): singleton rate ~3%, mean pool size ~19, effective labels ~131.

### Observation Vocabulary

Each codebook entry + position context maps to a contiguous `observation_id`. These IDs are the *only* interface the Decoder sees — it never accesses raw tokens or latent vectors.

### Form-Conditioned HMM

A left-to-right Hidden Markov Model is trained per form (Sonata, Ternary, etc.). Hidden states correspond to formal sections. The HMM learns which observation IDs tend to appear in each section. Warm-start initialization prevents state permutation. Automatic state-to-section naming aligns learned states with template sections.

### Harmonic Engine

The Renderer uses a **Markov bridge** to plan harmonic progressions: a backward-gravity algorithm that guides chord-degree transitions toward cadential targets (perfect/half/open) without forcing deterministic steps. Style-specific transition matrices (classic, romantic, jazz, rock) are defined in `config/harmony_matrix.json`.

## Diagnostics and Analysis

Every stage produces structured diagnostic output. Key reports:

```bash
# Distribution report: compare actual vs. HMM emission distributions
PYTHONPATH=src python3 -m src.analyze_hmm_distribution \
  --model-dir <model dir> --output-dir <dir to save reports>

# Quality report: heuristic GOOD/WARN/BAD assessment
PYTHONPATH=src python3 -m src.analyze_model_quality \
  --model-dir <model dir> --output-dir <dir to save reports>

# VAE encoder investigation
PYTHONPATH=src python3 -m src.analyze_vae_encoder \
  --music-dir <model dir> --output-dir <dir to save reports>
```

Diagnostics include: codebook singleton ratio, mean pool size, HMM emission entropy, TV distance / JS divergence between actual and learned distributions, boundary interval statistics, observation diversity metrics.

## Results Summary

| Metric | Edit-Distance Baseline | VAE (k=192) |
|---|---|---|
| Singleton ratio | 60–84% | **~3%** |
| Mean pool size | 1.0–3.0 | **~19** |
| Listening quality | Baseline | **Better** |

The candidate selector reduces mean boundary interval by ~40% (11.4 → 6.8 semitones) without harming observation diversity.

## References

- Kingma & Welling, "Auto-Encoding Variational Bayes," ICLR 2014.
- Rabiner, "A Tutorial on Hidden Markov Models," Proceedings of the IEEE, 1989.
- Cuthbert & Ariza, "music21: A Toolkit for Computer-Aided Musicology," ISMIR 2010.
- Roberts et al., "A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music," ICML 2018.

## License
