# HMM Music Structure Experiment

This package is isolated from `sample` and implements:

1. `src/grid_tokenizer.py`: MIDI/music files to fixed bar token grids.
2. `src/edit_distance_matrix.py`: token-level Levenshtein distance and RBF affinity.
3. `src/spectral_bar_cluster.py`: spectral clustering into micro bar labels.
4. `src/section_hmm.py`: discrete HMM training, decoding, and sampling.
5. `src/train.py`: full model-training CLI.
6. `src/generate.py`: generation CLI that writes both JSON and MIDI.

Example:

```powershell
D:\miniconda3\envs\learning\python.exe D:\workspace\musicai\hmm\src\train.py `
  --music-dir D:\workspace\musicai\datasets\solocello `
  --model-dir D:\workspace\musicai\hmm\generated\solocello_model `
  --n-bar-clusters 8 `
  --n-sections 4 `
  --verbose

D:\miniconda3\envs\learning\python.exe D:\workspace\musicai\hmm\src\generate.py `
  --model-dir D:\workspace\musicai\hmm\generated\solocello_model `
  --output-json D:\workspace\musicai\hmm\generated\solocello_sample.json `
  --output-midi D:\workspace\musicai\hmm\generated\solocello_sample.mid `
  --measures 32 `
  --seed 42
```
