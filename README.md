# diarization — `pm-image-cli audio record --diarize`

Speaker diarization library for `pm-image-cli`. Answers "who spoke when?"
by assigning each Whisper transcript segment a `SPEAKER_NN` label without
requiring any pre-registered voice profiles.

---

## Repository layout

```
include/diarization/   Public headers  — compile with -Iinclude
  AudioChunk.h
  ISpeakerEmbeddingModel.h
  ModelMetadata.h
  SpeakerVerifier.h
  SpeakerCluster.h
  SpeakerClusterManager.h
  LabelSmoother.h
  DiarizationEngine.h
  TranscriptFormatter.h

src/                   Core implementation
  DiarizationEngine.cpp
  SpeakerClusterManager.cpp
  LabelSmoother.cpp
  TranscriptFormatter.cpp

models/                ONNX model adapters + shared utilities
  WeSpeakerEcapaModel.h/.cpp    WeSpeaker ECAPA-TDNN (FBANK input)
  SpeechBrainEcapaModel.h/.cpp  SpeechBrain ECAPA-TDNN (PCM or FBANK input)
  SpeakerModelFactory.h/.cpp    Path-based model routing
  SpeakerVerifier.h/.cpp        High-level verification API
  EcapaOnnxModel.h/.cpp         Shared ONNX session wrapper
  FBankFrontEnd.h               Header-only 80-dim log-mel filterbank

adapters/              Integration shims
  AudioRecordCommand.h
  WhisperAdapter.h

integration/           Application-layer helpers
  AssistantMerger.h
  DiarizationCli.h
  DiarizationStatus.h

tests/                 Unit / integration / acceptance tests
  verification_trials.cpp   Official VoxCeleb1 verification evaluation

tools/                 Standalone utilities
  check_trials.cpp          Validate trial list against local dataset
  embed_wav.cpp             Inspect embeddings and FBANK for a single WAV

testdata/audio/        WAV samples for testing and benchmarking
bench/                 Benchmark harness
wespeaker/             ONNX model weights (not committed by default)

docs/
  voxceleb_evaluation.md    VoxCeleb1 evaluation protocol and tooling
  voxceleb_results.md       Official VoxCeleb1 evaluation results log
```

## What's included

| Path                                           | Purpose                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| `src/DiarizationEngine.cpp`                    | Core pipeline: embed → cluster → label                              |
| `src/SpeakerClusterManager.cpp`                | Cosine-similarity cluster manager                                   |
| `src/LabelSmoother.cpp`                        | Post-pass to remove single-segment speaker noise                    |
| `src/TranscriptFormatter.cpp`                  | Inline / SRT / VTT / JSON output formatters                         |
| `include/diarization/ModelMetadata.h`          | `ModelMetadata` struct + `describe()` (shape introspection)         |
| `include/diarization/ISpeakerEmbeddingModel.h` | Abstract base for speaker embedding models (`embed`, `inspect`)     |
| `include/diarization/SpeakerVerifier.h`        | High-level speaker verification API (factory-backed)                |
| `models/WeSpeakerEcapaModel.h/.cpp`            | WeSpeaker ECAPA-TDNN: FBANK input `[1,T,80]`, embedding `[1,192]`   |
| `models/SpeechBrainEcapaModel.h/.cpp`          | SpeechBrain ECAPA-TDNN: auto-detects raw PCM or FBANK input         |
| `models/SpeakerModelFactory.h/.cpp`            | Path-based routing: detects WeSpeaker / SpeechBrain from filename   |
| `models/FBankFrontEnd.h`                       | Header-only 80-dim log-mel filterbank (16 kHz, 25 ms/10 ms frames)  |
| `adapters/AudioRecordCommand.h`                | Integration shim for `pm-image-cli`                                 |
| `integration/AssistantMerger.h`                | Injects ASSISTANT TTS segments; clips overlapping diarized segments |
| `integration/DiarizationStatus.h`              | Formats `audio record status` output                                |
| `adapters/WhisperAdapter.h`                    | Bridge from `whisper_context*` to `WhisperSegment`                  |
| `integration/DiarizationCli.h`                 | CLI argument struct (`--diarize`, `--speaker-model`, …)             |

---

## Quick start

```bash
# Stub model (no ONNX required) — unit + integration + acceptance tests
bash ci_run.sh 2>&1 | tee ci_run.log
```

## Build Workflow

Configure once:

```bash
cmake -S . -B build
```

Then build individual targets:

```bash
cmake --build build --target check_trials
cmake --build build --target embed_wav
cmake --build build --target verification_trials
cmake --build build --target integration_tests_real
cmake --build build --target diarization_bench_real
```

or build everything:

```bash
cmake --build build
```

---

## Output formats

```
pm-image-cli audio record --diarize --format inline
pm-image-cli audio record --diarize --format srt
pm-image-cli audio record --diarize --format vtt
pm-image-cli audio record --diarize --format json
```

---

## Speaker models

Pass any supported ONNX model via `--speaker-model <path>`. The model type is
auto-detected by `SpeakerModelFactory` from the filename:

| Keyword in path                     | Model class             | Input                                                          |
| ----------------------------------- | ----------------------- | -------------------------------------------------------------- |
| `wespeaker`, `voxceleb`, `ecapa512` | `WeSpeakerEcapaModel`   | FBANK `[1, T, 80]`                                             |
| `speechbrain`                       | `SpeechBrainEcapaModel` | Raw PCM `[1, T]` or FBANK `[1, T, 80]` (auto-detected at load) |
| _(anything else)_                   | `WeSpeakerEcapaModel`   | FBANK `[1, T, 80]`                                             |

### Validated models

| Model                       | Embedding dim | VoxCeleb1 EER |    AUC | RTF (112 s) | Peak RSS |
| --------------------------- | ------------: | ------------: | -----: | ----------- | -------- |
| `voxceleb_ECAPA512_LM.onnx` |           192 |          2.1% | 0.9969 | 0.0909      | ~121 MB  |

**Download:**
[`voxceleb_ECAPA512_LM.onnx`](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/blob/main/voxceleb_ECAPA512_LM.onnx)
on HuggingFace — `Wespeaker/wespeaker-ecapa-tdnn512-LM`.
Place the file at `wespeaker/voxceleb_ECAPA512_LM.onnx` inside the repo root.

### Model introspection

After loading, call `SpeakerVerifier::inspect()` to retrieve input/output shapes:

```cpp
SpeakerVerifier verifier;
verifier.load("wespeaker/voxceleb_ECAPA512_LM.onnx");
auto meta = verifier.inspect();
std::cout << meta.describe();
// Input:  feats [1, ?, 80]
// Output: embs [1, 192]
```

`ModelMetadata` is also accessible via `ISpeakerEmbeddingModel::inspect()` on
any model class directly.

**Note:** `GetShape()` on static type info crashes on piper's ORT 1.22 build
(dynamic `-1` dims cause `GetDimensionsCount` to return garbage). Fixed in
both `WeSpeakerEcapaModel::load()` and `SpeechBrainEcapaModel::load()` via a
calibration forward pass with silent input.

---

## CI results (21 Jun 2026)

| Step                               | Result                                           |
| ---------------------------------- | ------------------------------------------------ |
| unit_tests                         | ✅ 75 passed                                     |
| integration_tests (stub)           | ✅ 201 620 passed                                |
| integration_tests_real (ONNX)      | ✅ 201 642 passed                                |
| acceptance_test                    | ✅ all 4 formats + AssistantMerger               |
| bench stub (112 s)                 | ✅ RTF 0.0007, 32 MB RSS                         |
| bench real (112 s)                 | ✅ RTF 0.0909, 121 MB RSS                        |
| speaker_verification (LibriSpeech) | ✅ EER 13.3% @ threshold 0.781 (10 speakers)     |
| fbank_validation                   | ✅ C++ vs Python cosine = 1.000000 (no mismatch) |
| voxceleb1_verification             | ✅ EER 2.1%, AUC 0.9969 (37,720 official trials) |

---

## Test audio

`.wav` audio samples used for benchmarking and real-model validation (in `testdata/audio/`):

- `M_1011_13y10m_1.wav` — 112 s, 44 100 Hz mono
- `M_1017_11y8m_1.wav` — 300 s, 44 100 Hz mono

Downloaded from <https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/>

---

## Speaker verification benchmark

### Primary: VoxCeleb1 official verification protocol

Evaluated with `tests/verification_trials` against the official VoxCeleb1
verification trial list (37,720 trials), using the official VoxCeleb1 audio dataset
and `voxceleb_ECAPA512_LM.onnx`.

| Metric                 |                           Value |
| ---------------------- | ------------------------------: |
| Dataset                | VoxCeleb1 official verification |
| Trials                 |                          37,720 |
| Unique utterances      |                           4,715 |
| Same-speaker mean      |                           0.672 |
| Different-speaker mean |                           0.128 |
| Separability gap       |                           0.544 |
| AUC (ROC)              |                          0.9969 |
| EER                    |                            2.1% |
| Recommended threshold  |                           0.413 |
| Evaluation time        |      ~57 min (4,715 embeddings) |

Results use raw cosine similarity without adaptive score normalization (AS-Norm).

The recommended cosine similarity threshold of **0.413** is derived from the
official VoxCeleb1 verification protocol and is suitable as a production
starting point. Full results and distribution plots are in
[`docs/voxceleb_results.md`](docs/voxceleb_results.md).

**Comparison against published results:**

| Implementation                       | VoxCeleb1-O EER |
| ------------------------------------ | --------------: |
| Published WeSpeaker ECAPA-TDNN512-LM |          0.878% |
| This C++ implementation (raw cosine) |            2.1% |

The 1.22 percentage point gap relative to the published WeSpeaker result is expected:
the published figure uses score normalisation and a tuned training pipeline.
This implementation uses raw cosine similarity with no score normalisation.

### Secondary: LibriSpeech sanity check

Measured with `tests/verification_test` on 10 LibriSpeech test-clean speakers
(3 clips each, 30 total). Retained as a **regression test for the embedding
pipeline** — not as a threshold reference.

| Metric                        | Value             |
| ----------------------------- | ----------------- |
| Same-speaker cosine mean      | 0.864 (σ = 0.088) |
| Different-speaker cosine mean | 0.709 (σ = 0.071) |
| EER threshold                 | 0.781             |
| EER                           | ~13.3%            |

The high different-speaker floor (0.709) is expected for LibriSpeech
test-clean (clean read speech, consistent recording conditions). It does not
indicate a model or frontend defect.

Production verification thresholds are characterised using the official
VoxCeleb1 verification protocol. LibriSpeech results are retained only as a
regression test for the embedding pipeline.

---

## Inference pipeline

```
16 kHz mono WAV
  ↓
80-dimensional log-mel FBANK
  25 ms frames, 10 ms hop
  pre-emphasis 0.97
  ↓
Utterance-level feature mean normalization (CMN)
  ↓
WeSpeaker ECAPA-TDNN ONNX (voxceleb_ECAPA512_LM)
  ↓
192-dimensional L2-normalised embedding
  ↓
Cosine similarity
```

### FBANK implementation validation

The custom C++ FBANK implementation in `models/FBankFrontEnd.h` was validated
against a NumPy reference (`tests/compare_embeddings.py`):

```
py/Hann  vs C++         cosine = 1.000000  ✓ MATCH
py/Hann  vs py/Hamming  cosine = 0.981780  (window choice, for reference)
```

During validation it was discovered that the official WeSpeaker inference
also applies **utterance-level cepstral mean normalisation (CMN)** after
FBANK extraction. Adding this step brought the C++ implementation into full
alignment with the reference and reduced VoxCeleb1 EER from ~34.7% to 2.1%.

---

## Evaluation tools

```
tools/check_trials
    Validate a VoxCeleb-style trial list against a local audio directory.
    Reports: Trials in file / Resolvable trials / Missing file pairs.
    Build: g++ -std=c++20 -O2 tools/check_trials.cpp -o tools/check_trials

tools/embed_wav
    Embed a single WAV file and print the 192-D embedding vector.
    Useful for inspecting FBANK extraction and model output.

tests/verification_trials
    Full speaker verification evaluation using the official VoxCeleb1
    trial list. Reports EER, AUC, ROC summary, and recommended threshold.
    See docs/voxceleb_evaluation.md for build and run instructions.
```

### Embedding cache

To minimise evaluation time, each unique utterance is embedded exactly once.
Subsequent trial pairs reuse the cached embedding in memory, eliminating
redundant ONNX inference. The official VoxCeleb1 verification set therefore
requires only 4,715 embedding computations for 37,720 verification trials.

---

## Status

| Requirement                                   | Status |
| --------------------------------------------- | ------ |
| Standalone diarization engine                 | ✅     |
| Whisper integration                           | ✅     |
| ONNX speaker model support                    | ✅     |
| Multi-model support (WeSpeaker + SpeechBrain) | ✅     |
| Path-based model factory                      | ✅     |
| Model metadata / shape introspection          | ✅     |
| Shared FBANK front-end                        | ✅     |
| FBANK validated vs Python reference           | ✅     |
| Utterance-level CMN                           | ✅     |
| Clustering                                    | ✅     |
| Label smoothing                               | ✅     |
| JSON / SRT / VTT / inline output              | ✅     |
| `audio record status`                         | ✅     |
| Assistant/TTS merge                           | ✅     |
| Acceptance test                               | ✅     |
| Real-model integration test                   | ✅     |
| Speaker verification benchmark framework      | ✅     |
| Official VoxCeleb1 verification evaluation    | ✅     |
| Benchmark (stub + real)                       | ✅     |
| Documentation                                 | ✅     |

Real ECAPA-TDNN model (`voxceleb_ECAPA512_LM.onnx`, dim=192) validated:

- loads and embeds correctly via `SpeakerVerifier` (factory-routed)
- `inspect()` returns correct shape: `Input: feats [1, ?, 80]  Output: embs [1, 192]`
- same-chunk cosine similarity = 1.0000
- short-clip embedding L2 norm = 1.0000
- cross-chunk cosine (first vs last second) = 0.5144
- 112 s file processed at RTF 0.0909 (11× real-time), 121 MB RSS
- FBANK implementation validated vs Python reference (cosine = 1.000000)
- VoxCeleb1 official verification: EER 2.1%, AUC 0.9969, threshold 0.413

---

## Next steps

- **Evaluate additional WeSpeaker architectures** — ECAPA-1024, ResNet34,
  CAM++; compare EER and AUC against the current ECAPA-512 baseline.
- **Compare against SpeechBrain models** — run the same VoxCeleb1 trial
  protocol against SpeechBrain ECAPA-TDNN weights.
- **Implement adaptive score normalisation (AS-Norm)** — score normalisation
  is the primary remaining gap between this implementation (2.1% EER) and
  the published WeSpeaker result (0.878% EER).
- **Characterise performance under degraded conditions** — noisy environments,
  far-field microphones, telephone-bandwidth audio.
- **Separate verification and diarization thresholds** — the cosine threshold
  in `SpeakerClusterManager` (clustering) and the verification threshold
  (binary same/different-speaker decision) address different problems and
  should be tuned independently.
- **Evaluate on ARM platforms** — characterise RTF and memory footprint on
  embedded deployment targets.
