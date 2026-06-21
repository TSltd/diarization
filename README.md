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
testdata/audio/        WAV samples for testing and benchmarking
bench/                 Benchmark harness
wespeaker/             ONNX model weights (not committed by default)
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

| Model                       | Dim | RTF (112 s file) | Peak RSS |
| --------------------------- | --- | ---------------- | -------- |
| `voxceleb_ECAPA512_LM.onnx` | 192 | 0.0909 (11×RT)   | ~121 MB  |

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

| Step                          | Result                             |
| ----------------------------- | ---------------------------------- |
| unit_tests                    | ✅ 75 passed                       |
| integration_tests (stub)      | ✅ 201 620 passed                  |
| integration_tests_real (ONNX) | ✅ 201 642 passed                  |
| acceptance_test               | ✅ all 4 formats + AssistantMerger |
| bench stub (112 s)            | ✅ RTF 0.0007, 32 MB RSS           |
| bench real (112 s)            | ✅ RTF 0.0909, 121 MB RSS          |

---

## Test audio

`.wav` audio samples used for benchmarking and real-model validation (in `testdata/audio/`):

- `M_1011_13y10m_1.wav` — 112 s, 44 100 Hz mono
- `M_1017_11y8m_1.wav` — 300 s, 44 100 Hz mono

Downloaded from <https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/>

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
| Clustering                                    | ✅     |
| Label smoothing                               | ✅     |
| JSON / SRT / VTT / inline output              | ✅     |
| `audio record status`                         | ✅     |
| Assistant/TTS merge                           | ✅     |
| Acceptance test                               | ✅     |
| Real-model integration test                   | ✅     |
| Benchmark (stub + real)                       | ✅     |
| Documentation                                 | ✅     |

Real ECAPA-TDNN model (`voxceleb_ECAPA512_LM.onnx`, dim=192) validated:

- loads and embeds correctly via `SpeakerVerifier` (factory-routed)
- `inspect()` returns correct shape: `Input: feats [1, ?, 80]  Output: embs [1, 192]`
- same-chunk cosine similarity = 1.0000
- short-clip embedding L2 norm = 1.0000
- cross-chunk cosine (first vs last second) = 0.5144
- 112 s file processed at RTF 0.0909 (11× real-time), 121 MB RSS
