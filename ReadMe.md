# diarization — `pm-image-cli audio record --diarize`

Speaker diarization library for `pm-image-cli`. Answers "who spoke when?"
by assigning each Whisper transcript segment a `SPEAKER_NN` label without
requiring any pre-registered voice profiles.

---

## What's included

| File                           | Purpose                                                             |
| ------------------------------ | ------------------------------------------------------------------- |
| `DiarizationEngine.h/.cpp`     | Core pipeline: embed → cluster → label                              |
| `SpeakerClusterManager.h/.cpp` | Cosine-similarity cluster manager                                   |
| `LabelSmoother.h/.cpp`         | Post-pass to remove single-segment speaker noise                    |
| `TranscriptFormatter.h/.cpp`   | Inline / SRT / VTT / JSON output formatters                         |
| `WeSpeakerEcapaModel.h/.cpp`   | ONNX ECAPA-TDNN speaker embedding model                             |
| `AudioRecordCommand.h`         | Integration shim for `pm-image-cli`                                 |
| `AssistantMerger.h`            | Injects ASSISTANT TTS segments; clips overlapping diarized segments |
| `DiarizationStatus.h`          | Formats `audio record status` output                                |
| `WhisperAdapter.h`             | Bridge from `whisper_context*` to `WhisperSegment`                  |
| `DiarizationCli.h`             | CLI argument struct (`--diarize`, `--speaker-model`, …)             |

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

## Speaker model

Drop any WeSpeaker-style ONNX model at `wespeaker/<name>.onnx` and pass
`--speaker-model wespeaker/<name>.onnx`. The validated model is:

| Model                       | Dim | RTF (112 s file) | Peak RSS |
| --------------------------- | --- | ---------------- | -------- |
| `voxceleb_ECAPA512_LM.onnx` | 192 | 0.049 (20×RT)    | ~122 MB  |

**Download:**
[`voxceleb_ECAPA512_LM.onnx`](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/blob/main/voxceleb_ECAPA512_LM.onnx)
on HuggingFace — `Wespeaker/wespeaker-ecapa-tdnn512-LM`.
Place the file at `wespeaker/voxceleb_ECAPA512_LM.onnx` inside the repo root.

**Note:** `GetShape()` on static type info crashes on piper's ORT 1.22 build
(dynamic `-1` dims cause `GetDimensionsCount` to return garbage). Fixed in
`WeSpeakerEcapaModel::load()` via a calibration forward pass with silent input.

---

## CI results (20 Jun 2026)

| Step                          | Result                             |
| ----------------------------- | ---------------------------------- |
| unit_tests                    | ✅ 57 passed                       |
| integration_tests (stub)      | ✅ 201 620 passed                  |
| integration_tests_real (ONNX) | ✅ 201 627 passed                  |
| acceptance_test               | ✅ all 4 formats + AssistantMerger |
| bench stub (112 s)            | ✅ RTF 0.0002, 32 MB RSS           |
| bench real (112 s)            | ✅ RTF 0.049, 122 MB RSS           |

---

## Test audio

`.wav` audio samples used for benchmarking and real-model validation:

- `M_1011_13y10m_1.wav` — 112 s, 44 100 Hz mono
- `M_1017_11y8m_1.wav` — 300 s, 44 100 Hz mono

Downloaded from <https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/>

---

## Status

| Requirement                      | Status |
| -------------------------------- | ------ |
| Standalone diarization engine    | ✅     |
| Whisper integration              | ✅     |
| ONNX speaker model support       | ✅     |
| Clustering                       | ✅     |
| Label smoothing                  | ✅     |
| JSON / SRT / VTT / inline output | ✅     |
| `audio record status`            | ✅     |
| Assistant/TTS merge              | ✅     |
| Acceptance test                  | ✅     |
| Real-model integration test      | ✅     |
| Benchmark (stub + real)          | ✅     |
| Documentation                    | ✅     |

Real ECAPA-TDNN model (`voxceleb_ECAPA512_LM.onnx`, dim=192) validated:

- loads and embeds correctly
- same-chunk cosine similarity = 1.0000
- 12.6 s test file processed in 489 ms (25.8× real-time)
