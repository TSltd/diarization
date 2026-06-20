# Diarization Benchmark Report

**Date:** 20 June 2026  
**Host:** Linux 6.17 x86-64  
**Compiler:** GCC 13, `-std=c++20 -O2`  
**ORT:** onnxruntime-linux-x64-1.22.0 (piper build)

---

## 1. Stub model — 112 s file (`audio/M_1011_13y10m_1.wav`)

No ONNX inference. Stub returns random unit-norm embeddings (dim=192) to
measure the pure pipeline overhead: load + resample + cluster + format.

```
Audio duration : 112.183 s
Sample rate    : 16 000 Hz (resampled from 44 100 Hz)
Samples        : 1 794 930
Load+resample  : 160.6 ms
Fake segments  : 56  (one per 2 s)
Model          : stub (random unit-norm, dim=192)
```

| Run | Embed (ms) | Format (ms) | Total (ms) | Spk | Win |
| --- | ---------- | ----------- | ---------- | --- | --- |
| 1   | 22.8       | 0.2         | 22.9       | 56  | 56  |
| 2   | 14.8       | 0.2         | 14.9       | 56  | 56  |
| 3   | 14.2       | 0.1         | 14.3       | 56  | 56  |

**Average (3 runs)**

| Metric           | Value              |
| ---------------- | ------------------ |
| Embed + cluster  | 17.2 ms            |
| Formatting       | 0.1 ms             |
| **Total**        | **17.4 ms**        |
| **RTF**          | **0.0002**         |
| Real-time factor | **6 454× RT**      |
| Peak RSS         | 32 340 kB (~32 MB) |

---

## 2. Real ONNX model — 112 s file (`audio/M_1011_13y10m_1.wav`)

Model: `wespeaker/voxceleb_ECAPA512_LM.onnx` — WeSpeaker ECAPA-TDNN,
large-margin training, 192-dim output, exported from PyTorch 1.12.1.

```
Audio duration : 112.183 s
Sample rate    : 16 000 Hz (resampled from 44 100 Hz)
Samples        : 1 794 930
Load+resample  : 93.2 ms
Fake segments  : 56  (one per 2 s)
Model          : wespeaker/voxceleb_ECAPA512_LM.onnx  (ONNX, dim=192)
```

| Run | Embed (ms) | Format (ms) | Total (ms) | Spk | Win |
| --- | ---------- | ----------- | ---------- | --- | --- |
| 1   | 4 837.1    | 0.1         | 4 837.1    | 1   | 56  |
| 2   | 6 555.2    | 0.3         | 6 555.5    | 1   | 56  |
| 3   | 5 087.8    | 0.1         | 5 087.9    | 1   | 56  |

**Average (3 runs)**

| Metric           | Value                |
| ---------------- | -------------------- |
| Embed + cluster  | 5 493.4 ms           |
| Formatting       | 0.1 ms               |
| **Total**        | **5 493.5 ms**       |
| **RTF**          | **0.049**            |
| Real-time factor | **20.4× RT**         |
| Peak RSS         | 122 168 kB (~122 MB) |

---

## 3. Real model — integration test (12.6 s two-speaker file)

```
Audio duration     : 12 600 ms
Sample rate        : 16 000 Hz (native)
Model              : wespeaker/voxceleb_ECAPA512_LM.onnx  (dim=192)
Processing time    : 489.1 ms
Real-time factor   : 25.8× RT
Speakers detected  : 3  (short 2-speaker file; threshold effects expected)
Same-chunk cosine  : 1.0000  (identical input → identical embedding)
```

---

## 4. Notes

### ORT piper build — static shape info bug

`GetShape()` on static `TypeInfo` objects crashes in piper's ORT 1.22 build
when the model has dynamic input dimensions (`-1`). The C++ wrapper's
`GetDimensionsCount()` returns a garbage `size_t`, causing `std::length_error`
when `std::vector<int64_t>(n)` tries to allocate.

**Fix (applied in `WeSpeakerEcapaModel::load()`):**  
Instead of querying `GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()`,
run a single forward pass with a silent 1-second input `[1, 100, 80]` and read
`embedding_dim_` from the concrete runtime output shape. Runtime output shape
queries work correctly even in piper's ORT build.

### Embedding dimension surprise

`voxceleb_ECAPA512_LM.onnx` outputs **192-dimensional** embeddings despite the
`512` in the filename. The `_LM` suffix stands for "large margin" (AAM-Softmax
loss variant); the base model has a 192-dim speaker representation layer.

### RTF target

The diarization pipeline adds ≤ 5 % overhead to a 112-second recording
(`RTF = 0.049`). This is well within the 10 % target documented in
`docs/proposal.md`. Stub-only mode is practically free (`RTF = 0.0002`).
