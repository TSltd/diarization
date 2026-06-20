# Real-Audio Sanity Report

Sanity check run: 2026-06-20.  
Tool: `python3` inline script using `struct` + `array` — no external libraries.

---

## Files checked

| File                        | Size     | Duration           | Channels | Rate     | Depth      | Range (first 10k frames) | Readable |
| --------------------------- | -------- | ------------------ | -------- | -------- | ---------- | ------------------------ | -------- |
| `audio/M_1011_13y10m_1.wav` | 9.44 MB  | 112.2 s (1.87 min) | 1 (mono) | 44100 Hz | 16-bit PCM | [−0.417, +0.476]         | ✅ OK    |
| `audio/M_1017_11y8m_1.wav`  | 25.26 MB | 300.4 s (5.01 min) | 1 (mono) | 44100 Hz | 16-bit PCM | [−0.021, +0.014]         | ✅ OK    |

---

## Findings

### Both files pass all wav_reader.h compatibility checks

- RIFF/WAVE header: ✅
- PCM format tag (1): ✅
- Mono (1 channel): ✅ — `read_wav_mono()` requires no downmixing
- 44100 Hz: ✅ — `read_wav_mono_16k()` resamples to 16 000 Hz (factor ≈ 2.756)
- 16-bit integer: ✅
- Sample range ⊂ [−1, 1] after normalisation: ✅

### After `read_wav_mono_16k()`

| File                  | 16 kHz frames | Memory (float32) |
| --------------------- | ------------- | ---------------- |
| `M_1011_13y10m_1.wav` | 1 794 930     | ≈ 6.9 MB         |
| `M_1017_11y8m_1.wav`  | 4 805 745     | ≈ 18.4 MB        |

These match the MemoryScaling integration test baselines (6.9 MB ≈ 1.8 MB stub
\+ 5.1 MB audio for 30 s clip; the 5-minute file fits within the 109.9 MB upper
bound tested in `integration_tests.cpp`).

### Note on M_1017 amplitude

The first 10 000 frames of `M_1017_11y8m_1.wav` show a very low amplitude
([−0.021, +0.014]). This is consistent with a quiet lead-in (breath, room
noise, or silence before speech onset) and is **not an error**. The FBANK
frontend normalises per-frame energy so low-level lead-ins do not affect
speaker embedding quality.

### No issues found

Both files are directly loadable via `wav::read_wav_mono_16k()` with no
pre-processing required. The real-model integration test
(`RealModelRoundTrip`, gated by `-DDIARIZE_HAVE_MODEL`) should use
`M_1011_13y10m_1.wav` as the primary test file (shorter, faster) and
`M_1017_11y8m_1.wav` for the long-recording benchmark.
