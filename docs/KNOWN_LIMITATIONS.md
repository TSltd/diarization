# Known Limitations

This document records the current constraints, accuracy caveats, and
architecture decisions that users and integrators should be aware of.

---

## 1. Speaker accuracy degrades below ~1 s segments

The ECAPA-TDNN / x-vector models are trained on utterances of 2–10 s.
Whisper sometimes produces very short segments (<1 s) for filler words or
partial sentences. The engine drops any segment shorter than
`speaker_min_ms` (default 800 ms) from embedding entirely and marks it
`UNKNOWN`, which propagates until LabelSmoother corrects it if the
surrounding context is consistent.

**Workaround:** Increase `--speaker-min-ms` or use Whisper in
`--word-timestamps` mode and merge short tokens before passing to the engine.

---

## 2. Overlapping speech is not supported

The cosine-similarity clustering model assumes exactly one active speaker
per window. Crosstalk or simultaneous speech causes the window to be
assigned to one speaker (the louder one dominates the embedding), and the
other speaker is silently ignored.

**Workaround:** None currently. A future version could run multi-label
classification or use a separation front-end.

---

## 3. Speaker count is unknown at start — accuracy improves with more audio

The online clustering creates a new speaker cluster only when no existing
cluster exceeds `speaker_threshold` (default 0.72). Early in a recording,
two acoustically similar speakers may be merged into one cluster before
enough evidence separates them. Accuracy improves over longer recordings
as centroids converge.

**Workaround:** Use `--speaker-max N` to cap the number of speakers when
the count is known in advance. For offline batch processing, consider
re-running clustering after the full audio is seen (not yet implemented).

---

## 4. No cross-recording speaker identity

`DiarizationEngine::reset()` must be called between independent recordings.
Speaker labels (SPEAKER_00, SPEAKER_01, …) are assigned in order of first
appearance within a single call to `process()` and have no meaning across
calls. The same physical speaker may receive different labels in two
separate runs.

---

## 5. ONNX model compatibility is not validated at runtime

`WeSpeakerEcapaModel` (and `EcapaOnnxModel` / `XVectorOnnxModel`) load any
`.onnx` file that is passed to `--speaker-model`. If the file has different
input/output node names or tensor shapes, the error surfaces as an ONNX
Runtime exception at the first `embed()` call, not at `load()`.

**Workaround:** Always verify that the model was exported from the
[WeSpeaker VoxCeleb2 recipe](https://wespeaker-1256283475.cos.ap-beijing.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.onnx)
or configure `input_name` / `output_name` in your model wrapper.

---

## 6. No speaker enrolment / named speakers

`--speaker-enroll` is reserved in the CLI struct (`DiarizationCliArgs`) but
is not yet implemented. All speakers receive opaque labels (SPEAKER_00,
SPEAKER_01, …). A future version will support an enrolment directory of
short reference clips mapped to named labels.

---

## 7. Sample-rate conversion is linear interpolation only

`tests/wav_reader.h::resample_mono()` uses linear interpolation to
downsample from 44 100 Hz (the real audio files) to 16 000 Hz. This is
fast but introduces minor aliasing. For production use a proper
anti-aliasing filter (e.g. libsoxr or libresample) should be used before
passing audio to the engine.

---

## 8. Thread safety

`DiarizationEngine` is not thread-safe. Concurrent calls to `process()`
on the same instance will cause data races in `SpeakerClusterManager`.
Use one engine instance per thread, or guard with a mutex.

---

## 9. Whisper confidence values are segment-level only

`WhisperSegment::confidence` is filled from `whisper_full_get_segment_prob()`,
which is the mean token log-probability exponentiated. It is a rough proxy
for ASR quality, not a calibrated probability. Values close to 0 reliably
indicate noise or silence segments; values near 1 do not guarantee correct
transcription.

---

## 10. LabelSmoother does not span segment boundaries

The smoother operates on a flat list of labelled segments. It will not
merge two consecutive SPEAKER_00 segments if there is a SPEAKER_01 segment
of any length between them, even if the SPEAKER_01 segment is only 1 ms.
Setting `--smoother-min-ms` too high can cause legitimate short speaker
turns to be relabelled.

---

## 11. Memory: embeddings are not stored long-term

Only cluster centroids are retained (one `std::vector<float>` per speaker,
192-dim for ECAPA-512). Raw per-window embeddings are discarded after
cluster assignment. This means the clustering cannot be re-optimised
offline (e.g. k-means post-pass) without re-running the full inference.

---

## 12. No support for multi-channel audio

`AudioBuffer` stores interleaved samples but `DiarizationEngine::slice()`
assumes `channels == 1`. Passing stereo audio will produce incorrect
timing and garbage embeddings. Downmix to mono before calling `process()`.
