Implement speaker diarization / speaker differentiation for the existing `pm-image-cli audio record` command.

Context:

- Existing stack: Win32 C++ CLI, whisper.cpp, ONNX Runtime, llama.cpp already integrated.
- Do **not** use Silero VAD.
- Do **not** depend on WebRTC APM for this feature.
- Goal: produce transcripts with stable speaker labels such as `SPEAKER_00`, `SPEAKER_01`.
- Assistant/TTS voice should be tagged from internal TTS playback events, not detected acoustically.

Existing command:

```bash
pm-image-cli audio record --provider whisper --model dist/models/ggml-base.en.bin --backend gpu --duration 3000
```

Add diarization support.

New CLI flags:

```txt
--diarize
--speaker-model TEXT          ONNX speaker embedding model path
--speaker-threshold FLOAT     Same-speaker cosine threshold, default 0.72
--speaker-window-ms INT       Embedding window size, default 1500
--speaker-hop-ms INT          Embedding hop size, default 750
--speaker-min-ms INT          Minimum speech segment for embedding, default 800
--speaker-max INT             Optional maximum expected speakers, 0 = auto
--speaker-enroll TEXT         Optional folder/file with known speaker WAV samples
--speaker-format TEXT         one of: inline,json,srt,vtt
```

Pipeline:

```txt
capture PCM s16le mono 16 kHz
  -> whisper.cpp transcription with timestamps
  -> speech chunks from whisper segments/token timestamps
  -> ONNX speaker embedding per speech chunk/window
  -> cosine similarity + online/offline clustering
  -> merge transcript segments with speaker labels
  -> output text/json/srt/vtt
```

Do not add a heavy ML framework. Use ONNX Runtime only for speaker embeddings.

Speaker model:

- Support ECAPA-TDNN / x-vector style ONNX models.
- Input should be normalized mono PCM or log-mel features depending on the selected model.
- Implement model adapter interface:

```cpp
class ISpeakerEmbeddingModel {
public:
    virtual ~ISpeakerEmbeddingModel() = default;
    virtual bool load(const std::string& model_path) = 0;
    virtual std::vector<float> embed(const AudioChunk& chunk) = 0;
    virtual int sample_rate() const = 0;
    virtual int embedding_dim() const = 0;
};
```

Add clustering:

```cpp
struct SpeakerCluster {
    std::string label;              // SPEAKER_00
    std::vector<float> centroid;    // normalized
    int segment_count = 0;
};
```

Algorithm:

1. Use Whisper timestamps to create candidate speech chunks.
2. Ignore chunks shorter than `--speaker-min-ms`.
3. Split long chunks into windows of `--speaker-window-ms` with `--speaker-hop-ms`.
4. Compute speaker embedding for each valid window.
5. Normalize every embedding to unit length.
6. Compare embedding with existing cluster centroids using cosine similarity.
7. If best score >= `--speaker-threshold`, assign to that cluster.
8. Otherwise create new cluster unless `--speaker-max` is reached.
9. Update centroid with running mean or exponential moving average.
10. Smooth short label flips by merging isolated segments surrounded by the same speaker.

Implement:

```cpp
SpeakerId assign_speaker(const std::vector<float>& embedding);
float cosine_similarity(std::span<const float> a, std::span<const float> b);
void update_centroid(SpeakerCluster& cluster, const std::vector<float>& embedding);
```

Transcript segment format:

```cpp
struct TranscriptSegment {
    int64_t start_ms;
    int64_t end_ms;
    std::string speaker;   // SPEAKER_00, SPEAKER_01, ASSISTANT, UNKNOWN
    std::string text;
    float confidence;
};
```

JSON output with `--json`:

```json
{
  "provider": "whisper",
  "model": "dist/models/ggml-base.en.bin",
  "duration_ms": 3000,
  "diarization": {
    "enabled": true,
    "speaker_model": "dist/models/speaker-ecapa.onnx",
    "threshold": 0.72,
    "speakers": [
      { "id": "SPEAKER_00", "segments": 4 },
      { "id": "SPEAKER_01", "segments": 2 }
    ]
  },
  "segments": [
    {
      "start_ms": 120,
      "end_ms": 1840,
      "speaker": "SPEAKER_00",
      "text": "Open the settings.",
      "confidence": 0.91
    }
  ],
  "text": "SPEAKER_00: Open the settings."
}
```

SRT output:

```txt
1
00:00:00,120 --> 00:00:01,840
[SPEAKER_00] Open the settings.
```

VTT output:

```txt
WEBVTT

00:00:00.120 --> 00:00:01.840
<SPEAKER_00> Open the settings.
```

Plain text output:

```txt
[SPEAKER_00 00:00.120-00:01.840] Open the settings.
[SPEAKER_01 00:02.100-00:03.500] Yes, do it.
```

Integrate with existing `audio record` behavior:

- `--diarize` implies `--stt`.
- `--diarize` requires `--provider whisper` for now.
- `--diarize` works with live mic recording and `--from-wav`.
- If `--text-out` is set, write selected speaker format.
- If `--json` is set, print/write structured JSON.
- Existing non-diarized recording behavior must remain unchanged.
- `audio record status` should show diarization enabled, speaker model path, and current known speaker count.

Assistant/TTS handling:

- Do not identify assistant/TTS through the speaker model.
- Since the app owns TTS playback, inject assistant events directly:

```cpp
record_assistant_segment(start_ms, end_ms, generated_text);
```

Then merge human mic transcript and assistant TTS transcript chronologically.

Implementation phases:

1. Offline diarization for `--from-wav`.
2. Diarized output for normal `audio record --duration`.
3. Optional online diarization with delayed stabilization.

Acceptance tests:

1. `--from-wav test_two_speakers.wav --provider whisper --model dist/models/ggml-base.en.bin --diarize --speaker-model dist/models/speaker-ecapa.onnx --json` produces at least two speaker labels.
2. Same speaker across multiple utterances keeps the same label.
3. Short silence does not create a new speaker.
4. `--speaker-format srt` produces valid SRT.
5. `--speaker-format vtt` produces valid WebVTT.
6. Existing command without `--diarize` behaves exactly as before.
