# VoxCeleb Evaluation

## Purpose

The LibriSpeech verification benchmark provides a useful sanity check for speaker embeddings, but it is not representative of real-world speaker verification deployments. LibriSpeech contains clean, read speech recorded under relatively consistent conditions, whereas ECAPA-TDNN models are typically trained and evaluated on VoxCeleb-style data collected from interviews, broadcasts, and online videos.

This evaluation measures speaker-verification performance using the official VoxCeleb1 verification protocol and is used to:

- Validate ONNX model integration
- Verify FBANK frontend correctness
- Measure Equal Error Rate (EER)
- Estimate deployment thresholds
- Compare alternative speaker-embedding models

---

# Dataset

## Audio

Dataset location:

```text
testdata/voxceleb_trials/audio/voxceleb1/
```

Expected layout:

```text
voxceleb1/
├── id10001/
│   ├── youtube_id/
│   │   ├── 00001.wav
│   │   ├── 00002.wav
│   │   └── ...
│   └── ...
├── id10002/
└── ...
```

Properties:

- Real-world interview and broadcast audio
- Multiple recording sessions per speaker
- Multiple recording environments
- Variable utterance duration
- Original VoxCeleb1 directory hierarchy preserved

---

## Trial Protocol

Official VoxCeleb1 verification trial list:

```text
testdata/voxceleb_trials/trials.txt
```

Format:

```text
1 voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
0 voxceleb1/id10934/R1twEbtsG1w/00003.wav voxceleb1/id10453/2q9ZA1_VeGM/00003.wav
```

Where:

```text
1 = same speaker
0 = different speakers
```

Paths are resolved relative to the supplied audio base directory.

---

## Trial Validation

Before running the benchmark, verify that the audio corpus matches the trial protocol:

```bash
g++ -std=c++20 -O2 tools/check_trials.cpp -o tools/check_trials

./tools/check_trials \
    testdata/voxceleb_trials/trials.txt \
    testdata/voxceleb_trials/audio
```

Expected output:

```text
Trials in file:     263486
Resolvable trials:  263486
Missing file pairs: 0
```

Any missing trial files should be investigated before interpreting benchmark results.

---

# Evaluation Pipeline

```text
WAV
  ↓
Resample 16 kHz
  ↓
FBANK (80 mel)
  ↓
ECAPA-TDNN ONNX
  ↓
192-D embedding
  ↓
L2 normalization
  ↓
Cosine similarity
```

Speaker embeddings are computed as:

```cpp
auto emb = verifier.embed(path);
```

Verification scores are computed as:

```cpp
float score = cosine(embA, embB);
```

All scores are evaluated against the ground-truth labels provided by the official trial protocol.

---

# Benchmark Configuration

## Acoustic Frontend

```text
Sample rate:      16 kHz
Mel bins:         80
Pre-emphasis:     0.97
Window:           Hann
FFT size:         512
Compression:      Natural logarithm
```

## Embeddings

```text
Dimension:        192
Normalization:    L2
```

## Scoring

```text
Metric:           Cosine similarity
Threshold sweep:  0.000 – 1.000
Step size:        0.001
```

---

# Metrics

## Same-Speaker Distribution

Computed across all positive trials.

Reported:

- Mean
- Standard deviation
- Minimum
- Maximum

## Different-Speaker Distribution

Computed across all negative trials.

Reported:

- Mean
- Standard deviation
- Minimum
- Maximum

## Equal Error Rate (EER)

For every threshold:

```text
False Accept Rate (FAR)
False Reject Rate (FRR)
```

are computed.

The EER threshold is defined as:

```text
FAR ≈ FRR
```

Reported:

```text
EER threshold
EER percentage
FAR
FRR
```

---

## Area Under Curve (AUC)

Computed via trapezoidal integration of the ROC curve.

Interpretation:

```text
0.50 = random guessing
1.00 = perfect separation
```

---

# Supported Models

The benchmark framework supports any implementation of:

```cpp
ISpeakerEmbeddingModel
```

Current implementations:

| Model                     | Status    |
| ------------------------- | --------- |
| WeSpeaker ECAPA-TDNN      | Verified  |
| SpeechBrain ECAPA-TDNN    | Supported |
| Custom ECAPA ONNX exports | Supported |

Model metadata is reported automatically through:

```cpp
inspect()
```

and includes:

- Input tensor name
- Output tensor name
- Input shape
- Output shape
- Embedding dimension

---

# Evaluation Binary

## Build

```bash
ORT=/home/dan/Documents/piper1-gpl/libpiper/lib/onnxruntime-linux-x64-1.22.0

g++ -std=c++20 -O2 -Iinclude -I. -DDIARIZE_HAVE_MODEL \
    -I$ORT/include \
    tests/verification_trials.cpp \
    src/DiarizationEngine.cpp \
    src/SpeakerClusterManager.cpp \
    src/LabelSmoother.cpp \
    src/TranscriptFormatter.cpp \
    models/WeSpeakerEcapaModel.cpp \
    models/EcapaOnnxModel.cpp \
    models/SpeechBrainEcapaModel.cpp \
    models/SpeakerModelFactory.cpp \
    models/SpeakerVerifier.cpp \
    -L$ORT/lib \
    -Wl,-rpath,$ORT/lib \
    -lonnxruntime \
    -o tests/verification_trials
```

## Run

```bash
./tests/verification_trials \
    wespeaker/voxceleb_ECAPA512_LM.onnx \
    testdata/voxceleb_trials/trials.txt \
    testdata/voxceleb_trials/audio
```

---

# Embedding Cache

Each unique WAV file is embedded exactly once.

Subsequent references reuse the cached embedding.

Benefits:

- Eliminates redundant ONNX inference
- Reduces runtime dramatically
- Makes large trial lists practical

Reported statistics:

```text
Unique utterances
Cache hits
Cache misses
Embedding time
Average ms/utterance
```

---

# Reference Results

Published results for the WeSpeaker ECAPA model:

```text
Model:
ECAPA_TDNN_GLOB_c512-ASTP-emb192
```

| LM  | AS-Norm | VoxCeleb1-O EER |
| --- | ------- | --------------- |
| No  | No      | 1.069%          |
| No  | Yes     | 0.957%          |
| Yes | No      | 0.878%          |
| Yes | Yes     | 0.782%          |

These values provide a reference point for implementation validation.

Direct reproduction is not expected unless evaluation conditions exactly match the published setup.

---

# Validation Criteria

Guideline only:

```text
Excellent:
    EER within 2× published value

Good:
    EER within 5× published value

Needs investigation:
    EER > 10× published value
```

Large deviations should trigger inspection of:

- Dataset integrity
- Trial protocol
- Frontend implementation
- Model export
- Score normalization

---

# Expected Outcome

Compared with LibriSpeech:

- Same-speaker scores should remain high
- Different-speaker scores should separate more clearly
- EER should decrease substantially
- Recommended thresholds should better reflect production behaviour

A successful evaluation should demonstrate meaningful separation between positive and negative score distributions and produce EER values consistent with published ECAPA verification results.

---

# Deliverables

The verification executable reports:

- Total trials
- Positive trials
- Negative trials
- Unique utterances embedded
- Cache statistics
- Missing files
- Same-speaker distribution
- Different-speaker distribution
- ASCII histograms
- ROC summary
- EER threshold
- EER value
- FAR / FRR
- AUC
- Recommended threshold

Results should be recorded in:

```text
docs/voxceleb_results.md
```

along with:

- Model name
- Model metadata
- Embedding dimension
- Evaluation date
- Trial counts
- Threshold recommendation
- Hardware used
- Runtime statistics

---
