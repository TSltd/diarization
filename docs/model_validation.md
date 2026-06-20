# Speaker Model Validation Record

Maintain one entry per tested model. Update `Verified` to `yes` after a
successful `RealModelRoundTrip` integration test run.

---

## voxceleb_ECAPA512_LM.onnx ← **validated in-repo model**

| Field                    | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| **Model name**           | `voxceleb_ECAPA512_LM.onnx`                                                                        |
| **Source**               | WeSpeaker VoxCeleb2 ECAPA-TDNN, large-margin (AAM-Softmax) recipe                                  |
| **Location**             | `wespeaker/voxceleb_ECAPA512_LM.onnx` (in-repo, 24 MB)                                             |
| **Training data**        | VoxCeleb 1 + 2 (7 205 speakers)                                                                    |
| **Architecture**         | ECAPA-TDNN, 192-dim output (despite "512" in filename — `_LM` uses a 192-dim representation layer) |
| **Embedding dimension**  | **192**                                                                                            |
| **Input node**           | `feats`                                                                                            |
| **Input shape**          | `[1, T, 80]` — batch=1, T=variable frames, 80 mel bins; dynamic dims cause piper ORT static bug    |
| **Output node**          | `embs`                                                                                             |
| **Output shape**         | `[1, 192]` (runtime-concrete; static shape info crashes piper ORT — see benchmark_report.md)       |
| **Frontend**             | 80-dim log-mel FBANK: 25 ms frame / 10 ms hop, 20–7600 Hz, pre-emphasis 0.97                       |
| **Required sample rate** | 16 000 Hz                                                                                          |
| **Wrapper class**        | `WeSpeakerEcapaModel` (`WeSpeakerEcapaModel.h/.cpp`)                                               |
| **Verified**             | **yes — 20 Jun 2026**                                                                              |
| **Test audio**           | `audio/M_1011_13y10m_1.wav` (112 s), `test_two_speakers.wav` (12.6 s)                              |
| **Cosine (same chunk)**  | 1.0000 (identical input → identical L2-normalised embedding)                                       |
| **RTF (112 s file)**     | 0.049 (20.4× real-time), Peak RSS 122 MB                                                           |
| **Notes**                | ORT piper build: use calibration forward pass in `load()` to discover dim; skip static shape query |

---

## voxceleb_ECAPA512.onnx

| Field                    | Value                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| **Model name**           | `voxceleb_ECAPA512.onnx`                                                                          |
| **Source**               | WeSpeaker VoxCeleb2 ECAPA-TDNN recipe                                                             |
| **Download URL**         | `https://wespeaker-1256283475.cos.ap-beijing.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.onnx` |
| **Training data**        | VoxCeleb 1 + 2 (7 205 speakers)                                                                   |
| **Architecture**         | ECAPA-TDNN, 512-dim output                                                                        |
| **Embedding dimension**  | 512                                                                                               |
| **Input node**           | `feats`                                                                                           |
| **Input shape**          | `[1, T, 80]` — batch=1, T=variable frames, 80 mel bins                                            |
| **Output node**          | `embs`                                                                                            |
| **Output shape**         | `[1, 512]`                                                                                        |
| **Frontend**             | 80-dim log-mel FBANK: 25 ms frame / 10 ms hop, 20–7600 Hz, pre-emphasis 0.97                      |
| **Required sample rate** | 16 000 Hz                                                                                         |
| **Wrapper class**        | `WeSpeakerEcapaModel` (`WeSpeakerEcapaModel.h/.cpp`)                                              |
| **Verified**             | pending (run `RealModelRoundTrip` with `-DDIARIZE_HAVE_MODEL`)                                    |
| **Test audio**           | `audio/M_1011_13y10m_1.wav` (112 s), `audio/M_1017_11y8m_1.wav` (300 s)                           |
| **Notes**                | Node names configurable via `WeSpeakerEcapaModel::set_node_names()`                               |

### Verification procedure

```bash
# 1. Download model
curl -L -o models/voxceleb_ECAPA512.onnx \
  https://wespeaker-1256283475.cos.ap-beijing.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.onnx

# 2. Build integration tests with real model support
g++ -std=c++20 -O2 -DDIARIZE_HAVE_MODEL \
    -I. \
    tests/integration_tests.cpp \
    DiarizationEngine.cpp SpeakerClusterManager.cpp \
    LabelSmoother.cpp TranscriptFormatter.cpp \
    WeSpeakerEcapaModel.cpp \
    -lonnxruntime \
    -o tests/integration_tests_real

# 3. Run with real model
./tests/integration_tests_real \
    --wav audio/M_1011_13y10m_1.wav \
    --model models/voxceleb_ECAPA512.onnx

# 4. Update 'Verified' field above to 'yes' and record date + git SHA.
```

### Expected output shape (onnxruntime inspection)

```python
import onnxruntime as rt, numpy as np
sess = rt.InferenceSession("models/voxceleb_ECAPA512.onnx")
# Input:  [1, T, 80]  float32
# Output: [1, 512]    float32
print(sess.get_inputs()[0].name)   # "feats"
print(sess.get_outputs()[0].name)  # "embs"
inp = np.zeros((1, 300, 80), dtype=np.float32)  # 3 s @ 10 ms/frame
emb = sess.run(None, {"feats": inp})[0]
print(emb.shape)  # (1, 512)
```

---

## Adding a new model

Copy the template below and fill in all fields before setting `Verified: yes`.

```
## <model_filename>.onnx

| Field | Value |
| --- | --- |
| **Model name** | |
| **Source** | |
| **Download URL** | |
| **Training data** | |
| **Architecture** | |
| **Embedding dimension** | |
| **Input node** | |
| **Input shape** | |
| **Output node** | |
| **Output shape** | |
| **Frontend** | |
| **Required sample rate** | |
| **Wrapper class** | |
| **Verified** | no |
| **Test audio** | |
| **Notes** | |
```
