#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ISpeakerEmbeddingModel.h"

// Forward-declare the ONNX Runtime session type to avoid pulling in the full
// onnxruntime_cxx_api.h in headers that only need the interface.
namespace Ort { class Session; class Env; class SessionOptions; }

// ---------------------------------------------------------------------------
// EcapaOnnxModel
// ---------------------------------------------------------------------------

/// ONNX Runtime–backed ECAPA-TDNN speaker embedding model.
///
/// Expected ONNX graph contract:
///   Input  "input"  : [1, T]  float32  (normalised mono PCM at 16 kHz)
///   Output "output" : [1, D]  float32  (raw embedding, will be L2-normalised)
///
/// The model preprocesses audio internally (or via a fixed log-mel frontend
/// baked into the ONNX graph).  If the graph expects log-mel features rather
/// than raw PCM, derive a subclass and override embed() to call a mel
/// front-end before the ONNX session.
class EcapaOnnxModel : public ISpeakerEmbeddingModel {
public:
    EcapaOnnxModel();
    ~EcapaOnnxModel() override;

    bool load(const std::string& model_path) override;
    std::vector<float> embed(const AudioChunk& chunk) override;
    int  sample_rate()   const override { return 16000; }
    int  embedding_dim() const override { return embedding_dim_; }

private:
    /// Run ONNX inference on a float32 input tensor and return the raw output.
    std::vector<float> run_inference(const std::vector<float>& pcm);

    /// L2-normalise a vector in-place.
    static void l2_normalize(std::vector<float>& v);

    // ONNX Runtime objects (pimpl-lite via raw pointers to avoid including
    // the heavy onnxruntime_cxx_api.h in this header).
    struct OrtState;
    std::unique_ptr<OrtState> ort_;

    int embedding_dim_ = 192; ///< Updated after load() inspects the model output shape
};

// ---------------------------------------------------------------------------
// XVectorOnnxModel
// ---------------------------------------------------------------------------

/// ONNX Runtime–backed x-vector speaker embedding model.
///
/// Expected ONNX graph contract:
///   Input  "input"  : [1, F, T]  float32  (log-mel filterbank, F=80 frames)
///   Output "output" : [1, D]     float32  (raw embedding)
///
/// This class adds a simple log-mel frontend before running the ONNX session.
class XVectorOnnxModel : public ISpeakerEmbeddingModel {
public:
    XVectorOnnxModel();
    ~XVectorOnnxModel() override;

    bool load(const std::string& model_path) override;
    std::vector<float> embed(const AudioChunk& chunk) override;
    int  sample_rate()   const override { return 16000; }
    int  embedding_dim() const override { return embedding_dim_; }

private:
    /// Compute 80-dim log-mel filterbank features from raw PCM.
    /// Returns a [F, T] row-major float32 matrix where F=num_mel_bins.
    std::vector<float> compute_log_mel(const std::vector<float>& pcm,
                                       int sample_rate,
                                       int* out_num_frames) const;

    static void l2_normalize(std::vector<float>& v);

    struct OrtState;
    std::unique_ptr<OrtState> ort_;

    int embedding_dim_ = 512;
    int num_mel_bins_  = 80;
};
