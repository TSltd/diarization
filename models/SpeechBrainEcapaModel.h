#pragma once

#include <memory>
#include <string>
#include <vector>

#include <diarization/ISpeakerEmbeddingModel.h>
#include <diarization/ModelMetadata.h>

// Forward-declare ONNX Runtime types to keep the header lightweight.
namespace Ort { class Session; class Env; class SessionOptions; }

/// @brief Speaker embedding model for SpeechBrain ECAPA-TDNN ONNX exports.
///
/// ## Supported graph variants
///
/// SpeechBrain ONNX exports come in two flavours, detected automatically at
/// load() time by inspecting the input tensor rank:
///
/// | Variant    | Input name | Input shape | Preprocessing       |
/// |------------|------------|-------------|---------------------|
/// | Raw PCM    | wavs       | [1, T]      | none (graph-internal) |
/// | FBank      | feats      | [1, T, 80]  | C++ FBANK front-end |
///
/// After load(), call inspect() to see which variant was detected.
///
/// ## ONNX graph contract (typical SpeechBrain ECAPA-TDNN export)
/// | Node   | Name         | Shape    | dtype   |
/// |--------|--------------|----------|---------|
/// | Input  | "wavs"       | [1, T]   | float32 |  ← raw PCM variant
/// | Input  | "feats"      | [1, T,80]| float32 |  ← FBank variant
/// | Output | "embeddings" | [1, D]   | float32 |
///
/// @note load() must succeed before calling embed() / inspect().
class SpeechBrainEcapaModel : public ISpeakerEmbeddingModel {
public:
    SpeechBrainEcapaModel();
    ~SpeechBrainEcapaModel() override;

    // Non-copyable (owns an ONNX session)
    SpeechBrainEcapaModel(const SpeechBrainEcapaModel&)            = delete;
    SpeechBrainEcapaModel& operator=(const SpeechBrainEcapaModel&) = delete;

    // ISpeakerEmbeddingModel ---------------------------------------------------
    bool               load(const std::string& model_path) override;
    std::vector<float> embed(const AudioChunk& chunk)      override;
    int                sample_rate()   const override { return 16000; }
    int                embedding_dim() const override { return embedding_dim_; }
    ModelMetadata      inspect()       const override { return metadata_; }

private:
    enum class InputMode {
        RawPCM,   ///< Input is flat float32 PCM samples, shape [1, T]
        FBank80,  ///< Input is 80-dim log-mel features,   shape [1, T, 80]
    };

    // ---- Preprocessing -------------------------------------------------------

    /// Run inference with a raw-PCM input tensor [1, T].
    std::vector<float> run_pcm(const std::vector<float>& pcm);

    /// Run inference with a pre-computed FBANK tensor [1, n_frames, 80].
    std::vector<float> run_fbank(const std::vector<float>& feats,
                                  int64_t n_frames);

    static void l2_normalize(std::vector<float>& v);

    // ---- ONNX session --------------------------------------------------------

    struct OrtState;
    std::unique_ptr<OrtState> ort_;

    // ---- State ---------------------------------------------------------------

    InputMode   input_mode_    = InputMode::RawPCM;
    std::string input_node_;
    std::string output_node_;
    int         embedding_dim_ = 192;
    ModelMetadata metadata_;

    // FBANK front-end (constructed on first use in FBank80 mode)
    struct FBankImpl;
    std::unique_ptr<FBankImpl> fbank_;
};
