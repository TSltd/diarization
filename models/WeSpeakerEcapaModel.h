#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ISpeakerEmbeddingModel.h"

// Forward-declare ONNX Runtime types to keep the header lightweight.
namespace Ort { class Session; class Env; class SessionOptions; }

/// @brief Concrete speaker embedding model targeting wespeaker-style ECAPA-TDNN exports.
///
/// ## Validated against
///   wespeaker VoxCeleb2 ECAPA-TDNN 512
///   https://wespeaker-1256283475.cos.ap-beijing.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.onnx
///
/// ## ONNX graph contract
/// | Node      | Name      | Shape      | dtype   |
/// |-----------|-----------|------------|---------|
/// | Input     | "feats"   | [1, T, 80] | float32 |
/// | Output    | "embs"    | [1, D]     | float32 |
///
/// Where T = number of 10 ms frames in the utterance and D is the embedding
/// dimension (256 for ECAPA256, 512 for ECAPA512).
///
/// ## Front-end parameters (must match the Python recipe used for training)
/// - Sample rate   : 16 000 Hz
/// - Pre-emphasis  : 0.97
/// - Frame length  : 25 ms  (400 samples)
/// - Frame shift   : 10 ms  (160 samples)
/// - FFT size      : 512
/// - Mel bins      : 80
/// - Freq range    : 20 – 7 600 Hz
/// - Window        : Hann
/// - Normalisation : none (no CMVN)
///
/// ## Input / output tensor name overrides
/// Some ONNX exports use different node names.  Pass the actual names via the
/// constructor if the defaults ("feats" / "embs") do not match your model.
class WeSpeakerEcapaModel : public ISpeakerEmbeddingModel {
public:
    /// @param input_node_name   Name of the ONNX input node  (default "feats")
    /// @param output_node_name  Name of the ONNX output node (default "embs")
    explicit WeSpeakerEcapaModel(
        std::string input_node_name  = "feats",
        std::string output_node_name = "embs");

    ~WeSpeakerEcapaModel() override;

    // ISpeakerEmbeddingModel ---------------------------------------------------
    bool               load(const std::string& model_path) override;
    std::vector<float> embed(const AudioChunk& chunk)      override;
    int                sample_rate()   const override { return 16000; }
    int                embedding_dim() const override { return embedding_dim_; }

private:
    // ---- FBANK front-end ---------------------------------------------------

    /// Compute 80-dim log-mel filterbank features from normalised mono PCM.
    ///
    /// Returns a row-major matrix of shape [n_frames, 80].
    /// The caller owns the returned buffer.
    std::vector<float> compute_fbank(const std::vector<float>& pcm) const;

    /// Radix-2 Cooley–Tukey FFT (in-place, power-of-two size).
    /// re/im are the real and imaginary parts of the signal.
    static void fft(std::vector<float>& re, std::vector<float>& im);

    /// Build the mel filterbank matrix [num_bins × (fft_size/2+1)] once
    /// and cache it.  Thread-safe after construction.
    void build_filterbank();

    // ---- ONNX session ------------------------------------------------------

    std::vector<float> run_inference(const std::vector<float>& feats,
                                     int64_t n_frames);

    static void l2_normalize(std::vector<float>& v);

    // ---- State -------------------------------------------------------------

    struct OrtState;
    std::unique_ptr<OrtState> ort_;

    std::string input_node_;
    std::string output_node_;

    int embedding_dim_ = 256;

    // Front-end constants.
    static constexpr int   kSampleRate  = 16000;
    static constexpr int   kFrameLen    = 400;   // 25 ms
    static constexpr int   kFrameShift  = 160;   // 10 ms
    static constexpr int   kFftSize     = 512;
    static constexpr int   kMelBins     = 80;
    static constexpr float kPreEmph     = 0.97f;
    static constexpr float kFreqLow     = 20.0f;
    static constexpr float kFreqHigh    = 7600.0f;

    // Cached mel filterbank matrix [kMelBins × (kFftSize/2+1)].
    std::vector<float> mel_fb_;
    // Cached Hann window [kFrameLen].
    std::vector<float> hann_win_;
};
