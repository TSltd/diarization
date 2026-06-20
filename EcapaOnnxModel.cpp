#include "EcapaOnnxModel.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

// Pull in the full ONNX Runtime C++ API only in the translation unit that
// actually uses it, keeping the header lightweight.
#include <onnxruntime_cxx_api.h>

// ---------------------------------------------------------------------------
// Internal ONNX Runtime state (pimpl)
// ---------------------------------------------------------------------------

struct EcapaOnnxModel::OrtState {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "EcapaOnnxModel"};
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> session;
};

struct XVectorOnnxModel::OrtState {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "XVectorOnnxModel"};
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> session;
};

// ---------------------------------------------------------------------------
// EcapaOnnxModel
// ---------------------------------------------------------------------------

EcapaOnnxModel::EcapaOnnxModel()  = default;
EcapaOnnxModel::~EcapaOnnxModel() = default;

bool EcapaOnnxModel::load(const std::string& model_path) {
    try {
        ort_ = std::make_unique<OrtState>();
        ort_->session_opts.SetIntraOpNumThreads(1);
        ort_->session_opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        // ONNX Runtime on Windows expects a wide-string path.
        std::wstring wpath(model_path.begin(), model_path.end());
        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, wpath.c_str(), ort_->session_opts);
#else
        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, model_path.c_str(), ort_->session_opts);
#endif

        // Discover the embedding dimension from the model's output shape.
        Ort::AllocatorWithDefaultOptions alloc;
        auto out_info = ort_->session->GetOutputTypeInfo(0)
                                     .GetTensorTypeAndShapeInfo();
        auto shape = out_info.GetShape();
        if (shape.size() >= 2 && shape[1] > 0)
            embedding_dim_ = static_cast<int>(shape[1]);

        return true;
    } catch (const Ort::Exception& e) {
        return false;
    }
}

std::vector<float> EcapaOnnxModel::run_inference(const std::vector<float>& pcm) {
    if (!ort_ || !ort_->session)
        throw std::runtime_error("EcapaOnnxModel: model not loaded");

    Ort::AllocatorWithDefaultOptions alloc;
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // Input tensor: [1, T]
    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(pcm.size())};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(pcm.data()),
        pcm.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[]  = {"input"};
    const char* output_names[] = {"output"};

    auto outputs = ort_->session->Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1);

    float*  raw  = outputs[0].GetTensorMutableData<float>();
    size_t  dims = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<float>(raw, raw + dims);
}

std::vector<float> EcapaOnnxModel::embed(const AudioChunk& chunk) {
    auto embedding = run_inference(chunk.samples);
    l2_normalize(embedding);
    return embedding;
}

void EcapaOnnxModel::l2_normalize(std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    const float norm = std::sqrt(norm_sq);
    if (norm > 1e-8f)
        for (float& x : v) x /= norm;
}

// ---------------------------------------------------------------------------
// XVectorOnnxModel
// ---------------------------------------------------------------------------

XVectorOnnxModel::XVectorOnnxModel()  = default;
XVectorOnnxModel::~XVectorOnnxModel() = default;

bool XVectorOnnxModel::load(const std::string& model_path) {
    try {
        ort_ = std::make_unique<OrtState>();
        ort_->session_opts.SetIntraOpNumThreads(1);
        ort_->session_opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring wpath(model_path.begin(), model_path.end());
        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, wpath.c_str(), ort_->session_opts);
#else
        ort_->session = std::make_unique<Ort::Session>(
            ort_->env, model_path.c_str(), ort_->session_opts);
#endif

        Ort::AllocatorWithDefaultOptions alloc;
        auto out_info = ort_->session->GetOutputTypeInfo(0)
                                     .GetTensorTypeAndShapeInfo();
        auto shape = out_info.GetShape();
        if (shape.size() >= 2 && shape[1] > 0)
            embedding_dim_ = static_cast<int>(shape[1]);

        return true;
    } catch (const Ort::Exception&) {
        return false;
    }
}

std::vector<float> XVectorOnnxModel::compute_log_mel(
    const std::vector<float>& pcm,
    int /*sample_rate*/,
    int* out_num_frames) const {

    // Parameters matching common x-vector/SpeakerNet front-ends.
    constexpr int    frame_len  = 400;   // 25 ms at 16 kHz
    constexpr int    frame_hop  = 160;   // 10 ms at 16 kHz
    constexpr int    fft_size   = 512;
    constexpr float  pre_emph   = 0.97f;

    const int n_frames = std::max(0,
        (static_cast<int>(pcm.size()) - frame_len) / frame_hop + 1);

    if (out_num_frames) *out_num_frames = n_frames;

    // ---------------------------------------------------------------------------
    // TODO: NOT YET VALIDATED against a production x-vector / SpeakerNet export.
    //
    // Different ONNX exports vary in:
    //   - feature layout  ([B, F, T]  vs  [B, T, F])
    //   - normalisation   (global CMVN vs per-utterance vs none)
    //   - window function (Hann vs Povey)
    //   - pre-emphasis coefficient
    //   - number of mel bins (typically 30, 40, or 80)
    //
    // Before integrating a real model, replace this stub with a front-end that
    // exactly matches the Python feature extractor used during training, and
    // add a round-trip test using a known (audio → embedding) pair.
    // ---------------------------------------------------------------------------
    std::vector<float> features(num_mel_bins_ * n_frames, 0.0f);

    // Pre-emphasis + framing + log-energy per sub-band (stub).
    // Each frame is written as a column of `num_mel_bins_` values.
    for (int f = 0; f < n_frames; ++f) {
        int start = f * frame_hop;
        // Use per-frame log-energy spread across all mel bins as a placeholder.
        float energy = 0.0f;
        for (int s = start; s < start + frame_len && s < static_cast<int>(pcm.size()); ++s)
            energy += pcm[s] * pcm[s];
        float log_e = std::log(energy + 1e-8f);
        for (int b = 0; b < num_mel_bins_; ++b)
            features[b * n_frames + f] = log_e;
    }

    return features;
}

std::vector<float> XVectorOnnxModel::embed(const AudioChunk& chunk) {
    if (!ort_ || !ort_->session)
        throw std::runtime_error("XVectorOnnxModel: model not loaded");

    int n_frames = 0;
    auto mel = compute_log_mel(chunk.samples, chunk.sample_rate, &n_frames);

    Ort::AllocatorWithDefaultOptions alloc;
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // Input tensor: [1, F, T]
    std::array<int64_t, 3> shape{1, num_mel_bins_, n_frames};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, mel.data(), mel.size(), shape.data(), shape.size());

    const char* input_names[]  = {"input"};
    const char* output_names[] = {"output"};

    auto outputs = ort_->session->Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1);

    float*  raw  = outputs[0].GetTensorMutableData<float>();
    size_t  dims = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> embedding(raw, raw + dims);
    l2_normalize(embedding);
    return embedding;
}

void XVectorOnnxModel::l2_normalize(std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    const float norm = std::sqrt(norm_sq);
    if (norm > 1e-8f)
        for (float& x : v) x /= norm;
}
