#include "SpeechBrainEcapaModel.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

#include "FBankFrontEnd.h"

// ---------------------------------------------------------------------------
// Internal types (pimpl)
// ---------------------------------------------------------------------------

struct SpeechBrainEcapaModel::OrtState {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "SpeechBrainEcapaModel"};
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> session;
};

/// Thin wrapper so FBankFrontEnd.h is not exposed in the public header.
struct SpeechBrainEcapaModel::FBankImpl {
    FBankFrontEnd fe;
};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

SpeechBrainEcapaModel::SpeechBrainEcapaModel()  = default;
SpeechBrainEcapaModel::~SpeechBrainEcapaModel() = default;

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

bool SpeechBrainEcapaModel::load(const std::string& model_path) {
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

        // ---- Discover input node name ----------------------------------------
        {
            auto raw = ort_->session->GetInputNameAllocated(0, alloc);
            input_node_ = raw.get();
        }

        // ---- Detect input mode from tensor rank ------------------------------
        // rank 2 → [batch, time]       → raw PCM (feature extraction is in-graph)
        // rank 3 → [batch, time, bins] → external FBANK front-end required
        {
            size_t input_rank = 2; // default: raw PCM
            try {
                auto in_info = ort_->session->GetInputTypeInfo(0)
                                            .GetTensorTypeAndShapeInfo();
                input_rank = in_info.GetDimensionsCount();
            } catch (...) {
                // Fallback: guess from node name convention
                input_rank = (input_node_ == "feats") ? 3 : 2;
            }

            if (input_rank == 3) {
                input_mode_ = InputMode::FBank80;
                fbank_ = std::make_unique<FBankImpl>();
            } else {
                input_mode_ = InputMode::RawPCM;
            }

            std::printf("[SpeechBrainEcapaModel] input_mode = %s  (node: \"%s\")\n",
                        (input_mode_ == InputMode::FBank80) ? "FBank80" : "RawPCM",
                        input_node_.c_str());
        }

        // ---- Discover output node name ---------------------------------------
        {
            auto raw = ort_->session->GetOutputNameAllocated(0, alloc);
            output_node_ = raw.get();
        }

        // ---- Calibration pass → discover embedding dimension ----------------
        // Run a silent input through the graph; GetShape() on the runtime output
        // tensor is reliable, unlike GetShape() on static type info (dynamic dims
        // can return garbage on some ORT builds).
        {
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const char* in_n[]  = { input_node_.c_str()  };
            const char* out_n[] = { output_node_.c_str() };

            if (input_mode_ == InputMode::FBank80) {
                // 100 frames of silence → [1, 100, 80]
                const int T = 100;
                std::vector<float> silence(T * 80, 0.0f);
                std::array<int64_t, 3> cal_shape{1, T, 80};
                auto cal_in = Ort::Value::CreateTensor<float>(
                    mem, silence.data(), silence.size(),
                    cal_shape.data(), cal_shape.size());
                auto cal_out = ort_->session->Run(
                    Ort::RunOptions{nullptr}, in_n, &cal_in, 1, out_n, 1);
                auto shape = cal_out[0].GetTensorTypeAndShapeInfo().GetShape();
                if      (shape.size() >= 2 && shape[1] > 0) embedding_dim_ = static_cast<int>(shape[1]);
                else if (shape.size() == 1 && shape[0] > 0) embedding_dim_ = static_cast<int>(shape[0]);
            } else {
                // 1 second of silence → [1, 16000]
                const int T = 16000;
                std::vector<float> silence(T, 0.0f);
                std::array<int64_t, 2> cal_shape{1, T};
                auto cal_in = Ort::Value::CreateTensor<float>(
                    mem, silence.data(), silence.size(),
                    cal_shape.data(), cal_shape.size());
                auto cal_out = ort_->session->Run(
                    Ort::RunOptions{nullptr}, in_n, &cal_in, 1, out_n, 1);
                auto shape = cal_out[0].GetTensorTypeAndShapeInfo().GetShape();
                if      (shape.size() >= 2 && shape[1] > 0) embedding_dim_ = static_cast<int>(shape[1]);
                else if (shape.size() == 1 && shape[0] > 0) embedding_dim_ = static_cast<int>(shape[0]);
            }
        }

        // ---- Populate ModelMetadata ------------------------------------------
        metadata_.input_name  = input_node_;
        metadata_.output_name = output_node_;
        try {
            auto in_info = ort_->session->GetInputTypeInfo(0)
                                        .GetTensorTypeAndShapeInfo();
            metadata_.input_shape = in_info.GetShape();
        } catch (...) {
            metadata_.input_shape = (input_mode_ == InputMode::FBank80)
                                    ? std::vector<int64_t>{1, -1, 80}
                                    : std::vector<int64_t>{1, -1};
        }
        metadata_.output_shape = {1, static_cast<int64_t>(embedding_dim_)};
        metadata_.loaded = true;

        std::printf("[SpeechBrainEcapaModel] %s\n", metadata_.describe().c_str());
        return true;

    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "SpeechBrainEcapaModel::load OrtException: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SpeechBrainEcapaModel::load std::exception: %s\n", e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// embed()
// ---------------------------------------------------------------------------

std::vector<float> SpeechBrainEcapaModel::embed(const AudioChunk& chunk) {
    if (chunk.empty()) return {};

    if (chunk.sample_rate != 16000)
        throw std::runtime_error(
            "SpeechBrainEcapaModel: expected 16 kHz audio, got "
            + std::to_string(chunk.sample_rate) + " Hz");

    std::vector<float> embedding;

    if (input_mode_ == InputMode::FBank80) {
        if (!fbank_)
            throw std::runtime_error(
                "SpeechBrainEcapaModel: FBank80 mode but front-end not initialised");
        auto feats = fbank_->fe.compute(chunk.samples);
        if (feats.empty()) return {};
        const int64_t n_frames = static_cast<int64_t>(feats.size()) / 80;
        embedding = run_fbank(feats, n_frames);
    } else {
        embedding = run_pcm(chunk.samples);
    }

    l2_normalize(embedding);
    return embedding;
}

// ---------------------------------------------------------------------------
// run_pcm() — raw PCM input [1, T]
// ---------------------------------------------------------------------------

std::vector<float> SpeechBrainEcapaModel::run_pcm(const std::vector<float>& pcm) {
    if (!ort_ || !ort_->session)
        throw std::runtime_error("SpeechBrainEcapaModel: model not loaded");

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> in_shape{1, static_cast<int64_t>(pcm.size())};
    auto in_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(pcm.data()), pcm.size(),
        in_shape.data(), in_shape.size());

    const char* in_n[]  = { input_node_.c_str()  };
    const char* out_n[] = { output_node_.c_str() };
    auto outputs = ort_->session->Run(
        Ort::RunOptions{nullptr}, in_n, &in_tensor, 1, out_n, 1);

    const float* raw = outputs[0].GetTensorData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t dims = static_cast<size_t>(embedding_dim_);
    if      (shape.size() >= 2 && shape[1] > 0) dims = static_cast<size_t>(shape[1]);
    else if (shape.size() == 1 && shape[0] > 0) dims = static_cast<size_t>(shape[0]);
    return std::vector<float>(raw, raw + dims);
}

// ---------------------------------------------------------------------------
// run_fbank() — FBANK input [1, T, 80]
// ---------------------------------------------------------------------------

std::vector<float> SpeechBrainEcapaModel::run_fbank(const std::vector<float>& feats,
                                                      int64_t n_frames) {
    if (!ort_ || !ort_->session)
        throw std::runtime_error("SpeechBrainEcapaModel: model not loaded");

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 3> in_shape{1, n_frames, 80};
    auto in_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(feats.data()), feats.size(),
        in_shape.data(), in_shape.size());

    const char* in_n[]  = { input_node_.c_str()  };
    const char* out_n[] = { output_node_.c_str() };
    auto outputs = ort_->session->Run(
        Ort::RunOptions{nullptr}, in_n, &in_tensor, 1, out_n, 1);

    const float* raw = outputs[0].GetTensorData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t dims = static_cast<size_t>(embedding_dim_);
    if      (shape.size() >= 2 && shape[1] > 0) dims = static_cast<size_t>(shape[1]);
    else if (shape.size() == 1 && shape[0] > 0) dims = static_cast<size_t>(shape[0]);
    return std::vector<float>(raw, raw + dims);
}

// ---------------------------------------------------------------------------
// l2_normalize()
// ---------------------------------------------------------------------------

void SpeechBrainEcapaModel::l2_normalize(std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    const float norm = std::sqrt(norm_sq);
    if (norm > 1e-8f)
        for (float& x : v) x /= norm;
}
