#include "WeSpeakerEcapaModel.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

// ---------------------------------------------------------------------------
// Internal ONNX Runtime state (pimpl)
// ---------------------------------------------------------------------------

struct WeSpeakerEcapaModel::OrtState {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "WeSpeakerEcapaModel"};
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> session;
};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

WeSpeakerEcapaModel::WeSpeakerEcapaModel(
    std::string input_node_name,
    std::string output_node_name)
    : input_node_ (std::move(input_node_name))
    , output_node_(std::move(output_node_name)) {

    build_filterbank();
}

WeSpeakerEcapaModel::~WeSpeakerEcapaModel() = default;

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

bool WeSpeakerEcapaModel::load(const std::string& model_path) {
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

        // Auto-detect embedding dimension by running a tiny silent forward pass.
        // NOTE: piper's ORT build crashes when calling GetShape() on STATIC type
        // info (dynamic -1 dims cause GetDimensionsCount to return garbage).
        // GetShape() on a RUNTIME output tensor works correctly.
        // Use 100 frames of silence [1, 100, 80] as calibration input.
        {
            const int T = 100;
            std::vector<float> silence(T * kMelBins, 0.0f);
            std::array<int64_t, 3> cal_shape{1, T, kMelBins};
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto cal_in = Ort::Value::CreateTensor<float>(
                mem, silence.data(), silence.size(),
                cal_shape.data(), cal_shape.size());
            const char* in_n[]  = { input_node_.c_str() };
            const char* out_n[] = { output_node_.c_str() };
            auto cal_out = ort_->session->Run(
                Ort::RunOptions{nullptr}, in_n, &cal_in, 1, out_n, 1);
            auto run_shape = cal_out[0].GetTensorTypeAndShapeInfo().GetShape();
            if (run_shape.size() >= 2 && run_shape[1] > 0)
                embedding_dim_ = static_cast<int>(run_shape[1]);
            else if (run_shape.size() == 1 && run_shape[0] > 0)
                embedding_dim_ = static_cast<int>(run_shape[0]);
        }

        // ---- Populate ModelMetadata -----------------------------------------
        metadata_.input_name  = input_node_;
        metadata_.output_name = output_node_;
        // Try to read input static shape; on piper's ORT build GetShape() can
        // return garbage for dynamic dims — wrap in try/catch and fall back to
        // the known WeSpeaker contract [1, T, 80].
        try {
            auto in_info = ort_->session->GetInputTypeInfo(0)
                                        .GetTensorTypeAndShapeInfo();
            metadata_.input_shape = in_info.GetShape();
        } catch (...) {
            metadata_.input_shape = {1, -1, kMelBins};
        }
        metadata_.output_shape = {1, static_cast<int64_t>(embedding_dim_)};
        metadata_.loaded = true;

        return true;
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "WeSpeakerEcapaModel::load OrtException: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "WeSpeakerEcapaModel::load std::exception: %s\n", e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// embed()
// ---------------------------------------------------------------------------

std::vector<float> WeSpeakerEcapaModel::embed(const AudioChunk& chunk) {
    if (chunk.empty())
        return {};

    // Resample check (we only support 16 kHz input).
    if (chunk.sample_rate != kSampleRate)
        throw std::runtime_error(
            "WeSpeakerEcapaModel: expected 16 kHz audio, got "
            + std::to_string(chunk.sample_rate) + " Hz");

    // 1. Compute FBANK features → [n_frames, 80]
    auto feats = compute_fbank(chunk.samples);
    if (feats.empty()) return {};

    const int64_t n_frames = static_cast<int64_t>(feats.size()) / kMelBins;

    // 2. Run ONNX inference
    auto embedding = run_inference(feats, n_frames);

    // 3. L2-normalise
    l2_normalize(embedding);
    return embedding;
}

// ---------------------------------------------------------------------------
// FBANK front-end
// ---------------------------------------------------------------------------

void WeSpeakerEcapaModel::build_filterbank() {
    const int n_fft_bins = kFftSize / 2 + 1; // 257

    // Hann window
    hann_win_.resize(kFrameLen);
    for (int i = 0; i < kFrameLen; ++i)
        hann_win_[i] = 0.5f * (1.0f - std::cos(
            2.0f * std::numbers::pi_v<float> * i / (kFrameLen - 1)));

    // --- Mel filterbank -------------------------------------------------
    // Convert Hz ↔ mel using the standard O'Shaughnessy formula.
    auto hz_to_mel = [](float f) { return 2595.0f * std::log10(1.0f + f / 700.0f); };
    auto mel_to_hz = [](float m) { return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f); };

    const float mel_low  = hz_to_mel(kFreqLow);
    const float mel_high = hz_to_mel(kFreqHigh);

    // (kMelBins + 2) equally spaced mel points.
    const int n_points = kMelBins + 2;
    std::vector<float> mel_pts(n_points);
    for (int i = 0; i < n_points; ++i)
        mel_pts[i] = mel_low + (mel_high - mel_low) * i / (n_points - 1);

    // Convert mel points → FFT bin indices.
    std::vector<int> bins(n_points);
    for (int i = 0; i < n_points; ++i) {
        float hz = mel_to_hz(mel_pts[i]);
        bins[i] = static_cast<int>(
            std::floor((kFftSize + 1) * hz / kSampleRate));
        bins[i] = std::clamp(bins[i], 0, n_fft_bins - 1);
    }

    // Build filterbank matrix [kMelBins × n_fft_bins], row-major.
    mel_fb_.assign(static_cast<size_t>(kMelBins) * n_fft_bins, 0.0f);
    for (int m = 0; m < kMelBins; ++m) {
        const int left   = bins[m];
        const int centre = bins[m + 1];
        const int right  = bins[m + 2];

        // Rising slope
        for (int k = left; k < centre; ++k) {
            float val = (k - left + 1.0f) / (centre - left + 1.0f);
            mel_fb_[static_cast<size_t>(m) * n_fft_bins + k] = val;
        }
        // Falling slope
        for (int k = centre; k <= right; ++k) {
            float val = (right - k + 1.0f) / (right - centre + 1.0f);
            mel_fb_[static_cast<size_t>(m) * n_fft_bins + k] = val;
        }
    }
}

std::vector<float> WeSpeakerEcapaModel::compute_fbank(
    const std::vector<float>& pcm) const {

    if (static_cast<int>(pcm.size()) < kFrameLen)
        return {};

    const int n_frames = (static_cast<int>(pcm.size()) - kFrameLen) / kFrameShift + 1;
    const int n_fft_bins = kFftSize / 2 + 1;

    // Output: row-major [n_frames, kMelBins]
    std::vector<float> out(static_cast<size_t>(n_frames) * kMelBins);

    // Reusable FFT buffers
    std::vector<float> re(kFftSize), im(kFftSize);

    for (int f = 0; f < n_frames; ++f) {
        const int start = f * kFrameShift;

        // --- Pre-emphasis + Hann window + zero-pad to kFftSize ---
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);

        re[0] = pcm[start] * hann_win_[0]; // first sample: no pre-emphasis carry
        for (int s = 1; s < kFrameLen; ++s) {
            float emph = pcm[start + s] - kPreEmph * pcm[start + s - 1];
            re[s] = emph * hann_win_[s];
        }

        // --- FFT ---
        fft(re, im);

        // --- Power spectrum: |X[k]|² ---
        // We only need the positive half [0, n_fft_bins).
        // Store temporarily in re[0..n_fft_bins-1].
        for (int k = 0; k < n_fft_bins; ++k)
            re[k] = re[k] * re[k] + im[k] * im[k];

        // --- Apply mel filterbank → log energy ---
        for (int m = 0; m < kMelBins; ++m) {
            float energy = 0.0f;
            const float* fb_row = mel_fb_.data() + static_cast<size_t>(m) * n_fft_bins;
            for (int k = 0; k < n_fft_bins; ++k)
                energy += fb_row[k] * re[k];
            out[static_cast<size_t>(f) * kMelBins + m] = std::log(std::max(energy, 1e-10f));
        }
    }

    // Utterance-level CMN (mean subtraction only)
    for (int m = 0; m < kMelBins; ++m) {
        float mean = 0.0f;

        for (int f = 0; f < n_frames; ++f)
            mean += out[f * kMelBins + m];

        mean /= static_cast<float>(n_frames);

        for (int f = 0; f < n_frames; ++f)
            out[f * kMelBins + m] -= mean;
    }

    return out;
}

// ---------------------------------------------------------------------------
// Radix-2 Cooley–Tukey FFT (in-place)
// ---------------------------------------------------------------------------

void WeSpeakerEcapaModel::fft(std::vector<float>& re, std::vector<float>& im) {
    const int N = static_cast<int>(re.size());
    assert((N & (N - 1)) == 0 && "FFT size must be a power of two");

    // Bit-reversal permutation
    {
        int j = 0;
        for (int i = 1; i < N; ++i) {
            int bit = N >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
    }

    // Butterfly stages
    for (int len = 2; len <= N; len <<= 1) {
        const float ang  = -2.0f * std::numbers::pi_v<float> / static_cast<float>(len);
        const float w_re = std::cos(ang);
        const float w_im = std::sin(ang);

        for (int i = 0; i < N; i += len) {
            float t_re = 1.0f, t_im = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                const int u = i + k;
                const int v = i + k + len / 2;

                const float v_re = re[v] * t_re - im[v] * t_im;
                const float v_im = re[v] * t_im + im[v] * t_re;

                re[v] = re[u] - v_re;
                im[v] = im[u] - v_im;
                re[u] = re[u] + v_re;
                im[u] = im[u] + v_im;

                // Advance twiddle factor
                const float nt_re = t_re * w_re - t_im * w_im;
                t_im = t_re * w_im + t_im * w_re;
                t_re = nt_re;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX inference
// ---------------------------------------------------------------------------

std::vector<float> WeSpeakerEcapaModel::run_inference(
    const std::vector<float>& feats,
    int64_t n_frames) {

    if (!ort_ || !ort_->session)
        throw std::runtime_error("WeSpeakerEcapaModel: model not loaded");

    auto mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // Input tensor: [1, T, 80]  (batch=1, frames, mel_bins)
    std::array<int64_t, 3> in_shape{1, n_frames, kMelBins};
    auto in_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(feats.data()),
        feats.size(),
        in_shape.data(),
        in_shape.size());

    const char* in_names[]  = { input_node_.c_str()  };
    const char* out_names[] = { output_node_.c_str() };

    auto outputs = ort_->session->Run(
        Ort::RunOptions{nullptr},
        in_names,  &in_tensor, 1,
        out_names, 1);

    const float* raw = outputs[0].GetTensorData<float>();

    // GetElementCount() returns UINT64_MAX for dynamic-dimension outputs (_LM
    // variants of wespeaker models may leave the output shape symbolic).
    // Use the concrete shape after inference instead.
    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t dims = static_cast<size_t>(embedding_dim_); // safe fallback
    if (out_shape.size() >= 2 && out_shape[1] > 0) {
        dims = static_cast<size_t>(out_shape[1]);
        embedding_dim_ = static_cast<int>(dims); // update if not yet known
    } else if (out_shape.size() == 1 && out_shape[0] > 0) {
        dims = static_cast<size_t>(out_shape[0]);
        embedding_dim_ = static_cast<int>(dims);
    }

    return std::vector<float>(raw, raw + dims);
}

void WeSpeakerEcapaModel::l2_normalize(std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    const float norm = std::sqrt(norm_sq);
    if (norm > 1e-8f)
        for (float& x : v) x /= norm;
}
