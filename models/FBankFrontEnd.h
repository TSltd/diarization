#pragma once

/// @file FBankFrontEnd.h
/// @brief Header-only 80-dim log-mel filterbank front-end.
///
/// Parameters match the WeSpeaker / SpeechBrain-ECAPA recipe:
///   - Sample rate   : 16 000 Hz
///   - Pre-emphasis  : 0.97
///   - Frame length  : 25 ms  (400 samples)
///   - Frame shift   : 10 ms  (160 samples)
///   - FFT size      : 512
///   - Mel bins      : 80
///   - Freq range    : 20 – 7 600 Hz
///   - Window        : Hann
///   - Normalisation : none (no CMVN)
///
/// Output layout: row-major [n_frames × 80] float32.
///
/// Thread-safety: `compute()` is const and safe to call from multiple threads
/// once the object is constructed.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <vector>

class FBankFrontEnd {
public:
    static constexpr int   kSampleRate  = 16000;
    static constexpr int   kFrameLen    = 400;   // 25 ms
    static constexpr int   kFrameShift  = 160;   // 10 ms
    static constexpr int   kFftSize     = 512;
    static constexpr int   kMelBins     = 80;
    static constexpr float kPreEmph     = 0.97f;
    static constexpr float kFreqLow     = 20.0f;
    static constexpr float kFreqHigh    = 7600.0f;

    FBankFrontEnd() { build_filterbank(); }

    /// Compute log-mel filterbank features from normalised mono PCM.
    /// Returns row-major [n_frames × kMelBins].  Returns an empty vector if
    /// @p pcm is shorter than one frame.
    std::vector<float> compute(const std::vector<float>& pcm) const {
        if (static_cast<int>(pcm.size()) < kFrameLen)
            return {};

        const int n_frames   = (static_cast<int>(pcm.size()) - kFrameLen) / kFrameShift + 1;
        const int n_fft_bins = kFftSize / 2 + 1;

        std::vector<float> out(static_cast<size_t>(n_frames) * kMelBins);
        std::vector<float> re(kFftSize), im(kFftSize);

        for (int f = 0; f < n_frames; ++f) {
            const int start = f * kFrameShift;

            // Pre-emphasis + Hann window + zero-pad to kFftSize
            std::fill(re.begin(), re.end(), 0.0f);
            std::fill(im.begin(), im.end(), 0.0f);

            re[0] = pcm[start] * hann_win_[0];
            for (int s = 1; s < kFrameLen; ++s) {
                float emph = pcm[start + s] - kPreEmph * pcm[start + s - 1];
                re[s] = emph * hann_win_[s];
            }

            fft(re, im);

            // Power spectrum |X[k]|² in the positive half
            for (int k = 0; k < n_fft_bins; ++k)
                re[k] = re[k] * re[k] + im[k] * im[k];

            // Mel filterbank → log energy
            for (int m = 0; m < kMelBins; ++m) {
                float energy = 0.0f;
                const float* row = mel_fb_.data() + static_cast<size_t>(m) * n_fft_bins;
                for (int k = 0; k < n_fft_bins; ++k)
                    energy += row[k] * re[k];
                out[static_cast<size_t>(f) * kMelBins + m] = std::log(std::max(energy, 1e-10f));
            }
        }
        return out;
    }

    /// Number of frames that will be produced for a given PCM length.
    static int num_frames(int n_samples) {
        if (n_samples < kFrameLen) return 0;
        return (n_samples - kFrameLen) / kFrameShift + 1;
    }

private:
    // ── Radix-2 Cooley–Tukey FFT (in-place, power-of-two size) ──────────────
    static void fft(std::vector<float>& re, std::vector<float>& im) {
        const int N = static_cast<int>(re.size());
        assert((N & (N - 1)) == 0 && "FFT size must be a power of two");

        // Bit-reversal permutation
        {
            int j = 0;
            for (int i = 1; i < N; ++i) {
                int bit = N >> 1;
                for (; j & bit; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
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
                    const int u = i + k, v = i + k + len / 2;
                    const float v_re = re[v] * t_re - im[v] * t_im;
                    const float v_im = re[v] * t_im + im[v] * t_re;
                    re[v] = re[u] - v_re;  im[v] = im[u] - v_im;
                    re[u] = re[u] + v_re;  im[u] = im[u] + v_im;
                    const float nt_re = t_re * w_re - t_im * w_im;
                    t_im = t_re * w_im + t_im * w_re;
                    t_re = nt_re;
                }
            }
        }
    }

    // ── Mel filterbank matrix construction ───────────────────────────────────
    void build_filterbank() {
        const int n_fft_bins = kFftSize / 2 + 1;

        // Hann window
        hann_win_.resize(kFrameLen);
        for (int i = 0; i < kFrameLen; ++i)
            hann_win_[i] = 0.5f * (1.0f - std::cos(
                2.0f * std::numbers::pi_v<float> * i / (kFrameLen - 1)));

        // Hz ↔ mel (O'Shaughnessy)
        auto hz_to_mel = [](float f) { return 2595.0f * std::log10(1.0f + f / 700.0f); };
        auto mel_to_hz = [](float m) { return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f); };

        const float mel_low  = hz_to_mel(kFreqLow);
        const float mel_high = hz_to_mel(kFreqHigh);

        const int n_points = kMelBins + 2;
        std::vector<float> mel_pts(n_points);
        for (int i = 0; i < n_points; ++i)
            mel_pts[i] = mel_low + (mel_high - mel_low) * i / (n_points - 1);

        std::vector<int> bins(n_points);
        for (int i = 0; i < n_points; ++i) {
            float hz = mel_to_hz(mel_pts[i]);
            bins[i] = static_cast<int>(std::floor((kFftSize + 1) * hz / kSampleRate));
            bins[i] = std::clamp(bins[i], 0, n_fft_bins - 1);
        }

        mel_fb_.assign(static_cast<size_t>(kMelBins) * n_fft_bins, 0.0f);
        for (int m = 0; m < kMelBins; ++m) {
            const int left = bins[m], centre = bins[m + 1], right = bins[m + 2];
            for (int k = left;   k < centre; ++k)
                mel_fb_[static_cast<size_t>(m) * n_fft_bins + k] =
                    (k - left + 1.0f) / (centre - left + 1.0f);
            for (int k = centre; k <= right;  ++k)
                mel_fb_[static_cast<size_t>(m) * n_fft_bins + k] =
                    (right - k + 1.0f) / (right - centre + 1.0f);
        }
    }

    std::vector<float> mel_fb_;    // [kMelBins × (kFftSize/2+1)]
    std::vector<float> hann_win_;  // [kFrameLen]
};
