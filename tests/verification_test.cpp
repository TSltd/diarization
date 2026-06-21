// tests/verification_test.cpp
//
// Speaker verification accuracy test.
//
// Loads every speaker??_?.wav from testdata/verification/ (or a path given
// on the command line), embeds each clip with the supplied ONNX model, and
// computes pairwise cosine similarities.  Prints:
//
//   • per-pair same-speaker scores
//   • per-pair different-speaker scores
//   • distribution statistics (min / max / mean / σ)
//   • EER threshold and FAR/FRR at the current threshold (0.72)
//
// Build (from repo root):
//   g++ -std=c++20 -O2 -Iinclude -I. -DDIARIZE_HAVE_MODEL \
//       -I<ORT>/include \
//       tests/verification_test.cpp \
//       src/DiarizationEngine.cpp src/SpeakerClusterManager.cpp \
//       src/LabelSmoother.cpp src/TranscriptFormatter.cpp \
//       models/WeSpeakerEcapaModel.cpp models/EcapaOnnxModel.cpp \
//       models/SpeechBrainEcapaModel.cpp models/SpeakerModelFactory.cpp \
//       models/SpeakerVerifier.cpp \
//       -L<ORT>/lib -Wl,-rpath,<ORT>/lib -lonnxruntime \
//       -o tests/verification_test
//
// Run:
//   ./tests/verification_test wespeaker/voxceleb_ECAPA512_LM.onnx
//   ./tests/verification_test wespeaker/voxceleb_ECAPA512_LM.onnx testdata/verification

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <diarization/AudioChunk.h>
#include <diarization/SpeakerVerifier.h>
#include "tests/wav_reader.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string bar(float v, int width = 40) {
    int filled = static_cast<int>(std::round(v * width));
    filled = std::max(0, std::min(filled, width));
    return std::string(filled, '#') + std::string(width - filled, '-');
}

struct Stats {
    float   vmin   = 0.0f;
    float   vmax   = 0.0f;
    float   mean   = 0.0f;
    float   stddev = 0.0f;
    int     count  = 0;

    static Stats of(const std::vector<float>& v) {
        Stats s;
        s.count = static_cast<int>(v.size());
        if (v.empty()) return s;
        s.vmin  = *std::min_element(v.begin(), v.end());
        s.vmax  = *std::max_element(v.begin(), v.end());
        s.mean  = std::accumulate(v.begin(), v.end(), 0.0f) / s.count;
        float var = 0.0f;
        for (float x : v) var += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(var / s.count);
        return s;
    }
};

// Equal-error rate: sweep threshold and find where FAR ≈ FRR.
// Returns {eer_threshold, eer_value}.
static std::pair<float, float> compute_eer(
    const std::vector<float>& same,       // same-speaker cosines
    const std::vector<float>& diff)       // different-speaker cosines
{
    float best_t   = 0.5f;
    float best_eer = 1.0f;

    for (int step = 0; step <= 1000; ++step) {
        float t   = step * 0.001f;
        // FRR: fraction of same-speaker pairs that fall *below* threshold
        int frr_n = static_cast<int>(
            std::count_if(same.begin(), same.end(), [t](float v){ return v < t; }));
        float frr = static_cast<float>(frr_n) / static_cast<float>(same.size());

        // FAR: fraction of diff-speaker pairs that are *above* threshold
        int far_n = static_cast<int>(
            std::count_if(diff.begin(), diff.end(), [t](float v){ return v >= t; }));
        float far = static_cast<float>(far_n) / static_cast<float>(diff.size());

        float eer = std::abs(far - frr);
        if (eer < best_eer) {
            best_eer = eer;
            best_t   = t;
        }
    }
    // approximate EER value at that threshold
    int frr_n = static_cast<int>(
        std::count_if(same.begin(), same.end(), [best_t](float v){ return v < best_t; }));
    float frr = static_cast<float>(frr_n) / static_cast<float>(same.size());
    int far_n = static_cast<int>(
        std::count_if(diff.begin(), diff.end(), [best_t](float v){ return v >= best_t; }));
    float far = static_cast<float>(far_n) / static_cast<float>(diff.size());
    float eer = (frr + far) * 0.5f;
    return { best_t, eer };
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // ── arguments ─────────────────────────────────────────────────────────────
    if (argc < 2) {
        std::cerr << "Usage: verification_test <model.onnx> [testdata/verification]\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string data_dir   = (argc >= 3) ? argv[2] : "testdata/verification";
    const float CURRENT_THRESHOLD = 0.72f;

    std::cout
        << "═══════════════════════════════════════════════════════\n"
        << "  Speaker Verification Test\n"
        << "  model: " << model_path << "\n"
        << "  data:  " << data_dir  << "\n"
        << "═══════════════════════════════════════════════════════\n\n";

    // ── check data dir ────────────────────────────────────────────────────────
    if (!fs::is_directory(data_dir)) {
        std::cerr << "  ERROR: " << data_dir << " is not a directory.\n"
                  << "  Run:  python tests/gen_verification_wavs.py\n";
        return 1;
    }

    // ── scan WAV files ────────────────────────────────────────────────────────
    // Filename convention: speakerNN_X.wav
    //   NN = zero-padded speaker index (01..99)
    //   X  = clip label (a..z)
    const std::regex filename_re(R"(speaker(\d+)_([a-z])\.wav)");

    // speaker_label → { clip_label → path }
    std::map<std::string, std::map<std::string, fs::path>> speaker_clips;

    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() != ".wav") continue;
        const std::string fname = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(fname, m, filename_re)) continue;
        const std::string spk  = "speaker" + m[1].str();
        const std::string clip = m[2].str();
        speaker_clips[spk][clip] = entry.path();
    }

    if (speaker_clips.empty()) {
        std::cerr << "  ERROR: no speaker??_?.wav files found in " << data_dir << "\n"
                  << "  Run:  python tests/gen_verification_wavs.py\n";
        return 1;
    }

    int total_clips = 0;
    for (const auto& [spk, clips] : speaker_clips)
        total_clips += static_cast<int>(clips.size());

    std::cout
        << "┌─ Dataset\n"
        << "│  speakers: " << speaker_clips.size() << "\n"
        << "│  clips:    " << total_clips << "\n"
        << "└────────────────────────────────────────────────────────\n\n";

    // ── load model ────────────────────────────────────────────────────────────
    std::cout << "┌─ Loading model\n│  " << model_path << "\n";
    SpeakerVerifier verifier;
    if (!verifier.load(model_path)) {
        std::cerr << "  ERROR: failed to load model\n";
        return 1;
    }
    auto meta = verifier.inspect();
    if (meta.loaded) {
        std::cout << "│  " << meta.describe() << "\n";
    }
    std::cout << "└────────────────────────────────────────────────────────\n\n";

    // ── embed all clips ───────────────────────────────────────────────────────
    std::cout << "┌─ Embedding clips\n";

    // speaker_label → clip_label → embedding
    std::map<std::string, std::map<std::string, std::vector<float>>> embeddings;

    int n_done = 0;
    auto t0 = Clock::now();

    for (const auto& [spk, clips] : speaker_clips) {
        for (const auto& [clip, path] : clips) {
            ++n_done;
            try {
                AudioBuffer buf = wav::read_wav_mono_16k(path.string());
                AudioChunk chunk;
                chunk.samples     = buf.samples;
                chunk.sample_rate = buf.sample_rate;
                chunk.start_ms    = 0;
                chunk.end_ms      = static_cast<int64_t>(
                    buf.samples.size() * 1000 / buf.sample_rate);

                auto emb = verifier.embed(chunk);
                embeddings[spk][clip] = std::move(emb);

                std::cout << "│  [" << std::setw(3) << n_done << "/" << total_clips << "] "
                          << spk << "_" << clip << ".wav  "
                          << std::fixed << std::setprecision(1)
                          << (chunk.end_ms / 1000.0f) << " s  ✓\n";
            } catch (const std::exception& ex) {
                std::cout << "│  [" << std::setw(3) << n_done << "/" << total_clips << "] "
                          << spk << "_" << clip << ".wav  ERROR: " << ex.what() << "\n";
            }
        }
    }

    auto t1 = Clock::now();
    float embed_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    std::cout << "│\n│  Total embed time: " << std::fixed << std::setprecision(0)
              << embed_ms << " ms  (" << std::setprecision(0)
              << (embed_ms / n_done) << " ms/clip)\n";
    std::cout << "└────────────────────────────────────────────────────────\n\n";

    // ── compute pairwise cosine similarities ──────────────────────────────────
    std::cout << "┌─ Pairwise cosine similarities\n│\n";

    std::vector<float> same_scores;
    std::vector<float> diff_scores;

    // Collect speakers as sorted list
    std::vector<std::string> spk_list;
    for (const auto& [spk, _] : embeddings) spk_list.push_back(spk);
    std::sort(spk_list.begin(), spk_list.end());

    // ── same-speaker ─────────────────────────────────────────────────────────
    std::cout << "│  ── Same-speaker pairs ─────────────────────────────\n";
    for (const auto& spk : spk_list) {
        const auto& clips = embeddings.at(spk);
        std::vector<std::string> clip_list;
        for (const auto& [c, _] : clips) clip_list.push_back(c);
        std::sort(clip_list.begin(), clip_list.end());

        std::cout << "│  " << spk << ":";
        for (size_t i = 0; i < clip_list.size(); ++i) {
            for (size_t j = i + 1; j < clip_list.size(); ++j) {
                const auto& ea = clips.at(clip_list[i]);
                const auto& eb = clips.at(clip_list[j]);
                if (ea.empty() || eb.empty()) continue;
                float cos = std::inner_product(ea.begin(), ea.end(), eb.begin(), 0.0f);
                same_scores.push_back(cos);
                std::cout << "  " << clip_list[i] << "×" << clip_list[j]
                          << "=" << std::fixed << std::setprecision(3) << cos;
            }
        }
        std::cout << "\n";
    }

    // ── different-speaker ─────────────────────────────────────────────────────
    std::cout << "│\n│  ── Different-speaker pairs ────────────────────────\n";
    // One representative clip per speaker (clip 'a') vs all others
    for (size_t i = 0; i < spk_list.size(); ++i) {
        const auto& clips_i = embeddings.at(spk_list[i]);
        if (clips_i.empty()) continue;
        const auto& ref_clip_i = clips_i.begin()->second;  // first clip of speaker i
        const auto& ref_lbl_i  = clips_i.begin()->first;

        for (size_t j = i + 1; j < spk_list.size(); ++j) {
            const auto& clips_j = embeddings.at(spk_list[j]);
            if (clips_j.empty()) continue;

            // Use all clips of speaker j vs ref clip of speaker i
            for (const auto& [lbl_j, emb_j] : clips_j) {
                if (emb_j.empty()) continue;
                float cos = std::inner_product(
                    ref_clip_i.begin(), ref_clip_i.end(), emb_j.begin(), 0.0f);
                diff_scores.push_back(cos);
            }
        }
    }

    // Print a compact sample of cross-speaker scores
    {
        int printed = 0;
        for (size_t i = 0; i < spk_list.size() && printed < 20; ++i) {
            const auto& clips_i = embeddings.at(spk_list[i]);
            if (clips_i.empty()) continue;
            const auto& ea = clips_i.begin()->second;
            const auto& la = clips_i.begin()->first;

            for (size_t j = i + 1; j < spk_list.size() && printed < 20; ++j) {
                const auto& clips_j = embeddings.at(spk_list[j]);
                if (clips_j.empty()) continue;
                const auto& eb = clips_j.begin()->second;
                const auto& lb = clips_j.begin()->first;
                float cos = std::inner_product(ea.begin(), ea.end(), eb.begin(), 0.0f);
                std::cout << "│  " << spk_list[i] << "_" << la
                          << " × " << spk_list[j] << "_" << lb
                          << "  = " << std::fixed << std::setprecision(3) << cos << "\n";
                ++printed;
            }
        }
        if (diff_scores.size() > 20) {
            std::cout << "│  … (" << diff_scores.size() << " pairs total)\n";
        }
    }

    std::cout << "└────────────────────────────────────────────────────────\n\n";

    // ── distributions ─────────────────────────────────────────────────────────
    if (same_scores.empty() || diff_scores.empty()) {
        std::cerr << "  ERROR: not enough clips to compute distributions.\n";
        return 1;
    }

    const Stats ss = Stats::of(same_scores);
    const Stats ds = Stats::of(diff_scores);

    std::cout << "┌─ Distributions\n│\n";

    auto print_hist = [&](const std::vector<float>& scores, const std::string& label) {
        const int  N_BINS = 20;
        const float lo    = 0.0f;
        const float hi    = 1.0f;
        const float bw    = (hi - lo) / N_BINS;
        std::vector<int> bins(N_BINS, 0);
        for (float v : scores) {
            int b = static_cast<int>((v - lo) / bw);
            b = std::max(0, std::min(N_BINS - 1, b));
            ++bins[b];
        }
        int peak = *std::max_element(bins.begin(), bins.end());
        std::cout << "│  " << label << "\n";
        for (int b = N_BINS - 1; b >= 0; --b) {
            float lo_b = lo + b * bw;
            float hi_b = lo_b + bw;
            float frac = (peak > 0) ? static_cast<float>(bins[b]) / peak : 0.0f;
            std::cout << "│  " << std::fixed << std::setprecision(2)
                      << lo_b << "–" << hi_b << "  "
                      << bar(frac, 30) << "  " << bins[b] << "\n";
        }
        std::cout << "│\n";
    };

    print_hist(same_scores, "Same-speaker  (" + std::to_string(ss.count) + " pairs)");
    print_hist(diff_scores, "Different-speaker  (" + std::to_string(ds.count) + " pairs)");

    std::cout << "│  Same-speaker:       "
              << "min=" << std::fixed << std::setprecision(3) << ss.vmin
              << "  max=" << ss.vmax
              << "  mean=" << ss.mean
              << "  σ=" << ss.stddev
              << "  n=" << ss.count << "\n";
    std::cout << "│  Different-speaker:  "
              << "min=" << std::fixed << std::setprecision(3) << ds.vmin
              << "  max=" << ds.vmax
              << "  mean=" << ds.mean
              << "  σ=" << ds.stddev
              << "  n=" << ds.count << "\n";
    std::cout << "└────────────────────────────────────────────────────────\n\n";

    // ── EER + threshold analysis ──────────────────────────────────────────────
    auto [eer_t, eer_v] = compute_eer(same_scores, diff_scores);

    // Midpoint between means
    float mid_t = (ss.mean + ds.mean) * 0.5f;

    // FAR / FRR at current threshold
    int far_n = static_cast<int>(
        std::count_if(diff_scores.begin(), diff_scores.end(),
                      [](float v){ return v >= 0.72f; }));
    int frr_n = static_cast<int>(
        std::count_if(same_scores.begin(), same_scores.end(),
                      [](float v){ return v <  0.72f; }));
    float far = static_cast<float>(far_n) / static_cast<float>(diff_scores.size());
    float frr = static_cast<float>(frr_n) / static_cast<float>(same_scores.size());

    std::cout << "┌─ Threshold analysis\n│\n"
              << "│  EER threshold:    " << std::fixed << std::setprecision(3)
              << eer_t << "   (EER ≈ " << std::setprecision(1) << eer_v * 100.0f << "%)\n"
              << "│  Midpoint:         " << std::fixed << std::setprecision(3) << mid_t
              << "   (mean_same=" << std::setprecision(3) << ss.mean
              << "  mean_diff=" << ds.mean << ")\n"
              << "│\n"
              << "│  Current threshold (0.72):\n"
              << "│    FAR (diff-speaker accepted):  "
              << std::setw(4) << far_n << " / " << diff_scores.size()
              << "  (" << std::fixed << std::setprecision(1) << far * 100.0f << "%)"
              << "  [lower is better]\n"
              << "│    FRR (same-speaker rejected):  "
              << std::setw(4) << frr_n << " / " << same_scores.size()
              << "  (" << std::fixed << std::setprecision(1) << frr * 100.0f << "%)"
              << "  [lower is better]\n"
              << "│\n";

    // Verdict
    bool separable = (ss.mean - ds.mean) > 0.2f;
    bool threshold_ok = (far < 0.10f) && (frr < 0.20f);
    std::cout << "│  Separability gap:  "
              << std::fixed << std::setprecision(3) << (ss.mean - ds.mean)
              << "  (" << (separable ? "✓ good separation" : "⚠ low separation") << ")\n";
    std::cout << "│  Threshold 0.72:    "
              << (threshold_ok ? "✓ reasonable (FAR<10%, FRR<20%)"
                               : "⚠ consider adjusting (see EER threshold above)")
              << "\n"
              << "└────────────────────────────────────────────────────────\n\n";

    // ── exit code ─────────────────────────────────────────────────────────────
    // Pass if same-speaker mean > different-speaker mean + margin (basic sanity)
    if (ss.mean <= ds.mean + 0.05f) {
        std::cerr << "  FAIL: same-speaker mean (" << std::fixed << std::setprecision(3)
                  << ss.mean << ") is not clearly above different-speaker mean ("
                  << ds.mean << ")\n";
        return 1;
    }

    std::cout << "  ✓  verification_test PASSED\n\n";
    return 0;
}
