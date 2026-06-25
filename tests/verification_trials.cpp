// tests/verification_trials.cpp
//
// Speaker verification evaluation on a trials file.
//
// Reads a trials file of the form:
//   1 id10020/000000.wav id10020/000001.wav   (same-speaker)
//   0 id10020/000000.wav id10023/000004.wav   (different-speaker)
//
// Embeds every unique WAV exactly once (cached), computes cosine similarity
// for each pair, then reports:
//
//   • Same / different-speaker cosine distributions
//   • ASCII histograms
//   • EER threshold and EER value
//   • FAR / FRR at the EER threshold
//   • Recommended threshold
//
// Build (from repo root):
//   ORT=/home/dan/Documents/piper1-gpl/libpiper/lib/onnxruntime-linux-x64-1.22.0
//   g++ -std=c++20 -O2 -Iinclude -I. -DDIARIZE_HAVE_MODEL \
//       -I$ORT/include \
//       tests/verification_trials.cpp \
//       src/DiarizationEngine.cpp src/SpeakerClusterManager.cpp \
//       src/LabelSmoother.cpp src/TranscriptFormatter.cpp \
//       models/WeSpeakerEcapaModel.cpp models/EcapaOnnxModel.cpp \
//       models/SpeechBrainEcapaModel.cpp models/SpeakerModelFactory.cpp \
//       models/SpeakerVerifier.cpp \
//       -L$ORT/lib -Wl,-rpath,$ORT/lib -lonnxruntime \
//       -o tests/verification_trials
//
// Run:
//   ./tests/verification_trials \
//       wespeaker/voxceleb_ECAPA512_LM.onnx \
//       testdata/voxceleb_trials/trials.txt \
//       ~/Datasets/voxceleb1

#include <filesystem>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <diarization/AudioChunk.h>
#include <diarization/SpeakerVerifier.h>
#include "tests/wav_reader.h"

using Clock = std::chrono::steady_clock;



namespace fs = std::filesystem;

fs::path resolve(const fs::path& base, const std::string& trial_path)
{
    fs::path tp(trial_path);

    if (!tp.empty() && *tp.begin() == "voxceleb1")
        tp = tp.lexically_relative("voxceleb1");

    return base / tp;
}

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string bar(float v, int width = 40) {
    int filled = static_cast<int>(std::round(v * width));
    filled = std::max(0, std::min(filled, width));
    return std::string(filled, '#') + std::string(width - filled, '-');
}

struct Stats {
    float vmin   = 0.0f;
    float vmax   = 0.0f;
    float mean   = 0.0f;
    float stddev = 0.0f;
    int   count  = 0;

    static Stats of(const std::vector<float>& v) {
        Stats s;
        s.count = static_cast<int>(v.size());
        if (v.empty()) return s;
        s.vmin   = *std::min_element(v.begin(), v.end());
        s.vmax   = *std::max_element(v.begin(), v.end());
        s.mean   = std::accumulate(v.begin(), v.end(), 0.0f) / s.count;
        float var = 0.0f;
        for (float x : v) var += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(var / s.count);
        return s;
    }
};

// AUC via trapezoidal rule over the ROC curve.
// Sweep threshold 0.000–1.000 in 0.001 steps; collect (FPR, TPR) pairs.
// FPR = FAR, TPR = 1 - FRR.
static float compute_auc(
    const std::vector<float>& same,
    const std::vector<float>& diff)
{
    // Build ROC points (descending threshold → ascending FPR)
    struct RocPt { float fpr, tpr; };
    std::vector<RocPt> pts;
    pts.reserve(1002);
    pts.push_back({ 0.0f, 0.0f });  // threshold=1.0 → nothing accepted

    for (int step = 1000; step >= 0; --step) {
        float t = step * 0.001f;
        int far_n = static_cast<int>(
            std::count_if(diff.begin(), diff.end(), [t](float v){ return v >= t; }));
        int frr_n = static_cast<int>(
            std::count_if(same.begin(), same.end(), [t](float v){ return v < t; }));
        float fpr = static_cast<float>(far_n) / static_cast<float>(diff.size());
        float tpr = 1.0f - static_cast<float>(frr_n) / static_cast<float>(same.size());
        pts.push_back({ fpr, tpr });
    }

    pts.push_back({ 1.0f, 1.0f });  // threshold=0.0 → everything accepted

    // Trapezoidal rule
    float auc = 0.0f;
    for (size_t i = 1; i < pts.size(); ++i) {
        float dx = pts[i].fpr - pts[i-1].fpr;
        if (dx > 0.0f)
            auc += dx * (pts[i].tpr + pts[i-1].tpr) * 0.5f;
    }
    return auc;
}

// Sweep threshold 0.000–1.000 in steps of 0.001.
// Returns {eer_threshold, eer_value}.
static std::pair<float, float> compute_eer(
    const std::vector<float>& same,
    const std::vector<float>& diff)
{
    float best_t   = 0.5f;
    float best_gap = 1.0f;

    for (int step = 0; step <= 1000; ++step) {
        float t = step * 0.001f;

        // FRR: same-speaker pairs rejected (score < threshold)
        int frr_n = static_cast<int>(
            std::count_if(same.begin(), same.end(), [t](float v){ return v < t; }));
        float frr = static_cast<float>(frr_n) / static_cast<float>(same.size());

        // FAR: different-speaker pairs accepted (score >= threshold)
        int far_n = static_cast<int>(
            std::count_if(diff.begin(), diff.end(), [t](float v){ return v >= t; }));
        float far = static_cast<float>(far_n) / static_cast<float>(diff.size());

        float gap = std::abs(far - frr);
        if (gap < best_gap) {
            best_gap = gap;
            best_t   = t;
        }
    }

    // Compute average EER at that threshold
    int frr_n = static_cast<int>(
        std::count_if(same.begin(), same.end(), [best_t](float v){ return v < best_t; }));
    float frr = static_cast<float>(frr_n) / static_cast<float>(same.size());
    int far_n = static_cast<int>(
        std::count_if(diff.begin(), diff.end(), [best_t](float v){ return v >= best_t; }));
    float far = static_cast<float>(far_n) / static_cast<float>(diff.size());

    return { best_t, (frr + far) * 0.5f };
}

static void print_histogram(const std::vector<float>& scores,
                             const std::string& label,
                             int n_bins = 20)
{
    const float lo = 0.0f, hi = 1.0f;
    const float bw = (hi - lo) / n_bins;
    std::vector<int> bins(n_bins, 0);
    for (float v : scores) {
        int b = static_cast<int>((v - lo) / bw);
        bins[std::max(0, std::min(n_bins - 1, b))]++;
    }
    int peak = *std::max_element(bins.begin(), bins.end());
    std::cout << "│  " << label << "\n";
    for (int b = n_bins - 1; b >= 0; --b) {
        float lo_b = lo + b * bw, hi_b = lo_b + bw;
        float frac = (peak > 0) ? static_cast<float>(bins[b]) / peak : 0.0f;
        std::cout << "│  " << std::fixed << std::setprecision(2)
                  << lo_b << "–" << hi_b << "  "
                  << bar(frac, 30) << "  " << bins[b] << "\n";
    }
    std::cout << "│\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr
            << "Usage: verification_trials <model.onnx> <trials.txt> <audio_base_dir>\n"
            << "  model.onnx      — ONNX speaker embedding model\n"
            << "  trials.txt      — generated_trials.txt (label pathA pathB)\n"
            << "  audio_base_dir  — base directory; paths in trials are relative to this\n";
        return 1;
    }

    const std::string model_path   = argv[1];
    const std::string trials_path  = argv[2];
    const std::string audio_base   = argv[3];

    std::cout
        << "═══════════════════════════════════════════════════════\n"
        << "  VoxCeleb-style Speaker Verification Trials\n"
        << "  model:  " << model_path  << "\n"
        << "  trials: " << trials_path << "\n"
        << "  audio:  " << audio_base  << "\n"
        << "═══════════════════════════════════════════════════════\n\n";

    // ── Read trials file ───────────────────────────────────────────────────────
    struct Trial {
        int         label;    // 1 = same, 0 = different
        std::string pathA;
        std::string pathB;
    };

    std::vector<Trial> trials;
    int n_pos = 0, n_neg = 0;

    {
        std::ifstream f(trials_path);
        if (!f) {
            std::cerr << "ERROR: cannot open trials file: " << trials_path << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            Trial t;
            if (!(ss >> t.label >> t.pathA >> t.pathB)) continue;
            trials.push_back(t);
            if (t.label == 1) ++n_pos; else ++n_neg;
        }
    }

    if (trials.empty()) {
        std::cerr << "ERROR: no trials loaded from " << trials_path << "\n";
        return 1;
    }

    std::cout << "┌─ Trials loaded\n"
              << "│  total:     " << trials.size() << "\n"
              << "│  positive:  " << n_pos  << "  (same-speaker)\n"
              << "│  negative:  " << n_neg  << "  (different-speaker)\n"
              << "└────────────────────────────────────────────────────────\n\n";

    // ── Load model ────────────────────────────────────────────────────────────
    std::cout << "┌─ Loading model\n│  " << model_path << "\n";
    SpeakerVerifier verifier;
    if (!verifier.load(model_path)) {
        std::cerr << "  ERROR: failed to load model\n";
        return 1;
    }
    auto meta = verifier.inspect();
    if (meta.loaded)
        std::cout << "│  " << meta.describe() << "\n";
    std::cout << "└────────────────────────────────────────────────────────\n\n";

    // ── Build unique path set and embed each once ─────────────────────────────
    std::cout << "┌─ Embedding unique utterances\n";

    // Collect unique relative paths
    std::vector<std::string> unique_paths;
    {
        std::vector<std::string> all_paths;
        all_paths.reserve(trials.size() * 2);
        for (const auto& t : trials) {
            all_paths.push_back(t.pathA);
            all_paths.push_back(t.pathB);
        }
        std::sort(all_paths.begin(), all_paths.end());
        all_paths.erase(std::unique(all_paths.begin(), all_paths.end()), all_paths.end());
        unique_paths = std::move(all_paths);
    }

    std::cout << "│  unique WAVs: " << unique_paths.size() << "\n│\n";

    // Embed each
    std::map<std::string, std::vector<float>> embeddings;

    int n_ok = 0;
    int n_fail = 0;          // embedding/read failures
    int missing_files = 0;   // file does not exist
    auto t0 = Clock::now();

    for (size_t i = 0; i < unique_paths.size(); ++i) {
        const std::string& rel  = unique_paths[i];
       
        const fs::path resolved = resolve(audio_base, rel);

        if (!fs::exists(resolved)) {
            ++missing_files;
            
            if (missing_files <= 10)
                std::cerr << "│  MISSING: "
                        << resolved.lexically_normal().string()
                        << "\n";
            continue;
        }


        try {
            AudioBuffer buf = wav::read_wav_mono_16k(resolved.string());

            if (buf.samples.empty())
                throw std::runtime_error("empty audio");

            AudioChunk chunk;
            chunk.samples     = buf.samples;
            chunk.sample_rate = buf.sample_rate;
            chunk.start_ms    = 0;
            chunk.end_ms      = static_cast<int64_t>(
                buf.samples.size() * 1000 / buf.sample_rate);

            auto emb = verifier.embed(chunk);
            embeddings[resolved.lexically_normal().string()] = std::move(emb);
            ++n_ok;

            if (i % 100 == 0 || i == unique_paths.size() - 1) {
                std::cout << "│  [" << std::setw(4) << (i + 1)
                          << "/" << unique_paths.size() << "]  " << rel << "\n";
            }
        } catch (const std::exception& ex) {
            std::cerr << "│  WARN: " << rel
                    << " — " << ex.what() << "\n";
            ++n_fail;
        }
    }

    auto t1 = Clock::now();
    float embed_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const int cache_misses = n_ok;   // one disk read per unique path
    const int cache_hits   = static_cast<int>(trials.size()) * 2 - cache_misses;

    std::cout
        << "│  Embeddings computed: " << cache_misses << "\n"
        << "│  Cache hits:          " << cache_hits   << "\n"
        << "│  Cache misses:        " << cache_misses << "\n";

    if (missing_files > 0)
        std::cout << "│  Missing WAV files:   " << missing_files << "\n";

    if (n_fail > 0)
        std::cout << "│  Embed failures:      " << n_fail << "\n";

    std::cout << "│  Time:   " << std::fixed << std::setprecision(0)
                << embed_ms << " ms  ("
                << std::setprecision(1) << (embed_ms / std::max(1, n_ok)) << " ms/utt)\n"
                << "└────────────────────────────────────────────────────────\n\n";

    // ── Score all trials ──────────────────────────────────────────────────────
    std::cout << "┌─ Scoring trials\n│\n";

    std::vector<float> same_scores, diff_scores;
    int n_skipped = 0;

    for (const auto& t : trials) {
        auto keyA = resolve(audio_base, t.pathA).lexically_normal().string();
        auto keyB = resolve(audio_base, t.pathB).lexically_normal().string();

        auto itA = embeddings.find(keyA);
        auto itB = embeddings.find(keyB);
        if (itA == embeddings.end() || itB == embeddings.end()) {
            ++n_skipped;
            continue;
        }
        const auto& ea = itA->second;
        const auto& eb = itB->second;
        if (ea.empty() || eb.empty()) { ++n_skipped; continue; }

        float cos = std::inner_product(ea.begin(), ea.end(), eb.begin(), 0.0f);
        if (t.label == 1) same_scores.push_back(cos);
        else              diff_scores.push_back(cos);
    }

    std::cout << "│  same-speaker pairs scored:  " << same_scores.size() << "\n"
              << "│  diff-speaker pairs scored:  " << diff_scores.size() << "\n"
              << "│  Missing WAV files:          " << missing_files << "\n"
                << "│  Embed failures:            " << n_fail << "\n";
    if (n_skipped > 0)
        std::cout << "│  skipped (missing embedding): " << n_skipped << "\n";
    std::cout << "└────────────────────────────────────────────────────────\n\n";

    if (same_scores.empty() || diff_scores.empty()) {
        std::cerr << "ERROR: not enough scored pairs to compute statistics.\n";
        return 1;
    }

    // ── Distributions ─────────────────────────────────────────────────────────
    const Stats ss = Stats::of(same_scores);
    const Stats ds = Stats::of(diff_scores);

    std::cout << "┌─ Distributions\n│\n";
    print_histogram(same_scores,
        "Same-speaker (" + std::to_string(ss.count) + " pairs)");
    print_histogram(diff_scores,
        "Different-speaker (" + std::to_string(ds.count) + " pairs)");

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
    std::cout << "│\n│  Separability gap:   "
              << std::fixed << std::setprecision(3) << (ss.mean - ds.mean) << "\n";
    std::cout << "└────────────────────────────────────────────────────────\n\n";

    // ── EER + AUC + threshold analysis ────────────────────────────────────────
    auto [eer_t, eer_v] = compute_eer(same_scores, diff_scores);
    const float auc     = compute_auc(same_scores, diff_scores);
    const float mid_t   = (ss.mean + ds.mean) * 0.5f;

    // FAR / FRR at EER threshold
    int far_n_eer = static_cast<int>(
        std::count_if(diff_scores.begin(), diff_scores.end(),
                      [eer_t](float v){ return v >= eer_t; }));
    int frr_n_eer = static_cast<int>(
        std::count_if(same_scores.begin(), same_scores.end(),
                      [eer_t](float v){ return v <  eer_t; }));
    float far_eer = static_cast<float>(far_n_eer) / diff_scores.size();
    float frr_eer = static_cast<float>(frr_n_eer) / same_scores.size();

    std::cout << "┌─ Threshold analysis\n│\n"
              << "│  EER threshold:  " << std::fixed << std::setprecision(3) << eer_t
              << "   (EER ≈ " << std::setprecision(1) << eer_v * 100.0f << "%)\n"
              << "│    FAR: " << std::setw(6) << far_n_eer << " / " << diff_scores.size()
              << "  (" << std::fixed << std::setprecision(1) << far_eer * 100.0f << "%)\n"
              << "│    FRR: " << std::setw(6) << frr_n_eer << " / " << same_scores.size()
              << "  (" << std::fixed << std::setprecision(1) << frr_eer * 100.0f << "%)\n"
              << "│\n"
              << "│  AUC (ROC):      " << std::fixed << std::setprecision(4) << auc
              << "   (0.5 = random, 1.0 = perfect)\n"
              << "│\n"
              << "│  Midpoint (mean_same+mean_diff)/2:  "
              << std::fixed << std::setprecision(3) << mid_t << "\n"
              << "│\n";

    // ROC summary: FAR/FRR at a few representative thresholds
    std::cout << "│  ROC summary:\n"
              << "│  " << std::setw(8) << "Thresh"
              << "  " << std::setw(8) << "FAR%"
              << "  " << std::setw(8) << "FRR%"
              << "\n";
    for (float t : { 0.40f, 0.50f, 0.60f, 0.65f, 0.70f, 0.72f,
                     0.75f, 0.78f, 0.80f, 0.85f, 0.90f }) {
        int fa = static_cast<int>(
            std::count_if(diff_scores.begin(), diff_scores.end(),
                          [t](float v){ return v >= t; }));
        int fr = static_cast<int>(
            std::count_if(same_scores.begin(), same_scores.end(),
                          [t](float v){ return v <  t; }));
        std::string marker = (std::abs(t - eer_t) < 0.006f) ? "  <- EER" : "";
        std::cout << "│  " << std::fixed << std::setprecision(2) << std::setw(8) << t
                  << "  " << std::setw(7) << std::setprecision(1)
                  << (100.0f * fa / diff_scores.size()) << "%"
                  << "  " << std::setw(7) << std::setprecision(1)
                  << (100.0f * fr / same_scores.size()) << "%"
                  << marker << "\n";
    }
    std::cout << "│\n";

    // Verdict
    const bool good_sep   = (ss.mean - ds.mean) > 0.15f;
    const float rec_t     = eer_t;

    std::cout << "│  Recommended threshold (EER):  "
              << std::fixed << std::setprecision(3) << rec_t << "\n"
              << "│  Separability:                 "
              << (good_sep ? "✓ good" : "⚠ poor") << "  (gap = "
              << std::fixed << std::setprecision(3) << (ss.mean - ds.mean) << ")\n"
              << "└────────────────────────────────────────────────────────\n\n";

    // ── Exit code ─────────────────────────────────────────────────────────────
    if (!good_sep) {
        std::cerr << "FAIL: insufficient separability between same and different speakers.\n";
        return 1;
    }

    std::cout << "  ✓  verification_trials PASSED\n\n";
    return 0;
}
