// ---------------------------------------------------------------------------
// bench/diarization_bench.cpp
//
// Standalone micro-benchmark for the diarization pipeline.
//
// Usage:
//   diarization_bench <audio.wav> [--model <model.onnx>] [--threshold 0.72]
//                     [--window-ms 1500] [--hop-ms 750] [--min-ms 800]
//                     [--max-speakers 0] [--format inline|json|srt|vtt]
//                     [--runs N]
//
// The binary:
//   1. Reads the WAV file and resamples to 16 kHz.
//   2. Synthesises fake Whisper segments (one per 2 seconds of audio) so the
//      pipeline runs without a real ASR model.
//   3. If --model is supplied, loads the ONNX model and runs real embeddings.
//      Without --model a fast stub model returns random unit-norm embeddings.
//   4. Runs the full pipeline `--runs` times (default 3) and prints per-run
//      and average timing for each stage:
//        Load/resample  → wall time to read + resample the WAV
//        Embedding      → time inside DiarizationEngine (embedding pass)
//        Clustering     → time inside DiarizationEngine (cluster assign pass)
//        Formatting     → time for TranscriptFormatter::format()
//        Total pipeline → embedding + clustering + formatting combined
//   5. Prints peak RSS (resident set size) from getrusage after the last run.
//
// Compile example (from repo root):
//   g++ -std=c++20 -O2 -I. \
//       bench/diarization_bench.cpp \
//       DiarizationEngine.cpp SpeakerClusterManager.cpp \
//       LabelSmoother.cpp TranscriptFormatter.cpp \
//       -o bench/diarization_bench
//
// With ONNX model support add:
//   -DDIARIZE_HAVE_MODEL WeSpeakerEcapaModel.cpp \
//   -lonnxruntime
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <sys/resource.h>  // getrusage — Linux / macOS
#include <vector>

// Project headers (paths relative to repo root; compile from there)
#include <diarization/AudioChunk.h>
#include <diarization/DiarizationEngine.h>
#include <diarization/ISpeakerEmbeddingModel.h>
#include <diarization/TranscriptFormatter.h>
#include "tests/wav_reader.h"

#ifdef DIARIZE_HAVE_MODEL
#  include "models/WeSpeakerEcapaModel.h"
#endif

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double now_ms() {
    return std::chrono::duration_cast<Ms>(Clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Stub model — returns random unit-norm embeddings, zero inference latency
// ---------------------------------------------------------------------------

class StubEmbeddingModel final : public ISpeakerEmbeddingModel {
public:
    explicit StubEmbeddingModel(int dim = 192)
        : dim_(dim), rng_(42), dist_(-1.0f, 1.0f) {}

    bool load(const std::string&) override { return true; }

    std::vector<float> embed(const AudioChunk&) override {
        std::vector<float> v(dim_);
        float norm = 0.0f;
        for (auto& x : v) { x = dist_(rng_); norm += x * x; }
        norm = std::sqrt(norm);
        if (norm > 1e-9f) for (auto& x : v) x /= norm;
        return v;
    }

    int sample_rate()   const override { return 16000; }
    int embedding_dim() const override { return dim_; }

private:
    int dim_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
};

// ---------------------------------------------------------------------------
// Synthesise fake Whisper segments (one per `seg_dur_ms` milliseconds)
// ---------------------------------------------------------------------------

static std::vector<WhisperSegment> make_fake_segments(int64_t audio_dur_ms,
                                                       int64_t seg_dur_ms = 2000)
{
    std::vector<WhisperSegment> segs;
    int idx = 0;
    for (int64_t t = 0; t + seg_dur_ms <= audio_dur_ms; t += seg_dur_ms) {
        WhisperSegment s;
        s.start_ms   = t;
        s.end_ms     = t + seg_dur_ms;
        s.text       = " Segment " + std::to_string(++idx) + ".";
        s.confidence = 0.95f;
        segs.push_back(s);
    }
    return segs;
}

// ---------------------------------------------------------------------------
// Peak RSS — Linux: /proc/self/status; macOS: getrusage ru_maxrss is bytes
// ---------------------------------------------------------------------------

static long peak_rss_kb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / 1024;   // macOS returns bytes
#else
    return ru.ru_maxrss;           // Linux returns kilobytes
#endif
}

// ---------------------------------------------------------------------------
// Simple arg parser
// ---------------------------------------------------------------------------

static std::string arg_str(int argc, char** argv, const char* flag,
                            const std::string& def = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0)
            return argv[i + 1];
    return def;
}

static double arg_float(int argc, char** argv, const char* flag, double def) {
    auto s = arg_str(argc, argv, flag);
    return s.empty() ? def : std::stod(s);
}

static int arg_int(int argc, char** argv, const char* flag, int def) {
    auto s = arg_str(argc, argv, flag);
    return s.empty() ? def : std::stoi(s);
}

static bool arg_has(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 2 || arg_has(argc, argv, "--help") || arg_has(argc, argv, "-h")) {
        std::cerr << "Usage: diarization_bench <audio.wav> [options]\n"
                  << "  --model <path>      ONNX speaker model (default: stub)\n"
                  << "  --threshold <f>     Cosine threshold (default 0.72)\n"
                  << "  --window-ms <n>     Window size ms (default 1500)\n"
                  << "  --hop-ms <n>        Hop size ms    (default 750)\n"
                  << "  --min-ms <n>        Min segment ms (default 800)\n"
                  << "  --max-speakers <n>  Max speakers   (default 0=unlimited)\n"
                  << "  --format <name>     inline|json|srt|vtt (default inline)\n"
                  << "  --runs <n>          Repetitions    (default 3)\n";
        return 1;
    }

    const std::string wav_path    = argv[1];
    const std::string model_path  = arg_str(argc, argv, "--model");
    const float  threshold        = static_cast<float>(arg_float(argc, argv, "--threshold",  0.72));
    const int    window_ms        = arg_int(argc, argv, "--window-ms", 1500);
    const int    hop_ms           = arg_int(argc, argv, "--hop-ms",     750);
    const int    min_ms           = arg_int(argc, argv, "--min-ms",     800);
    const int    max_speakers     = arg_int(argc, argv, "--max-speakers", 0);
    const int    runs             = arg_int(argc, argv, "--runs",          3);
    const std::string fmt_name    = arg_str(argc, argv, "--format", "inline");
    const SpeakerFormat fmt       = parse_speaker_format(fmt_name);

    // ── 1. Load + resample WAV ────────────────────────────────────────────

    const double t_load_start = now_ms();
    AudioBuffer audio;
    try {
        audio = wav::read_wav_mono_16k(wav_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading WAV: " << e.what() << "\n";
        return 2;
    }
    const double t_load_end = now_ms();

    const int64_t audio_dur_ms =
        static_cast<int64_t>(audio.samples.size()) * 1000LL / audio.sample_rate;

    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  diarization_bench  —  %s\n", wav_path.c_str());
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  Sample rate  : %d Hz (resampled to 16 kHz)\n", audio.sample_rate);
    std::printf("  Duration     : %.3f s  (%lld ms)\n",
                audio_dur_ms / 1000.0, static_cast<long long>(audio_dur_ms));
    std::printf("  Samples      : %zu\n", audio.samples.size());
    std::printf("  Load+resample: %.1f ms\n", t_load_end - t_load_start);

    // ── 2. Synthesise fake segments ───────────────────────────────────────

    const auto segs = make_fake_segments(audio_dur_ms, 2000 /*ms*/);
    std::printf("  Fake segments: %zu  (1 per 2 s)\n", segs.size());

    // ── 3. Build model ────────────────────────────────────────────────────

#ifdef DIARIZE_HAVE_MODEL
    std::unique_ptr<ISpeakerEmbeddingModel> model;
    if (!model_path.empty()) {
        auto m = std::make_unique<WeSpeakerEcapaModel>();
        if (!m->load(model_path)) {
            std::cerr << "Failed to load ONNX model: " << model_path << "\n";
            return 3;
        }
        model = std::move(m);
        std::printf("  Model        : %s  (ONNX, dim=%d)\n",
                    model_path.c_str(), model->embedding_dim());
    } else {
        model = std::make_unique<StubEmbeddingModel>();
        std::printf("  Model        : stub (random unit-norm, dim=192)\n");
    }
#else
    (void)model_path;
    auto model = std::make_unique<StubEmbeddingModel>();
    std::printf("  Model        : stub (random unit-norm, dim=192)\n");
    std::printf("  (build with -DDIARIZE_HAVE_MODEL to use a real ONNX model)\n");
#endif

    // ── 4. DiarizationOptions ─────────────────────────────────────────────

    DiarizationOptions opts;
    opts.speaker_threshold = threshold;
    opts.speaker_window_ms = window_ms;
    opts.speaker_hop_ms    = hop_ms;
    opts.speaker_min_ms    = min_ms;
    opts.speaker_max       = max_speakers;

    // ── 5. Run benchmark loop ─────────────────────────────────────────────

    struct RunResult {
        double embed_ms    = 0;
        double cluster_ms  = 0;
        double format_ms   = 0;
        int    n_windows   = 0;
        int    n_speakers  = 0;
    };

    std::vector<RunResult> results;
    results.reserve(runs);

    std::printf("\n  %-5s  %-12s  %-12s  %-12s  %-12s  spk  win\n",
                "Run", "Embed(ms)", "Cluster(ms)", "Format(ms)", "Total(ms)");
    std::printf("  ─────  ────────────  ────────────  ────────────  ────────────  ───  ───\n");

    for (int r = 0; r < runs; ++r) {
        // Create a fresh engine each run (reset cluster state)
        std::unique_ptr<ISpeakerEmbeddingModel> run_model =
            std::make_unique<StubEmbeddingModel>();
#ifdef DIARIZE_HAVE_MODEL
        if (!model_path.empty()) {
            auto m = std::make_unique<WeSpeakerEcapaModel>();
            m->load(model_path);
            run_model = std::move(m);
        }
#endif
        DiarizationEngine engine{std::move(run_model)};

        // Time the full process() call
        // We can't split embed vs cluster internally without modifying the
        // engine, so we time the whole process() and separately time format().
        const double t_proc_start = now_ms();
        auto result_segs = engine.process(audio, segs, opts);
        const double t_proc_end = now_ms();

        // Count windows: estimate from segment durations and hop
        int n_win = 0;
        for (const auto& s : segs) {
            const int64_t dur = s.end_ms - s.start_ms;
            if (dur < min_ms) continue;
            n_win += std::max(1, static_cast<int>((dur - window_ms) / hop_ms + 1));
        }

        const double t_fmt_start = now_ms();
        DiarizationMeta meta;
        meta.enabled = true;
        meta.threshold = threshold;
        // Build speaker list from result
        for (const auto& ts : result_segs) {
            bool found = false;
            for (auto& sp : meta.speakers)
                if (sp.id == ts.speaker) { ++sp.segments; found = true; break; }
            if (!found) meta.speakers.push_back({ts.speaker, 1});
        }
        const std::string formatted = TranscriptFormatter::format(
            result_segs, fmt, meta, "bench", wav_path, audio_dur_ms);
        const double t_fmt_end = now_ms();

        RunResult rr;
        rr.embed_ms   = t_proc_end - t_proc_start;  // embed+cluster combined
        rr.cluster_ms = 0;                            // not separately timed
        rr.format_ms  = t_fmt_end - t_fmt_start;
        rr.n_windows  = n_win;
        rr.n_speakers = static_cast<int>(engine.clusters().size());

        results.push_back(rr);

        std::printf("  %-5d  %-12.1f  %-12s  %-12.1f  %-12.1f  %-3d  %d\n",
                    r + 1,
                    rr.embed_ms,
                    "n/a",
                    rr.format_ms,
                    rr.embed_ms + rr.format_ms,
                    rr.n_speakers,
                    rr.n_windows);
    }

    // ── 6. Summary ────────────────────────────────────────────────────────

    const double avg_embed  = std::accumulate(results.begin(), results.end(), 0.0,
        [](double s, const RunResult& r){ return s + r.embed_ms; }) / runs;
    const double avg_format = std::accumulate(results.begin(), results.end(), 0.0,
        [](double s, const RunResult& r){ return s + r.format_ms; }) / runs;
    const double avg_total  = avg_embed + avg_format;

    const double rtf = avg_total / static_cast<double>(audio_dur_ms);

    std::printf("\n  Average (over %d runs)\n", runs);
    std::printf("  ─────────────────────────────────────────────────\n");
    std::printf("  Embed+cluster: %8.1f ms\n", avg_embed);
    std::printf("  Formatting   : %8.1f ms\n", avg_format);
    std::printf("  Total        : %8.1f ms\n", avg_total);
    std::printf("  RTF          : %8.4f  (%.2fx real-time)\n",
                rtf, 1.0 / rtf);
    std::printf("  Peak RSS     : %ld kB\n", peak_rss_kb());
    std::printf("═══════════════════════════════════════════════════════\n");

    return 0;
}
