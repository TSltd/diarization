/// Integration tests for the diarization pipeline.
///
/// These tests exercise the full path from AudioBuffer → DiarizationEngine →
/// TranscriptFormatter without a live microphone or a real Whisper model.
/// A stub embedding model is used so the suite can run in CI without a
/// downloaded ONNX file.
///
/// Tests that require a real ONNX model are gated behind the preprocessor
/// symbol DIARIZE_HAVE_MODEL.  Build with:
///
//   g++ -std=c++20 -DDIARIZE_HAVE_MODEL -I.. \
//       integration_tests.cpp \
//       ../SpeakerClusterManager.cpp \
//       ../LabelSmoother.cpp \
//       ../TranscriptFormatter.cpp \
//       ../DiarizationEngine.cpp \
//       ../WeSpeakerEcapaModel.cpp \
//       -lonnxruntime \
//       -o run_integration && ./run_integration \
//       dist/models/speaker-ecapa.onnx \
//       test_two_speakers.wav

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "wav_reader.h"            // wav::read_wav_mono
#include <diarization/DiarizationEngine.h>
#include <diarization/ISpeakerEmbeddingModel.h>
#include <diarization/TranscriptFormatter.h>

#ifdef DIARIZE_HAVE_MODEL
#include "models/WeSpeakerEcapaModel.h"
#include "models/SpeakerModelFactory.h"
#include <diarization/SpeakerVerifier.h>
#endif

// ---------------------------------------------------------------------------
// Test runner — deferred execution so main() sets globals before tests run.
// ---------------------------------------------------------------------------

#include <functional>

static int g_pass = 0;
static int g_fail = 0;

struct TestCase {
    const char*             name;
    std::function<void()>   fn;
};
static std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> r;
    return r;
}

#define EXPECT(cond) \
    do { \
        if (!(cond)) { \
            std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail; \
        } else { \
            ++g_pass; \
        } \
    } while (0)

// TEST() now only registers the function; execution happens in main().
#define TEST(name) \
    static void test_##name(); \
    struct _Reg_##name { \
        _Reg_##name() { test_registry().push_back({#name, test_##name}); } \
    } _reg_##name; \
    static void test_##name()

// ---------------------------------------------------------------------------
// Global paths (populated from argv in main())
// ---------------------------------------------------------------------------

static std::string g_model_path;
static std::string g_wav_path;

// ---------------------------------------------------------------------------
// Stub embedding model
// ---------------------------------------------------------------------------

/// Generates a distinct unit embedding per call based on a cycling index.
/// The first N/2 calls return embeddings in the "A" half of embedding space,
/// the second N/2 calls return embeddings in the "B" half.
class SegmentIndexModel : public ISpeakerEmbeddingModel {
public:
    explicit SegmentIndexModel(int total_segments)
        : total_(total_segments) {}

    bool load(const std::string&) override { return true; }

    std::vector<float> embed(const AudioChunk&) override {
        // First half of segments → speaker A direction
        // Second half            → speaker B direction
        float sign = (call_ < total_ / 2) ? 1.0f : -1.0f;
        ++call_;
        // 4-dim embedding pointing in opposite directions
        return { sign, 0.f, 0.f, 0.f };
    }

    int sample_rate()   const override { return 16000; }
    int embedding_dim() const override { return 4; }

private:
    int total_;
    int call_ = 0;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static AudioBuffer make_audio(int64_t duration_ms, int sample_rate = 16000) {
    AudioBuffer buf;
    buf.sample_rate = sample_rate;
    buf.channels    = 1;
    buf.samples.assign(static_cast<size_t>(duration_ms * sample_rate / 1000), 0.0f);
    return buf;
}

static std::vector<WhisperSegment> make_uniform_segments(
    int n, int64_t total_ms) {

    std::vector<WhisperSegment> segs(n);
    const int64_t seg_ms = total_ms / n;
    for (int i = 0; i < n; ++i) {
        segs[i].start_ms   = i * seg_ms;
        segs[i].end_ms     = (i + 1) * seg_ms;
        segs[i].text       = "segment " + std::to_string(i);
        segs[i].confidence = 0.9f;
    }
    return segs;
}

// ---------------------------------------------------------------------------
// TEST: FullPipelineStub — engine + formatter, no real model
// ---------------------------------------------------------------------------
TEST(FullPipelineStub) {
    // 4 segments: first 2 → SPEAKER_00, last 2 → SPEAKER_01
    const int N = 4;
    auto model_ptr = std::unique_ptr<ISpeakerEmbeddingModel>(
        new SegmentIndexModel{N});
    DiarizationEngine engine{std::move(model_ptr)};

    AudioBuffer audio = make_audio(8000); // 8 seconds
    auto segs = make_uniform_segments(N, 8000);

    DiarizationOptions opts;
    opts.speaker_min_ms    = 800;
    opts.speaker_window_ms = 4000; // > segment → one embedding per segment
    opts.speaker_hop_ms    = 2000;
    opts.speaker_threshold = 0.5f;

    auto results = engine.process(audio, segs, opts);

    EXPECT(results.size() == static_cast<size_t>(N));

    // Collect unique speakers
    std::set<std::string> speakers;
    for (const auto& r : results)
        if (r.speaker != "UNKNOWN") speakers.insert(r.speaker);

    EXPECT(speakers.size() == 2u);

    // First two segments should share a label, last two should share a label.
    EXPECT(results[0].speaker == results[1].speaker);
    EXPECT(results[2].speaker == results[3].speaker);
    EXPECT(results[0].speaker != results[2].speaker);

    // --- Format as JSON ---
    DiarizationMeta meta;
    meta.enabled       = true;
    meta.speaker_model = "(stub)";
    meta.threshold     = opts.speaker_threshold;
    for (auto& sp : speakers)
        meta.speakers.push_back({sp, 0});

    std::string json = TranscriptFormatter::format_json(
        results, meta, "whisper", "(none)", 8000);

    EXPECT(!json.empty());
    EXPECT(json.find("\"segments\"") != std::string::npos);
    EXPECT(json.find("\"SPEAKER_00\"") != std::string::npos);

    // --- Format as SRT ---
    std::string srt = TranscriptFormatter::format_srt(results);
    EXPECT(srt.find("SPEAKER_00") != std::string::npos);
    EXPECT(srt.find("-->") != std::string::npos);

    // --- Format as VTT ---
    std::string vtt = TranscriptFormatter::format_vtt(results);
    EXPECT(vtt.substr(0, 7) == "WEBVTT\n");
}

// ---------------------------------------------------------------------------
// TEST: WavReaderSmoke — read a WAV produced by gen_test_wav.py
// ---------------------------------------------------------------------------
TEST(WavReaderSmoke) {
    if (g_wav_path.empty()) {
        std::printf("  SKIP  WavReaderSmoke: no WAV path provided\n");
        return;
    }

    AudioBuffer buf = wav::read_wav_mono(g_wav_path);

    EXPECT(buf.sample_rate == 16000);
    EXPECT(buf.channels == 1);
    // The generated WAV is ~15 seconds → ~240 000 samples.
    EXPECT(buf.samples.size() > 10000u);

    // Samples must be in [-1, 1].
    for (float s : buf.samples)
        EXPECT(s >= -1.0f && s <= 1.0f);
}

// ---------------------------------------------------------------------------
// TEST: MemoryScaling — measure peak sample-buffer size at 30s / 5min / 30min
// ---------------------------------------------------------------------------
TEST(MemoryScaling) {
    // This test does NOT run the engine; it only verifies that allocating and
    // immediately releasing AudioBuffers at different sizes works without
    // crashing and that the expected byte counts are reasonable.
    struct Case { const char* label; int64_t dur_ms; size_t expected_bytes; };
    const Case cases[] = {
        { "30s",  30'000,     30'000LL * 16000 / 1000 * 4 },  // ~1.9 MB
        { "5min", 300'000,   300'000LL * 16000 / 1000 * 4 }, // ~18.3 MB
        { "30min",1'800'000,1'800'000LL* 16000 / 1000 * 4 }, // ~110 MB
    };

    for (const auto& c : cases) {
        const size_t n_samples = static_cast<size_t>(c.dur_ms) * 16000 / 1000;
        EXPECT(n_samples * sizeof(float) == c.expected_bytes);

        // Allocate and immediately release — tests that malloc doesn't fail.
        {
            std::vector<float> v(n_samples, 0.0f);
            EXPECT(v.size() == n_samples);
        } // released here

        std::printf("  INFO  %s: %zu samples, %.1f MB\n",
            c.label, n_samples,
            static_cast<double>(n_samples * sizeof(float)) / (1024.0 * 1024.0));
    }
}

// ---------------------------------------------------------------------------
// TEST: RealModelRoundTrip — needs a real ONNX model + two-speaker WAV
// ---------------------------------------------------------------------------
#ifdef DIARIZE_HAVE_MODEL
TEST(RealModelRoundTrip) {
    if (g_model_path.empty() || g_wav_path.empty()) {
        std::printf("  SKIP  RealModelRoundTrip: set model + wav paths\n");
        return;
    }

    // Load the model.
    WeSpeakerEcapaModel model;
    bool ok = model.load(g_model_path);
    EXPECT(ok);
    if (!ok) {
        std::printf("  SKIP  RealModelRoundTrip: model load failed\n");
        return;
    }

    EXPECT(model.embedding_dim() > 0);
    std::printf("  INFO  embedding_dim = %d\n", model.embedding_dim());

    // Load the two-speaker WAV.
    AudioBuffer audio = wav::read_wav_mono(g_wav_path);
    EXPECT(audio.sample_rate == 16000);
    EXPECT(!audio.samples.empty());

    // Build a simple 1-second stride segmentation to mock Whisper.
    const int64_t dur_ms = static_cast<int64_t>(audio.samples.size()) * 1000 / audio.sample_rate;
    const int64_t stride = 2000; // 2-second segments
    std::vector<WhisperSegment> wsegs;
    for (int64_t t = 0; t + stride <= dur_ms; t += stride) {
        WhisperSegment s;
        s.start_ms   = t;
        s.end_ms     = t + stride;
        s.text       = "...";
        s.confidence = 0.9f;
        wsegs.push_back(s);
    }

    auto model_ptr = std::unique_ptr<ISpeakerEmbeddingModel>(
        new WeSpeakerEcapaModel{});
    model_ptr->load(g_model_path);
    DiarizationEngine engine{std::move(model_ptr)};

    auto t0 = std::chrono::steady_clock::now();
    auto results = engine.process(audio, wsegs);
    auto t1 = std::chrono::steady_clock::now();

    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  INFO  processed %lld ms audio in %.1f ms (%.1fx RT)\n",
        static_cast<long long>(dur_ms), elapsed_ms,
        static_cast<double>(dur_ms) / elapsed_ms);

    // At least 2 distinct speaker labels expected for a two-speaker file.
    std::set<std::string> speakers;
    for (const auto& r : results)
        if (r.speaker != "UNKNOWN") speakers.insert(r.speaker);

    EXPECT(speakers.size() >= 2u);
    std::printf("  INFO  detected %zu speaker(s)\n", speakers.size());

    // Embedding round-trip: same utterance embedded twice → cosine ≥ 0.90
    if (audio.samples.size() >= 32000) { // at least 2 s
        AudioChunk chunk_a, chunk_b;
        chunk_a.sample_rate = 16000; chunk_a.start_ms = 0;   chunk_a.end_ms = 1000;
        chunk_b.sample_rate = 16000; chunk_b.start_ms = 0;   chunk_b.end_ms = 1000;
        chunk_a.samples = std::vector<float>(audio.samples.begin(),
                                             audio.samples.begin() + 16000);
        chunk_b.samples = chunk_a.samples; // identical audio

        WeSpeakerEcapaModel round_trip_model;
        round_trip_model.load(g_model_path);
        auto ea = round_trip_model.embed(chunk_a);
        auto eb = round_trip_model.embed(chunk_b);

        EXPECT(ea.size() == eb.size());

        float dot = 0.0f;
        for (size_t i = 0; i < ea.size(); ++i) dot += ea[i] * eb[i];

        std::printf("  INFO  same-chunk cosine similarity = %.4f\n", dot);
        EXPECT(dot >= 0.90f); // identical audio must produce nearly identical embeddings
    }
}

// ---------------------------------------------------------------------------
// SpeakerVerifier via factory
// ---------------------------------------------------------------------------

TEST(SpeakerVerifier_factory_load_and_inspect) {
    if (g_model_path.empty()) {
        std::printf("  SKIP  (no model path)\n");
        return;
    }

    // Factory should pick WeSpeakerEcapaModel for voxceleb_ECAPA512_LM.onnx
    EXPECT(SpeakerModelFactory::detect_flavor(g_model_path) ==
           SpeakerModelFactory::Flavor::WeSpeaker);

    SpeakerVerifier sv;
    EXPECT(sv.load(g_model_path));

    auto meta = sv.inspect();
    EXPECT(meta.loaded);
    std::printf("  INFO  model graph:\n        %s\n",
                meta.describe().c_str());

    EXPECT(!meta.input_name.empty());
    EXPECT(!meta.output_name.empty());
    // Output should be [1, D] with D > 0
    EXPECT(meta.output_shape.size() >= 2);
    EXPECT(meta.output_shape.back() > 0);
}

TEST(SpeakerVerifier_same_chunk_cosine) {
    if (g_model_path.empty() || g_wav_path.empty()) {
        std::printf("  SKIP  (no model or wav path)\n");
        return;
    }

    auto audio = wav::read_wav_mono_16k(g_wav_path);
    if (audio.samples.size() < 16000) {
        std::printf("  SKIP  (audio too short)\n");
        return;
    }

    SpeakerVerifier sv;
    EXPECT(sv.load(g_model_path));

    // Extract the first second twice — identical audio must yield cosine ≥ 0.90
    AudioChunk chunk_a, chunk_b;
    chunk_a.sample_rate = 16000;
    chunk_a.samples = std::vector<float>(audio.samples.begin(),
                                          audio.samples.begin() + 16000);
    chunk_b = chunk_a;

    float cosine = sv.similarity(chunk_a, chunk_b);
    std::printf("  INFO  same-chunk cosine (SpeakerVerifier) = %.4f\n", cosine);
    EXPECT(cosine >= 0.90f);

    // verify() with a low threshold should pass
    EXPECT(sv.verify(chunk_a, chunk_b, 0.72f));
}

TEST(SpeakerVerifier_short_clip_valid_embedding) {
    if (g_model_path.empty()) {
        std::printf("  SKIP  (no model path)\n");
        return;
    }

    // 0.5 s of near-silence (8000 samples — well above one FBANK frame)
    AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.samples.assign(8000, 0.01f);

    SpeakerVerifier sv;
    EXPECT(sv.load(g_model_path));

    auto emb = sv.embed(chunk);
    EXPECT(!emb.empty());
    std::printf("  INFO  short-clip embedding dim = %zu\n", emb.size());

    // L2 norm should be ≈ 1.0 (model outputs normalised vectors)
    float norm = 0.0f;
    for (float x : emb) norm += x * x;
    norm = std::sqrt(norm);
    std::printf("  INFO  short-clip embedding L2 norm = %.4f\n", norm);
    EXPECT(norm > 0.9f && norm < 1.1f);
}

TEST(SpeakerVerifier_cross_chunk_cosine_bounded) {
    if (g_model_path.empty() || g_wav_path.empty()) {
        std::printf("  SKIP  (no model or wav path)\n");
        return;
    }

    auto audio = wav::read_wav_mono_16k(g_wav_path);
    if (audio.samples.size() < 64000) {
        std::printf("  SKIP  (audio too short for cross-chunk test)\n");
        return;
    }

    SpeakerVerifier sv;
    EXPECT(sv.load(g_model_path));

    // First second vs last second — may or may not be the same speaker
    AudioChunk a, b;
    a.sample_rate = b.sample_rate = 16000;
    a.samples = std::vector<float>(audio.samples.begin(),
                                    audio.samples.begin() + 16000);
    b.samples = std::vector<float>(audio.samples.end() - 16000,
                                    audio.samples.end());

    float cosine = sv.similarity(a, b);
    std::printf("  INFO  cross-chunk cosine (first vs last second) = %.4f\n", cosine);

    // L2-normalised vectors: cosine is always in [-1, 1]
    EXPECT(cosine >= -1.0f && cosine <= 1.0f);
}
#endif // DIARIZE_HAVE_MODEL

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Optional: argv[1] = model path, argv[2] = wav path
    if (argc > 1) g_model_path = argv[1];
    if (argc > 2) g_wav_path   = argv[2];

    // Run all registered tests now that globals are set.
    for (const auto& tc : test_registry()) {
        std::printf("[ RUN  ] %s\n", tc.name);
        tc.fn();
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
