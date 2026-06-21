/// Unit tests for the diarization pipeline.
///
/// Uses a tiny self-contained test runner so the file compiles without any
/// external test framework.  Each TEST(...) block runs independently; the
/// process exits with a non-zero code if any assertion fails.
///
/// Compile (example):
//   g++ -std=c++20 -I.. tests.cpp
//       ../SpeakerClusterManager.cpp
//       ../LabelSmoother.cpp
//       ../TranscriptFormatter.cpp
//       ../DiarizationEngine.cpp
//       -o run_tests && ./run_tests

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// Headers under test
#include <diarization/AudioChunk.h>
#include <diarization/DiarizationEngine.h>
#include <diarization/ISpeakerEmbeddingModel.h>
#include <diarization/LabelSmoother.h>
#include <diarization/ModelMetadata.h>
#include <diarization/SpeakerCluster.h>
#include <diarization/SpeakerClusterManager.h>
#include <diarization/TranscriptFormatter.h>

// models/SpeakerModelFactory.h contains detect_flavor() inline — no ONNX link needed
#include "models/SpeakerModelFactory.h"

// ---------------------------------------------------------------------------
// Minimal test runner
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond) \
    do { \
        if (!(cond)) { \
            std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail; \
        } else { \
            ++g_pass; \
        } \
    } while (0)

#define TEST(name) \
    static void test_##name(); \
    struct _Reg_##name { \
        _Reg_##name() { \
            std::printf("[ RUN  ] " #name "\n"); \
            test_##name(); \
        } \
    } _reg_##name; \
    static void test_##name()

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an L2-normalised embedding from a direction vector.
static std::vector<float> unit(std::vector<float> v) {
    float sq = 0.f;
    for (float x : v) sq += x * x;
    float n = std::sqrt(sq);
    for (float& x : v) x /= n;
    return v;
}

/// A stub model that returns a fixed embedding for every chunk.
class FixedEmbeddingModel : public ISpeakerEmbeddingModel {
public:
    explicit FixedEmbeddingModel(std::vector<float> emb)
        : emb_(std::move(emb)) {}

    bool load(const std::string&) override { return true; }
    std::vector<float> embed(const AudioChunk&) override { return emb_; }
    int sample_rate()   const override { return 16000; }
    int embedding_dim() const override { return static_cast<int>(emb_.size()); }

private:
    std::vector<float> emb_;
};

/// Build a mono AudioBuffer containing `duration_ms` ms of silence at 16 kHz.
static AudioBuffer make_silent_buffer(int64_t duration_ms) {
    AudioBuffer buf;
    buf.sample_rate = 16000;
    buf.channels    = 1;
    buf.samples.assign(static_cast<size_t>(duration_ms * 16000 / 1000), 0.0f);
    return buf;
}

/// Build a WhisperSegment spanning [start_ms, end_ms).
static WhisperSegment make_seg(int64_t start_ms, int64_t end_ms,
                               const std::string& text = "hello",
                               float conf = 0.9f) {
    WhisperSegment s;
    s.start_ms   = start_ms;
    s.end_ms     = end_ms;
    s.text       = text;
    s.confidence = conf;
    return s;
}

// ---------------------------------------------------------------------------
// TEST: CosineSimilarity
// ---------------------------------------------------------------------------
TEST(CosineSimilarity) {
    // Two identical unit vectors → similarity == 1.
    auto a = unit({1.f, 0.f, 0.f});
    auto b = unit({1.f, 0.f, 0.f});

    // Exercise through SpeakerClusterManager by checking that two identical
    // embeddings always land in the same cluster.
    SpeakerClusterManager mgr(ClusteringOptions{0.5f, 0});
    std::string lbl1 = mgr.assign(a);
    std::string lbl2 = mgr.assign(b);
    EXPECT(lbl1 == lbl2); // same cluster

    // Orthogonal vectors → similarity == 0, below any reasonable threshold.
    auto c = unit({0.f, 1.f, 0.f});
    SpeakerClusterManager mgr2(ClusteringOptions{0.5f, 0});
    std::string lbl_a = mgr2.assign(a);
    std::string lbl_c = mgr2.assign(c);
    EXPECT(lbl_a != lbl_c); // different clusters

    // Opposite vector → similarity == -1, definitely different cluster.
    auto d = unit({-1.f, 0.f, 0.f});
    SpeakerClusterManager mgr3(ClusteringOptions{0.5f, 0});
    std::string lbl_aa = mgr3.assign(a);
    std::string lbl_d  = mgr3.assign(d);
    EXPECT(lbl_aa != lbl_d);
}

// ---------------------------------------------------------------------------
// TEST: ClusterCreation
// ---------------------------------------------------------------------------
TEST(ClusterCreation) {
    SpeakerClusterManager mgr;

    // First assignment should always create SPEAKER_00.
    auto emb = unit({1.f, 0.f, 0.f});
    std::string lbl = mgr.assign(emb);
    EXPECT(lbl == "SPEAKER_00");
    EXPECT(mgr.clusters().size() == 1u);

    // Clearly different embedding should create SPEAKER_01.
    auto emb2 = unit({0.f, 1.f, 0.f});
    std::string lbl2 = mgr.assign(emb2);
    EXPECT(lbl2 == "SPEAKER_01");
    EXPECT(mgr.clusters().size() == 2u);
}

// ---------------------------------------------------------------------------
// TEST: ClusterReuse
// ---------------------------------------------------------------------------
TEST(ClusterReuse) {
    // Embeddings that are very close (cosine ≈ 1) should map to the same cluster.
    SpeakerClusterManager mgr(ClusteringOptions{0.72f, 0});

    auto base = unit({1.0f, 0.0f, 0.0f});
    // A slightly perturbed version (angle << threshold).
    auto near = unit({1.0f, 0.01f, 0.01f});

    std::string lbl1 = mgr.assign(base);
    std::string lbl2 = mgr.assign(near);
    EXPECT(lbl1 == lbl2);
    EXPECT(mgr.clusters().size() == 1u);

    // The centroid update should not have zeroed out the cluster.
    EXPECT(!mgr.clusters()[0].centroid.empty());
}

// ---------------------------------------------------------------------------
// TEST: MaxSpeakersLimit
// ---------------------------------------------------------------------------
TEST(MaxSpeakersLimit) {
    // With max_speakers == 2, a third distinct speaker must map to one of the
    // existing two clusters, not create a third.
    SpeakerClusterManager mgr(ClusteringOptions{0.99f, 2}); // very tight threshold

    auto e0 = unit({1.0f,  0.0f, 0.0f});
    auto e1 = unit({0.0f,  1.0f, 0.0f});
    auto e2 = unit({0.0f,  0.0f, 1.0f}); // orthogonal to both

    mgr.assign(e0);
    mgr.assign(e1);
    EXPECT(mgr.clusters().size() == 2u);

    mgr.assign(e2); // should NOT create a third cluster
    EXPECT(mgr.clusters().size() == 2u);
}

// ---------------------------------------------------------------------------
// TEST: LabelSmoothing
// ---------------------------------------------------------------------------
TEST(LabelSmoothing) {
    LabelSmoother smoother(500); // flip threshold: 500 ms

    // A 300 ms SPEAKER_01 sandwiched between two SPEAKER_00 segments.
    std::vector<TranscriptSegment> segs = {
        {0,    2000, "SPEAKER_00", "Hello",  0.9f},
        {2000, 2300, "SPEAKER_01", "yeah",   0.5f}, // 300 ms flip
        {2300, 4000, "SPEAKER_00", "anyway", 0.9f},
    };

    smoother.smooth(segs);

    EXPECT(segs[0].speaker == "SPEAKER_00");
    EXPECT(segs[1].speaker == "SPEAKER_00"); // relabelled
    EXPECT(segs[2].speaker == "SPEAKER_00");

    // A 600 ms flip (> threshold) should NOT be relabelled.
    std::vector<TranscriptSegment> segs2 = {
        {0,    2000, "SPEAKER_00", "Hello",  0.9f},
        {2000, 2600, "SPEAKER_01", "yeah",   0.5f}, // 600 ms — keep
        {2600, 4000, "SPEAKER_00", "anyway", 0.9f},
    };

    smoother.smooth(segs2);
    EXPECT(segs2[1].speaker == "SPEAKER_01"); // unchanged

    // ASSISTANT labels must never be relabelled.
    std::vector<TranscriptSegment> segs3 = {
        {0,    2000, "SPEAKER_00", "Hello",      0.9f},
        {2000, 2200, "ASSISTANT",  "Sure thing", 1.0f}, // 200 ms
        {2200, 4000, "SPEAKER_00", "Thanks",     0.9f},
    };

    smoother.smooth(segs3);
    EXPECT(segs3[1].speaker == "ASSISTANT"); // protected
}

// ---------------------------------------------------------------------------
// TEST: WindowGeneration
// ---------------------------------------------------------------------------
TEST(WindowGeneration) {
    // 4-second chunk at 16 kHz → 64000 samples.
    AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.start_ms    = 0;
    chunk.end_ms      = 4000;
    chunk.samples.assign(64000, 0.0f);

    // window=1500 ms, hop=750 ms:
    //   window = 24000 samples, hop = 12000 samples
    //   windows start at 0, 12000, 24000, 36000, 40000 (last full window at 40000+24000=64000)
    //   expected: floor((64000 - 24000) / 12000) + 1 = 4 windows
    // Actually: start positions: 0, 12000, 24000, 36000 → 4 windows (36000+24000=60000 ≤ 64000), 
    //           48000+24000=72000 > 64000 → stop. So 4 windows.

    // Use DiarizationEngine's make_windows via a full process call.
    // Instead, test indirectly: build a DiarizationEngine with a fixed model and
    // count how many embed() calls are made.
    class CountingModel : public ISpeakerEmbeddingModel {
    public:
        mutable int call_count = 0;
        bool load(const std::string&) override { return true; }
        std::vector<float> embed(const AudioChunk&) override {
            ++call_count;
            return unit({1.f, 0.f, 0.f});
        }
        int sample_rate()   const override { return 16000; }
        int embedding_dim() const override { return 3; }
        static std::vector<float> unit(std::vector<float> v) {
            float sq = 0.f; for (float x : v) sq += x*x;
            float n = std::sqrt(sq); for (float& x : v) x /= n;
            return v;
        }
    };

    auto* raw_model  = new CountingModel();
    auto  model_ptr1 = std::unique_ptr<ISpeakerEmbeddingModel>(raw_model);
    DiarizationEngine engine{std::move(model_ptr1)};

    AudioBuffer audio = make_silent_buffer(4000);
    std::vector<WhisperSegment> wsegs = { make_seg(0, 4000) };

    DiarizationOptions opts;
    opts.speaker_min_ms    = 800;
    opts.speaker_window_ms = 1500;
    opts.speaker_hop_ms    = 750;

    engine.process(audio, wsegs, opts);

    // 4 windows expected for a 4000 ms segment with 1500/750 window/hop.
    EXPECT(raw_model->call_count == 4);
}

// ---------------------------------------------------------------------------
// TEST: TranscriptMerge
// ---------------------------------------------------------------------------
TEST(TranscriptMerge) {
    // Two segments with clearly different embeddings should get different labels.
    auto model_a = unit({1.f, 0.f, 0.f});
    auto model_b = unit({0.f, 1.f, 0.f});

    // Alternating model: first segment → emb_a, second → emb_b.
    class AlternatingModel : public ISpeakerEmbeddingModel {
    public:
        std::vector<float> a, b;
        mutable int idx = 0;
        bool load(const std::string&) override { return true; }
        std::vector<float> embed(const AudioChunk&) override {
            return (idx++ % 2 == 0) ? a : b;
        }
        int sample_rate()   const override { return 16000; }
        int embedding_dim() const override { return 3; }
    };

    auto* raw = new AlternatingModel();
    raw->a    = model_a;
    raw->b    = model_b;

    auto model_ptr2 = std::unique_ptr<ISpeakerEmbeddingModel>(raw);
    DiarizationEngine engine{std::move(model_ptr2)};

    // Two long segments (2 s each) so they meet the min_ms threshold.
    AudioBuffer audio = make_silent_buffer(4000);
    std::vector<WhisperSegment> wsegs = {
        make_seg(0,    2000, "Hello"),
        make_seg(2000, 4000, "World"),
    };

    DiarizationOptions opts;
    opts.speaker_min_ms    = 800;
    opts.speaker_window_ms = 4000; // window > segment → each segment embeds once
    opts.speaker_hop_ms    = 2000;
    opts.speaker_threshold = 0.72f;

    auto results = engine.process(audio, wsegs, opts);

    EXPECT(results.size() == 2u);
    EXPECT(results[0].text == "Hello");
    EXPECT(results[1].text == "World");
    // The two segments must have different speaker labels.
    EXPECT(results[0].speaker != results[1].speaker);
    EXPECT(results[0].speaker == "SPEAKER_00");
    EXPECT(results[1].speaker == "SPEAKER_01");
}

// ---------------------------------------------------------------------------
// TEST: CentroidRunningMean
// ---------------------------------------------------------------------------
TEST(CentroidRunningMean) {
    // Verify the incremental mean stays unit-length and converges correctly.

    SpeakerClusterManager mgr(ClusteringOptions{-1.0f, 0}); // accept everything

    // Assign the same unit vector 5 times.  Centroid must remain that vector.
    auto e = unit({1.0f, 0.0f, 0.0f});
    for (int i = 0; i < 5; ++i) mgr.assign(e);

    EXPECT(mgr.clusters().size() == 1u);
    const auto& c = mgr.clusters()[0].centroid;
    // Centroid must still be approximately (1, 0, 0).
    EXPECT(std::abs(c[0] - 1.0f) < 1e-5f);
    EXPECT(std::abs(c[1]) < 1e-5f);
    EXPECT(std::abs(c[2]) < 1e-5f);

    // After 1 assignment of (1,0,0) and 1 of (0,1,0),
    // raw mean = (0.5, 0.5, 0.0), normalised = (√2/2, √2/2, 0).
    SpeakerClusterManager mgr2(ClusteringOptions{-1.0f, 0});
    auto ex = unit({1.0f, 0.0f, 0.0f});
    auto ey = unit({0.0f, 1.0f, 0.0f});
    mgr2.assign(ex);
    mgr2.assign(ey);

    EXPECT(mgr2.clusters().size() == 1u);
    const auto& c2 = mgr2.clusters()[0].centroid;

    // Centroid must be unit-length.
    float norm_sq = 0.f;
    for (float v : c2) norm_sq += v * v;
    EXPECT(std::abs(norm_sq - 1.0f) < 1e-5f);

    // Components should be equal (45° bisector).
    const float expected = std::sqrt(2.0f) / 2.0f; // ≈ 0.7071
    EXPECT(std::abs(c2[0] - expected) < 1e-4f);
    EXPECT(std::abs(c2[1] - expected) < 1e-4f);
    EXPECT(std::abs(c2[2]) < 1e-5f);
}

// ---------------------------------------------------------------------------
// TEST: SpeakerTransitionInSegment
// ---------------------------------------------------------------------------
TEST(SpeakerTransitionInSegment) {
    // Documents the known majority-vote limitation:
    //
    //   Whisper returns one 4-second segment spanning two speakers.
    //   The embedding model alternates A/B across four windows (2-2 tie).
    //
    //   Expected behaviour: the engine assigns one stable, deterministic label
    //   (std::map iteration order = alphabetical = SPEAKER_00 wins ties).
    //   The segment will NOT be split — that would require a different algorithm
    //   (e.g. change-point detection).  This test documents the current behaviour
    //   so a future change-point pass can be verified against it.

    class AltModel : public ISpeakerEmbeddingModel {
    public:
        mutable int idx = 0;
        bool load(const std::string&) override { return true; }
        std::vector<float> embed(const AudioChunk&) override {
            // Even windows → SPEAKER_00 direction, odd → SPEAKER_01 direction.
            return (idx++ % 2 == 0)
                ? unit({1.0f, 0.0f, 0.0f})   // A
                : unit({0.0f, 1.0f, 0.0f});   // B
        }
        int sample_rate()   const override { return 16000; }
        int embedding_dim() const override { return 3; }
        static std::vector<float> unit(std::vector<float> v) {
            float sq = 0.f; for (float x : v) sq += x*x;
            float n = std::sqrt(sq); for (float& x : v) x /= n;
            return v;
        }
    };

    auto model_ptr = std::unique_ptr<ISpeakerEmbeddingModel>(new AltModel{});
    DiarizationEngine engine{std::move(model_ptr)};

    AudioBuffer audio = make_silent_buffer(4000);
    // Single 4-second segment (span covers both speakers).
    std::vector<WhisperSegment> wsegs = { make_seg(0, 4000, "mixed") };

    DiarizationOptions opts;
    opts.speaker_min_ms    = 800;
    opts.speaker_window_ms = 1500;  // 4 windows: A B A B
    opts.speaker_hop_ms    = 750;
    opts.speaker_threshold = 0.72f;

    auto results = engine.process(audio, wsegs, opts);

    EXPECT(results.size() == 1u);
    // Result must be a valid speaker label (not UNKNOWN) and must be stable.
    EXPECT(results[0].speaker == "SPEAKER_00" || results[0].speaker == "SPEAKER_01");
    // Document: with a 2-2 tie and std::map alpha ordering, SPEAKER_00 wins.
    EXPECT(results[0].speaker == "SPEAKER_00");
}

// ---------------------------------------------------------------------------
// TEST: JsonSchemaFields
// ---------------------------------------------------------------------------
TEST(JsonSchemaFields) {
    // Verify the JSON output contains every field name specified in the ticket.

    std::vector<TranscriptSegment> segs = {
        {120, 1840, "SPEAKER_00", "Open the settings.", 0.91f},
        {2100, 3500, "SPEAKER_01", "Yes, do it.", 0.88f},
    };

    DiarizationMeta meta;
    meta.enabled       = true;
    meta.speaker_model = "dist/models/speaker-ecapa.onnx";
    meta.threshold     = 0.72f;
    meta.speakers      = {{"SPEAKER_00", 4}, {"SPEAKER_01", 2}};

    std::string json = TranscriptFormatter::format_json(
        segs, meta,
        "whisper",
        "dist/models/ggml-base.en.bin",
        3000);

    // Top-level required fields.
    EXPECT(json.find("\"provider\"")    != std::string::npos);
    EXPECT(json.find("\"model\"")       != std::string::npos);
    EXPECT(json.find("\"duration_ms\"") != std::string::npos);
    EXPECT(json.find("\"diarization\"") != std::string::npos);
    EXPECT(json.find("\"segments\"")    != std::string::npos);
    EXPECT(json.find("\"text\"")        != std::string::npos);

    // Diarization sub-object fields.
    EXPECT(json.find("\"enabled\"")      != std::string::npos);
    EXPECT(json.find("\"speaker_model\"")!= std::string::npos);
    EXPECT(json.find("\"threshold\"")    != std::string::npos);
    EXPECT(json.find("\"speakers\"")     != std::string::npos);

    // Speaker entry fields.
    EXPECT(json.find("\"id\"")       != std::string::npos);
    EXPECT(json.find("\"segments\"") != std::string::npos);

    // Segment entry fields.
    EXPECT(json.find("\"start_ms\"")   != std::string::npos);
    EXPECT(json.find("\"end_ms\"")     != std::string::npos);
    EXPECT(json.find("\"speaker\"")    != std::string::npos);
    EXPECT(json.find("\"confidence\"") != std::string::npos);

    // Exact values from the ticket example.
    EXPECT(json.find("\"whisper\"")  != std::string::npos);
    EXPECT(json.find("\"SPEAKER_00\"") != std::string::npos);
    EXPECT(json.find("\"SPEAKER_01\"") != std::string::npos);
    EXPECT(json.find("3000")           != std::string::npos);

    // Plain-text field is the concatenated speaker: text form.
    EXPECT(json.find("SPEAKER_00: Open the settings.") != std::string::npos);
}

// ---------------------------------------------------------------------------
// ModelMetadata
// ---------------------------------------------------------------------------

TEST(ModelMetadata_describe_unloaded) {
    ModelMetadata m;
    // Default-constructed: loaded = false → fixed placeholder string
    EXPECT(m.describe() == "(model not loaded)");
}

TEST(ModelMetadata_describe_loaded) {
    ModelMetadata m;
    m.input_name   = "feats";
    m.output_name  = "embs";
    m.input_shape  = {1, -1, 80};   // -1 = dynamic dim → rendered as '?'
    m.output_shape = {1, 192};
    m.loaded       = true;
    auto d = m.describe();
    // Should contain both node names
    EXPECT(d.find("feats") != std::string::npos);
    EXPECT(d.find("embs")  != std::string::npos);
    // Dynamic dim rendered as '?'
    EXPECT(d.find('?') != std::string::npos);
    // Known dim present
    EXPECT(d.find("192") != std::string::npos);
    EXPECT(d.find("80")  != std::string::npos);
}

// ---------------------------------------------------------------------------
// SpeakerModelFactory — detect_flavor() is inline; no ONNX linkage required
// ---------------------------------------------------------------------------

TEST(SpeakerModelFactory_detect_wespeaker) {
    using F = SpeakerModelFactory::Flavor;
    EXPECT(SpeakerModelFactory::detect_flavor("wespeaker/voxceleb_ECAPA512_LM.onnx") == F::WeSpeaker);
    EXPECT(SpeakerModelFactory::detect_flavor("models/voxceleb_ecapa.onnx")          == F::WeSpeaker);
    EXPECT(SpeakerModelFactory::detect_flavor("ECAPA512_model.onnx")                 == F::WeSpeaker);
    // Case-insensitive
    EXPECT(SpeakerModelFactory::detect_flavor("WeSpeaker/ECAPA.onnx")                == F::WeSpeaker);
}

TEST(SpeakerModelFactory_detect_speechbrain) {
    using F = SpeakerModelFactory::Flavor;
    EXPECT(SpeakerModelFactory::detect_flavor("speechbrain_ecapa.onnx")             == F::SpeechBrain);
    EXPECT(SpeakerModelFactory::detect_flavor("models/SpeechBrain_TDNN.onnx")       == F::SpeechBrain);
    EXPECT(SpeakerModelFactory::detect_flavor("/data/SpeechBrain/model.onnx")       == F::SpeechBrain);
    // Case-insensitive
    EXPECT(SpeakerModelFactory::detect_flavor("SPEECHBRAIN_ECAPA.onnx")             == F::SpeechBrain);
}

TEST(SpeakerModelFactory_detect_unknown) {
    using F = SpeakerModelFactory::Flavor;
    EXPECT(SpeakerModelFactory::detect_flavor("some_model.onnx")        == F::Unknown);
    EXPECT(SpeakerModelFactory::detect_flavor("")                        == F::Unknown);
    EXPECT(SpeakerModelFactory::detect_flavor("ecapa_tdnn.onnx")        == F::Unknown);
    EXPECT(SpeakerModelFactory::detect_flavor("speaker_embedding.onnx") == F::Unknown);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    // All TEST() blocks run via static constructors above.
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
