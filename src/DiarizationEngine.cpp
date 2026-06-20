#include "DiarizationEngine.h"

#include <algorithm>
#include <map>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DiarizationEngine::DiarizationEngine(
    std::unique_ptr<ISpeakerEmbeddingModel> model)
    : model_(std::move(model)) {

    if (!model_)
        throw std::invalid_argument("DiarizationEngine: model must not be null");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<TranscriptSegment> DiarizationEngine::process(
    const AudioBuffer&                 audio,
    const std::vector<WhisperSegment>& whisper_segments,
    const DiarizationOptions&          options) {

    // Reconfigure the cluster manager with the current options.
    cluster_manager_ = SpeakerClusterManager({
        options.speaker_threshold,
        options.speaker_max
    });

    // Reconfigure the label smoother.
    smoother_ = LabelSmoother(options.smoother_min_ms);

    std::vector<TranscriptSegment> results;
    results.reserve(whisper_segments.size());

    for (const auto& wseg : whisper_segments) {
        const int64_t dur_ms = wseg.end_ms - wseg.start_ms;

        // --- Step 1: Build the output segment (text fields copied now). ----
        TranscriptSegment out;
        out.start_ms   = wseg.start_ms;
        out.end_ms     = wseg.end_ms;
        out.text       = wseg.text;
        out.confidence = wseg.confidence;
        out.speaker    = "UNKNOWN";

        // --- Step 2: Skip segments that are too short to embed. ------------
        if (dur_ms < options.speaker_min_ms) {
            results.push_back(std::move(out));
            continue;
        }

        // --- Step 3: Slice audio for this segment. -------------------------
        AudioChunk segment_audio = slice(audio, wseg.start_ms, wseg.end_ms);
        if (segment_audio.empty()) {
            results.push_back(std::move(out));
            continue;
        }

        // --- Step 4: Generate overlapping windows. -------------------------
        auto windows = make_windows(segment_audio,
                                    options.speaker_window_ms,
                                    options.speaker_hop_ms);
        if (windows.empty())
            windows.push_back(segment_audio); // fall back to whole chunk

        // --- Step 5: Embed each window and assign to a cluster. ------------
        std::map<std::string, int> votes;
        for (const auto& win : windows) {
            try {
                auto embedding = model_->embed(win);
                if (!embedding.empty()) {
                    std::string label = cluster_manager_.assign(embedding);
                    votes[label]++;
                }
            } catch (...) {
                // Embedding failure for one window is non-fatal.
            }
        }

        // --- Step 6: Majority vote across all windows. ---------------------
        if (!votes.empty()) {
            auto best = std::max_element(votes.begin(), votes.end(),
                [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });
            out.speaker = best->first;
        }

        results.push_back(std::move(out));
    }

    // --- Step 7: Smooth isolated label flips. ------------------------------
    smoother_.smooth(results);

    return results;
}

const std::vector<SpeakerCluster>& DiarizationEngine::clusters() const {
    return cluster_manager_.clusters();
}

void DiarizationEngine::reset() {
    cluster_manager_.reset();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

AudioChunk DiarizationEngine::slice(const AudioBuffer& audio,
                                    int64_t start_ms, int64_t end_ms) {
    if (audio.sample_rate <= 0 || audio.samples.empty())
        return {};

    // For multi-channel audio, take channel 0 (left) as the mono signal.
    const int channels = std::max(1, audio.channels);

    const int64_t start_sample = start_ms * audio.sample_rate / 1000;
    const int64_t end_sample   = end_ms   * audio.sample_rate / 1000;

    // The total number of mono frames available.
    const int64_t total_frames = static_cast<int64_t>(
        audio.samples.size()) / channels;

    const int64_t clamped_start = std::clamp(start_sample, int64_t{0}, total_frames);
    const int64_t clamped_end   = std::clamp(end_sample,   int64_t{0}, total_frames);

    if (clamped_start >= clamped_end) return {};

    AudioChunk chunk;
    chunk.sample_rate = audio.sample_rate;
    chunk.start_ms    = start_ms;
    chunk.end_ms      = end_ms;
    chunk.samples.reserve(static_cast<size_t>(clamped_end - clamped_start));

    for (int64_t f = clamped_start; f < clamped_end; ++f)
        chunk.samples.push_back(audio.samples[f * channels]); // channel 0

    return chunk;
}

std::vector<AudioChunk> DiarizationEngine::make_windows(
    const AudioChunk& chunk, int window_ms, int hop_ms) {

    if (window_ms <= 0 || hop_ms <= 0 || chunk.sample_rate <= 0)
        return {};

    const int win_samples = window_ms * chunk.sample_rate / 1000;
    const int hop_samples = hop_ms    * chunk.sample_rate / 1000;
    const int total       = static_cast<int>(chunk.samples.size());

    if (total <= win_samples) {
        // Chunk is shorter than one window — return it as-is.
        return {};
    }

    std::vector<AudioChunk> windows;
    for (int start = 0; start + win_samples <= total; start += hop_samples) {
        AudioChunk win;
        win.sample_rate = chunk.sample_rate;
        win.start_ms    = chunk.start_ms + static_cast<int64_t>(start) * 1000 / chunk.sample_rate;
        win.end_ms      = win.start_ms   + window_ms;
        win.samples     = std::vector<float>(
            chunk.samples.begin() + start,
            chunk.samples.begin() + start + win_samples);
        windows.push_back(std::move(win));
    }
    return windows;
}
