#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <diarization/AudioChunk.h>
#include <diarization/ISpeakerEmbeddingModel.h>
#include <diarization/LabelSmoother.h>
#include <diarization/SpeakerClusterManager.h>
#include <diarization/TranscriptFormatter.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct DiarizationOptions {
    float   speaker_threshold = 0.72f;  ///< Cosine similarity threshold
    int     speaker_window_ms = 1500;   ///< Embedding window size
    int     speaker_hop_ms    = 750;    ///< Window hop size
    int     speaker_min_ms    = 800;    ///< Minimum segment length to embed
    int     speaker_max       = 0;      ///< Max speakers (0 = unlimited)
    int64_t smoother_min_ms   = 500;    ///< Flip duration threshold for LabelSmoother
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Orchestrates the full diarization pipeline:
///
///   WhisperSegment list + AudioBuffer
///     → candidate speech chunks (filtered by speaker_min_ms)
///     → overlapping windows (speaker_window_ms / speaker_hop_ms)
///     → speaker embeddings via ISpeakerEmbeddingModel
///     → cluster assignment via SpeakerClusterManager
///     → majority-vote label per WhisperSegment
///     → label smoothing via LabelSmoother
///     → std::vector<TranscriptSegment>
///
/// The engine does **not** handle ASSISTANT/TTS injection — that is merged by
/// the application layer after process() returns.
class DiarizationEngine {
public:
    explicit DiarizationEngine(
        std::unique_ptr<ISpeakerEmbeddingModel> model);

    /// Run diarization and return labelled transcript segments.
    /// The returned segments are in chronological order and have the same text
    /// as the input WhisperSegments.
    std::vector<TranscriptSegment> process(
        const AudioBuffer&                  audio,
        const std::vector<WhisperSegment>&  whisper_segments,
        const DiarizationOptions&           options = {});

    /// Read-only access to the current speaker clusters (e.g. for status output).
    const std::vector<SpeakerCluster>& clusters() const;

    /// Reset all cluster state (call between independent recordings).
    void reset();

private:
    /// Slice a window of audio from the buffer.
    static AudioChunk slice(const AudioBuffer& audio,
                            int64_t start_ms, int64_t end_ms);

    /// Generate overlapping windows from a chunk.
    static std::vector<AudioChunk> make_windows(const AudioChunk& chunk,
                                                int window_ms,
                                                int hop_ms);

    std::unique_ptr<ISpeakerEmbeddingModel> model_;
    SpeakerClusterManager                  cluster_manager_;
    LabelSmoother                          smoother_;
};
