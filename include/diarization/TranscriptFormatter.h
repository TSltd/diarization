#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A single Whisper output segment (no speaker info yet).
struct WhisperSegment {
    int64_t     start_ms   = 0;
    int64_t     end_ms     = 0;
    std::string text;
    float       confidence = 0.0f;
};

/// A transcript segment after diarization — has a speaker label.
struct TranscriptSegment {
    int64_t     start_ms   = 0;
    int64_t     end_ms     = 0;
    std::string speaker;   ///< "SPEAKER_00", "SPEAKER_01", "ASSISTANT", "UNKNOWN"
    std::string text;
    float       confidence = 0.0f;
};

/// Raw decoded audio.
struct AudioBuffer {
    int                sample_rate = 16000;
    int                channels    = 1;
    std::vector<float> samples;    ///< Interleaved if channels > 1
};

// ---------------------------------------------------------------------------
// Output formats
// ---------------------------------------------------------------------------

enum class SpeakerFormat {
    Inline, ///< [SPEAKER_00 00:00.120-00:01.840] text
    Json,   ///< Full JSON object (see ticket schema)
    Srt,    ///< SubRip subtitles
    Vtt,    ///< WebVTT
};

/// Parse a format name string ("inline", "json", "srt", "vtt").
/// Throws std::invalid_argument for unknown values.
SpeakerFormat parse_speaker_format(const std::string& name);

// ---------------------------------------------------------------------------
// Formatter
// ---------------------------------------------------------------------------

/// Holds diarization metadata for the JSON "diarization" object.
struct DiarizationMeta {
    bool        enabled       = true;
    std::string speaker_model;
    float       threshold     = 0.72f;

    struct SpeakerInfo {
        std::string id;
        int         segments = 0;
    };
    std::vector<SpeakerInfo> speakers;
};

class TranscriptFormatter {
public:
    /// Render segments as the four supported output formats.
    static std::string format_inline(const std::vector<TranscriptSegment>& segs);
    static std::string format_srt   (const std::vector<TranscriptSegment>& segs);
    static std::string format_vtt   (const std::vector<TranscriptSegment>& segs);

    /// Full JSON output including the diarization metadata block.
    static std::string format_json(
        const std::vector<TranscriptSegment>& segs,
        const DiarizationMeta&                meta,
        const std::string&                    provider,
        const std::string&                    model_path,
        int64_t                               duration_ms);

    /// Dispatch to the correct formatter.
    static std::string format(
        const std::vector<TranscriptSegment>& segs,
        SpeakerFormat                         fmt,
        const DiarizationMeta&                meta    = {},
        const std::string&                    provider = "whisper",
        const std::string&                    model_path = "",
        int64_t                               duration_ms = 0);

private:
    /// Format a millisecond timestamp as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT).
    static std::string ms_to_timestamp(int64_t ms, char sep);

    /// Format a millisecond timestamp as MM:SS.mmm (inline).
    static std::string ms_to_short(int64_t ms);
};
