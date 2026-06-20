#pragma once

// ---------------------------------------------------------------------------
// WhisperAdapter.h
//
// Bridges whisper.cpp output to the types expected by DiarizationEngine /
// TranscriptFormatter.
//
// Key whisper.cpp timestamp facts
// ────────────────────────────────
//  • whisper_full_get_segment_t0(ctx, i)  → int64_t  centiseconds  (1/100 s)
//  • whisper_full_get_segment_t1(ctx, i)  → int64_t  centiseconds
//  • Multiply by 10 to obtain milliseconds.
//  • whisper_full_get_segment_prob(ctx, i) → float in [0, 1]
//  • whisper_full_get_segment_text(ctx, i) → const char*  (UTF-8, always non-null)
//
// The adapter exposes a single function convert_segments() that accepts the
// raw whisper_context* and returns a std::vector<WhisperSegment> ready for
// DiarizationEngine::process().  An optional filter is applied:
//   • Empty or whitespace-only text → dropped
//   • end_ms <= start_ms            → dropped (malformed output)
//   • Duplicate consecutive segments (same text + start) → dropped
//
// whisper.cpp header is guarded so this file compiles in unit tests that do
// not link against whisper.
// ---------------------------------------------------------------------------

#include "TranscriptFormatter.h"   // WhisperSegment

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward-declare whisper types so we can use them without including the
// full whisper.h in every TU.  When whisper.h IS available (i.e. at link
// time in the real binary) the struct layout is compatible because we only
// use the opaque pointer through the whisper_* API.
// ---------------------------------------------------------------------------
#ifndef WHISPER_H
struct whisper_context;   // opaque — only used as a pointer below
#endif

// ---------------------------------------------------------------------------
// centiseconds → milliseconds
// ---------------------------------------------------------------------------

/// Convert a raw whisper timestamp (centiseconds) to milliseconds.
inline constexpr int64_t whisper_ts_to_ms(int64_t centiseconds) noexcept {
    return centiseconds * 10;
}

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

/// Returns true if the string is empty or contains only ASCII whitespace.
inline bool is_blank(const std::string& s) noexcept {
    return std::all_of(s.begin(), s.end(),
                       [](unsigned char c){ return std::isspace(c); });
}

// ---------------------------------------------------------------------------
// convert_segments — main entry point
// ---------------------------------------------------------------------------

#ifdef WHISPER_H   // only compiled when whisper.h is available

#include "whisper.h"

/// Extract all segments from a completed whisper_full() run and return them
/// as WhisperSegment objects with millisecond timestamps.
///
/// @param ctx        A whisper_context* after a successful whisper_full() call.
/// @param min_dur_ms Segments shorter than this are discarded (default 0 = keep all).
/// @return           Chronologically ordered, filtered segments.
inline std::vector<WhisperSegment> convert_segments(whisper_context* ctx,
                                                    int64_t min_dur_ms = 0)
{
    if (!ctx) return {};

    const int n = whisper_full_n_segments(ctx);
    if (n <= 0) return {};

    std::vector<WhisperSegment> out;
    out.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const char* raw_text = whisper_full_get_segment_text(ctx, i);
        std::string text = raw_text ? raw_text : "";

        // Drop blank segments
        if (is_blank(text)) continue;

        const int64_t start_ms = whisper_ts_to_ms(whisper_full_get_segment_t0(ctx, i));
        const int64_t end_ms   = whisper_ts_to_ms(whisper_full_get_segment_t1(ctx, i));

        // Drop malformed (zero or negative duration)
        if (end_ms <= start_ms) continue;

        // Drop segments below minimum duration
        if ((end_ms - start_ms) < min_dur_ms) continue;

        // Drop exact duplicate of the previous segment (whisper occasionally
        // repeats a segment at the same timestamp when repetition-penalty is off)
        if (!out.empty() &&
            out.back().start_ms == start_ms &&
            out.back().text     == text)
        {
            continue;
        }

        WhisperSegment seg;
        seg.start_ms   = start_ms;
        seg.end_ms     = end_ms;
        seg.text       = std::move(text);
        seg.confidence = whisper_full_get_segment_prob(ctx, i);
        out.push_back(std::move(seg));
    }

    return out;
}

#endif  // WHISPER_H

// ---------------------------------------------------------------------------
// Offline / test helper — build WhisperSegments from raw values without
// needing a live whisper_context.  Applies the same filtering as above.
// ---------------------------------------------------------------------------

struct RawSegment {
    int64_t     t0_cs;          ///< centiseconds (raw whisper timestamp)
    int64_t     t1_cs;          ///< centiseconds (raw whisper timestamp)
    std::string text;
    float       prob = 1.0f;
};

/// Convert a hand-built list of RawSegment (e.g. for testing or offline replay)
/// into WhisperSegments, applying the same filtering rules as convert_segments().
inline std::vector<WhisperSegment> convert_raw_segments(
    const std::vector<RawSegment>& raw,
    int64_t min_dur_ms = 0)
{
    std::vector<WhisperSegment> out;
    out.reserve(raw.size());

    for (const auto& r : raw) {
        if (is_blank(r.text)) continue;

        const int64_t start_ms = whisper_ts_to_ms(r.t0_cs);
        const int64_t end_ms   = whisper_ts_to_ms(r.t1_cs);

        if (end_ms <= start_ms) continue;
        if ((end_ms - start_ms) < min_dur_ms) continue;

        if (!out.empty() &&
            out.back().start_ms == start_ms &&
            out.back().text     == r.text)
        {
            continue;
        }

        WhisperSegment seg;
        seg.start_ms   = start_ms;
        seg.end_ms     = end_ms;
        seg.text       = r.text;
        seg.confidence = r.prob;
        out.push_back(std::move(seg));
    }

    return out;
}
