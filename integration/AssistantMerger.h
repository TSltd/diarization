#pragma once

// ---------------------------------------------------------------------------
// integration/AssistantMerger.h
//
// Application-layer merge of ASSISTANT (TTS) segments into a diarized
// transcript.
//
// The diarization engine intentionally does not know about "ASSISTANT".  It
// returns segments labelled SPEAKER_XX / UNKNOWN.  The application layer
// holds a TTS event log — a list of time-ranges during which the assistant
// was speaking — and calls merge() to:
//
//   1. Inject ASSISTANT segments for every TTS range.
//   2. Clip or drop any diarized segments that overlap an ASSISTANT range.
//   3. Sort the combined list chronologically.
//
// Terminology:
//   • "diarized segments"  — output of DiarizationEngine::process()
//   • "assistant segments" — from the TTS event log (record_assistant_segment)
//   • "merged transcript"  — chronological union, ready for TranscriptFormatter
//
// Example
// ───────
//   // Build assistant event log during recording:
//   AssistantMerger::EventLog tts_log;
//   tts_log.push_back({12500, 14200, "[Assistant: playing audio]"});
//
//   // After diarization:
//   auto merged = AssistantMerger::merge(diarized, tts_log);
//   auto output = TranscriptFormatter::format(merged, SpeakerFormat::Json, ...);
// ---------------------------------------------------------------------------

#include <diarization/TranscriptFormatter.h>   // TranscriptSegment

#include <algorithm>
#include <string>
#include <vector>

class AssistantMerger {
public:
    // -----------------------------------------------------------------------
    // TTS event — one span of assistant speech
    // -----------------------------------------------------------------------

    struct Event {
        int64_t     start_ms = 0;
        int64_t     end_ms   = 0;
        std::string text;               ///< Optional label / TTS text excerpt
        float       confidence = 1.0f;  ///< Always 1.0 for injected segments
    };

    using EventLog = std::vector<Event>;

    // -----------------------------------------------------------------------
    // record_assistant_segment()
    //
    // Called by the application layer whenever the TTS engine starts a new
    // utterance.  Appends to the event log that will later be passed to merge().
    // -----------------------------------------------------------------------

    static void record_assistant_segment(EventLog& log,
                                         int64_t start_ms,
                                         int64_t end_ms,
                                         const std::string& text = "") {
        if (end_ms <= start_ms) return;
        log.push_back({start_ms, end_ms, text, 1.0f});
    }

    // -----------------------------------------------------------------------
    // merge()
    //
    // Merges diarized speaker segments with the assistant TTS event log.
    //
    // Rules:
    //   • Any diarized segment that is FULLY inside an assistant range is
    //     dropped (the assistant was talking, not the user).
    //   • Any diarized segment that PARTIALLY overlaps an assistant range is
    //     clipped to the non-overlapping portion.  If the remaining portion
    //     is shorter than min_dur_ms it is dropped.
    //   • ASSISTANT segments are injected for every event in `events`.
    //   • The result is sorted by start_ms.
    //
    // @param diarized   Output of DiarizationEngine::process().
    // @param events     TTS event log (may be unsorted on entry).
    // @param min_dur_ms Minimum duration to keep a clipped segment (ms).
    // @return           Merged, chronologically ordered transcript.
    // -----------------------------------------------------------------------

    static std::vector<TranscriptSegment> merge(
        const std::vector<TranscriptSegment>& diarized,
        const EventLog&                       events,
        int64_t                               min_dur_ms = 200)
    {
        // Sort events by start time (defensive copy)
        EventLog sorted_events = events;
        std::sort(sorted_events.begin(), sorted_events.end(),
                  [](const Event& a, const Event& b){
                      return a.start_ms < b.start_ms;
                  });

        std::vector<TranscriptSegment> out;
        out.reserve(diarized.size() + sorted_events.size());

        // 1. Process each diarized segment against all assistant ranges.
        for (const auto& seg : diarized) {
            std::vector<TranscriptSegment> pieces = clip(seg, sorted_events, min_dur_ms);
            for (auto& p : pieces)
                out.push_back(std::move(p));
        }

        // 2. Inject ASSISTANT segments.
        for (const auto& ev : sorted_events) {
            TranscriptSegment aseg;
            aseg.start_ms   = ev.start_ms;
            aseg.end_ms     = ev.end_ms;
            aseg.speaker    = "ASSISTANT";
            aseg.text       = ev.text.empty() ? "[Assistant]" : ev.text;
            aseg.confidence = ev.confidence;
            out.push_back(std::move(aseg));
        }

        // 3. Sort chronologically.
        std::sort(out.begin(), out.end(),
                  [](const TranscriptSegment& a, const TranscriptSegment& b){
                      return a.start_ms < b.start_ms;
                  });

        return out;
    }

    // -----------------------------------------------------------------------
    // merge_inplace()  — convenience overload that modifies the vector.
    // -----------------------------------------------------------------------

    static void merge_inplace(std::vector<TranscriptSegment>& diarized,
                               const EventLog& events,
                               int64_t min_dur_ms = 200) {
        diarized = merge(diarized, events, min_dur_ms);
    }

private:
    // Clip a single diarized segment against all assistant events.
    // Returns 0, 1 or 2 pieces depending on overlap.
    static std::vector<TranscriptSegment> clip(
        const TranscriptSegment& seg,
        const EventLog&          events,
        int64_t                  min_dur_ms)
    {
        // We work with a list of surviving ranges [start, end).
        struct Range { int64_t s, e; };
        std::vector<Range> ranges = {{seg.start_ms, seg.end_ms}};

        for (const auto& ev : events) {
            std::vector<Range> next;
            for (const auto& r : ranges) {
                // No overlap
                if (ev.end_ms <= r.s || ev.start_ms >= r.e) {
                    next.push_back(r);
                    continue;
                }
                // Left piece
                if (r.s < ev.start_ms)
                    next.push_back({r.s, ev.start_ms});
                // Right piece
                if (ev.end_ms < r.e)
                    next.push_back({ev.end_ms, r.e});
                // Fully covered → dropped (no push)
            }
            ranges = std::move(next);
        }

        // Convert surviving ranges back to TranscriptSegments
        std::vector<TranscriptSegment> result;
        for (const auto& r : ranges) {
            if (r.e - r.s < min_dur_ms) continue;
            TranscriptSegment piece = seg;  // copy text/speaker/confidence
            piece.start_ms = r.s;
            piece.end_ms   = r.e;
            result.push_back(std::move(piece));
        }
        return result;
    }
};
