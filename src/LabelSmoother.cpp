#include "LabelSmoother.h"

#include "TranscriptFormatter.h"

// ---------------------------------------------------------------------------

LabelSmoother::LabelSmoother(int64_t min_flip_ms)
    : min_flip_ms_(min_flip_ms) {}

// ---------------------------------------------------------------------------

void LabelSmoother::smooth(std::vector<TranscriptSegment>& segs) const {
    // We need at least three segments for a flip to be possible.
    if (segs.size() < 3) return;

    // Protected labels that are never relabelled.
    auto is_protected = [](const std::string& lbl) {
        return lbl == "ASSISTANT" || lbl == "UNKNOWN";
    };

    // Single-pass scan.  We may need multiple passes if relabelling creates
    // new flip candidates, so repeat until stable.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i + 1 < segs.size(); ++i) {
            const auto& prev = segs[i - 1];
            auto&       cur  = segs[i];
            const auto& next = segs[i + 1];

            if (is_protected(cur.speaker)) continue;

            const int64_t dur = cur.end_ms - cur.start_ms;
            if (dur < min_flip_ms_
                && prev.speaker == next.speaker
                && cur.speaker  != prev.speaker) {
                cur.speaker = prev.speaker;
                changed     = true;
            }
        }
    }
}
