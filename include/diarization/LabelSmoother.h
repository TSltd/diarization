#pragma once

#include <cstdint>
#include <vector>

// Forward declaration — defined in TranscriptFormatter.h
struct TranscriptSegment;

/// Post-processing pass that eliminates isolated speaker-label flips.
///
/// A "flip" is a single short segment sandwiched between two segments that
/// both carry the same speaker label.  If the middle segment is shorter than
/// `min_flip_ms` it is relabelled to match its neighbours.
///
/// Example:
///   SPEAKER_00  (2 000 ms)
///   SPEAKER_01  (  300 ms)   <- isolated, < min_flip_ms  → relabelled
///   SPEAKER_00  (1 500 ms)
///
/// Result:
///   SPEAKER_00  (2 000 ms)
///   SPEAKER_00  (  300 ms)
///   SPEAKER_00  (1 500 ms)
class LabelSmoother {
public:
    /// @param min_flip_ms  Segments shorter than this (in ms) are candidates
    ///                     for relabelling.  Defaults to 500 ms.
    explicit LabelSmoother(int64_t min_flip_ms = 500);

    /// Apply in-place smoothing to a sequence of TranscriptSegments.
    /// Segments whose speaker is "ASSISTANT" or "UNKNOWN" are never relabelled.
    void smooth(std::vector<TranscriptSegment>& segments) const;

private:
    int64_t min_flip_ms_;
};
