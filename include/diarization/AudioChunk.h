#pragma once

#include <cstdint>
#include <vector>

/// A slice of mono float PCM audio with timing metadata.
/// Samples are normalised to [-1.0, 1.0], sample_rate is in Hz.
struct AudioChunk {
    std::vector<float> samples;   ///< Mono PCM, normalised [-1, 1]
    int      sample_rate = 16000; ///< Hz
    int64_t  start_ms    = 0;     ///< Start offset within the original recording
    int64_t  end_ms      = 0;     ///< End offset within the original recording

    /// Duration in milliseconds derived from the sample count.
    int64_t duration_ms() const {
        if (sample_rate <= 0) return 0;
        return static_cast<int64_t>(samples.size()) * 1000 / sample_rate;
    }

    bool empty() const { return samples.empty(); }
};
