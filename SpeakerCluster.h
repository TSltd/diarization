#pragma once

#include <string>
#include <vector>

/// Represents one detected speaker as an online cluster.
/// The centroid is kept L2-normalised after every update so that cosine
/// similarity can be computed as a simple dot product.
struct SpeakerCluster {
    std::string          label;         ///< e.g. "SPEAKER_00", "SPEAKER_01"
    std::vector<float>   centroid;      ///< Running mean, L2-normalised
    int                  segment_count = 0; ///< Number of windows assigned so far
};
