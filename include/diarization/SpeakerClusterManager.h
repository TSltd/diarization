#pragma once

#include <span>
#include <string>
#include <vector>

#include <diarization/SpeakerCluster.h>

/// Clustering parameters forwarded from DiarizationOptions.
struct ClusteringOptions {
    float threshold  = 0.72f; ///< Cosine similarity to reuse an existing cluster
    int   max_speakers = 0;   ///< 0 = unlimited
};

/// Manages the set of online speaker clusters.
///
/// Each call to assign() either maps the embedding to the nearest existing
/// cluster whose centroid has cosine similarity >= threshold, or creates a new
/// cluster (subject to the max_speakers cap).  The cluster centroid is updated
/// after every successful assignment.
class SpeakerClusterManager {
public:
    explicit SpeakerClusterManager(const ClusteringOptions& opts = {});

    /// Find or create the best-matching cluster for the given L2-normalised
    /// embedding.  Returns the cluster label ("SPEAKER_00", …).
    std::string assign(const std::vector<float>& embedding);

    /// Read-only view of all current clusters.
    const std::vector<SpeakerCluster>& clusters() const { return clusters_; }

    /// Remove all clusters (useful between independent recordings).
    void reset();

private:
    /// Dot product of two unit vectors == cosine similarity.
    static float cosine_similarity(std::span<const float> a,
                                   std::span<const float> b);

    /// Incorporate a new embedding into the cluster centroid via running mean,
    /// then re-normalise to unit length.
    static void update_centroid(SpeakerCluster& cluster,
                                const std::vector<float>& embedding);

    /// Format a zero-padded speaker label for index i.
    static std::string make_label(int i);

    ClusteringOptions          opts_;
    std::vector<SpeakerCluster> clusters_;
};
