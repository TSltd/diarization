#include <diarization/SpeakerClusterManager.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SpeakerClusterManager::SpeakerClusterManager(const ClusteringOptions& opts)
    : opts_(opts) {}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string SpeakerClusterManager::assign(const std::vector<float>& embedding) {
    if (embedding.empty())
        throw std::invalid_argument("assign: embedding must not be empty");

    // Find the existing cluster with the highest cosine similarity.
    int   best_idx   = -1;
    float best_score = -1.0f;

    for (int i = 0; i < static_cast<int>(clusters_.size()); ++i) {
        float score = cosine_similarity(embedding, clusters_[i].centroid);
        if (score > best_score) {
            best_score = score;
            best_idx   = i;
        }
    }

    if (best_idx >= 0 && best_score >= opts_.threshold) {
        // Re-use the existing cluster.
        update_centroid(clusters_[best_idx], embedding);
        return clusters_[best_idx].label;
    }

    // Create a new cluster unless we've reached the speaker cap.
    const int cap = opts_.max_speakers;
    if (cap > 0 && static_cast<int>(clusters_.size()) >= cap) {
        // Hard cap reached – assign to the closest cluster regardless of score.
        update_centroid(clusters_[best_idx], embedding);
        return clusters_[best_idx].label;
    }

    SpeakerCluster nc;
    nc.label         = make_label(static_cast<int>(clusters_.size()));
    nc.centroid      = embedding; // already L2-normalised by the caller
    nc.segment_count = 1;
    clusters_.push_back(std::move(nc));
    return clusters_.back().label;
}

void SpeakerClusterManager::reset() {
    clusters_.clear();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

float SpeakerClusterManager::cosine_similarity(std::span<const float> a,
                                                std::span<const float> b) {
    assert(a.size() == b.size());
    // Both vectors are unit-length, so cosine similarity == dot product.
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        dot += a[i] * b[i];
    // Clamp to [-1, 1] to guard against floating-point drift.
    return std::clamp(dot, -1.0f, 1.0f);
}

void SpeakerClusterManager::update_centroid(SpeakerCluster&          cluster,
                                             const std::vector<float>& embedding) {
    assert(cluster.centroid.size() == embedding.size());

    const int n = ++cluster.segment_count;

    // Running mean: new_centroid = old_centroid + (embedding - old_centroid) / n
    float norm_sq = 0.0f;
    for (size_t i = 0; i < cluster.centroid.size(); ++i) {
        cluster.centroid[i] += (embedding[i] - cluster.centroid[i]) / static_cast<float>(n);
        norm_sq += cluster.centroid[i] * cluster.centroid[i];
    }

    // Re-normalise to unit length.
    const float norm = std::sqrt(norm_sq);
    if (norm > 1e-8f) {
        for (float& v : cluster.centroid)
            v /= norm;
    }
}

std::string SpeakerClusterManager::make_label(int i) {
    return std::format("SPEAKER_{:02d}", i);
}
