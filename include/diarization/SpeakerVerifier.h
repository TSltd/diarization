#pragma once

#include <memory>
#include <string>
#include <vector>

#include <diarization/AudioChunk.h>
#include <diarization/ISpeakerEmbeddingModel.h>
#include <diarization/ModelMetadata.h>

/// @brief High-level speaker verification helper.
///
/// Wraps WeSpeakerEcapaModel to provide three convenience operations:
///   - embed()      — extract a speaker embedding from an audio chunk
///   - similarity() — cosine similarity between two chunks
///   - verify()     — hard accept/reject decision against a threshold
///
/// Usage:
/// @code
///   SpeakerVerifier sv;
///   sv.load("wespeaker/voxceleb_ECAPA512_LM.onnx");
///   bool same = sv.verify(chunkA, chunkB);       // threshold = 0.72
///   float score = sv.similarity(chunkA, chunkB); // raw cosine in [-1, 1]
/// @endcode
///
/// @note load() must be called before embed(), similarity(), or verify().
class SpeakerVerifier {
public:
    SpeakerVerifier();
    ~SpeakerVerifier();

    // Non-copyable (owns an ONNX session)
    SpeakerVerifier(const SpeakerVerifier&)            = delete;
    SpeakerVerifier& operator=(const SpeakerVerifier&) = delete;

    SpeakerVerifier(SpeakerVerifier&&)            = default;
    SpeakerVerifier& operator=(SpeakerVerifier&&) = default;

    /// Load the ONNX speaker model from @p model_path.
    /// The correct model class (WeSpeaker vs SpeechBrain) is selected
    /// automatically by SpeakerModelFactory based on keywords in the path.
    /// @return true on success, false on I/O or model error.
    bool load(const std::string& model_path);

    /// Return metadata about the loaded graph (node names, shapes).
    /// Returns an unloaded ModelMetadata if load() has not been called.
    ModelMetadata inspect() const;

    /// Compute an L2-normalised speaker embedding for @p chunk.
    /// Requires load() to have been called successfully.
    std::vector<float> embed(const AudioChunk& chunk);

    /// Cosine similarity between the speaker embeddings of @p a and @p b.
    /// Returns a value in [-1, 1].  1.0 = identical speaker, -1.0 = opposite.
    /// Requires load() to have been called successfully.
    float similarity(const AudioChunk& a, const AudioChunk& b);

    /// Returns true if similarity(a, b) >= @p threshold.
    /// Default threshold of 0.72 is a reasonable operating point for
    /// voxceleb_ECAPA512_LM; tune per-deployment as needed.
    bool verify(const AudioChunk& a, const AudioChunk& b,
                float threshold = 0.72f);

private:
    std::unique_ptr<ISpeakerEmbeddingModel> model_;
};
