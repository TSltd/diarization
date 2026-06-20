#pragma once

#include <string>
#include <vector>

#include "AudioChunk.h"

/// Abstract interface for speaker embedding models.
/// A concrete implementation (e.g. EcapaOnnxModel) loads an ONNX model and
/// converts an AudioChunk into a fixed-length L2-normalised embedding vector.
class ISpeakerEmbeddingModel {
public:
    virtual ~ISpeakerEmbeddingModel() = default;

    /// Load the model from disk.  Returns true on success.
    virtual bool load(const std::string& model_path) = 0;

    /// Compute a speaker embedding for the given audio chunk.
    /// The returned vector is L2-normalised (unit length).
    virtual std::vector<float> embed(const AudioChunk& chunk) = 0;

    /// Expected input sample rate in Hz (usually 16000).
    virtual int sample_rate() const = 0;

    /// Dimensionality of the output embedding vector.
    virtual int embedding_dim() const = 0;
};
