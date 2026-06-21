#include <diarization/SpeakerVerifier.h>

#include <numeric>
#include <stdexcept>

#include "SpeakerModelFactory.h"

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

// model_ starts null; load() creates the right subclass via the factory.
SpeakerVerifier::SpeakerVerifier()  = default;
SpeakerVerifier::~SpeakerVerifier() = default;

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

bool SpeakerVerifier::load(const std::string& model_path) {
    // Factory selects WeSpeakerEcapaModel or SpeechBrainEcapaModel based on
    // keywords in the path (e.g. "speechbrain" → SpeechBrainEcapaModel).
    model_ = SpeakerModelFactory::create(model_path);
    return model_ && model_->load(model_path);
}

// ---------------------------------------------------------------------------
// inspect()
// ---------------------------------------------------------------------------

ModelMetadata SpeakerVerifier::inspect() const {
    if (!model_) return {};
    return model_->inspect();
}

// ---------------------------------------------------------------------------
// embed()
// ---------------------------------------------------------------------------

std::vector<float> SpeakerVerifier::embed(const AudioChunk& chunk) {
    if (!model_)
        throw std::runtime_error("SpeakerVerifier: call load() before embed()");
    return model_->embed(chunk);
}

// ---------------------------------------------------------------------------
// similarity()
// ---------------------------------------------------------------------------

float SpeakerVerifier::similarity(const AudioChunk& a, const AudioChunk& b) {
    const auto ea = embed(a);
    const auto eb = embed(b);
    // WeSpeaker / SpeechBrain both return L2-normalised vectors, so the dot
    // product equals cosine similarity directly.
    return std::inner_product(ea.begin(), ea.end(), eb.begin(), 0.0f);
}

// ---------------------------------------------------------------------------
// verify()
// ---------------------------------------------------------------------------

bool SpeakerVerifier::verify(const AudioChunk& a, const AudioChunk& b,
                              float threshold) {
    return similarity(a, b) >= threshold;
}
