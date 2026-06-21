#include "SpeakerModelFactory.h"

#include "SpeechBrainEcapaModel.h"
#include "WeSpeakerEcapaModel.h"

std::unique_ptr<ISpeakerEmbeddingModel>
SpeakerModelFactory::create(const std::string& model_path) {
    switch (detect_flavor(model_path)) {
        case Flavor::SpeechBrain:
            return std::make_unique<SpeechBrainEcapaModel>();

        case Flavor::WeSpeaker:
        case Flavor::Unknown:
        default:
            return std::make_unique<WeSpeakerEcapaModel>();
    }
}
