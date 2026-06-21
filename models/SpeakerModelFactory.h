#pragma once

#include <cctype>
#include <memory>
#include <string>

#include <diarization/ISpeakerEmbeddingModel.h>

/// @brief Factory that constructs the correct ISpeakerEmbeddingModel subclass
/// from an ONNX model path.
///
/// Detection is based on keywords in the file path; no file I/O is performed
/// in create() itself — the model is NOT loaded.  Call ->load(path) on the
/// returned object to open the ONNX session.
///
/// | Keyword(s) in path (case-insensitive)        | Model created           |
/// |----------------------------------------------|-------------------------|
/// | "speechbrain"                                 | SpeechBrainEcapaModel   |
/// | "wespeaker" / "voxceleb" / "ecapa512"         | WeSpeakerEcapaModel     |
/// | (none of the above)                           | WeSpeakerEcapaModel     |
///
/// Example:
/// @code
///   auto model = SpeakerModelFactory::create(model_path);
///   if (!model->load(model_path))
///       throw std::runtime_error("failed to load " + model_path);
///   auto emb = model->embed(chunk);
/// @endcode
class SpeakerModelFactory {
public:
    /// Flavour detected from the model path string.
    enum class Flavor { WeSpeaker, SpeechBrain, Unknown };

    /// Detect the model flavour from keywords in @p model_path.
    /// Pure string matching — no file I/O, no ONNX dependency.
    /// Defined inline so callers can use it without linking SpeakerModelFactory.cpp.
    static Flavor detect_flavor(const std::string& model_path) {
        // Lower-case copy for case-insensitive matching
        std::string lower;
        lower.reserve(model_path.size());
        for (unsigned char c : model_path)
            lower.push_back(static_cast<char>(std::tolower(c)));

        if (lower.find("speechbrain") != std::string::npos)
            return Flavor::SpeechBrain;

        if (lower.find("wespeaker")  != std::string::npos ||
            lower.find("voxceleb")   != std::string::npos ||
            lower.find("ecapa512")   != std::string::npos)
            return Flavor::WeSpeaker;

        return Flavor::Unknown;  // treated as WeSpeaker in create()
    }

    /// Construct (but do NOT load) the appropriate model object.
    /// Returns a non-null unique_ptr in all cases.
    /// Defined in SpeakerModelFactory.cpp (links WeSpeaker + SpeechBrain objects).
    static std::unique_ptr<ISpeakerEmbeddingModel>
    create(const std::string& model_path);
};
