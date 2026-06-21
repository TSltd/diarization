#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

/// Runtime description of an ONNX speaker embedding model's graph contract.
///
/// Populated by ISpeakerEmbeddingModel::inspect() after a successful load().
/// Dynamic dimensions are reported as -1.
///
/// Example output of describe():
/// @code
///   Input:  wavs [1, ?]
///   Output: embeddings [1, 192]
/// @endcode
struct ModelMetadata {
    std::string              input_name;    ///< ONNX input node name
    std::string              output_name;   ///< ONNX output node name
    std::vector<int64_t>     input_shape;   ///< Static input shape (-1 = dynamic dim)
    std::vector<int64_t>     output_shape;  ///< Static output shape (-1 = dynamic dim)
    bool                     loaded = false;///< False until load() succeeds

    /// Human-readable summary, e.g. for logging or --debug output.
    std::string describe() const {
        if (!loaded) return "(model not loaded)";

        auto fmt_shape = [](const std::vector<int64_t>& s) {
            std::ostringstream ss;
            ss << '[';
            for (size_t i = 0; i < s.size(); ++i) {
                if (i) ss << ", ";
                if (s[i] < 0) ss << '?'; else ss << s[i];
            }
            ss << ']';
            return ss.str();
        };

        std::ostringstream out;
        out << "Input:  " << input_name  << " " << fmt_shape(input_shape)  << '\n'
            << "Output: " << output_name << " " << fmt_shape(output_shape);
        return out.str();
    }
};
