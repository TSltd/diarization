// tools/embed_wav.cpp
//
// Minimal tool: embed one WAV file with the ONNX model and print the
// L2-normalised embedding to stdout, one float per line.
//
// Used by compare_embeddings.py to get the C++ reference embedding.
//
// Build (from repo root):
/* 

g++ -std=c++20 -O2 -Iinclude -I. -DDIARIZE_HAVE_MODEL \
    -I$ORT/include \
    tools/embed_wav.cpp \
    src/DiarizationEngine.cpp \
    src/SpeakerClusterManager.cpp \
    src/LabelSmoother.cpp \
    src/TranscriptFormatter.cpp \
    models/WeSpeakerEcapaModel.cpp \
    models/EcapaOnnxModel.cpp \
    models/SpeechBrainEcapaModel.cpp \
    models/SpeakerModelFactory.cpp \
    models/SpeakerVerifier.cpp \
    -L$ORT/lib \
    -Wl,-rpath,$ORT/lib \
    -lonnxruntime \
    -o tools/embed_wav

 */
//
// Usage:
//   ./tools/embed_wav <model.onnx> <audio.wav>
//   ./tools/embed_wav <model.onnx> <audio.wav> --fbank   (print FBANK too)

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <diarization/AudioChunk.h>
#include <diarization/SpeakerVerifier.h>
#include "models/FBankFrontEnd.h"
#include "tests/wav_reader.h"

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: embed_wav <model.onnx> <audio.wav> [--fbank]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string wav_path   = argv[2];
    const bool dump_fbank = (argc >= 4 && std::string(argv[3]) == "--fbank");

    // Load model
    SpeakerVerifier verifier;
    if (!verifier.load(model_path)) {
        std::cerr << "ERROR: failed to load model: " << model_path << "\n";
        return 1;
    }

    // Read WAV
    AudioBuffer buf = wav::read_wav_mono_16k(wav_path);
    if (buf.samples.empty()) {
        std::cerr << "ERROR: empty audio: " << wav_path << "\n";
        return 1;
    }

    // Optionally dump FBANK features
    if (dump_fbank) {
        FBankFrontEnd fb;
        auto feats = fb.compute(buf.samples);
        int n_frames = static_cast<int>(feats.size()) / FBankFrontEnd::kMelBins;
        std::cerr << "FBANK: " << n_frames << " frames × "
                  << FBankFrontEnd::kMelBins << " bins\n";
        // Print first 3 frames to stderr
        for (int f = 0; f < std::min(n_frames, 3); ++f) {
            std::cerr << "  frame[" << f << "]:";
            for (int m = 0; m < 8; ++m)
                std::cerr << " " << std::fixed << std::setprecision(3)
                          << feats[f * FBankFrontEnd::kMelBins + m];
            std::cerr << " ...\n";
        }
    }

    // Embed
    AudioChunk chunk;
    chunk.samples     = buf.samples;
    chunk.sample_rate = buf.sample_rate;
    chunk.start_ms    = 0;
    chunk.end_ms      = static_cast<int64_t>(
        buf.samples.size() * 1000 / buf.sample_rate);

    std::vector<float> emb = verifier.embed(chunk);

    // Print embedding — one float per line, full precision
    std::cout << std::setprecision(10);
    for (float v : emb)
        std::cout << v << "\n";

    return 0;
}
