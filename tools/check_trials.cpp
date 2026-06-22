// tools/check_trials.cpp
//
// Inspects a VoxCeleb-style trials file and reports how many trials
// reference audio files that are actually present on disk.
//
// Usage:
//   check_trials <trials.txt> <audio_base_dir>
//
// Output:
//   Trials in file:     263485
//   Resolvable trials:      57
//   Missing file pairs: 263428
//
// Build (no ONNX dependency):
//   g++ -std=c++20 -O2 tools/check_trials.cpp -o tools/check_trials
//
// Example:
//   ./tools/check_trials \
//       testdata/voxceleb_trials/trials.txt \
//       testdata/voxceleb_trials/audio/voxceleb1_cd

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr
            << "Usage: check_trials <trials.txt> <audio_base_dir>\n"
            << "  trials.txt     — trial list (label pathA pathB per line)\n"
            << "  audio_base_dir — base directory; paths in trials are relative to it\n";
        return 1;
    }

    const std::string trials_path = argv[1];
    const fs::path    audio_base  = argv[2];

    std::ifstream f(trials_path);
    if (!f) {
        std::cerr << "ERROR: cannot open trials file: " << trials_path << "\n";
        return 1;
    }

    long long n_total      = 0;
    long long n_resolvable = 0;
    long long n_missing    = 0;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        int         label;
        std::string pathA, pathB;
        if (!(ss >> label >> pathA >> pathB)) continue;

        ++n_total;

        const bool a_ok = fs::exists(audio_base / pathA);
        const bool b_ok = fs::exists(audio_base / pathB);

        if (a_ok && b_ok)
            ++n_resolvable;
        else
            ++n_missing;
    }

    std::cout
        << "Trials in file:     " << n_total      << "\n"
        << "Resolvable trials:  " << n_resolvable  << "\n"
        << "Missing file pairs: " << n_missing     << "\n";

    return (n_resolvable > 0) ? 0 : 1;
}
