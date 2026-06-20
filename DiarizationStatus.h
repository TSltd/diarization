#pragma once

// ---------------------------------------------------------------------------
// DiarizationStatus.h
//
// Formats the `audio record status` output block.
//
// The ticket requires:
//
//   Diarization: enabled
//   Speaker model: /path/to/model.onnx
//   Known speakers: 2 (SPEAKER_00: 8 segments, SPEAKER_01: 5 segments)
//
// Usage:
//   DiarizationStatus st;
//   st.enabled       = dia_args.diarize;
//   st.speaker_model = dia_args.speaker_model;
//   for (const auto& c : engine.clusters())
//       st.add_cluster(c);
//   std::cout << st.format() << "\n";
// ---------------------------------------------------------------------------

#include "DiarizationCli.h"    // DiarizationCliArgs
#include "SpeakerCluster.h"    // SpeakerCluster

#include <sstream>
#include <string>
#include <vector>

struct DiarizationStatus {
    // -----------------------------------------------------------------------
    // Inputs (filled by the application before calling format())
    // -----------------------------------------------------------------------

    bool        enabled       = false;
    std::string speaker_model;
    float       threshold     = 0.72f;

    struct ClusterInfo {
        std::string label;
        int         segment_count = 0;
    };
    std::vector<ClusterInfo> clusters;

    // -----------------------------------------------------------------------
    // Populate from a DiarizationCliArgs + live cluster list
    // -----------------------------------------------------------------------

    static DiarizationStatus from_args_and_clusters(
        const DiarizationCliArgs&          args,
        const std::vector<SpeakerCluster>& live_clusters)
    {
        DiarizationStatus st;
        st.enabled       = args.diarize;
        st.speaker_model = args.speaker_model;
        st.threshold     = args.speaker_threshold;
        for (const auto& c : live_clusters)
            st.clusters.push_back({c.label, c.segment_count});
        return st;
    }

    // -----------------------------------------------------------------------
    // Convenience helper used during recording
    // -----------------------------------------------------------------------

    void add_cluster(const SpeakerCluster& c) {
        clusters.push_back({c.label, c.segment_count});
    }

    // -----------------------------------------------------------------------
    // format() — multi-line status string
    //
    // Output (example):
    //   Diarization:    enabled
    //   Speaker model:  /models/voxceleb_ECAPA512.onnx
    //   Threshold:      0.72
    //   Known speakers: 2 (SPEAKER_00: 8 seg, SPEAKER_01: 5 seg)
    // -----------------------------------------------------------------------

    std::string format() const {
        std::ostringstream os;

        os << "Diarization:    " << (enabled ? "enabled" : "disabled") << "\n";

        if (!enabled) return os.str();

        os << "Speaker model:  "
           << (speaker_model.empty() ? "(none)" : speaker_model) << "\n";
        os << "Threshold:      " << threshold << "\n";

        if (clusters.empty()) {
            os << "Known speakers: 0\n";
        } else {
            os << "Known speakers: " << clusters.size() << " (";
            for (std::size_t i = 0; i < clusters.size(); ++i) {
                if (i > 0) os << ", ";
                os << clusters[i].label << ": " << clusters[i].segment_count << " seg";
            }
            os << ")\n";
        }

        return os.str();
    }

    // -----------------------------------------------------------------------
    // format_compact() — single-line version for log output
    //
    // Example: "diarize=on model=ecapa512 spk=2 [SPEAKER_00:8 SPEAKER_01:5]"
    // -----------------------------------------------------------------------

    std::string format_compact() const {
        std::ostringstream os;
        os << "diarize=" << (enabled ? "on" : "off");
        if (!enabled) return os.str();

        if (!speaker_model.empty()) {
            // Basename only for compact form
            const auto slash = speaker_model.find_last_of("/\\");
            const std::string base = (slash == std::string::npos)
                ? speaker_model
                : speaker_model.substr(slash + 1);
            os << " model=" << base;
        }
        os << " spk=" << clusters.size();
        if (!clusters.empty()) {
            os << " [";
            for (std::size_t i = 0; i < clusters.size(); ++i) {
                if (i > 0) os << " ";
                os << clusters[i].label << ":" << clusters[i].segment_count;
            }
            os << "]";
        }
        return os.str();
    }
};
