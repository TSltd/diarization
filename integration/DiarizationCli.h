#pragma once

/// CLI flag definitions and option conversion for the diarization feature.
///
/// This header decouples the diarization engine from the specific CLI parsing
/// library used by pm-image-cli (e.g. CLI11, cxxopts, getopt).  The
/// application layer fills a `DiarizationCliArgs` struct from argv and then
/// calls `to_diarization_options()` to get the engine-facing options.
///
/// ## Wiring into pm-image-cli (pseudocode)
///
///   // 1. Declare args struct alongside other audio-record options:
///   DiarizationCliArgs dia_args;
///
///   // 2. Register CLI flags (CLI11 example):
///   app.add_flag  ("--diarize",            dia_args.diarize);
///   app.add_option("--speaker-model",      dia_args.speaker_model);
///   app.add_option("--speaker-threshold",  dia_args.speaker_threshold);
///   app.add_option("--speaker-window-ms",  dia_args.speaker_window_ms);
///   app.add_option("--speaker-hop-ms",     dia_args.speaker_hop_ms);
///   app.add_option("--speaker-min-ms",     dia_args.speaker_min_ms);
///   app.add_option("--speaker-max",        dia_args.speaker_max);
///   app.add_option("--speaker-enroll",     dia_args.speaker_enroll);
///   app.add_option("--speaker-format",     dia_args.speaker_format)
///       ->check(CLI::IsMember({"inline","json","srt","vtt"}));
///
///   // 3. After parsing, build the engine:
///   if (dia_args.diarize) {
///       auto model = std::make_unique<WeSpeakerEcapaModel>();
///       if (!model->load(dia_args.speaker_model))
///           throw std::runtime_error("Failed to load speaker model");
///       DiarizationEngine engine(std::move(model));
///
///       // After whisper finishes:
///       auto segments = engine.process(audio_buf, whisper_segs,
///                           to_diarization_options(dia_args));
///       // Inject ASSISTANT segments from TTS event log, then format.
///       fmt_output = TranscriptFormatter::format(
///                        segments,
///                        parse_speaker_format(dia_args.speaker_format), ...);
///   }

#include <string>
#include "DiarizationEngine.h"   // DiarizationOptions
#include "TranscriptFormatter.h" // SpeakerFormat, parse_speaker_format

// ---------------------------------------------------------------------------
// CLI argument bag  (one-to-one with the ticket's --speaker-* flags)
// ---------------------------------------------------------------------------

struct DiarizationCliArgs {
    /// Master on/off switch.  --diarize implies --stt.
    bool diarize = false;

    /// Path to the ONNX speaker embedding model (required if diarize=true).
    std::string speaker_model;

    /// Cosine similarity threshold for cluster reuse.
    float speaker_threshold = 0.72f;

    /// Embedding window size in ms.
    int speaker_window_ms = 1500;

    /// Embedding hop size in ms.
    int speaker_hop_ms = 750;

    /// Minimum speech segment length to embed (shorter segments get UNKNOWN).
    int speaker_min_ms = 800;

    /// Maximum expected speakers (0 = auto / unlimited).
    int speaker_max = 0;

    /// Optional path to a folder or file containing known speaker WAV samples.
    /// Not yet implemented — reserved for --speaker-enroll.
    std::string speaker_enroll;

    /// Output format: "inline" | "json" | "srt" | "vtt"
    std::string speaker_format = "inline";
};

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Build a DiarizationOptions struct from the parsed CLI arguments.
inline DiarizationOptions to_diarization_options(const DiarizationCliArgs& a) {
    DiarizationOptions opts;
    opts.speaker_threshold = a.speaker_threshold;
    opts.speaker_window_ms = a.speaker_window_ms;
    opts.speaker_hop_ms    = a.speaker_hop_ms;
    opts.speaker_min_ms    = a.speaker_min_ms;
    opts.speaker_max       = a.speaker_max;
    return opts;
}

/// Parse and validate the --speaker-format flag.
/// Throws std::invalid_argument for unknown values (same as parse_speaker_format).
inline SpeakerFormat diarize_output_format(const DiarizationCliArgs& a) {
    return parse_speaker_format(a.speaker_format);
}

/// Validate that all required fields are present when --diarize is set.
/// Returns an empty string on success, or an error message on failure.
inline std::string validate_diarize_args(const DiarizationCliArgs& a) {
    if (!a.diarize) return {};
    if (a.speaker_model.empty())
        return "--diarize requires --speaker-model <path.onnx>";
    if (a.speaker_threshold < 0.0f || a.speaker_threshold > 1.0f)
        return "--speaker-threshold must be in [0, 1]";
    if (a.speaker_window_ms <= 0)
        return "--speaker-window-ms must be > 0";
    if (a.speaker_hop_ms <= 0 || a.speaker_hop_ms > a.speaker_window_ms)
        return "--speaker-hop-ms must be > 0 and <= --speaker-window-ms";
    if (a.speaker_min_ms < 0)
        return "--speaker-min-ms must be >= 0";
    if (a.speaker_max < 0)
        return "--speaker-max must be >= 0";
    return {};
}
