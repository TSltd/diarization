#pragma once

// ---------------------------------------------------------------------------
// adapters/AudioRecordCommand.h
//
// Drop-in integration layer for pm-image-cli's `audio record` command.
//
// This header wires together the full diarization pipeline:
//   DiarizationCliArgs  →  DiarizationEngine  →  AssistantMerger
//                       →  TranscriptFormatter  →  DiarizationStatus
//
// The class is intentionally thin: it does no I/O itself and owns no thread.
// The pm-image-cli recording loop calls into it at four call-sites:
//
//   1. AudioRecordCommand::configure()   — after argv parsing, before record
//   2. AudioRecordCommand::on_tts_start() — whenever TTS begins playing
//   3. AudioRecordCommand::on_tts_end()   — whenever TTS finishes
//   4. AudioRecordCommand::finish()       — after whisper_full() returns
//
// Dependency headers that must be available at the call-site:
//   #include "integration/AssistantMerger.h"
//   #include "integration/DiarizationCli.h"
//   #include <diarization/DiarizationEngine.h>
//   #include "integration/DiarizationStatus.h"
//   #include <diarization/TranscriptFormatter.h>
//   #include "adapters/WhisperAdapter.h"
//   #include "models/WeSpeakerEcapaModel.h"   (or any ISpeakerEmbeddingModel)
// ---------------------------------------------------------------------------

#include "integration/AssistantMerger.h"
#include "integration/DiarizationCli.h"
#include <diarization/DiarizationEngine.h>
#include "integration/DiarizationStatus.h"
#include <diarization/TranscriptFormatter.h>
#include "adapters/WhisperAdapter.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Forward-declare the model so callers can supply their own model factory
// without pulling models/WeSpeakerEcapaModel.h into every TU.
class ISpeakerEmbeddingModel;

// ---------------------------------------------------------------------------
// AudioRecordCommand
// ---------------------------------------------------------------------------

class AudioRecordCommand {
public:
    // -----------------------------------------------------------------------
    // 1. configure()
    //
    // Call once after argv parsing.  Validates the CLI args and, if --diarize
    // is set, constructs the DiarizationEngine with the given model.
    //
    // @param args        Filled-in CLI args struct.
    // @param model       Speaker embedding model.  May be nullptr if
    //                    args.diarize == false.
    // @throws std::runtime_error if validation fails or model is required
    //         but null.
    // -----------------------------------------------------------------------

    void configure(const DiarizationCliArgs& args,
                   std::unique_ptr<ISpeakerEmbeddingModel> model = nullptr) {
        args_ = args;

        const std::string err = validate_diarize_args(args);
        if (!err.empty())
            throw std::runtime_error(err);

        if (args.diarize) {
            if (!model)
                throw std::runtime_error(
                    "AudioRecordCommand: diarize=true but no model provided");
            engine_ = std::make_unique<DiarizationEngine>(std::move(model));
        }
    }

    // -----------------------------------------------------------------------
    // 2. on_tts_start() / on_tts_end()
    //
    // Called by the TTS subsystem around each assistant utterance.
    // Records the wall-clock time so the segment can be added to the event log.
    //
    // Usage in pm-image-cli:
    //   cmd.on_tts_start(recording_start_epoch_ms, tts_text);
    //   // ... TTS plays ...
    //   cmd.on_tts_end(recording_start_epoch_ms);
    // -----------------------------------------------------------------------

    void on_tts_start(int64_t recording_start_ms, const std::string& text = "") {
        tts_text_         = text;
        tts_start_offset_ = wall_ms() - recording_start_ms;
    }

    void on_tts_end(int64_t recording_start_ms) {
        const int64_t end_offset = wall_ms() - recording_start_ms;
        if (end_offset > tts_start_offset_) {
            AssistantMerger::record_assistant_segment(
                tts_log_, tts_start_offset_, end_offset, tts_text_);
        }
    }

    // -----------------------------------------------------------------------
    // 3. finish()
    //
    // Call after whisper_full() completes.  If --diarize is set, runs the
    // full pipeline and returns the formatted transcript string.
    //
    // The audio must already be at 16 kHz mono in `audio_buf`.
    //
    // Example (real integration with whisper.cpp):
    //
    //   #include "whisper.h"
    //   ...
    //   whisper_full(ctx, wparams, audio_buf.samples.data(),
    //                (int)audio_buf.samples.size());
    //
    //   auto output = cmd.finish(audio_buf, ctx);   // whisper_context* ctx
    //   std::cout << output;
    //
    // Without whisper.cpp (testing / stub use):
    //
    //   auto output = cmd.finish(audio_buf, fake_segments);
    // -----------------------------------------------------------------------

    // Overload 1: takes a whisper_context* (real integration path)
#ifdef WHISPER_H
    std::string finish(const AudioBuffer&  audio_buf,
                       whisper_context*    ctx,
                       int64_t             duration_ms = 0) {
        const auto whisper_segs = convert_segments(ctx);
        return finish_impl(audio_buf, whisper_segs, duration_ms);
    }
#endif

    // Overload 2: takes pre-built WhisperSegments (testing / offline path)
    std::string finish(const AudioBuffer&                 audio_buf,
                       const std::vector<WhisperSegment>& whisper_segs,
                       int64_t                            duration_ms = 0) {
        return finish_impl(audio_buf, whisper_segs, duration_ms);
    }

    // -----------------------------------------------------------------------
    // status()
    //
    // Returns the `audio record status` string at any point after configure().
    // Called by the `pm-image-cli audio record status` sub-command.
    //
    // Output format:
    //   Diarization:    enabled
    //   Speaker model:  /models/voxceleb_ECAPA512.onnx
    //   Threshold:      0.72
    //   Known speakers: 2 (SPEAKER_00: 8 seg, SPEAKER_01: 5 seg)
    // -----------------------------------------------------------------------

    std::string status() const {
        const std::vector<SpeakerCluster> live =
            engine_ ? engine_->clusters() : std::vector<SpeakerCluster>{};
        return DiarizationStatus::from_args_and_clusters(args_, live).format();
    }

    // -----------------------------------------------------------------------
    // reset()
    //
    // Clear cluster state and TTS log between independent recordings.
    // -----------------------------------------------------------------------

    void reset() {
        if (engine_) engine_->reset();
        tts_log_.clear();
    }

private:
    // -----------------------------------------------------------------------
    // Internal finish pipeline
    // -----------------------------------------------------------------------

    std::string finish_impl(const AudioBuffer&                 audio_buf,
                            const std::vector<WhisperSegment>& whisper_segs,
                            int64_t                            duration_ms) {
        if (!args_.diarize || !engine_) {
            // Diarization disabled — fall back to plain inline text.
            std::string plain;
            for (const auto& s : whisper_segs) plain += s.text;
            return plain;
        }

        // Step 1: diarize
        auto diarized = engine_->process(
            audio_buf, whisper_segs, to_diarization_options(args_));

        // Step 2: inject ASSISTANT segments from the TTS event log
        if (!tts_log_.empty())
            AssistantMerger::merge_inplace(diarized, tts_log_);

        // Step 3: build metadata for JSON output
        DiarizationMeta meta;
        meta.enabled       = true;
        meta.speaker_model = args_.speaker_model;
        meta.threshold     = args_.speaker_threshold;
        for (const auto& c : engine_->clusters())
            meta.speakers.push_back({c.label, c.segment_count});

        if (duration_ms == 0 && !audio_buf.samples.empty())
            duration_ms = static_cast<int64_t>(audio_buf.samples.size())
                          * 1000LL / audio_buf.sample_rate;

        // Step 4: format
        return TranscriptFormatter::format(
            diarized,
            diarize_output_format(args_),
            meta,
            "whisper",
            args_.speaker_model,
            duration_ms);
    }

    static int64_t wall_ms() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()).count();
    }

    // State
    DiarizationCliArgs                      args_;
    std::unique_ptr<DiarizationEngine>      engine_;
    AssistantMerger::EventLog               tts_log_;
    int64_t                                 tts_start_offset_ = 0;
    std::string                             tts_text_;
};
