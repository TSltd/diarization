#include <diarization/TranscriptFormatter.h>

#include <algorithm>
#include <format>
#include <map>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

SpeakerFormat parse_speaker_format(const std::string& name) {
    if (name == "inline") return SpeakerFormat::Inline;
    if (name == "json")   return SpeakerFormat::Json;
    if (name == "srt")    return SpeakerFormat::Srt;
    if (name == "vtt")    return SpeakerFormat::Vtt;
    throw std::invalid_argument("Unknown speaker format: " + name);
}

// HH:MM:SS<sep>mmm
std::string TranscriptFormatter::ms_to_timestamp(int64_t ms, char sep) {
    int64_t h   = ms / 3'600'000;
    int64_t m   = (ms % 3'600'000) / 60'000;
    int64_t s   = (ms % 60'000) / 1000;
    int64_t rem = ms % 1000;
    return std::format("{:02}:{:02}:{:02}{}{:03}", h, m, s, sep, rem);
}

// MM:SS.mmm
std::string TranscriptFormatter::ms_to_short(int64_t ms) {
    int64_t m   = ms / 60'000;
    int64_t s   = (ms % 60'000) / 1000;
    int64_t rem = ms % 1000;
    return std::format("{:02}:{:02}.{:03}", m, s, rem);
}

// ---------------------------------------------------------------------------
// Inline
// ---------------------------------------------------------------------------

std::string TranscriptFormatter::format_inline(
    const std::vector<TranscriptSegment>& segs) {

    std::ostringstream out;
    for (const auto& seg : segs) {
        out << '[' << seg.speaker << ' '
            << ms_to_short(seg.start_ms) << '-'
            << ms_to_short(seg.end_ms)   << "] "
            << seg.text << '\n';
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// SRT
// ---------------------------------------------------------------------------

std::string TranscriptFormatter::format_srt(
    const std::vector<TranscriptSegment>& segs) {

    std::ostringstream out;
    int idx = 1;
    for (const auto& seg : segs) {
        out << idx++ << '\n'
            << ms_to_timestamp(seg.start_ms, ',') << " --> "
            << ms_to_timestamp(seg.end_ms,   ',') << '\n'
            << '[' << seg.speaker << "] " << seg.text << "\n\n";
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// VTT
// ---------------------------------------------------------------------------

std::string TranscriptFormatter::format_vtt(
    const std::vector<TranscriptSegment>& segs) {

    std::ostringstream out;
    out << "WEBVTT\n\n";
    for (const auto& seg : segs) {
        out << ms_to_timestamp(seg.start_ms, '.') << " --> "
            << ms_to_timestamp(seg.end_ms,   '.') << '\n'
            << '<' << seg.speaker << "> " << seg.text << "\n\n";
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

// Minimal JSON escaping (handles the characters likely to appear in transcripts).
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

std::string TranscriptFormatter::format_json(
    const std::vector<TranscriptSegment>& segs,
    const DiarizationMeta&                meta,
    const std::string&                    provider,
    const std::string&                    model_path,
    int64_t                               duration_ms) {

    // Build plain-text field (SPEAKER_XX: text …)
    std::string plain;
    for (const auto& seg : segs) {
        if (!plain.empty()) plain += ' ';
        plain += seg.speaker + ": " + seg.text;
    }

    std::ostringstream out;
    out << "{\n";
    out << std::format("  \"provider\": \"{}\",\n", json_escape(provider));
    out << std::format("  \"model\": \"{}\",\n",    json_escape(model_path));
    out << std::format("  \"duration_ms\": {},\n",  duration_ms);

    // diarization block
    out << "  \"diarization\": {\n";
    out << std::format("    \"enabled\": {},\n", meta.enabled ? "true" : "false");
    out << std::format("    \"speaker_model\": \"{}\",\n", json_escape(meta.speaker_model));
    out << std::format("    \"threshold\": {:.2f},\n", meta.threshold);
    out << "    \"speakers\": [\n";
    for (size_t i = 0; i < meta.speakers.size(); ++i) {
        const auto& sp = meta.speakers[i];
        out << std::format("      {{ \"id\": \"{}\", \"segments\": {} }}",
                           json_escape(sp.id), sp.segments);
        out << (i + 1 < meta.speakers.size() ? ",\n" : "\n");
    }
    out << "    ]\n";
    out << "  },\n";

    // segments array
    out << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& seg = segs[i];
        out << "    {\n";
        out << std::format("      \"start_ms\": {},\n",    seg.start_ms);
        out << std::format("      \"end_ms\": {},\n",      seg.end_ms);
        out << std::format("      \"speaker\": \"{}\",\n", json_escape(seg.speaker));
        out << std::format("      \"text\": \"{}\",\n",    json_escape(seg.text));
        out << std::format("      \"confidence\": {:.4f}\n", seg.confidence);
        out << "    }";
        out << (i + 1 < segs.size() ? ",\n" : "\n");
    }
    out << "  ],\n";

    out << std::format("  \"text\": \"{}\"\n", json_escape(plain));
    out << "}\n";
    return out.str();
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

std::string TranscriptFormatter::format(
    const std::vector<TranscriptSegment>& segs,
    SpeakerFormat                         fmt,
    const DiarizationMeta&                meta,
    const std::string&                    provider,
    const std::string&                    model_path,
    int64_t                               duration_ms) {

    switch (fmt) {
        case SpeakerFormat::Inline: return format_inline(segs);
        case SpeakerFormat::Srt:    return format_srt(segs);
        case SpeakerFormat::Vtt:    return format_vtt(segs);
        case SpeakerFormat::Json:   return format_json(segs, meta, provider,
                                                        model_path, duration_ms);
    }
    return {};
}
