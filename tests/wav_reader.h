#pragma once

/// Minimal RIFF/PCM WAV reader.
///
/// Handles the subset of WAV files produced by the test generator and by
/// standard audio tools:
///   - PCM (format tag 1) only
///   - 8, 16, 24, or 32-bit integer samples
///   - Any number of channels (downmixed to mono on read)
///   - Any sample rate
///
/// Does NOT handle WAVE_FORMAT_EXTENSIBLE, compressed formats, or RF64.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../TranscriptFormatter.h"  // AudioBuffer

namespace wav {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace detail {

inline uint16_t read_u16le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
inline uint32_t read_u32le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}

} // namespace detail

// ---------------------------------------------------------------------------
// read_wav()
// ---------------------------------------------------------------------------

/// Read a WAV file and return an AudioBuffer (float32, all channels kept).
/// Samples are normalised to [-1.0, 1.0].
///
/// @throws std::runtime_error on any format or I/O error.
inline AudioBuffer read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("wav::read_wav: cannot open " + path);

    // --- RIFF header (12 bytes) ---
    uint8_t riff[12];
    if (!f.read(reinterpret_cast<char*>(riff), 12) || f.gcount() != 12)
        throw std::runtime_error("wav::read_wav: truncated RIFF header");
    if (std::memcmp(riff, "RIFF", 4) != 0)
        throw std::runtime_error("wav::read_wav: not a RIFF file");
    if (std::memcmp(riff + 8, "WAVE", 4) != 0)
        throw std::runtime_error("wav::read_wav: RIFF type is not WAVE");

    // --- Scan chunks until we find "fmt " and "data" ---
    uint16_t audio_format  = 0;
    uint16_t num_channels  = 0;
    uint32_t sample_rate   = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size     = 0;
    bool     have_fmt      = false;
    bool     have_data     = false;

    while (f) {
        uint8_t chunk_id[4];
        uint8_t chunk_sz[4];
        if (!f.read(reinterpret_cast<char*>(chunk_id), 4)) break;
        if (!f.read(reinterpret_cast<char*>(chunk_sz), 4)) break;
        const uint32_t chunk_size = detail::read_u32le(chunk_sz);

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16)
                throw std::runtime_error("wav::read_wav: fmt chunk too small");
            uint8_t fmt_buf[16];
            f.read(reinterpret_cast<char*>(fmt_buf), 16);
            audio_format   = detail::read_u16le(fmt_buf);
            num_channels   = detail::read_u16le(fmt_buf + 2);
            sample_rate    = detail::read_u32le(fmt_buf + 4);
            bits_per_sample= detail::read_u16le(fmt_buf + 14);
            // Skip any extra fmt bytes
            if (chunk_size > 16)
                f.seekg(chunk_size - 16, std::ios::cur);
            have_fmt = true;

        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (!have_fmt)
                throw std::runtime_error("wav::read_wav: data chunk before fmt");
            if (audio_format != 1)
                throw std::runtime_error("wav::read_wav: only PCM (format=1) supported");
            if (num_channels == 0)
                throw std::runtime_error("wav::read_wav: zero channels");
            if (bits_per_sample != 8 && bits_per_sample != 16 &&
                bits_per_sample != 24 && bits_per_sample != 32)
                throw std::runtime_error("wav::read_wav: unsupported bit depth "
                    + std::to_string(bits_per_sample));

            data_size = chunk_size;
            have_data = true;
            break; // leave file position at start of data

        } else {
            // Unknown chunk — skip
            f.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!have_fmt)
        throw std::runtime_error("wav::read_wav: no fmt chunk");
    if (!have_data)
        throw std::runtime_error("wav::read_wav: no data chunk");

    // --- Read raw PCM bytes ---
    std::vector<uint8_t> raw(data_size);
    f.read(reinterpret_cast<char*>(raw.data()), data_size);
    const size_t bytes_read = static_cast<size_t>(f.gcount());

    const int bytes_per_sample = bits_per_sample / 8;
    const size_t n_samples = bytes_read / static_cast<size_t>(bytes_per_sample);

    // --- Convert to float32, normalised [-1, 1] ---
    AudioBuffer buf;
    buf.sample_rate = static_cast<int>(sample_rate);
    buf.channels    = static_cast<int>(num_channels);
    buf.samples.resize(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        const uint8_t* p = raw.data() + i * bytes_per_sample;
        float s = 0.0f;
        if (bits_per_sample == 8) {
            // 8-bit PCM is unsigned [0, 255]
            s = (static_cast<float>(p[0]) - 128.0f) / 128.0f;
        } else if (bits_per_sample == 16) {
            int16_t v;
            std::memcpy(&v, p, 2);
            s = static_cast<float>(v) / 32768.0f;
        } else if (bits_per_sample == 24) {
            int32_t v = static_cast<int32_t>(p[0]) |
                        (static_cast<int32_t>(p[1]) << 8) |
                        (static_cast<int32_t>(p[2]) << 16);
            if (v & 0x800000) v |= 0xFF000000; // sign-extend
            s = static_cast<float>(v) / 8388608.0f;
        } else { // 32-bit
            int32_t v;
            std::memcpy(&v, p, 4);
            s = static_cast<float>(v) / 2147483648.0f;
        }
        buf.samples[i] = s;
    }

    return buf;
}

// ---------------------------------------------------------------------------
// read_wav_mono()
// ---------------------------------------------------------------------------

/// Like read_wav() but always returns a mono AudioBuffer.
/// Multi-channel files are averaged to mono.
inline AudioBuffer read_wav_mono(const std::string& path) {
    AudioBuffer multi = read_wav(path);
    if (multi.channels == 1) return multi;

    const int ch = multi.channels;
    const size_t n_frames = multi.samples.size() / static_cast<size_t>(ch);

    AudioBuffer mono;
    mono.sample_rate = multi.sample_rate;
    mono.channels    = 1;
    mono.samples.resize(n_frames);

    for (size_t i = 0; i < n_frames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < ch; ++c)
            sum += multi.samples[i * ch + c];
        mono.samples[i] = sum / static_cast<float>(ch);
    }
    return mono;
}

// ---------------------------------------------------------------------------
// resample_mono()
// ---------------------------------------------------------------------------

/// Resample a mono float32 buffer from `src_rate` to `dst_rate` using
/// linear interpolation.  Suitable for downsampling from 44100 → 16000 Hz.
///
/// For production use a polyphase filter (e.g. libsamplerate); linear
/// interpolation is good enough for the FBANK front-end in integration tests.
inline std::vector<float> resample_mono(const std::vector<float>& in,
                                         int src_rate, int dst_rate) {
    if (src_rate == dst_rate) return in;
    if (in.empty() || src_rate <= 0 || dst_rate <= 0) return {};

    const double ratio     = static_cast<double>(src_rate) / dst_rate;
    const size_t n_out     = static_cast<size_t>(
        std::floor(static_cast<double>(in.size()) / ratio));

    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        double src_pos  = i * ratio;
        size_t idx0     = static_cast<size_t>(src_pos);
        size_t idx1     = idx0 + 1;
        float  frac     = static_cast<float>(src_pos - idx0);

        float s0 = in[idx0];
        float s1 = (idx1 < in.size()) ? in[idx1] : s0;
        out[i]   = s0 + frac * (s1 - s0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// read_wav_mono_16k()
// ---------------------------------------------------------------------------

/// Read a WAV file, downmix to mono, and resample to 16 000 Hz.
/// This is the preferred entry-point for the diarization engine since
/// WeSpeakerEcapaModel requires exactly 16 kHz input.
inline AudioBuffer read_wav_mono_16k(const std::string& path) {
    AudioBuffer buf = read_wav_mono(path);
    if (buf.sample_rate == 16000) return buf;

    buf.samples     = resample_mono(buf.samples, buf.sample_rate, 16000);
    buf.sample_rate = 16000;
    return buf;
}

} // namespace wav
