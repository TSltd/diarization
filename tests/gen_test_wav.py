#!/usr/bin/env python3
"""
Generate test WAV files for the diarization integration tests.

Produces two files:

  test_two_speakers.wav  (~15 s, mono, 16 kHz, PCM16)
      Four alternating "turns", each 3 seconds:
        Turn 0: Speaker A — 220 Hz sine + Gaussian noise (simulates speaker 1)
        Turn 1: Speaker B — 440 Hz sine + Gaussian noise (simulates speaker 2)
        Turn 2: Speaker A — repeat
        Turn 3: Speaker B — repeat

      The frequency difference is large enough for a spectral-feature embedding
      model to distinguish the two "speakers" in unit-test scenarios.
      For real-model validation use LibriSpeech or VoxConverse instead.

  test_single_speaker.wav  (~6 s, mono, 16 kHz, PCM16)
      Two turns of the same speaker (Speaker A, 220 Hz) to verify that the
      engine does NOT create a second speaker label when it isn't needed.

Usage:
  python3 gen_test_wav.py [--output-dir DIR]

Requirements:
  numpy (pip install numpy)

Outputs:
  <output_dir>/test_two_speakers.wav
  <output_dir>/test_single_speaker.wav
"""

import argparse
import math
import pathlib
import struct
import sys

try:
    import numpy as np
except ImportError:
    print("error: numpy is required — pip install numpy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# WAV writer (pure-Python, no external audio library needed)
# ---------------------------------------------------------------------------

def write_wav_pcm16(path: pathlib.Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a mono float32 array as a 16-bit PCM WAV file."""
    # Clip and convert to int16
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16   = (clipped * 32767.0).astype(np.int16)

    data_bytes = pcm16.tobytes()
    data_size  = len(data_bytes)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))   # file size - 8
        f.write(b"WAVE")
        # fmt  chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I",  16))               # chunk size
        f.write(struct.pack("<H",   1))               # PCM
        f.write(struct.pack("<H",   1))               # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))   # byte rate
        f.write(struct.pack("<H",   2))               # block align
        f.write(struct.pack("<H",  16))               # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data_bytes)

    print(f"  wrote {path}  ({len(pcm16) / sample_rate:.1f}s, {sample_rate} Hz, PCM16)")


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def sine_turn(freq_hz: float, duration_s: float, sample_rate: int,
              noise_amplitude: float = 0.05, rng: np.random.Generator = None) -> np.ndarray:
    """
    One "speaker turn": a sine wave at freq_hz with a small amount of white noise.

    The sine wave simulates a speaker's fundamental frequency; the noise makes
    successive identical turns slightly different (more realistic for embeddings).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = 0.5 * np.sin(2.0 * math.pi * freq_hz * t)
    signal += noise_amplitude * rng.standard_normal(len(signal))
    return signal.astype(np.float32)


def silence(duration_s: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


# ---------------------------------------------------------------------------
# Build test files
# ---------------------------------------------------------------------------

def build_two_speakers(sample_rate: int = 16000) -> np.ndarray:
    """
    Four 3-second turns: A B A B with 200 ms silence between turns.

    Turn timing (approximate):
      [0.0 – 3.0]   Speaker A  (220 Hz)
      [3.2 – 6.2]   Speaker B  (440 Hz)
      [6.4 – 9.4]   Speaker A  (220 Hz)
      [9.6 – 12.6]  Speaker B  (440 Hz)

    Total: ~13 seconds.
    """
    rng = np.random.default_rng(0)
    gap  = silence(0.2, sample_rate)
    turn = 3.0   # seconds per turn

    spk_a = lambda: sine_turn(220.0, turn, sample_rate, rng=rng)
    spk_b = lambda: sine_turn(440.0, turn, sample_rate, rng=rng)

    return np.concatenate([
        spk_a(), gap,
        spk_b(), gap,
        spk_a(), gap,
        spk_b(),
    ])


def build_single_speaker(sample_rate: int = 16000) -> np.ndarray:
    """
    Two 3-second turns from the same speaker (220 Hz) with a 200 ms gap.
    Total: ~6.2 seconds.
    """
    rng = np.random.default_rng(1)
    gap = silence(0.2, sample_rate)
    spk_a = lambda: sine_turn(220.0, 3.0, sample_rate, rng=rng)
    return np.concatenate([spk_a(), gap, spk_a()])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write WAV files into (default: current dir)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate

    print(f"Generating test WAV files in {out_dir.resolve()} at {sr} Hz …")

    write_wav_pcm16(out_dir / "test_two_speakers.wav",   build_two_speakers(sr),   sr)
    write_wav_pcm16(out_dir / "test_single_speaker.wav", build_single_speaker(sr), sr)

    print("\nDone.  To run integration tests:")
    print(f"  ./run_integration  ''  {out_dir}/test_two_speakers.wav")
    print(f"\nTo validate with a real model:")
    print(f"  ./run_integration  dist/models/speaker-ecapa.onnx  {out_dir}/test_two_speakers.wav")


if __name__ == "__main__":
    main()
