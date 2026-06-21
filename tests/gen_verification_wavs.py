#!/usr/bin/env python3
"""
gen_verification_wavs.py — populate testdata/verification/ with real speaker clips.

Downloads 10 speakers × 3 clips from the LibriSpeech test-clean corpus via
HuggingFace datasets (streaming mode — no full corpus download required).

Output layout
─────────────
  testdata/verification/
    speaker01_a.wav   speaker01_b.wav   speaker01_c.wav
    speaker02_a.wav   ...
    ...
    speaker10_a.wav   speaker10_b.wav   speaker10_c.wav

File format: mono, 16 kHz, PCM-16 WAV, 3–10 seconds.

Dependencies
────────────
  pip install datasets soundfile

Usage
─────
  python tests/gen_verification_wavs.py [--out-dir testdata/verification]
"""

import argparse
import io
import os
import sys
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
missing = []
try:
    from datasets import load_dataset, Audio as HFAudio
except ImportError:
    missing.append("datasets")
try:
    import soundfile as sf
except ImportError:
    missing.append("soundfile")
try:
    import numpy as np
except ImportError:
    missing.append("numpy")

if missing:
    print(f"Missing Python packages: {', '.join(missing)}")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_SPEAKERS     = 10
CLIPS_PER_SPEAKER   = 3
CLIP_LABELS         = ['a', 'b', 'c']
TARGET_SR           = 16_000
MIN_DURATION_S      = 3.0
MAX_DURATION_S      = 10.0

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "testdata" / "verification"),
                   help="Output directory (default: testdata/verification)")
    p.add_argument("--speakers", type=int, default=TARGET_SPEAKERS,
                   help="Number of speakers to collect (default: 10)")
    p.add_argument("--clips", type=int, default=CLIPS_PER_SPEAKER,
                   help="Clips per speaker (default: 3)")
    p.add_argument("--min-dur", type=float, default=MIN_DURATION_S,
                   help="Minimum clip duration in seconds (default: 3.0)")
    p.add_argument("--max-dur", type=float, default=MAX_DURATION_S,
                   help="Maximum clip duration in seconds (default: 10.0)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files")
    return p.parse_args()


def resample_linear(samples: "np.ndarray", src_rate: int, dst_rate: int) -> "np.ndarray":
    """Linear-interpolation resample (good enough for 16 kHz target)."""
    if src_rate == dst_rate:
        return samples
    ratio = src_rate / dst_rate
    n_out = int(len(samples) / ratio)
    indices = np.arange(n_out) * ratio
    idx0 = indices.astype(int)
    idx1 = np.clip(idx0 + 1, 0, len(samples) - 1)
    frac = (indices - idx0).astype(np.float32)
    return samples[idx0] + frac * (samples[idx1] - samples[idx0])


def pcm16(samples: "np.ndarray") -> "np.ndarray":
    """Convert float32 [-1, 1] → int16."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_speakers = args.speakers
    n_clips    = args.clips
    labels     = [chr(ord('a') + i) for i in range(n_clips)]

    print("══════════════════════════════════════════")
    print("  gen_verification_wavs.py")
    print(f"  target: {n_speakers} speakers × {n_clips} clips = {n_speakers * n_clips} files")
    print(f"  output: {out_dir}")
    print("══════════════════════════════════════════")
    print()

    # ── check which files already exist ──────────────────────────────────────
    needed = {}   # speaker_label -> list of missing clip labels
    for s in range(1, n_speakers + 1):
        spk = f"speaker{s:02d}"
        missing_clips = []
        for lbl in labels:
            path = out_dir / f"{spk}_{lbl}.wav"
            if not path.exists() or args.force:
                missing_clips.append(lbl)
            else:
                print(f"  [skip] {spk}_{lbl}.wav  (already exists)")
        if missing_clips:
            needed[spk] = missing_clips

    if not needed:
        print()
        print("  All files already present.  Use --force to re-download.")
        return 0

    total_needed = sum(len(v) for v in needed.values())
    print(f"  Need to download {total_needed} clip(s) for "
          f"{len(needed)} speaker(s).\n")

    # ── stream LibriSpeech test-clean ─────────────────────────────────────────
    print("  Connecting to HuggingFace (LibriSpeech test-clean, streaming)…")
    ds = None
    candidates = [
        # Newer HF datasets library requires namespace/name
        ("openslr/librispeech_asr", "clean",       "test"),
        ("openslr/librispeech_asr", "clean",       "test.clean"),
        # Legacy short name (older datasets versions)
        ("librispeech_asr",         "clean",       "test"),
        ("librispeech_asr",         "clean",       "test.clean"),
    ]
    last_exc = None
    for repo, cfg, split in candidates:
        try:
            ds = load_dataset(repo, name=cfg, split=split, streaming=True)
            print(f"  Using dataset: {repo}  config={cfg}  split={split}")
            break
        except Exception as exc:
            last_exc = exc

    if ds is None:
        print(f"  ERROR loading dataset: {last_exc}")
        sys.exit(1)

    # Disable HF audio decoding (avoids torchcodec dependency in new versions).
    # We get raw bytes and decode with soundfile ourselves.
    try:
        ds = ds.cast_column("audio", HFAudio(decode=False))
    except Exception:
        pass  # older datasets versions ignore this; they decode automatically

    print("  Streaming…  (this may take a minute on first run)\n")

    # speaker_id (int) → list of np.float32 arrays
    collected: dict[int, list] = {}

    all_done = False
    for item in ds:
        if all_done:
            break

        sid   = item["speaker_id"]          # int
        audio = item["audio"]

        # ── decode audio ────────────────────────────────────────────────────
        # With decode=False: audio = {"bytes": b"...", "path": "..."}
        # With decode=True  (old): audio = {"array": [...], "sampling_rate": N}
        if isinstance(audio, dict) and "bytes" in audio and audio["bytes"] is not None:
            arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32",
                              always_2d=False)
            arr = arr.astype(np.float32)
            if arr.ndim == 2:               # stereo → mono
                arr = arr.mean(axis=1)
        elif isinstance(audio, dict) and "array" in audio:
            arr = np.array(audio["array"], dtype=np.float32)
            sr  = int(audio["sampling_rate"])
        else:
            continue                        # unrecognised format, skip

        dur = len(arr) / sr

        # duration filter
        if dur < args.min_dur or dur > args.max_dur:
            continue

        # stop collecting new speakers once we have enough
        if sid not in collected:
            if len(collected) >= n_speakers:
                continue                     # skip speakers beyond our quota
            collected[sid] = []

        if len(collected[sid]) < n_clips:
            # resample if needed
            if sr != TARGET_SR:
                arr = resample_linear(arr, sr, TARGET_SR)
                sr  = TARGET_SR
            collected[sid].append(arr)

        # check done
        done = sum(1 for clips in collected.values() if len(clips) >= n_clips)
        if done >= n_speakers:
            all_done = True

    # ── write WAV files ───────────────────────────────────────────────────────
    print()
    written = 0
    for s_idx, (sid, clips) in enumerate(
            sorted(collected.items())[:n_speakers], start=1):
        spk = f"speaker{s_idx:02d}"
        for c_idx, arr in enumerate(clips[:n_clips]):
            lbl = labels[c_idx]
            out_path = out_dir / f"{spk}_{lbl}.wav"
            dur = len(arr) / TARGET_SR
            sf.write(str(out_path), pcm16(arr), TARGET_SR, subtype="PCM_16")
            print(f"  [saved] {spk}_{lbl}.wav  "
                  f"({dur:.1f} s,  LibriSpeech speaker {sid})")
            written += 1

    print()
    print(f"  ✓  {written} file(s) written to {out_dir}")

    # ── verify expected file count ────────────────────────────────────────────
    wav_count = len(list(out_dir.glob("speaker*.wav")))
    expected  = n_speakers * n_clips
    if wav_count >= expected:
        print(f"  ✓  {wav_count} WAV files present  ({expected} required)")
        return 0
    else:
        print(f"  ✗  Only {wav_count} WAV files present (need {expected})")
        print("     Not enough distinct speakers in the test-clean split.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
