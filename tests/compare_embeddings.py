#!/usr/bin/env python3
"""
tests/compare_embeddings.py
---------------------------
Compare C++ FBANK/embedding vs Python reference on speaker01_a.wav.

Steps:
  1. Compute FBANK with numpy replica of C++ FBankFrontEnd  (Hann window)
  2. Compute FBANK with torchaudio kaldi.fbank               (Hamming window – WeSpeaker reference)
  3. Run ONNX model in Python with both FBANK variants       → py_hann_emb, py_hamming_emb
  4. Run ./tools/embed_wav binary                            → cpp_emb
  5. Report cosine similarities

Expected outcomes:
  cosine(py_hann,    cpp)     ≈ 1.00  → front-end matches, no fix needed
  cosine(py_hamming, cpp)     ≈ 1.00  → C++ should use Hamming window
  cosine(py_hann,    py_hamming) shows the window difference in isolation
"""

import subprocess, sys, os
import numpy as np
import soundfile as sf
import onnxruntime as ort

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_PATH   = os.path.join(REPO_ROOT, "testdata/verification/speaker01_a.wav")
MODEL_PATH = os.path.join(REPO_ROOT, "wespeaker/voxceleb_ECAPA512_LM.onnx")
EMBED_BIN  = os.path.join(REPO_ROOT, "tools/embed_wav")

# ── helpers ───────────────────────────────────────────────────────────────────
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

# ── numpy FBANK  (exact replica of C++ FBankFrontEnd) ─────────────────────────
def cpp_fbank_numpy(pcm: np.ndarray, window: str = "hann") -> np.ndarray:
    """
    Parameters match FBankFrontEnd.h:
      sample_rate=16000, frame_len=400, frame_shift=160, fft_size=512,
      mel_bins=80, pre_emph=0.97, freq_low=20, freq_high=7600,
      mel_scale=O'Shaughnessy (2595*log10), log=natural, no CMVN
    """
    SR        = 16000
    FRAME_LEN = 400
    FRAME_SHF = 160
    FFT_SIZE  = 512
    MEL_BINS  = 80
    PRE_EMPH  = 0.97
    FREQ_LOW  = 20.0
    FREQ_HIGH = 7600.0

    n_fft_bins = FFT_SIZE // 2 + 1

    # Window
    if window == "hann":
        win = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1)))
    elif window == "hamming":
        win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))
    else:
        raise ValueError(f"Unknown window: {window}")

    # Mel filterbank (O'Shaughnessy)
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_low  = hz_to_mel(FREQ_LOW)
    mel_high = hz_to_mel(FREQ_HIGH)
    n_pts    = MEL_BINS + 2
    mel_pts  = np.linspace(mel_low, mel_high, n_pts)
    hz_pts   = mel_to_hz(mel_pts)
    bins     = np.floor((FFT_SIZE + 1) * hz_pts / SR).astype(int)
    bins     = np.clip(bins, 0, n_fft_bins - 1)

    fb = np.zeros((MEL_BINS, n_fft_bins), dtype=np.float32)
    for m in range(MEL_BINS):
        left, centre, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, centre):
            fb[m, k] = (k - left + 1.0) / (centre - left + 1.0)
        for k in range(centre, right + 1):
            fb[m, k] = (right - k + 1.0) / (right - centre + 1.0)

    # Framing
    n_frames = (len(pcm) - FRAME_LEN) // FRAME_SHF + 1
    feats = np.zeros((n_frames, MEL_BINS), dtype=np.float32)

    for f in range(n_frames):
        start = f * FRAME_SHF
        frame = pcm[start : start + FRAME_LEN].copy()

        # Pre-emphasis (matching C++: sample[0] not emphasised)
        emph = frame.copy()
        emph[1:] = frame[1:] - PRE_EMPH * frame[:-1]
        # emph[0] stays as frame[0]

        # Window
        windowed = emph * win

        # FFT
        spectrum = np.fft.rfft(windowed, n=FFT_SIZE)
        power    = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)

        # Mel + log
        energy = fb @ power
        feats[f] = np.log(np.maximum(energy, 1e-10))

    return feats  # [n_frames, 80]


# ── torchaudio kaldi.fbank (WeSpeaker reference) ──────────────────────────────
def torchaudio_fbank(wav_path: str) -> np.ndarray:
    try:
        import torch
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
    except ImportError:
        print("  [SKIP] torchaudio not available — skipping Hamming reference")
        return None

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000,
        window_type='hamming',
        use_log_fbank=True,
        use_power=True,
        high_freq=7600.0,
        low_freq=20.0,
    )
    return feats.numpy().astype(np.float32)  # [n_frames, 80]


# ── ONNX inference ─────────────────────────────────────────────────────────────
def onnx_embed(feats: np.ndarray, model_path: str) -> np.ndarray:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp  = feats[np.newaxis, :, :].astype(np.float32)   # [1, T, 80]
    out  = sess.run(None, {"feats": inp})
    emb  = out[0].flatten()
    return l2norm(emb)


# ── C++ embedding via embed_wav binary ────────────────────────────────────────
def cpp_embed(wav_path: str, model_path: str, embed_bin: str) -> np.ndarray:
    result = subprocess.run(
        [embed_bin, model_path, wav_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] embed_wav returned {result.returncode}")
        print(result.stderr)
        return None
    vals = [float(x) for x in result.stdout.strip().split("\n") if x.strip()]
    if not vals:
        print("  [ERROR] embed_wav produced no output")
        return None
    return l2norm(np.array(vals, dtype=np.float32))


# ── FBANK frame comparison ────────────────────────────────────────────────────
def compare_fbank_frames(fa: np.ndarray, fb_: np.ndarray,
                          name_a: str, name_b: str, n_show: int = 3) -> None:
    min_frames = min(len(fa), len(fb_))
    fa = fa[:min_frames]; fb_ = fb_[:min_frames]
    mae = np.mean(np.abs(fa - fb_))
    max_e = np.max(np.abs(fa - fb_))
    print(f"  FBANK diff  {name_a} vs {name_b}:  MAE={mae:.4f}  max={max_e:.4f}")
    for f in range(min(n_show, min_frames)):
        print(f"    frame[{f}]  {name_a}: {fa[f, :6].tolist()}")
        print(f"    frame[{f}]  {name_b}: {fb_[f, :6].tolist()}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  FBANK / Embedding comparison: C++ vs Python reference")
    print("=" * 62)

    # Check files exist
    for p, label in [(WAV_PATH, "WAV"), (MODEL_PATH, "ONNX model"), (EMBED_BIN, "embed_wav binary")]:
        if not os.path.exists(p):
            print(f"\n[ERROR] {label} not found: {p}")
            sys.exit(1)

    print(f"\nWAV   : {WAV_PATH}")
    print(f"Model : {MODEL_PATH}")

    # Load PCM
    pcm, sr = sf.read(WAV_PATH, dtype='float32')
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != 16000:
        print(f"[WARN] sample rate {sr}, expected 16000 — resample not implemented here")
    print(f"Audio : {len(pcm)} samples  ({len(pcm)/sr:.2f}s)  @ {sr} Hz\n")

    # ── 1. FBANK variants ──────────────────────────────────────────────────────
    print("── Step 1: FBANK computation ─────────────────────────────")
    feats_hann    = cpp_fbank_numpy(pcm, window="hann")
    feats_hamming = cpp_fbank_numpy(pcm, window="hamming")
    feats_torch   = torchaudio_fbank(WAV_PATH)

    print(f"  numpy/Hann    : {feats_hann.shape}")
    print(f"  numpy/Hamming : {feats_hamming.shape}")
    if feats_torch is not None:
        print(f"  torchaudio    : {feats_torch.shape}")

    compare_fbank_frames(feats_hann,    feats_hamming, "numpy/Hann", "numpy/Hamming")
    if feats_torch is not None:
        compare_fbank_frames(feats_hamming, feats_torch, "numpy/Hamming", "torchaudio")
        compare_fbank_frames(feats_hann,    feats_torch, "numpy/Hann",    "torchaudio")

    # ── 2. Python ONNX embeddings ──────────────────────────────────────────────
    print("\n── Step 2: Python ONNX embeddings ───────────────────────")
    emb_py_hann    = onnx_embed(feats_hann,    MODEL_PATH)
    emb_py_hamming = onnx_embed(feats_hamming, MODEL_PATH)
    emb_py_torch   = onnx_embed(feats_torch,   MODEL_PATH) if feats_torch is not None else None
    print(f"  py/Hann    dim={len(emb_py_hann)}  norm={np.linalg.norm(emb_py_hann):.6f}")
    print(f"  py/Hamming dim={len(emb_py_hamming)}  norm={np.linalg.norm(emb_py_hamming):.6f}")
    if emb_py_torch is not None:
        print(f"  py/torch   dim={len(emb_py_torch)}  norm={np.linalg.norm(emb_py_torch):.6f}")

    # ── 3. C++ embedding ──────────────────────────────────────────────────────
    print("\n── Step 3: C++ embedding (embed_wav binary) ──────────────")
    emb_cpp = cpp_embed(WAV_PATH, MODEL_PATH, EMBED_BIN)
    if emb_cpp is not None:
        print(f"  cpp dim={len(emb_cpp)}  norm={np.linalg.norm(emb_cpp):.6f}")

    # ── 4. Cosine comparisons ─────────────────────────────────────────────────
    print("\n── Step 4: Cosine similarity summary ────────────────────")
    results = [
        ("py/Hann    vs py/Hamming",  emb_py_hann,    emb_py_hamming),
        ("py/Hann    vs py/torch",    emb_py_hann,    emb_py_torch),
        ("py/Hamming vs py/torch",    emb_py_hamming, emb_py_torch),
    ]
    if emb_cpp is not None:
        results += [
            ("py/Hann    vs C++",     emb_py_hann,    emb_cpp),
            ("py/Hamming vs C++",     emb_py_hamming, emb_cpp),
            ("py/torch   vs C++",     emb_py_torch,   emb_cpp),
        ]

    for label, a, b in results:
        if a is None or b is None:
            print(f"  {label:<30}  SKIPPED (missing embedding)")
            continue
        sim = cosine(a, b)
        verdict = "✓ MATCH" if sim > 0.99 else ("~ close" if sim > 0.95 else "✗ MISMATCH")
        print(f"  {label:<30}  cosine = {sim:.6f}  {verdict}")

    # ── 5. Diagnosis ──────────────────────────────────────────────────────────
    print("\n── Step 5: Diagnosis ────────────────────────────────────")
    if emb_cpp is None:
        print("  Cannot diagnose: C++ binary failed.")
        return

    sim_hann    = cosine(emb_py_hann,    emb_cpp)
    sim_hamming = cosine(emb_py_hamming, emb_cpp)

    if sim_hann > 0.99:
        print("  ✓ C++ front-end (Hann) matches Python reference exactly.")
        print("    The high different-speaker cosine is likely a LibriSpeech")
        print("    characteristic, not a front-end bug. No fix needed.")
    elif sim_hamming > 0.99:
        print("  ✗ C++ Hann window does NOT match; Hamming window DOES.")
        print("    → Fix: change FBankFrontEnd.h to use Hamming window.")
        print("       hann_win_[i] = 0.54 - 0.46*cos(2π*i/(N-1))")
    elif sim_hamming > sim_hann:
        print(f"  ~ Hamming is closer ({sim_hamming:.4f}) than Hann ({sim_hann:.4f}).")
        print("    Partial mismatch — window is one factor but likely not the only one.")
    else:
        print(f"  ✗ Neither Hann ({sim_hann:.4f}) nor Hamming ({sim_hamming:.4f}) match well.")
        print("    The mismatch is deeper than the window function.")
        print("    Check: mel scale (Kaldi vs O'Shaughnessy), power vs magnitude,")
        print("           log base (natural vs log10), CMVN, normalisation.")


if __name__ == "__main__":
    main()
