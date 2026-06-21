#!/usr/bin/env bash
# ci_run.sh — full clean-build + test run
# Run from the repo root:  bash ci_run.sh 2>&1 | tee ci_run.log

set -euo pipefail
cd "$(dirname "$0")"

ORT=/home/dan/Documents/piper1-gpl/libpiper/lib/onnxruntime-linux-x64-1.22.0
ORT_INC="$ORT/include"
ORT_LIB="$ORT/lib"
MODEL="wespeaker/voxceleb_ECAPA512_LM.onnx"
AUDIO1="testdata/audio/M_1011_13y10m_1.wav"   # 112 s
AUDIO2="testdata/audio/M_1017_11y8m_1.wav"    # 300 s

CXX="g++ -std=c++20 -O2"
SRCS_CORE="src/DiarizationEngine.cpp src/SpeakerClusterManager.cpp src/LabelSmoother.cpp src/TranscriptFormatter.cpp"

PASS=0; FAIL=0

pass() { echo "  ✓  $1"; PASS=$((PASS+1)); }
fail() { echo "  ✗  $1"; FAIL=$((FAIL+1)); }

section() { echo; echo "══════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════"; }

# ─── 1. CLEAN ────────────────────────────────────────────────────────────────
section "1. Clean"
rm -f tests/unit_tests tests/integration_tests tests/integration_tests_real \
      tests/acceptance_test bench/diarization_bench bench/diarization_bench_real
echo "  cleaned."

# ─── 2. BUILD ────────────────────────────────────────────────────────────────
section "2. Build (from clean)"

echo "  [build] unit_tests..."
$CXX -Iinclude -I. \
  tests/tests.cpp $SRCS_CORE \
  -o tests/unit_tests 2>&1 \
  && pass "unit_tests built" || { fail "unit_tests BUILD FAILED"; exit 1; }

echo "  [build] integration_tests (stub)..."
$CXX -Iinclude -I. \
  tests/integration_tests.cpp $SRCS_CORE \
  -o tests/integration_tests 2>&1 \
  && pass "integration_tests built" || { fail "integration_tests BUILD FAILED"; exit 1; }

echo "  [build] integration_tests_real (ONNX model)..."
$CXX -Iinclude -I. -DDIARIZE_HAVE_MODEL \
  -I"$ORT_INC" \
  tests/integration_tests.cpp $SRCS_CORE \
  models/WeSpeakerEcapaModel.cpp models/EcapaOnnxModel.cpp \
  models/SpeechBrainEcapaModel.cpp models/SpeakerModelFactory.cpp models/SpeakerVerifier.cpp \
  -L"$ORT_LIB" -Wl,-rpath,"$ORT_LIB" -lonnxruntime \
  -o tests/integration_tests_real 2>&1 \
  && pass "integration_tests_real built" || fail "integration_tests_real BUILD FAILED"

echo "  [build] acceptance_test..."
$CXX -Iinclude -I. \
  tests/acceptance_test.cpp $SRCS_CORE \
  -o tests/acceptance_test 2>&1 \
  && pass "acceptance_test built" || { fail "acceptance_test BUILD FAILED"; exit 1; }

echo "  [build] diarization_bench (stub)..."
$CXX -Iinclude -I. \
  bench/diarization_bench.cpp $SRCS_CORE \
  -o bench/diarization_bench 2>&1 \
  && pass "diarization_bench built" || { fail "diarization_bench BUILD FAILED"; exit 1; }

echo "  [build] diarization_bench_real (ONNX model)..."
$CXX -Iinclude -I. -DDIARIZE_HAVE_MODEL \
  -I"$ORT_INC" \
  bench/diarization_bench.cpp $SRCS_CORE \
  models/WeSpeakerEcapaModel.cpp models/EcapaOnnxModel.cpp \
  models/SpeechBrainEcapaModel.cpp models/SpeakerModelFactory.cpp models/SpeakerVerifier.cpp \
  -L"$ORT_LIB" -Wl,-rpath,"$ORT_LIB" -lonnxruntime \
  -o bench/diarization_bench_real 2>&1 \
  && pass "diarization_bench_real built" || fail "diarization_bench_real BUILD FAILED"

ls -lh tests/unit_tests tests/integration_tests tests/acceptance_test bench/diarization_bench

# ─── 3. UNIT TESTS ───────────────────────────────────────────────────────────
section "3. Unit tests"
./tests/unit_tests \
  && pass "unit tests PASSED" || fail "unit tests FAILED"

# ─── 4. INTEGRATION TESTS (stub) ─────────────────────────────────────────────
section "4. Integration tests (stub model)"
./tests/integration_tests "" testdata/audio/test_two_speakers.wav \
  && pass "integration tests (stub) PASSED" || fail "integration tests (stub) FAILED"

# ─── 5. INTEGRATION TESTS (real model) ───────────────────────────────────────
section "5. Integration tests (real ONNX model)"
if [[ -x tests/integration_tests_real ]]; then
  ./tests/integration_tests_real "$MODEL" testdata/audio/test_two_speakers.wav \
    && pass "integration tests (real model) PASSED" \
    || fail "integration tests (real model) FAILED"
else
  echo "  (skipped — build failed)"
fi

# ─── 6. ACCEPTANCE TEST ──────────────────────────────────────────────────────
section "6. Acceptance test"
./tests/acceptance_test testdata/audio/test_two_speakers.wav --all-formats \
  && pass "acceptance test PASSED" || fail "acceptance test FAILED"

# ─── 7. BENCHMARK — stub, audio1 (112 s) ─────────────────────────────────────
section "7. Benchmark — stub model, $AUDIO1"
./bench/diarization_bench "$AUDIO1" --runs 3 \
  && pass "bench (stub, audio1) PASSED" || fail "bench (stub, audio1) FAILED"

# ─── 8. BENCHMARK — real model, audio1 ───────────────────────────────────────
section "8. Benchmark — real ONNX model, $AUDIO1"
if [[ -x bench/diarization_bench_real ]]; then
  ./bench/diarization_bench_real "$AUDIO1" --model "$MODEL" --runs 2 \
    && pass "bench (real model, audio1) PASSED" \
    || fail "bench (real model, audio1) FAILED"
else
  echo "  (skipped — build failed)"
fi

# ─── 9. FINAL SUMMARY ────────────────────────────────────────────────────────
section "CI Summary"
echo "  Pass: $PASS"
echo "  Fail: $FAIL"
if [[ $FAIL -eq 0 ]]; then
  echo "  ✓  ALL CHECKS PASSED"
  exit 0
else
  echo "  ✗  $FAIL CHECK(S) FAILED"
  exit 1
fi
