cmake_minimum_required(VERSION 3.20)
project(diarization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
set(ORT_ROOT "/home/dan/Documents/piper1-gpl/libpiper/lib/onnxruntime-linux-x64-1.22.0"
    CACHE PATH "Path to ONNX Runtime installation")

find_library(ORT_LIB onnxruntime HINTS "${ORT_ROOT}/lib" REQUIRED)

# ── Include paths ─────────────────────────────────────────────────────────────
# -Iinclude  → allows  #include <diarization/Foo.h>
# -I.        → allows  #include "models/Foo.h" etc. from repo root
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR})

# ── Core library ─────────────────────────────────────────────────────────────
add_library(diarization_core STATIC
    src/DiarizationEngine.cpp
    src/SpeakerClusterManager.cpp
    src/LabelSmoother.cpp
    src/TranscriptFormatter.cpp
)
target_include_directories(diarization_core PUBLIC include ${CMAKE_SOURCE_DIR})

# ── WeSpeaker ONNX model adapter ──────────────────────────────────────────────
add_library(diarization_model STATIC
    models/WeSpeakerEcapaModel.cpp
    models/EcapaOnnxModel.cpp
    models/SpeechBrainEcapaModel.cpp
    models/SpeakerModelFactory.cpp
    models/SpeakerVerifier.cpp
)
target_include_directories(diarization_model PUBLIC
    include
    ${CMAKE_SOURCE_DIR}
    "${ORT_ROOT}/include"
)
target_link_libraries(diarization_model PUBLIC ${ORT_LIB})

# ── Tests ─────────────────────────────────────────────────────────────────────

# unit_tests  (no ONNX)
add_executable(unit_tests tests/tests.cpp)
target_link_libraries(unit_tests PRIVATE diarization_core)
set_target_properties(unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY tests)

# integration_tests stub  (no ONNX)
add_executable(integration_tests tests/integration_tests.cpp)
target_link_libraries(integration_tests PRIVATE diarization_core)
set_target_properties(integration_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY tests)

# integration_tests_real  (real ONNX model)
add_executable(integration_tests_real tests/integration_tests.cpp)
target_compile_definitions(integration_tests_real PRIVATE DIARIZE_HAVE_MODEL)
target_link_libraries(integration_tests_real PRIVATE diarization_core diarization_model)
target_link_options(integration_tests_real PRIVATE
    "-Wl,-rpath,${ORT_ROOT}/lib"
)
set_target_properties(integration_tests_real PROPERTIES RUNTIME_OUTPUT_DIRECTORY tests)

# acceptance_test  (no ONNX)
add_executable(acceptance_test tests/acceptance_test.cpp)
target_link_libraries(acceptance_test PRIVATE diarization_core)
set_target_properties(acceptance_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY tests)

# ── Benchmarks ────────────────────────────────────────────────────────────────

# diarization_bench stub  (no ONNX)
add_executable(diarization_bench bench/diarization_bench.cpp)
target_link_libraries(diarization_bench PRIVATE diarization_core)
set_target_properties(diarization_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY bench)

# diarization_bench_real  (real ONNX model)
add_executable(diarization_bench_real bench/diarization_bench.cpp)
target_compile_definitions(diarization_bench_real PRIVATE DIARIZE_HAVE_MODEL)
target_link_libraries(diarization_bench_real PRIVATE diarization_core diarization_model)
target_link_options(diarization_bench_real PRIVATE
    "-Wl,-rpath,${ORT_ROOT}/lib"
)
set_target_properties(diarization_bench_real PROPERTIES RUNTIME_OUTPUT_DIRECTORY bench)

# ── Install (optional) ────────────────────────────────────────────────────────
install(DIRECTORY include/ DESTINATION include)
install(TARGETS diarization_core ARCHIVE DESTINATION lib)
