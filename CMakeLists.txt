cmake_minimum_required(VERSION 3.20)

project(diarization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

set(
    ORT_ROOT
    "/home/dan/Documents/piper1-gpl/libpiper/lib/onnxruntime-linux-x64-1.22.0"
    CACHE PATH
    "Path to ONNX Runtime installation"
)

find_library(
    ORT_LIB
    onnxruntime
    HINTS "${ORT_ROOT}/lib"
    REQUIRED
)

# ─────────────────────────────────────────────────────────────────────────────
# Include paths
# ─────────────────────────────────────────────────────────────────────────────

include_directories(include)
include_directories(${CMAKE_SOURCE_DIR})

# ─────────────────────────────────────────────────────────────────────────────
# Core library
# ─────────────────────────────────────────────────────────────────────────────

add_library(diarization_core STATIC
    src/DiarizationEngine.cpp
    src/SpeakerClusterManager.cpp
    src/LabelSmoother.cpp
    src/TranscriptFormatter.cpp
)

target_include_directories(diarization_core
    PUBLIC
        include
        ${CMAKE_SOURCE_DIR}
)

# ─────────────────────────────────────────────────────────────────────────────
# Speaker embedding / ONNX library
# ─────────────────────────────────────────────────────────────────────────────

add_library(diarization_model STATIC
    models/WeSpeakerEcapaModel.cpp
    models/EcapaOnnxModel.cpp
    models/SpeechBrainEcapaModel.cpp
    models/SpeakerModelFactory.cpp
    models/SpeakerVerifier.cpp
)

target_include_directories(diarization_model
    PUBLIC
        include
        ${CMAKE_SOURCE_DIR}
        ${ORT_ROOT}/include
)

target_link_libraries(diarization_model
    PUBLIC
        ${ORT_LIB}
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper for ONNX executables
# ─────────────────────────────────────────────────────────────────────────────

function(add_onnx_executable TARGET_NAME SOURCE_FILE OUTPUT_DIR)

    add_executable(${TARGET_NAME}
        ${SOURCE_FILE}
    )

    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            DIARIZE_HAVE_MODEL
    )

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            diarization_core
            diarization_model
    )

    target_link_options(${TARGET_NAME}
        PRIVATE
            "-Wl,-rpath,${ORT_ROOT}/lib"
    )

    set_target_properties(${TARGET_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )

endfunction()

# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (no ONNX)
# ─────────────────────────────────────────────────────────────────────────────

add_executable(unit_tests
    tests/tests.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        diarization_core
)

set_target_properties(unit_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY tests
)

# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────

add_executable(integration_tests
    tests/integration_tests.cpp
)

target_link_libraries(integration_tests
    PRIVATE
        diarization_core
)

set_target_properties(integration_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY tests
)

add_onnx_executable(
    integration_tests_real
    tests/integration_tests.cpp
    tests
)

# ─────────────────────────────────────────────────────────────────────────────
# Acceptance tests
# ─────────────────────────────────────────────────────────────────────────────

add_executable(acceptance_test
    tests/acceptance_test.cpp
)

target_link_libraries(acceptance_test
    PRIVATE
        diarization_core
)

set_target_properties(acceptance_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY tests
)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

add_executable(diarization_bench
    bench/diarization_bench.cpp
)

target_link_libraries(diarization_bench
    PRIVATE
        diarization_core
)

set_target_properties(diarization_bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY bench
)

add_onnx_executable(
    diarization_bench_real
    bench/diarization_bench.cpp
    bench
)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation tools
# ─────────────────────────────────────────────────────────────────────────────

add_executable(check_trials
    tools/check_trials.cpp
)

set_target_properties(check_trials
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY tools
)

add_onnx_executable(
    embed_wav
    tools/embed_wav.cpp
    tools
)

add_onnx_executable(
    verification_trials
    tests/verification_trials.cpp
    tests
)

# ─────────────────────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────────────────────

install(
    DIRECTORY include/
    DESTINATION include
)

install(
    TARGETS diarization_core
    ARCHIVE DESTINATION lib
)