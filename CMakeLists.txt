# GPU NAM — a standalone neural-amp player built on the Pulp SDK.
#
# Pulp is vendored as a git submodule at ./pulp, pinned to a known commit. After
# cloning this repo, fetch it with:
#
#   git submodule update --init --recursive
#
# The submodule provides the Pulp static libraries, its CMake helpers
# (pulp_add_plugin, PulpBundleRelocatable), and the generalized GPU WaveNet
# inference primitive (pulp::render's prepare_wavenet / wavenet_forward) that
# this plugin's opt-in GPU engine drives.

cmake_minimum_required(VERSION 3.24)
project(pulp-gpu-nam LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # A Debug build of a GPU/Skia UI is dramatically slower; default to Release.
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pulp/CMakeLists.txt")
    message(FATAL_ERROR
        "The Pulp submodule is missing. From the repo root run:\n"
        "  git submodule update --init --recursive")
endif()

# Where the Pulp submodule lives — used by src/ to locate Pulp's CMake helpers
# (Pulp's own scripts key off CMAKE_SOURCE_DIR, which here is THIS repo, so we
# pass the submodule root explicitly).
set(GPU_NAM_PULP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pulp" CACHE INTERNAL "Pulp submodule root")

# Build Pulp as a lean SDK: its GPU stack ON (GPU NAM needs Dawn + Skia
# Graphite), its own examples and test suite OFF (a plugin build should not pull
# in every Pulp example/test — GPU NAM ships its own tests below).
set(PULP_ENABLE_GPU     ON  CACHE BOOL "" FORCE)
set(PULP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PULP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

# GPU NAM's own test suite (independent of Pulp's). ON by default; fetches a
# private Catch2 so the plugin tests build without turning on Pulp's whole
# test tree.
option(GPU_NAM_BUILD_TESTS "Build GPU NAM's own test suite" ON)

add_subdirectory(pulp)

if(GPU_NAM_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.6.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    include(CTest)
    include(Catch)
endif()

add_subdirectory(src)
