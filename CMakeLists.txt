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

# The plugin (and Pulp's macOS view/render/standalone layers it links) compile
# Objective-C++ (.mm). Pulp enables OBJC/OBJCXX in its own project() call, but a
# language enabled in a subdirectory's project() doesn't provide compile rules to
# this top-level project's sibling targets — enable it here at the top level.
if(APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # A Debug build of a GPU/Skia UI is dramatically slower; default to Release.
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# GPU NAM's own test suite (independent of Pulp's). ON by default; fetches a
# private Catch2 so the plugin tests build without turning on Pulp's whole
# test tree.
option(GPU_NAM_BUILD_TESTS "Build GPU NAM's own test suite" ON)

# Where the Pulp SDK comes from: the vendored ./pulp submodule built from source
# (default), or a pre-installed Pulp SDK consumed via find_package(Pulp). The
# installed-SDK path lets a release build link a versioned, prebuilt SDK instead
# of recompiling Pulp's GPU stack (Dawn + Skia) from source. Point
# CMAKE_PREFIX_PATH at the SDK install prefix when enabling it.
option(GPU_NAM_USE_INSTALLED_PULP
    "Build against an installed Pulp SDK (find_package(Pulp)) instead of the ./pulp submodule" OFF)

if(GPU_NAM_USE_INSTALLED_PULP)
    # find_package(Pulp) provides the exported targets (pulp::format, pulp::render,
    # pulp::gpu-audio, pulp::signal, pulp::view, pulp::canvas, pulp::runtime) and the
    # CMake helpers this plugin uses — pulp_add_plugin() (PulpUtils) and the
    # relocatable-bundle helpers (PulpBundleRelocatable), both included by
    # PulpConfig.cmake — so none of the submodule wiring below runs.
    find_package(Pulp REQUIRED GLOBAL)
    message(STATUS "GPU NAM: building against installed Pulp SDK ${Pulp_VERSION} (${Pulp_DIR})")

    # The install export namespaces targets as Pulp:: ; the in-tree submodule
    # build exposes them as pulp:: . src/ refers to the lowercase names, so alias
    # the imported Pulp:: targets to pulp:: to make the two SDK sources
    # interchangeable. (find_package(... GLOBAL) promotes the imported targets to
    # global scope so they can be aliased.)
    foreach(_pulp_lib runtime audio format render gpu-audio signal
                      view canvas view-core view-script)
        if(TARGET Pulp::${_pulp_lib} AND NOT TARGET pulp::${_pulp_lib})
            add_library(pulp::${_pulp_lib} ALIAS Pulp::${_pulp_lib})
        endif()
    endforeach()
else()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pulp/CMakeLists.txt")
        message(FATAL_ERROR
            "The Pulp submodule is missing. From the repo root run:\n"
            "  git submodule update --init --recursive\n"
            "Or configure with -DGPU_NAM_USE_INSTALLED_PULP=ON to build against an "
            "installed Pulp SDK (set CMAKE_PREFIX_PATH to its install prefix).")
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

    add_subdirectory(pulp)

    # add_subdirectory(pulp) defines the pulp_add_plugin() function globally, but the
    # target names it resolves (Pulp::standalone, Pulp::format, …) are computed by
    # PulpUtils in Pulp's OWN subdirectory scope and don't propagate to this sibling
    # src/ subdirectory. Re-include PulpUtils here so those are resolved in the
    # consumer's top scope and inherited by src/. (This mirrors what find_package(Pulp)
    # does for installed-SDK consumers.)
    list(APPEND CMAKE_MODULE_PATH "${GPU_NAM_PULP_DIR}/tools/cmake")
    include(PulpUtils)
endif()

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
    # Enable testing directly rather than via include(CTest): the Pulp submodule
    # (add_subdirectory(pulp), above) is built with PULP_BUILD_TESTS=OFF, which
    # leaves the cached BUILD_TESTING=OFF. The stock CTest module only calls
    # enable_testing() when BUILD_TESTING is ON, so include(CTest) would silently
    # register zero tests. GPU NAM gates its own suite on GPU_NAM_BUILD_TESTS.
    enable_testing()
    include(Catch)
endif()

add_subdirectory(src)
