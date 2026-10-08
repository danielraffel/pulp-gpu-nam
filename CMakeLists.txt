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
option(GPU_NAM_BUILD_FORGE_ADAPTER
    "Expose an opt-in in-process Processor factory for Forge graph consumers" OFF)
option(GPU_NAM_NATIVE_HOST_PROBE
    "Add a diagnostic-only CLAP query and host test (never distribute this build)" OFF)
if(GPU_NAM_NATIVE_HOST_PROBE AND NOT GPU_NAM_BUILD_TESTS)
    message(FATAL_ERROR "GPU_NAM_NATIVE_HOST_PROBE requires GPU_NAM_BUILD_TESTS")
endif()

option(GPU_NAM_NORMAL_PHYSICAL_LIFECYCLE
    "Permit manual normal lifecycle testing against the pinned admission-safe SDK" OFF)
option(GPU_NAM_SILENT_DEVICE_PROBE
    "Build manual silent physical audio callback probe (never registered with CTest)" OFF)
if(GPU_NAM_NORMAL_PHYSICAL_LIFECYCLE AND NOT GPU_NAM_SILENT_DEVICE_PROBE)
    message(FATAL_ERROR "Normal physical lifecycle requires GPU_NAM_SILENT_DEVICE_PROBE")
endif()
if(GPU_NAM_SILENT_DEVICE_PROBE AND NOT GPU_NAM_NATIVE_HOST_PROBE)
    message(FATAL_ERROR "Silent device probe requires GPU_NAM_NATIVE_HOST_PROBE")
endif()

option(GPU_NAM_BUILD_GPU_AUDIO_CAPABILITY_PROBE
    "Build the optional Pulp GPU-audio capability compatibility probe" ON)

# Validation-only bridge for Pulp's provisional opaque shared WaveNet session.
# This remains OFF until the corresponding Pulp API lands. Enabling it requires
# a Pulp checkout containing GpuWaveNetSession; the committed submodule pin
# intentionally stays on the released legacy API.
option(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION
    "Build the provisional shared-memory WaveNet session adapter (default OFF)" OFF)
option(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET
    "Build SDK-owned stamped WaveNet consumer validation (requires new SDK)" OFF)
option(GPU_NAM_MAX_INFLIGHT_API
    "Enable the SDK multi-flight max_inflight field (requires that API in the installed SDK)" OFF)
option(GPU_NAM_EXPERIMENTAL_CPU_SHADOW_WORKER
    "Build the opt-in stateful CPU-shadow-worker readiness experiment" OFF)
if(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET AND GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
    message(FATAL_ERROR "Select only one experimental WaveNet adapter")
endif()
if(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
    add_compile_definitions(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET=1)
endif()
if(GPU_NAM_MAX_INFLIGHT_API)
    add_compile_definitions(PULP_GPU_WAVENET_MAX_INFLIGHT_API=1)
endif()
if(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
    add_compile_definitions(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION=1)
endif()

# Where the Pulp SDK comes from: the vendored ./pulp submodule built from source
# (default), or a pre-installed Pulp SDK consumed via find_package(Pulp). The
# installed-SDK path lets a release build link a versioned, prebuilt SDK instead
# of recompiling Pulp's GPU stack (Dawn + Skia) from source. Point
# CMAKE_PREFIX_PATH at the SDK install prefix when enabling it.
option(GPU_NAM_USE_INSTALLED_PULP
    "Build against an installed Pulp SDK (find_package(Pulp)) instead of the ./pulp submodule" OFF)

set(GPU_NAM_EXPECTED_PULP_SOURCE_SHA "" CACHE STRING
    "Optional exact installed Pulp SDK source commit for validation")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpuNamSdkContract.cmake")
if(NOT GPU_NAM_USE_INSTALLED_PULP)
    gpu_nam_validate_sdk_contract()
endif()

if(GPU_NAM_USE_INSTALLED_PULP)
    # find_package(Pulp) provides the exported targets (pulp::format, pulp::render,
    # pulp::gpu-audio, pulp::signal, pulp::view, pulp::canvas, pulp::runtime) and the
    # CMake helpers this plugin uses — pulp_add_plugin() (PulpUtils) and the
    # relocatable-bundle helpers (PulpBundleRelocatable), both included by
    # PulpConfig.cmake — so none of the submodule wiring below runs.
    find_package(Pulp REQUIRED GLOBAL)
    gpu_nam_validate_sdk_contract()
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

    # GPU NAM's engine needs the SDK's GPU stack. A Pulp SDK built without GPU
    # (no pulp::render / pulp::gpu-audio) would otherwise configure "successfully"
    # but silently skip every GpuNam plugin target in src/ (guarded on those
    # targets), leaving a build with only the auxiliary test binaries. The
    # submodule path forces PULP_ENABLE_GPU=ON, so fail loudly here to match.
    foreach(_pulp_gpu_req render gpu-audio)
        if(NOT TARGET pulp::${_pulp_gpu_req})
            message(FATAL_ERROR
                "GPU_NAM_USE_INSTALLED_PULP=ON but the installed Pulp SDK provides no "
                "pulp::${_pulp_gpu_req} target. GPU NAM requires a GPU-enabled SDK "
                "(built with PULP_REQUIRE_GPU_FOR_SDK=ON / PULP_ENABLE_GPU=ON). "
                "Install a GPU SDK, or drop -DGPU_NAM_USE_INSTALLED_PULP=ON to build "
                "the vendored submodule.")
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
    # Exact-provider validation is a configure-time provider identity check.
    # Keep Pulp's full unit-test tree disabled in the consumer: its test data
    # and helper paths are rooted at a standalone Pulp checkout, while GPU-NAM
    # owns the downstream validation suite. The exact proof still runs from
    # Pulp's GPU-audio provider CMake surface.
    set(PULP_BUILD_TESTS OFF CACHE BOOL "" FORCE)

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
    include(Catch)
endif()

# Keep CTest available for the standalone capability probe and for Pulp's exact
# provider suite. The latter is built by the vendored Pulp subdirectory when
# PULP_GPU_AUDIO_EXACT_PROVIDER_PROOF=ON, even if GPU-NAM's own Catch2 suite is
# disabled for a focused validation build.
if(GPU_NAM_BUILD_TESTS OR GPU_NAM_BUILD_GPU_AUDIO_CAPABILITY_PROBE OR
   PULP_GPU_AUDIO_EXACT_PROVIDER_PROOF)
    enable_testing()
endif()

add_subdirectory(src)

if(GPU_NAM_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME gpu-nam-sdk-contract
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_sdk_contract.py)
    # The production receipt runner is intentionally an explicit operator
    # command because it needs a distribution SDK and a real GPU provider.  Its
    # parser/positive/typed-negative contract is still a normal CTest gate so a
    # change cannot silently weaken the fail-closed receipt before an operator
    # runs the hardware-backed lane.
    add_test(NAME gpu-nam-production-receipt-contract
        COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/validation/test_gpu_nam_production_receipt.py)
    set_tests_properties(gpu-nam-production-receipt-contract PROPERTIES
        LABELS "validation;receipt")
endif()
