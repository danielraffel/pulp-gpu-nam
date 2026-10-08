# GPU NAM — fused block-parallel WaveNet neural-amp inference on the GPU.
#
# This is the plugin build for the standalone pulp-gpu-nam repo. Pulp is provided
# by the ./pulp submodule (added at the repo root); this file uses Pulp's targets
# (pulp::render, pulp::gpu-audio, …) and helpers (pulp_add_plugin,
# PulpBundleRelocatable) exactly as an in-tree Pulp example would.
#
# Per-architecture build toggles for the reusable NAM inference substrate
# (nam_runtime.hpp + the per-arch headers). All default ON; turn one OFF to build
# a smaller binary that supports only the subset you need — e.g.
# `-DGPU_NAM_WITH_LSTM=OFF`. The toggles pass GPU_NAM_WITH_<ARCH>=0 to the
# substrate + its tests; a define is emitted only for a DISABLED arch, so the
# default all-on build is byte-for-byte unchanged. (The full GPU plugin ships all
# architectures; the WaveNet GPU engine is WaveNet-specific, so a subset build
# that drops WaveNet targets the CPU inference layer, not the GPU plugin.)
option(GPU_NAM_WITH_WAVENET "GPU NAM: build WaveNet (A1) architecture" ON)
option(GPU_NAM_WITH_A2      "GPU NAM: build WaveNet A2 architecture" ON)
option(GPU_NAM_WITH_CONVNET "GPU NAM: build ConvNet architecture" ON)
option(GPU_NAM_WITH_LSTM    "GPU NAM: build LSTM architecture" ON)
option(GPU_NAM_WITH_LINEAR  "GPU NAM: build Linear architecture" ON)
option(GPU_NAM_WITH_KERAS   "GPU NAM: build RTNeural/Keras (.json) GRU/LSTM engine" ON)
set(_gpu_nam_arch_defs "")
foreach(_gn_arch WAVENET A2 CONVNET LSTM LINEAR KERAS)
    if(NOT GPU_NAM_WITH_${_gn_arch})
        list(APPEND _gpu_nam_arch_defs "GPU_NAM_WITH_${_gn_arch}=0")
    endif()
endforeach()
# Apply the toggles directory-wide so EVERY gpu-nam target that compiles the
# inference substrate — the plugin core, its format adapters, and the tests —
# sees a consistent GPU_NAM_WITH_<ARCH> set (ODR-safe: they all share one
# NamRuntime layout). A define is emitted only for a DISABLED arch, so the
# default all-on build passes nothing and is unchanged. The full GPU plugin
# additionally requires WaveNet (its GPU engine is WaveNet-specific) and
# #errors at compile time if WaveNet is turned off.
if(_gpu_nam_arch_defs)
    add_compile_definitions(${_gpu_nam_arch_defs})
endif()

if(TARGET Catch2::Catch2WithMain)
    add_executable(gpu-nam-gpu-cpu-diagnostic-options-test
        test_gpu_nam_gpu_cpu_diagnostic_options.cpp)
    target_link_libraries(gpu-nam-gpu-cpu-diagnostic-options-test
        PRIVATE Catch2::Catch2WithMain)
    target_include_directories(gpu-nam-gpu-cpu-diagnostic-options-test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME gpu-nam-gpu-cpu-diagnostic-options
        COMMAND gpu-nam-gpu-cpu-diagnostic-options-test)
endif()

# Optional licensing gate (default OFF) — a reference for paid plugins. The shipped
# MIT demo leaves it OFF, so nothing is restricted. Turn it ON and supply a public
# key to gate the opt-in GPU acceleration behind a valid license (the free tier
# falls back to the bit-exact CPU engine).
option(GPU_NAM_WITH_LICENSE "GPU NAM: gate the GPU engine behind a license (reference)" OFF)
set(GPU_NAM_LICENSE_PUBKEY "" CACHE STRING "PEM RSA public key for the GPU NAM license gate")
if(GPU_NAM_WITH_LICENSE)
    add_compile_definitions(GPU_NAM_WITH_LICENSE=1
        "GPU_NAM_LICENSE_PUBKEY=\"${GPU_NAM_LICENSE_PUBKEY}\"")
endif()

if(TARGET pulp::render)
    add_executable(gpu-nam-check wavenet_check.cpp)
    target_link_libraries(gpu-nam-check PRIVATE pulp::render)
    target_include_directories(gpu-nam-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if(TARGET Catch2::Catch2WithMain)
        add_executable(gpu-nam-test test_gpu_nam.cpp)
        target_link_libraries(gpu-nam-test PRIVATE pulp::render Catch2::Catch2WithMain)
        target_include_directories(gpu-nam-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME gpu-nam-test COMMAND gpu-nam-test)

        # Fused GPU NAM forward validated against the exact CPU oracle on the real
        # .nam capture. Needs both pulp::render (GPU primitive) and pulp::runtime
        # (choc::json, used by the .nam loader / oracle).
        if(TARGET pulp::runtime)
            add_executable(gpu-nam-gpu-test test_gpu_nam_real.cpp)
            target_link_libraries(gpu-nam-gpu-test
                PRIVATE pulp::render pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-gpu-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(NAME gpu-nam-gpu-test COMMAND gpu-nam-gpu-test)
        endif()
    endif()
endif()

# Bounded SDK compatibility probe. Newer Pulp SDKs may expose the authenticated
# backend-neutral capability report; older SDKs keep the existing staged path.
# The probe is deliberately independent of private shared-I/O headers and stays
# green on older pins while making capability absence explicit in its output.
if(GPU_NAM_BUILD_GPU_AUDIO_CAPABILITY_PROBE AND TARGET pulp::gpu-audio)
    add_executable(gpu-nam-gpu-audio-capability-probe gpu_audio_capability_probe.cpp)
    target_link_libraries(gpu-nam-gpu-audio-capability-probe PRIVATE pulp::gpu-audio)
    add_test(NAME gpu-nam-gpu-audio-capability-probe
        COMMAND gpu-nam-gpu-audio-capability-probe)
endif()

# Exact, reference-faithful NAM WaveNet CPU inference + .nam loader. This is the
# oracle the GPU primitive is validated against; it only needs choc::json (via
# pulp::runtime), not the GPU/render stack, so it builds independently of pulp::render.
if(TARGET pulp::runtime)
    add_executable(gpu-nam-model-check nam_model_check.cpp)
    target_link_libraries(gpu-nam-model-check PRIVATE pulp::runtime)
    target_include_directories(gpu-nam-model-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if(TARGET Catch2::Catch2WithMain)
        # Each architecture's test (loader + NamRuntime dispatch) is built only
        # when that architecture is enabled — a disabled arch's runtime dispatch
        # correctly rejects the model, which would fail the test by design.
        if(GPU_NAM_WITH_WAVENET)
            add_executable(gpu-nam-model-test test_nam_model.cpp)
            target_link_libraries(gpu-nam-model-test PRIVATE pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-model-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(NAME gpu-nam-model-test COMMAND gpu-nam-model-test)
        endif()

        # WaveNet A2 architecture (SlimmableContainer + windowed head) + dispatch.
        if(GPU_NAM_WITH_A2)
            add_executable(gpu-nam-a2-test test_nam_a2.cpp)
            target_link_libraries(gpu-nam-a2-test PRIVATE pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-a2-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(NAME gpu-nam-a2-test COMMAND gpu-nam-a2-test)
        endif()

        # ConvNet architecture (dilated conv blocks + batchnorm) + dispatch.
        if(GPU_NAM_WITH_CONVNET)
            add_executable(gpu-nam-convnet-test test_nam_convnet.cpp)
            target_link_libraries(gpu-nam-convnet-test PRIVATE pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-convnet-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(NAME gpu-nam-convnet-test COMMAND gpu-nam-convnet-test)
        endif()

        # LSTM architecture + NamRuntime dispatch (no GPU needed).
        if(GPU_NAM_WITH_LSTM)
            add_executable(gpu-nam-lstm-test test_nam_lstm.cpp)
            target_link_libraries(gpu-nam-lstm-test PRIVATE pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-lstm-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(NAME gpu-nam-lstm-test COMMAND gpu-nam-lstm-test)
        endif()

        # Linear architecture (single causal FIR) + NamRuntime dispatch (no GPU).
        if(GPU_NAM_WITH_LINEAR)
            add_executable(gpu-nam-linear-test test_nam_linear.cpp)
            target_link_libraries(gpu-nam-linear-test PRIVATE pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-linear-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(NAME gpu-nam-linear-test COMMAND gpu-nam-linear-test)
        endif()

        # RTNeural/Keras (.json) GRU + LSTM + Dense engine. A separate KerasRuntime
        # from the .nam NamRuntime (different file format + recurrent-only weight
        # layout); recurrent, so CPU-only. Needs only choc::json (pulp::runtime).
        if(GPU_NAM_WITH_KERAS)
            add_executable(gpu-nam-keras-test test_keras_runtime.cpp)
            target_link_libraries(gpu-nam-keras-test PRIVATE pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-keras-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(NAME gpu-nam-keras-test COMMAND gpu-nam-keras-test)
        endif()

        # Optional licensing gate (free-tier default, locks on an invalid license).
        add_executable(gpu-nam-license-test test_gpu_nam_license.cpp)
        target_link_libraries(gpu-nam-license-test PRIVATE pulp::runtime Catch2::Catch2WithMain)
        target_include_directories(gpu-nam-license-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME gpu-nam-license-test COMMAND gpu-nam-license-test)

        # Grace-period engine retirement (audio-thread-safe live swaps; no GPU).
        add_executable(gpu-nam-retire-test test_nam_retire_list.cpp)
        target_link_libraries(gpu-nam-retire-test PRIVATE Catch2::Catch2WithMain)
        target_include_directories(gpu-nam-retire-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME gpu-nam-retire-test COMMAND gpu-nam-retire-test)

        # Typed WaveNet preparation metadata.  This test covers the
        # backend-neutral descriptor and CPU-fallback contract without
        # constructing Dawn/Metal resources or claiming shared execution.
        add_executable(gpu-nam-prepared-program-test test_gpu_nam_prepared_program.cpp)
        target_link_libraries(gpu-nam-prepared-program-test
            PRIVATE pulp::gpu-audio Catch2::Catch2WithMain)
        target_include_directories(gpu-nam-prepared-program-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME gpu-nam-prepared-program-test COMMAND gpu-nam-prepared-program-test)

        # Output-level make-up math (Raw / Normalized / Calibrated). Pure header,
        # no pulp/GPU deps — just Catch2.
        add_executable(gpu-nam-output-cal-test test_output_calibration.cpp)
        target_link_libraries(gpu-nam-output-cal-test PRIVATE Catch2::Catch2WithMain)
        target_include_directories(gpu-nam-output-cal-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME gpu-nam-output-cal-test COMMAND gpu-nam-output-cal-test)
    endif()
endif()

# The shadow worker is a CPU-only readiness experiment. Keep it outside the
# GPU plugin guard so its ring/ordering test can run against a runtime-only SDK.
if(GPU_NAM_EXPERIMENTAL_CPU_SHADOW_WORKER AND TARGET pulp::runtime)
    add_executable(gpu-nam-cpu-shadow-worker
        test_gpu_nam_cpu_shadow_worker.cpp)
    target_link_libraries(gpu-nam-cpu-shadow-worker PRIVATE pulp::runtime)
    target_include_directories(gpu-nam-cpu-shadow-worker
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(gpu-nam-cpu-shadow-worker PRIVATE
        GPU_NAM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/example.nam")
    if(GPU_NAM_BUILD_TESTS)
        add_test(NAME gpu-nam-cpu-shadow-worker
            COMMAND gpu-nam-cpu-shadow-worker)
        set_tests_properties(gpu-nam-cpu-shadow-worker PROPERTIES
            TIMEOUT 30 RESOURCE_LOCK pulp_gpu)
    endif()
endif()

# ── GPU NAM plugin (VST3, AU v2, CLAP, Standalone) ──────────────────────────
# A Neural Amp Modeler player: loads an open-source .nam WaveNet capture and runs
# it through the inline CPU oracle (default) or the opt-in GPU audio runtime
# (GpuNamCloudNode on a GpuAudioTransport). Native GPU UI: input→output transfer
# curve + output spectrum + gain/mix/engine controls. Requires the GPU compute
# primitive (pulp::render) and the GPU audio runtime (pulp::gpu-audio).
if(TARGET pulp::render AND TARGET pulp::gpu-audio)
    set(GPU_NAM_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/example.nam")
    set(GPU_NAM_ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/assets/nam")

    # Single source of the plugin version. Feeds pulp_add_plugin (VST3/CLAP/AU/
    # Standalone plists), the custom AU plist (below), and the processor
    # descriptor (GPU_NAM_VERSION_STRING compile define) so the three can't drift
    # — the AU plist silently froze at 1.0.0 before this was centralized.
    set(GPU_NAM_VERSION "1.2.5")
    add_compile_definitions(GPU_NAM_VERSION_STRING="${GPU_NAM_VERSION}")

    set(_gpu_nam_experimental_sources)
    if(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
        list(APPEND _gpu_nam_experimental_sources gpu_nam_stamped_node.cpp)
        add_executable(gpu-nam-stamped-validation
            gpu_nam_stamped_node.cpp gpu_nam_stamped_paced.cpp test_gpu_nam_stamped_node.cpp)
        add_executable(gpu-nam-stamped-multichannel-validation
            gpu_nam_stamped_node.cpp test_gpu_nam_stamped_multichannel.cpp)
        target_link_libraries(gpu-nam-stamped-multichannel-validation PRIVATE pulp::gpu-audio)
        target_compile_definitions(gpu-nam-stamped-multichannel-validation PRIVATE
            GPU_NAM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/example.nam")
        if(GPU_NAM_BUILD_TESTS)
            add_test(NAME gpu-nam-stamped-multichannel-4096
                COMMAND gpu-nam-stamped-multichannel-validation)
            set_tests_properties(gpu-nam-stamped-multichannel-4096 PROPERTIES
                TIMEOUT 300 RESOURCE_LOCK pulp_gpu)
        endif()
        target_link_libraries(gpu-nam-stamped-validation PRIVATE pulp::gpu-audio)
        target_compile_definitions(gpu-nam-stamped-validation PRIVATE
            GPU_NAM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/example.nam")
        if(GPU_NAM_BUILD_TESTS)
            add_executable(gpu-nam-completion-options-test test_gpu_nam_completion_options.cpp)
            target_link_libraries(gpu-nam-completion-options-test PRIVATE pulp::gpu-audio)
            add_test(NAME gpu-nam-completion-options COMMAND gpu-nam-completion-options-test)
            add_executable(gpu-nam-paced-options-test test_gpu_nam_paced_options.cpp)
            target_compile_features(gpu-nam-paced-options-test PRIVATE cxx_std_20)
            add_test(NAME gpu-nam-paced-options COMMAND gpu-nam-paced-options-test)
            foreach(_lead 1 2 4 8)
                foreach(_frames 32 64 128 512)
                    add_test(NAME gpu-nam-stamped-${_frames}-lead-${_lead}
                        COMMAND gpu-nam-stamped-validation ${_lead} ${_frames})
                    set_tests_properties(gpu-nam-stamped-${_frames}-lead-${_lead} PROPERTIES
                        TIMEOUT 60 RESOURCE_LOCK pulp_gpu)
                endforeach()
            endforeach()
        endif()
    endif()
    if(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
        add_executable(gpu-nam-cpu-baseline gpu_nam_cpu_baseline.cpp)
        target_link_libraries(gpu-nam-cpu-baseline PRIVATE pulp::runtime)
        target_include_directories(gpu-nam-cpu-baseline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(gpu-nam-cpu-baseline PRIVATE
            GPU_NAM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/example.nam")
        list(APPEND _gpu_nam_experimental_sources gpu_nam_shared_session_node.cpp)
    endif()
    set(_gpu_nam_plugin_sources gpu_nam_view.cpp)
    if(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
        list(APPEND _gpu_nam_plugin_sources gpu_nam_stamped_node.cpp)
    elseif(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
        list(APPEND _gpu_nam_plugin_sources gpu_nam_shared_session_node.cpp)
    endif()

    pulp_add_plugin(GpuNam
        FORMATS           VST3 AU CLAP Standalone
        SOURCES           ${_gpu_nam_plugin_sources}
        PLUGIN_NAME       "GpuNam"
        BUNDLE_ID         "com.pulp.gpunam"
        MANUFACTURER      "Pulp"
        VERSION           "${GPU_NAM_VERSION}"
        CATEGORY          Effect
        PLUGIN_CODE       "GNam"
        MANUFACTURER_CODE "Pulp"
    )

    # The AU needs a custom AudioComponent display name ("Pulp: GPU NAM"), so it
    # can't use pulp_add_plugin's generated template. Configure the version fields
    # from GPU_NAM_VERSION and override the AU target's Info.plist, so the custom
    # plist tracks the project version instead of hardcoding it.
    if(TARGET GpuNam_AU)
        string(REPLACE "." ";" _gn_ver_parts "${GPU_NAM_VERSION}")
        list(LENGTH _gn_ver_parts _gn_ver_len)
        list(GET _gn_ver_parts 0 _gn_ver_major)
        set(_gn_ver_minor 0)
        set(_gn_ver_patch 0)
        if(_gn_ver_len GREATER 1)
            list(GET _gn_ver_parts 1 _gn_ver_minor)
        endif()
        if(_gn_ver_len GREATER 2)
            list(GET _gn_ver_parts 2 _gn_ver_patch)
        endif()
        math(EXPR GPU_NAM_AU_VERSION_INT
             "${_gn_ver_major} * 65536 + ${_gn_ver_minor} * 256 + ${_gn_ver_patch}")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.au.in"
            "${CMAKE_CURRENT_BINARY_DIR}/Info.plist.au"
            @ONLY)
        set_target_properties(GpuNam_AU PROPERTIES
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist.au")
    endif()

    # The processor's create_view() + GPU engine pull in the render / gpu-audio /
    # view / canvas / signal stack. Shipped plugin/standalone binaries find their
    # default model via the executable-relative bundled Resources copy below — they
    # deliberately do NOT bake the source-tree path (so they stay portable). The
    # source-tree fallback define is set only on the dev tools (test + shot), which
    # run from the build tree without a bundle.
    if(TARGET GpuNam_Core)
        target_include_directories(GpuNam_Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(GpuNam_Core PUBLIC
            pulp::render pulp::gpu-audio pulp::signal pulp::view pulp::canvas pulp::runtime)
        # Shipped formats must resolve assets from their executable-relative
        # Resources copy. Keep the shared plugin core free of a source-tree
        # absolute path; development-only tools below provide their own
        # GPU_NAM_ASSET_DIR fallback when they run outside a bundle.
    endif()

    # Bundle the default model into each format's Resources so the plugin finds it
    # at runtime next to the binary (the build-define is the source-tree fallback).
    foreach(_gn_target GpuNam_CLAP GpuNam_VST3 GpuNam_AU GpuNam_Standalone)
        if(TARGET ${_gn_target})
            if(APPLE)
                add_custom_command(TARGET ${_gn_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                        "$<TARGET_BUNDLE_DIR:${_gn_target}>/Contents/Resources"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GPU_NAM_MODEL}"
                        "$<TARGET_BUNDLE_DIR:${_gn_target}>/Contents/Resources/example.nam"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${GPU_NAM_ASSETS}"
                        "$<TARGET_BUNDLE_DIR:${_gn_target}>/Contents/Resources/assets/nam"
                    COMMENT "Bundling example.nam + UI assets into ${_gn_target}"
                    VERBATIM)
            else()
                add_custom_command(TARGET ${_gn_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${GPU_NAM_MODEL}"
                        "$<TARGET_FILE_DIR:${_gn_target}>/example.nam"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        "${GPU_NAM_ASSETS}"
                        "$<TARGET_FILE_DIR:${_gn_target}>/assets/nam"
                    COMMENT "Copying example.nam + UI assets next to ${_gn_target}"
                    VERBATIM)
            endif()
        endif()
    endforeach()

    # Self-contained, relocatable GPU bundles (the @loader_path fix) — fail the
    # build if any bundle bakes a build-tree dylib path. In installed-SDK mode
    # find_package(Pulp) (via PulpConfig) already defines these helpers; only the
    # submodule build needs to include the module from the source tree.
    if(NOT GPU_NAM_USE_INSTALLED_PULP)
        include(${GPU_NAM_PULP_DIR}/tools/cmake/PulpBundleRelocatable.cmake)
    endif()
    foreach(_gn_target GpuNam_CLAP GpuNam_VST3 GpuNam_AU GpuNam_Standalone)
        if(TARGET ${_gn_target})
            pulp_make_bundle_relocatable(${_gn_target})
            pulp_validate_bundle_relocatable(${_gn_target})
        endif()
    endforeach()

    # Headless editor screenshot. macOS-only dev tool (Skia raster capture).
    if(APPLE AND (TARGET pulp::view OR TARGET Pulp::view))
        add_executable(gpu-nam-shot screenshot.cpp gpu_nam_view.cpp
                       ${_gpu_nam_experimental_sources})
        target_link_libraries(gpu-nam-shot PRIVATE
            pulp::format pulp::audio pulp::render pulp::gpu-audio pulp::signal
            pulp::view pulp::canvas pulp::runtime)
        target_include_directories(gpu-nam-shot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(gpu-nam-shot PRIVATE
            GPU_NAM_DEFAULT_MODEL_PATH="${GPU_NAM_MODEL}"
            GPU_NAM_ASSET_DIR="${GPU_NAM_ASSETS}")
    endif()

    # Performance harness (CPU oracle vs GPU engine, realtime ratio across block
    # sizes). A dev tool like gpu-nam-shot; not built into the shipped plugin.
    add_executable(gpu-nam-bench bench.cpp gpu_nam_view.cpp
                   ${_gpu_nam_experimental_sources})
    target_link_libraries(gpu-nam-bench PRIVATE
        pulp::format pulp::render pulp::gpu-audio pulp::signal
        pulp::view pulp::canvas pulp::runtime)
    target_include_directories(gpu-nam-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(gpu-nam-bench PRIVATE
        GPU_NAM_DEFAULT_MODEL_PATH="${GPU_NAM_MODEL}"
        GPU_NAM_ASSET_DIR="${GPU_NAM_ASSETS}")

    # Processor tests (CPU + GPU engine, live switch, load_model). Links
    # gpu_nam_view.cpp so the processor's vtable (first non-inline virtual is
    # create_view()) resolves at link time; the view is never CALLED headless.
    if(GPU_NAM_BUILD_TESTS AND TARGET Catch2::Catch2WithMain)
        add_executable(gpu-nam-plugin-test test_gpu_nam_plugin.cpp gpu_nam_view.cpp
                        ${_gpu_nam_experimental_sources})
        target_link_libraries(gpu-nam-plugin-test PRIVATE
            pulp::format pulp::render pulp::gpu-audio pulp::signal
            pulp::view pulp::canvas pulp::runtime Catch2::Catch2WithMain)
        target_include_directories(gpu-nam-plugin-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(gpu-nam-plugin-test PRIVATE
            GPU_NAM_DEFAULT_MODEL_PATH="${GPU_NAM_MODEL}"
            GPU_NAM_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models"
            GPU_NAM_ASSET_DIR="${GPU_NAM_ASSETS}")
        catch_discover_tests(gpu-nam-plugin-test)

        if(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
            add_executable(gpu-nam-shared-session-adapter-test
                           test_gpu_nam_shared_session_adapter.cpp
                           gpu_nam_shared_session_node.cpp)
            target_link_libraries(gpu-nam-shared-session-adapter-test PRIVATE
                pulp::gpu-audio pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-shared-session-adapter-test PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR})
            target_compile_definitions(gpu-nam-shared-session-adapter-test PRIVATE
                GPU_NAM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/example.nam")
            add_test(NAME gpu-nam-shared-session-adapter-test
                     COMMAND gpu-nam-shared-session-adapter-test)

            # Short matched CPU/shared-GPU correctness and cadence diagnostic.
            # This reports delivery/fallback counts and callback CPU costs; it
            # is intentionally not a benchmark or a realtime claim.
            add_executable(gpu-nam-gpu-cpu-diagnostic
                           gpu_nam_gpu_cpu_diagnostic.cpp
                           gpu_nam_shared_session_node.cpp)
            target_link_libraries(gpu-nam-gpu-cpu-diagnostic PRIVATE
                pulp::gpu-audio pulp::runtime)
            target_include_directories(gpu-nam-gpu-cpu-diagnostic PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR})
            target_compile_definitions(gpu-nam-gpu-cpu-diagnostic PRIVATE
                GPU_NAM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/example.nam")
        endif()

        # Editor tests: faithful UI renders non-blank + composited (Skia), and
        # pointer input (toggles, knob drag) drives real parameters.
        if(APPLE AND (TARGET pulp::view OR TARGET Pulp::view))
            add_executable(gpu-nam-ui-test test_gpu_nam_ui.cpp gpu_nam_view.cpp
                            ${_gpu_nam_experimental_sources})
            target_link_libraries(gpu-nam-ui-test PRIVATE
                pulp::format pulp::render pulp::gpu-audio pulp::signal
                pulp::view pulp::canvas pulp::runtime Catch2::Catch2WithMain)
            target_include_directories(gpu-nam-ui-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            target_compile_definitions(gpu-nam-ui-test PRIVATE
                GPU_NAM_DEFAULT_MODEL_PATH="${GPU_NAM_MODEL}"
                GPU_NAM_ASSET_DIR="${GPU_NAM_ASSETS}")
            catch_discover_tests(gpu-nam-ui-test)
        endif()
    endif()

    # CLAP load smoke test — the plugin's shared object must dlopen cleanly.
    if(PULP_HAS_CLAP AND APPLE)
        add_test(NAME clap-dlopen-GpuNam
            COMMAND bash -c "python3 -c \"import ctypes; ctypes.CDLL('${CMAKE_BINARY_DIR}/CLAP/GpuNam.clap/Contents/MacOS/GpuNam'); print('OK')\"")
        set_tests_properties(clap-dlopen-GpuNam PROPERTIES LABELS "validation;clap" TIMEOUT 20)
    endif()

    # AU validation via auval. killall AudioComponentRegistrar forces a rescan of
    # the freshly-copied bundle (stale-cache trap codified in the auv2 skill).
    if(APPLE AND PULP_HAS_AUSDK)
        add_test(NAME auval-GpuNam
            COMMAND bash -c "d=\"$HOME/Library/Audio/Plug-Ins/Components/GpuNam.auvaltest.component\"; rm -rf \"$d\"; cp -R \"${CMAKE_BINARY_DIR}/AU/GpuNam.component\" \"$d\" && { killall -KILL AudioComponentRegistrar 2>/dev/null || true; sleep 1; auval -v aufx GNam Pulp 2>&1 | tee /dev/stderr | grep -q 'PASS'; }; rc=$?; rm -rf \"$d\"; exit $rc")
        set_tests_properties(auval-GpuNam PROPERTIES
            LABELS "validation;au"
            TIMEOUT 60
            ENVIRONMENT "PULP_DISABLE_PLUGIN_EDITOR=1;PULP_HEADLESS=1;PULP_TEST_MODE=1")
    endif()

    if(PULP_HAS_VST3)
        find_program(PLUGINVAL pluginval)
        if(PLUGINVAL)
            add_test(NAME pluginval-GpuNam-VST3
                COMMAND ${PLUGINVAL} --validate "${CMAKE_BINARY_DIR}/VST3/GpuNam.vst3" --strictness-level 5 --timeout-ms 15000)
            set_tests_properties(pluginval-GpuNam-VST3 PROPERTIES
                LABELS "validation;vst3"
                TIMEOUT 40
                ENVIRONMENT "PULP_DISABLE_PLUGIN_EDITOR=1;PULP_HEADLESS=1;PULP_TEST_MODE=1")
        endif()
    endif()
endif()

# This opt-in binary is acceptance instrumentation, not a distributable plugin.
if(GPU_NAM_NATIVE_HOST_PROBE)
    if(NOT APPLE OR NOT TARGET GpuNam_CLAP OR
       (NOT GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET AND
        NOT GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION))
        message(FATAL_ERROR
            "Native host probe requires the macOS CLAP target and an experimental "
            "WaveNet adapter (stamped or shared-session)")
    endif()
    target_compile_definitions(GpuNam_CLAP PRIVATE GPU_NAM_NATIVE_HOST_PROBE=1)
    target_link_options(GpuNam_CLAP PRIVATE "LINKER:-exported_symbol,_gpu_nam_host_probe_v1")
    add_executable(gpu-nam-clap-host-test test_gpu_nam_clap_host.cpp)
    target_include_directories(gpu-nam-clap-host-test PRIVATE "${PULP_CLAP_INCLUDE_DIR}")
    target_link_libraries(gpu-nam-clap-host-test PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(gpu-nam-clap-host-test GpuNam_CLAP)
    add_test(NAME gpu-nam-clap-host-acceptance
        COMMAND gpu-nam-clap-host-test $<TARGET_FILE:GpuNam_CLAP>)
    set_tests_properties(gpu-nam-clap-host-acceptance PROPERTIES TIMEOUT 60 RESOURCE_LOCK pulp_gpu)
endif()

if(GPU_NAM_SILENT_DEVICE_PROBE)
    if(NOT APPLE OR NOT GPU_NAM_USE_INSTALLED_PULP)
        message(FATAL_ERROR "Silent device probe requires macOS and a complete installed SDK")
    endif()
    add_executable(gpu-nam-silent-device-probe EXCLUDE_FROM_ALL gpu_nam_silent_device_probe.cpp)
    target_link_libraries(gpu-nam-silent-device-probe PRIVATE pulp::audio ${CMAKE_DL_LIBS})
    add_dependencies(gpu-nam-silent-device-probe GpuNam_CLAP)
    target_include_directories(gpu-nam-silent-device-probe PRIVATE
        "${PULP_CLAP_INCLUDE_DIR}")
    if(GPU_NAM_NORMAL_PHYSICAL_LIFECYCLE)
        if(NOT GPU_NAM_EXPECTED_PULP_SOURCE_SHA STREQUAL "7858d7edb5a2bbb568c03d7e740043e8c897c0ee")
            message(FATAL_ERROR "Normal physical lifecycle requires the exact reviewed admission-safe SDK7858")
        endif()
        execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            OUTPUT_VARIABLE _physical_source_sha OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _physical_git_result)
        execute_process(COMMAND git status --porcelain WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            OUTPUT_VARIABLE _physical_source_dirty OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _physical_status_result)
        if(NOT _physical_git_result EQUAL 0 OR NOT _physical_status_result EQUAL 0 OR NOT _physical_source_dirty STREQUAL "")
            message(FATAL_ERROR "Physical lifecycle requires checked source provenance")
        endif()
        target_compile_definitions(gpu-nam-silent-device-probe PRIVATE
            GPU_NAM_NORMAL_PHYSICAL_LIFECYCLE=1
            GPU_NAM_PHYSICAL_SOURCE_SHA="${_physical_source_sha}"
            GPU_NAM_PHYSICAL_SDK_SHA="${PULP_SDK_SOURCE_GIT_SHA}")
    endif()
    # Intentionally no add_test: opening a physical device is always manual.
endif()

if(GPU_NAM_BUILD_TESTS)
    add_executable(gpu-nam-delivery-status-test test_gpu_nam_delivery_status.cpp)
    target_compile_features(gpu-nam-delivery-status-test PRIVATE cxx_std_20)
    add_test(NAME gpu-nam-delivery-status COMMAND gpu-nam-delivery-status-test)
endif()

if(GPU_NAM_BUILD_TESTS)
    add_executable(gpu-nam-physical-lifecycle-test test_gpu_nam_physical_lifecycle.cpp)
    add_test(NAME gpu-nam-physical-lifecycle-contract COMMAND gpu-nam-physical-lifecycle-test)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME gpu-nam-physical-watchdog-contract
        COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/scripts/test_physical_lifecycle_watchdog.py)
endif()
