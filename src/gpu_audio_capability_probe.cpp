#include <pulp/gpu_audio/gpu_audio_transport.hpp>

#include <iostream>
#include <string>
#include <vector>

#if __has_include(<pulp/gpu_audio/gpu_audio_capability.hpp>)
#  include <pulp/gpu_audio/gpu_audio_capability.hpp>
#  define GPU_NAM_HAS_CAPABILITY_REPORT 1
#else
#  define GPU_NAM_HAS_CAPABILITY_REPORT 0
#endif

#if GPU_NAM_HAS_CAPABILITY_REPORT && __has_include(<pulp/gpu_audio/gpu_convolver.hpp>)
#  include <pulp/gpu_audio/gpu_convolver.hpp>
#  define GPU_NAM_HAS_PREPARE_PROBE 1
#else
#  define GPU_NAM_HAS_PREPARE_PROBE 0
#endif

int main(int argc, char** argv) {
#if GPU_NAM_HAS_PREPARE_PROBE
    const bool require_shared = argc > 1 && std::string(argv[1]) == "--require-shared";
#else
    (void)argc;
    (void)argv;
#endif
#if GPU_NAM_HAS_CAPABILITY_REPORT
    pulp::gpu_audio::GpuAudioTransport transport;
    const auto before = transport.capability_report();

    // A consumer must not infer shared memory from target availability alone.
    // Before prepare(), the report must fail closed and remain allocation-free.
    const bool fail_closed =
        before.path == pulp::gpu_audio::GpuAudioExecutionPath::Unavailable &&
        before.provider == pulp::gpu_audio::GpuAudioProvider::Unknown &&
        before.eligibility == pulp::gpu_audio::GpuAudioEligibility::Unavailable &&
        !before.prepared;
    std::cout << "capability_report=present\n"
              << "pre_prepare_path=" << static_cast<int>(before.path) << '\n'
              << "pre_prepare_provider=" << static_cast<int>(before.provider) << '\n'
              << "pre_prepare_eligibility=" << static_cast<int>(before.eligibility) << '\n'
              << "pre_prepare_prepared=" << (before.prepared ? "true" : "false") << '\n';

#if GPU_NAM_HAS_PREPARE_PROBE
    pulp::gpu_audio::GpuConvolver node(1, 32, 48000, std::vector<float>{1.0f}, 2);
    const bool node_prepared = node.prepare();
    pulp::gpu_audio::GpuAudioTransport::Config config;
    config.ring_blocks = 4;
    const bool transport_prepared = node_prepared && transport.prepare(&node, config);
    const auto after = transport.capability_report();
    const bool shared =
        transport_prepared && after.prepared &&
        after.path == pulp::gpu_audio::GpuAudioExecutionPath::SharedMemory &&
        after.provider == pulp::gpu_audio::GpuAudioProvider::Dawn &&
        after.eligibility == pulp::gpu_audio::GpuAudioEligibility::Eligible;
    std::cout << "prepare_probe=present\n"
              << "node_prepared=" << (node_prepared ? "true" : "false") << '\n'
              << "transport_prepared=" << (transport_prepared ? "true" : "false") << '\n'
              << "post_prepare_path=" << static_cast<int>(after.path) << '\n'
              << "post_prepare_provider=" << static_cast<int>(after.provider) << '\n'
              << "post_prepare_eligibility=" << static_cast<int>(after.eligibility) << '\n'
              << "post_prepare_prepared=" << (after.prepared ? "true" : "false") << '\n'
              << "shared_provider_selected=" << (shared ? "true" : "false") << '\n';
    if (require_shared && !shared)
        return 2;
#else
    if (require_shared)
        return 2;
#endif
    return fail_closed ? 0 : 1;
#else
    // Older Pulp SDKs intentionally retain the staged transport path. Keep
    // this probe green and make the capability absence explicit in its output.
    std::cout << "capability_report=unavailable\n";
    return argc > 1 && std::string(argv[1]) == "--require-shared" ? 2 : 0;
#endif
}
