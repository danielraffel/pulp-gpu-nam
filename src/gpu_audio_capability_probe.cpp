#include <pulp/gpu_audio/gpu_audio_transport.hpp>

#include <iostream>

#if __has_include(<pulp/gpu_audio/gpu_audio_capability.hpp>)
#  include <pulp/gpu_audio/gpu_audio_capability.hpp>
#  define GPU_NAM_HAS_CAPABILITY_REPORT 1
#else
#  define GPU_NAM_HAS_CAPABILITY_REPORT 0
#endif

int main() {
#if GPU_NAM_HAS_CAPABILITY_REPORT
    pulp::gpu_audio::GpuAudioTransport transport;
    const auto report = transport.capability_report();

    // A consumer must not infer shared memory from target availability alone.
    // Before prepare(), the report must fail closed and remain allocation-free.
    const bool fail_closed =
        report.path == pulp::gpu_audio::GpuAudioExecutionPath::Unavailable &&
        report.provider == pulp::gpu_audio::GpuAudioProvider::Unknown &&
        report.eligibility == pulp::gpu_audio::GpuAudioEligibility::Unavailable &&
        !report.prepared;
    std::cout << "capability_report=present\n"
              << "path=" << static_cast<int>(report.path) << '\n'
              << "provider=" << static_cast<int>(report.provider) << '\n'
              << "eligibility=" << static_cast<int>(report.eligibility) << '\n'
              << "prepared=" << (report.prepared ? "true" : "false") << '\n';
    return fail_closed ? 0 : 1;
#else
    // Older Pulp SDKs intentionally retain the staged transport path. Keep
    // this probe green and make the capability absence explicit in its output.
    std::cout << "capability_report=unavailable\n";
    return 0;
#endif
}
