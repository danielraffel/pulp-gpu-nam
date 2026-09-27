#pragma once
#include <cstdint>
// Consumer-owned test ABI. Only GPU_NAM_NATIVE_HOST_PROBE builds export it.
// Query after stop_processing and before deactivate; exactly one instance must
// exist. A zero return is the only indication that delivery counters are valid.
struct GpuNamHostProbeSnapshot {
    std::uint32_t size = sizeof(GpuNamHostProbeSnapshot);
    std::uint32_t version = 1;
    std::uint32_t active = 0;
    std::int32_t requested_engine = -1, prepared_engine = -1;
    std::uint32_t latency_samples = 0;
    std::uint64_t gpu_delivered = 0, cpu_fallback = 0, priming = 0, other = 0;
};
using GpuNamHostProbeQuery = int (*)(GpuNamHostProbeSnapshot*);
// Return codes: 0 snapshot; 1 malformed request; 2 SDK delivery query missing;
// 3 not exactly one plugin instance. Counters are undefined unless return is 0.
