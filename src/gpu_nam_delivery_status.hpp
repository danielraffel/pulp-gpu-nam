#pragma once
#include <cstdint>
#include <string>

namespace pulp::examples::nam {
struct GpuNamDeliveryStatus {
    bool available = false;
    std::uint64_t gpu_selected = 0, worker_selected = 0, cpu_fallback = 0,
                  silence = 0, passthrough = 0, priming = 0, invalid = 0;
};

// Dependent detection keeps ordinary UI builds compatible with older SDKs.
// Worker-produced counters are never substituted for callback selections.
template<class Transport>
GpuNamDeliveryStatus read_gpu_nam_delivery_status(const Transport& transport) {
    if constexpr (requires {
        transport.delivery_snapshot().gpu_blocks;
        transport.delivery_snapshot().worker_output_blocks;
        transport.delivery_snapshot().cpu_fallback_blocks;
        transport.delivery_snapshot().silence_blocks;
        transport.delivery_snapshot().passthrough_blocks;
        transport.delivery_snapshot().priming_blocks;
        transport.delivery_snapshot().invalid_blocks;
    }) {
        const auto s = transport.delivery_snapshot();
        return {true, s.gpu_blocks, s.worker_output_blocks, s.cpu_fallback_blocks,
                s.silence_blocks, s.passthrough_blocks, s.priming_blocks, s.invalid_blocks};
    }
    return {};
}

// UI/control lane only. Counts are independent live observations, not a coherent
// sample or a claim about numerical correctness, GPU time or deadline success.
inline std::string gpu_nam_delivery_label(bool active, const GpuNamDeliveryStatus& s) {
    if (!active) return "CPU engine active";
    if (!s.available) return "GPU path: delivery counts unavailable";
    if (s.worker_selected && !s.gpu_selected)
        return "Worker output " + std::to_string(s.worker_selected) +
               "; CPU fallback " + std::to_string(s.cpu_fallback);
    auto label = "GPU selected " + std::to_string(s.gpu_selected) +
                 "; CPU fallback " + std::to_string(s.cpu_fallback);
    if (s.worker_selected) label += "; worker output " + std::to_string(s.worker_selected);
    return label;
}
} // namespace pulp::examples::nam
