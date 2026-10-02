#pragma once

#include "nam_model.hpp"
#include "gpu_nam_completion_options.hpp"
#include <pulp/gpu_audio/gpu_wavenet_realtime_node.hpp>
#include <array>
#include <memory>
#include <vector>
#include <cstdint>

namespace pulp::examples {

// The SDK owns stamped GPU admission, completion, and retirement. This adapter
// owns only NAM model translation and the continuously advanced CPU fallback.
class GpuNamStampedNode final : public gpu_audio::GpuWaveNetRealtimeNode {
public:
    static std::unique_ptr<GpuNamStampedNode> create(
        const nam::NamModel& model, std::uint32_t channels,
        std::uint32_t block_size, std::uint32_t sample_rate,
        std::uint32_t lead_blocks, GpuNamCompletionOptions completion = {},
        std::uint32_t capacity = 16, std::uint32_t max_inflight = 1);

    bool prepare() override;
    void prime_fallback(const audio::BufferView<const float>& input,
                        std::uint32_t frames) noexcept override;
    void process_cpu_fallback(const audio::BufferView<const float>& input,
                              audio::BufferView<float>& output,
                              std::uint32_t frames) noexcept override;

    // Read only after callback/worker quiescence. Counts model invocations,
    // not GPU deliveries or CPU time saved.
    std::uint64_t cpu_model_calls() const noexcept { return cpu_model_calls_; }
    std::uint64_t fallback_reads() const noexcept { return fallback_reads_; }

private:
    GpuNamStampedNode(const Config& config, const nam::NamModel& model);
    nam::NamModel model_;
    static constexpr std::uint32_t kMaxNamChannels = 64;
    std::array<nam::NamModel, kMaxNamChannels> cpu_;
    std::array<std::vector<float>, kMaxNamChannels> ring_, due_;
    std::vector<float> zero_;
    std::uint32_t channels_, frames_, lead_, cursor_ = 0;
    std::uint64_t cpu_model_calls_ = 0, fallback_reads_ = 0;
    bool fallback_ready_ = false;
};
} // namespace pulp::examples
