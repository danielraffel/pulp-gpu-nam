#pragma once

// Experimental adapter for the provisional public Pulp WaveNet session.
//
// This path deliberately stays behind GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION.
// It is a validation bridge, not the default engine: the current public session
// owns one provider per mono stream and exposes serialized operations, so this
// adapter keeps the existing GpuAudioTransport and CPU fallback contracts while
// synchronously waiting on the worker for a bounded interval.

#include "nam_model.hpp"

#include <pulp/gpu_audio/gpu_audio_node.hpp>
#include <pulp/gpu_audio/gpu_wavenet.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace pulp::examples {

class GpuNamSharedSessionNode final : public gpu_audio::GpuAudioNode {
  public:
    GpuNamSharedSessionNode(std::uint32_t channels, std::uint32_t block_size,
                            std::uint32_t sample_rate, const nam::NamModel* model)
        : channels_(channels),
          block_size_(block_size),
          sample_rate_(sample_rate),
          model_(model) {}

    gpu_audio::GpuAudioNodeDescriptor descriptor() const override;
    bool prepare() override;
    void process_block(const audio::BufferView<const float>& input,
                       audio::BufferView<float>& output, std::uint32_t n) override;
    void process_cpu_fallback(const audio::BufferView<const float>& input,
                              audio::BufferView<float>& output,
                              std::uint32_t n) noexcept override;

    bool gpu_available() const noexcept {
        return prepared_ && channels_ > 0 && channels_ <= kNamChannels &&
               sessions_[0] != nullptr && (channels_ == 1 || sessions_[1] != nullptr);
    }

    // The public session intentionally does not expose adapter identity yet.
    // Keep the diagnostic explicit so callers do not confuse it with the legacy
    // blocking render::GpuCompute path.
    std::string backend() const {
        return gpu_available() ? "Dawn shared WaveNet (experimental)" : std::string{};
    }

    // Validation diagnostic: counts results whose public session disposition
    // was GpuDelivered. This is separate from gpu_available(), which only
    // proves that the session prepared successfully.
    std::uint64_t gpu_delivered_blocks() const noexcept { return gpu_delivered_blocks_; }

  private:
    static constexpr std::uint32_t kNamChannels = 2;
    static constexpr std::uint64_t kWorkerWaitBudgetNs = 2'000'000;

    static std::uint64_t now_ns() noexcept;
    void drain_ready(gpu_audio::GpuWaveNetSession& session,
                     std::vector<float>& scratch) noexcept;
    bool submit_and_collect(gpu_audio::GpuWaveNetSession& session,
                            std::span<const float> input, std::uint64_t sequence,
                            std::span<float> output) noexcept;

    std::uint32_t channels_ = 0;
    std::uint32_t block_size_ = 0;
    std::uint32_t sample_rate_ = 0;
    const nam::NamModel* model_ = nullptr;
    bool prepared_ = false;
    std::uint64_t sequence_ = 0;
    std::uint64_t gpu_delivered_blocks_ = 0;

    std::vector<std::vector<std::uint32_t>> dilations_;
    std::vector<gpu_audio::GpuWaveNetLayerDescriptor> descriptors_;
    std::array<std::unique_ptr<gpu_audio::GpuWaveNetSession>, kNamChannels> sessions_{};
    std::array<nam::NamModel, kNamChannels> worker_cpu_{};
    std::array<nam::NamModel, kNamChannels> realtime_cpu_{};
    std::array<std::vector<float>, kNamChannels> gpu_output_{};
};

} // namespace pulp::examples
