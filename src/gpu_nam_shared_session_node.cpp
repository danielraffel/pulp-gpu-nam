#include "gpu_nam_shared_session_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace pulp::examples {

std::uint64_t GpuNamSharedSessionNode::now_ns() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

gpu_audio::GpuAudioNodeDescriptor GpuNamSharedSessionNode::descriptor() const {
    gpu_audio::GpuAudioNodeDescriptor descriptor;
    descriptor.name = "NeuralAmp (experimental shared WaveNet)";
    descriptor.input_channels = channels_;
    descriptor.output_channels = channels_;
    descriptor.block_size = block_size_;
    descriptor.sample_rate = sample_rate_;
    descriptor.latency_blocks = 1;
    descriptor.miss_policy = gpu_audio::MissPolicy::CpuFallback;
    descriptor.supports_cpu_fallback = true;
    return descriptor;
}

bool GpuNamSharedSessionNode::prepare() {
    if (model_ == nullptr || channels_ == 0 || channels_ > kNamChannels ||
        block_size_ == 0 || sample_rate_ == 0)
        return false;

    const auto& model_layers = model_->arrays();
    if (model_layers.empty())
        return false;

    dilations_.clear();
    descriptors_.clear();
    dilations_.reserve(model_layers.size());
    descriptors_.reserve(model_layers.size());
    for (const auto& layer : model_layers) {
        if (layer.activation != "Tanh" || layer.input_size <= 0 ||
            layer.condition_size <= 0 || layer.channels <= 0 ||
            layer.kernel_size <= 0 || layer.head_size <= 0 || layer.dilations.empty())
            return false;
        dilations_.emplace_back(layer.dilations.begin(), layer.dilations.end());
        descriptors_.push_back({
            .input_size = static_cast<std::uint32_t>(layer.input_size),
            .condition_size = static_cast<std::uint32_t>(layer.condition_size),
            .channels = static_cast<std::uint32_t>(layer.channels),
            .kernel = static_cast<std::uint32_t>(layer.kernel_size),
            .head_size = static_cast<std::uint32_t>(layer.head_size),
            .dilation = 0,
            .gated = layer.gated,
            .head_bias = layer.head_bias,
            .tanh_activation = true,
            .dilations = dilations_.back(),
        });
    }

    fallback_zero_input_.assign(block_size_, 0.0f);

    const gpu_audio::GpuWaveNetDescriptor model_descriptor{
        .block_size = block_size_,
        .sample_rate = sample_rate_,
        .stream_instances = 1,
        .head_scale = model_->head_scale(),
        .layers = descriptors_,
        .weight_count = model_->weights_size(),
    };
    const std::span<const float> weights(model_->weights_data(), model_->weights_size());

    for (std::uint32_t channel = 0; channel < channels_; ++channel) {
        worker_cpu_[channel] = *model_;
        realtime_cpu_[channel] = *model_;
        worker_cpu_[channel].prewarm();
        realtime_cpu_[channel].prewarm();
        gpu_output_[channel].assign(block_size_, 0.0f);
        fallback_output_[channel].assign(block_size_, 0.0f);
        fallback_delay_[channel].assign(block_size_, 0.0f);

        auto result = gpu_audio::GpuWaveNetSession::create({
            .descriptor = model_descriptor,
            .weights = weights,
            .slots = 2,
        });
        if (!result) {
            sessions_[channel].reset();
            prepared_ = false;
            return false;
        }
        sessions_[channel] = std::move(result.session);
    }
    prepared_ = true;
    sequence_ = 0;
    gpu_delivered_blocks_ = 0;
    return true;
}

void GpuNamSharedSessionNode::prime_fallback(
    const audio::BufferView<const float>& input, std::uint32_t n) noexcept {
    if (!prepared_ || n != block_size_)
        return;

    // The transport's fixed one-block PDC means the delayed slot is the
    // substitute for a miss visible in this callback.  Copy it out before
    // advancing the state with the current input block.
    for (std::uint32_t channel = 0; channel < channels_; ++channel) {
        auto& due = fallback_output_[channel];
        auto& slot = fallback_delay_[channel];
        std::copy_n(slot.data(), n, due.data());
        // A missing input channel is a zero signal, but it still advances the
        // stateful WaveNet history.  Skipping process() here would make the
        // next real block resume from an old timeline.
        const float* source = channel < input.num_channels()
                                  ? input.channel_ptr(channel)
                                  : fallback_zero_input_.data();
        realtime_cpu_[channel].process(source, slot.data(), n);
    }
}

void GpuNamSharedSessionNode::drain_ready(gpu_audio::GpuWaveNetSession& session,
                                          std::vector<float>& scratch) noexcept {
    session.service(now_ns());
    while (session.receive(scratch)) {
        // A result that arrived after the worker's bounded wait has already
        // been replaced by the CPU output for its timeline slot. Consuming it
        // here releases the provider slot and keeps sequence intake moving.
    }
}

bool GpuNamSharedSessionNode::submit_and_collect(
    gpu_audio::GpuWaveNetSession& session, std::span<const float> input,
    std::uint64_t sequence, std::span<float> output) noexcept {
    const std::uint64_t start = now_ns();
    const std::uint64_t deadline = start + kWorkerWaitBudgetNs;
    if (!session.submit_block(input, sequence, deadline))
        return false;

    while (now_ns() < deadline) {
        session.service(now_ns());
        if (auto result = session.receive(output)) {
            if (result->sequence != sequence)
                continue;
            if (result->status == gpu_audio::GpuWaveNetBlockStatus::GpuDelivered)
                ++gpu_delivered_blocks_;
            return result->status == gpu_audio::GpuWaveNetBlockStatus::GpuDelivered;
        }
        std::this_thread::yield();
    }
    return false;
}

void GpuNamSharedSessionNode::process_block(const audio::BufferView<const float>& input,
                                            audio::BufferView<float>& output,
                                            std::uint32_t n) {
    if (!prepared_ || n != block_size_) {
        output.clear();
        return;
    }
    const std::uint64_t sequence = sequence_++;
    for (std::uint32_t channel = 0; channel < channels_; ++channel) {
        const float* source = channel < input.num_channels() ? input.channel_ptr(channel) : nullptr;
        float* destination = output.channel_ptr(channel);
        if (source == nullptr) {
            std::fill(destination, destination + n, 0.0f);
            continue;
        }

        // Produce the correct CPU result first. The shared session may miss its
        // bounded worker budget, in which case this exact output remains intact.
        worker_cpu_[channel].process(source, destination, n);
        auto& session = *sessions_[channel];
        drain_ready(session, gpu_output_[channel]);
        if (submit_and_collect(session,
                               std::span<const float>(source, n), sequence,
                               std::span<float>(gpu_output_[channel].data(), n))) {
            std::memcpy(destination, gpu_output_[channel].data(),
                        static_cast<std::size_t>(n) * sizeof(float));
        }
    }
}

void GpuNamSharedSessionNode::process_cpu_fallback(
    const audio::BufferView<const float>& /*input*/, audio::BufferView<float>& output,
    std::uint32_t n) noexcept {
    if (!prepared_ || n != block_size_ || output.num_samples() < n) {
        output.clear();
        return;
    }
    // Clear any excess samples too, so a larger host view cannot retain stale
    // tail data when the transport supplies a shorter block.
    output.clear();
    const std::uint32_t count = std::min<std::uint32_t>(channels_, output.num_channels());
    for (std::uint32_t channel = 0; channel < count; ++channel) {
        const float* source = fallback_output_[channel].data();
        std::copy_n(source, n, output.channel_ptr(channel));
    }
}

} // namespace pulp::examples
