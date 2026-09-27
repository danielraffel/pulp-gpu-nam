#include "gpu_nam_stamped_node.hpp"
#include <algorithm>
#include <limits>

namespace pulp::examples {
std::unique_ptr<GpuNamStampedNode> GpuNamStampedNode::create(
    const nam::NamModel& model, std::uint32_t channels,
    std::uint32_t frames, std::uint32_t sample_rate, std::uint32_t lead) {
    if (!channels || channels > 2 || !frames || !sample_rate || !lead || lead > 8 ||
        model.arrays().empty()) return {};
    const auto prewarm = model.prewarm_block_count(frames);
    if (prewarm > std::numeric_limits<std::uint32_t>::max()) return {};
    std::vector<std::vector<std::uint32_t>> dilations;
    std::vector<gpu_audio::GpuWaveNetLayerDescriptor> layers;
    dilations.reserve(model.arrays().size());
    layers.reserve(model.arrays().size());
    for (const auto& layer : model.arrays()) {
        if (layer.activation != "Tanh" || layer.input_size <= 0 ||
            layer.condition_size <= 0 || layer.channels <= 0 || layer.kernel_size <= 0 ||
            layer.head_size <= 0 || layer.dilations.empty() ||
            std::any_of(layer.dilations.begin(), layer.dilations.end(),
                        [](auto d) { return d <= 0; })) return {};
        dilations.emplace_back(layer.dilations.begin(), layer.dilations.end());
        layers.push_back({
            .input_size = static_cast<std::uint32_t>(layer.input_size),
            .condition_size = static_cast<std::uint32_t>(layer.condition_size),
            .channels = static_cast<std::uint32_t>(layer.channels),
            .kernel = static_cast<std::uint32_t>(layer.kernel_size),
            .head_size = static_cast<std::uint32_t>(layer.head_size),
            .dilation = 0, .gated = layer.gated, .head_bias = layer.head_bias,
            .tanh_activation = true, .dilations = dilations.back()});
    }
    Config config;
    config.session.descriptor = {
        .block_size = frames, .sample_rate = sample_rate, .stream_instances = 1,
        .head_scale = model.head_scale(), .layers = layers,
        .weight_count = model.weights_size()};
    config.session.weights = {model.weights_data(), model.weights_size()};
    config.channels = channels;
    config.lead_blocks = lead;
    config.capacity = 16;
    config.prewarm_blocks = static_cast<std::uint32_t>(prewarm);
    config.miss_policy = gpu_audio::MissPolicy::CpuFallback;
    config.supports_cpu_fallback = true;
    if (!gpu_audio::validate_gpu_wavenet_descriptor(config.session.descriptor).accepted())
        return {};
    // The base constructor copies spans before these temporary owners expire.
    return std::unique_ptr<GpuNamStampedNode>(new GpuNamStampedNode(config, model));
}

GpuNamStampedNode::GpuNamStampedNode(const Config& config, const nam::NamModel& model)
    : GpuWaveNetRealtimeNode(config), model_(model), channels_(config.channels),
      frames_(config.session.descriptor.block_size), lead_(config.lead_blocks) {}

bool GpuNamStampedNode::prepare() {
    fallback_ready_ = false;
    if (!release()) return false;
    zero_.assign(frames_, 0.f);
    for (std::uint32_t ch = 0; ch < channels_; ++ch) {
        cpu_[ch] = model_;
        cpu_[ch].prewarm_block_aligned(frames_);
        ring_[ch].assign(static_cast<std::size_t>(lead_) * frames_, 0.f);
        due_[ch].assign(frames_, 0.f);
    }
    cursor_ = 0;
    cpu_model_calls_ = fallback_reads_ = 0;
    fallback_ready_ = GpuWaveNetRealtimeNode::prepare();
    return fallback_ready_;
}

void GpuNamStampedNode::prime_fallback(const audio::BufferView<const float>& input,
                                      std::uint32_t frames) noexcept {
    if (!fallback_ready_ || frames != frames_) return;
    for (std::uint32_t ch = 0; ch < channels_; ++ch) {
        auto* slot = ring_[ch].data() + static_cast<std::size_t>(cursor_) * frames_;
        std::copy_n(slot, frames_, due_[ch].data());
        const auto* source = ch < input.num_channels() && input.num_samples() >= frames_
                                 ? input.channel_ptr(ch) : zero_.data();
        cpu_[ch].process(source, slot, frames_);
        ++cpu_model_calls_;
    }
    cursor_ = (cursor_ + 1) % lead_;
}

void GpuNamStampedNode::process_cpu_fallback(const audio::BufferView<const float>&,
                                            audio::BufferView<float>& output,
                                            std::uint32_t frames) noexcept {
    output.clear();
    if (!fallback_ready_ || frames != frames_ || output.num_samples() < frames_) return;
    ++fallback_reads_;
    for (std::size_t ch = 0; ch < std::min<std::size_t>(channels_, output.num_channels()); ++ch)
        std::copy_n(due_[ch].data(), frames_, output.channel_ptr(ch));
}
} // namespace pulp::examples
