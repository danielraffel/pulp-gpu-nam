#include "gpu_nam_stamped_node.hpp"
#include "nam_model.hpp"

#include <pulp/audio/buffer.hpp>
#include <pulp/gpu_audio/gpu_audio_transport.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main() {
    constexpr std::uint32_t channels = 4;
    constexpr std::uint32_t frames = 32;
    constexpr std::uint32_t lead = 4;
    constexpr std::uint32_t blocks = 4096;

    pulp::examples::nam::NamModel model;
    std::string error;
    if (!pulp::examples::nam::load_nam(GPU_NAM_MODEL_PATH, model, &error)) {
        std::cerr << "model_load_failed=" << error << "\n";
        return 1;
    }
    auto node = pulp::examples::GpuNamStampedNode::create(model, channels, frames, 48'000, lead);
    if (!node) {
        // A machine without the authenticated provider is an explicit skip for
        // this physical validation. Shape acceptance was still checked by the
        // create() call; never report a GPU result from this path.
        std::cout << "multichannel_provider_unavailable=1 channels=" << channels << "\n";
        return 0;
    }
    if (!node->prepare()) {
        std::cout << "multichannel_provider_unavailable=1 prepare_failed=1 channels="
                  << channels << "\n";
        return 0;
    }

    pulp::gpu_audio::GpuAudioTransport transport;
    if (!transport.prepare(node.get(), {.ring_blocks = 16, .run_worker_thread = false})) {
        std::cerr << "transport_prepare_failed=1\n";
        return 2;
    }

    std::array<pulp::examples::nam::NamModel, channels> oracle{model, model, model, model};
    for (auto& cpu : oracle) cpu.prewarm_block_aligned(frames);
    std::array<std::vector<float>, channels> input{}, output{};
    std::array<const float*, channels> in_ptrs{};
    std::array<float*, channels> out_ptrs{};
    std::vector<float> expected(std::size_t(blocks + lead) * channels * frames, 0.0f);
    for (auto& v : input) v.assign(frames, 0.0f);
    for (auto& v : output) v.assign(frames, 0.0f);
    pulp::audio::BufferView<const float> in(in_ptrs.data(), channels, frames);
    pulp::audio::BufferView<float> out(out_ptrs.data(), channels, frames);

    double max_error = 0.0;
    std::uint64_t nonfinite = 0;
    for (std::uint32_t block = 0; block < blocks + lead; ++block) {
        for (std::uint32_t ch = 0; ch < channels; ++ch) {
            for (std::uint32_t i = 0; i < frames; ++i) {
                const auto sample = static_cast<double>(block * frames + i);
                input[ch][i] = block < blocks
                    ? static_cast<float>(0.05 * std::sin((0.011 + 0.003 * ch) * sample) +
                                         0.01 * std::cos((0.017 + 0.002 * ch) * sample))
                    : 0.0f;
            }
            in_ptrs[ch] = input[ch].data();
            out_ptrs[ch] = output[ch].data();
            oracle[ch].process(input[ch].data(),
                expected.data() + (std::size_t(block) * channels + ch) * frames, frames);
        }
        transport.process(in, out, frames);
        const auto due = block >= lead ? block - lead : blocks + lead;
        if (block >= lead) {
            for (std::uint32_t ch = 0; ch < channels; ++ch)
                for (std::uint32_t i = 0; i < frames; ++i) {
                    const float want = expected[(std::size_t(due) * channels + ch) * frames + i];
                    const float got = output[ch][i];
                    if (!std::isfinite(got)) ++nonfinite;
                    max_error = std::max(max_error, std::abs(double(got) - want));
                }
        }
        // External-pump mode is intentional: it makes completion retirement
        // observable without introducing a second scheduling variable.
        transport.pump();
        // Give the provider a bounded observation window. This is a paced
        // lifecycle test, not a same-block deadline benchmark.
        std::this_thread::sleep_for(std::chrono::microseconds(667));
        transport.pump();
    }
    // Drain all records before release. A successful release is part of this
    // test: it proves every channel's provider/session owners can retire.
    for (int i = 0; i < 16; ++i) transport.pump();
    const auto stats = transport.stats();
    const auto delivery = transport.delivery_snapshot();
    const auto cpu_calls = node->cpu_model_calls();
    const auto fallback_reads = node->fallback_reads();
    const bool transport_released = (transport.release(), true);
    const bool node_released = node->release();

    std::cout << "channels=" << channels << " blocks=" << blocks
              << " max_error=" << max_error << " nonfinite=" << nonfinite
              << " produced=" << stats.produced_blocks
              << " misses=" << stats.miss_blocks
              << " gpu_blocks=" << delivery.gpu_blocks
              << " fallback_blocks=" << delivery.cpu_fallback_blocks
              << " cpu_model_calls=" << cpu_calls
              << " fallback_reads=" << fallback_reads
              << " transport_released=" << transport_released
              << " node_released=" << node_released << "\n";

    if (!transport_released || !node_released || nonfinite != 0 || max_error > 1e-4 ||
        cpu_calls != std::uint64_t(blocks + lead) * channels ||
        delivery.gpu_blocks < blocks / 4)
        return 3;
    return 0;
}
