#include "gpu_nam_stamped_node.hpp"
#include <pulp/gpu_audio/gpu_audio_transport.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    unsigned lead = 2;
    unsigned frames = 32;
    if (argc > 4) return 64;
    if (argc >= 2) {
        const std::string arg = argv[1];
        const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), lead);
        if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() ||
            (lead != 1 && lead != 2 && lead != 4 && lead != 8)) return 64;
    }
    if (argc >= 3) {
        const std::string arg = argv[2];
        const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), frames);
        if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() ||
            (frames != 32 && frames != 64 && frames != 128)) return 64;
    }
    using namespace pulp;
    constexpr unsigned channels = 2, blocks = 48;
    examples::nam::NamModel model;
    std::string error;
    if (!examples::nam::load_nam(argc == 4 ? argv[3] : GPU_NAM_MODEL_PATH, model, &error)) return 1;
    if (examples::GpuNamStampedNode::create(model, 3, frames, 48000, lead) ||
        examples::GpuNamStampedNode::create(model, channels, frames, 48000, 0)) return 2;
    auto node = examples::GpuNamStampedNode::create(model, channels, frames, 48000, lead);
    if (!node) return 3;
    if (!node->prepare()) {
        std::cout << "shared_prepare_failed=1 cause=unclassified\n";
        return 8;
    }
    std::array<examples::nam::NamModel, channels> oracle{model, model};
    for (auto& cpu : oracle) cpu.prewarm_block_aligned(frames);
    std::vector<float> history((blocks + lead) * channels * frames);
    std::array<std::vector<float>, channels> input{}, output{};
    for (unsigned ch = 0; ch < channels; ++ch) {
        input[ch].resize(frames);
        output[ch].resize(frames);
    }
    const float* ins[]{input[0].data(), input[1].data()};
    float* outs[]{output[0].data(), output[1].data()};
    audio::BufferView<const float> in(ins, channels, frames);
    audio::BufferView<float> out(outs, channels, frames);
    gpu_audio::GpuAudioTransport transport;
    if (!transport.prepare(node.get(), {.ring_blocks = 16, .run_worker_thread = false})) return 4;
    double max_error = 0;
    unsigned gpu_callbacks = 0;
    for (unsigned block = 0; block < blocks + lead; ++block) {
        for (unsigned ch = 0; ch < channels; ++ch) {
            for (unsigned i = 0; i < frames; ++i)
                input[ch][i] = block < blocks ? .1f * std::sin(float(block * frames + i) *
                                                              (.013f + ch * .007f)) : 0.f;
            oracle[ch].process(input[ch].data(),
                history.data() + (block * channels + ch) * frames, frames);
        }
        const auto reads_before = node->fallback_reads();
        transport.process(in, out, frames);
        if (block >= lead && node->fallback_reads() == reads_before) ++gpu_callbacks;
        for (unsigned ch = 0; ch < channels; ++ch) for (unsigned i = 0; i < frames; ++i) {
            const float expected = block < lead ? 0.f :
                history[((block - lead) * channels + ch) * frames + i];
            const double residual = std::abs(double(output[ch][i]) - expected);
            if (!std::isfinite(residual)) return 5;
            max_error = std::max(max_error, residual);
        }
        // Functional scheduling only, not a realtime deadline benchmark. Stop
        // servicing near the end to prove aligned fallback after actual hits.
        if (block < 32) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
            do {
                transport.pump();
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            } while (std::chrono::steady_clock::now() < until);
        }
    }
    const auto stats = transport.stats();
    transport.release();
    const auto calls = node->cpu_model_calls();
    const auto misses = node->fallback_reads();
    if (!node->release()) return 6;
    std::cout << "frames=" << frames << " lead=" << lead << " max_error=" << max_error
              << " cpu_model_calls=" << calls << " fallback_reads=" << misses
              << " gpu_callbacks=" << gpu_callbacks
              << " worker_produced=" << stats.produced_blocks << '\n';
    return max_error <= 1e-4 && calls == (blocks + lead) * channels &&
           misses > 0 && gpu_callbacks > 0 ? 0 : 7;
}
