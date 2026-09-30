#include "gpu_nam_cpu_shadow_worker.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kChannels = 2;
constexpr std::uint32_t kFrames = 32;
constexpr std::uint32_t kLead = 4;
constexpr std::uint32_t kBlocks = 96;

bool wait_for(std::uint64_t expected,
              const pulp::examples::GpuNamCpuShadowWorker& worker) {
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (worker.processed() < expected && Clock::now() < deadline)
        std::this_thread::yield();
    return worker.processed() >= expected;
}
}

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : GPU_NAM_MODEL_PATH;
    pulp::examples::nam::NamModel model;
    std::string error;
    if (!pulp::examples::nam::load_nam(model_path, model, &error)) {
        std::cerr << "load_error=" << error << '\n';
        return 1;
    }
    pulp::examples::GpuNamCpuShadowWorker worker(model, kChannels, kFrames, kLead);
    if (!worker.prepare() || !worker.start())
        return 2;
    pulp::examples::GpuNamCpuShadowWorker oversized(model, kChannels, kFrames, kLead, 65);
    if (oversized.prepare())
        return 10;

    std::array<std::vector<float>, kChannels> input, actual;
    for (auto& channel : input) channel.assign(kFrames, 0.f);
    for (auto& channel : actual) channel.assign(kFrames, 0.f);
    std::vector<float> history(static_cast<std::size_t>(kBlocks) * kChannels * kFrames,
                               0.f);
    std::array<pulp::examples::nam::NamModel, kChannels> oracle{model, model};
    for (auto& channel : oracle) channel.prewarm_block_aligned(kFrames);
    const float* input_ptr[kChannels] = {input[0].data(), input[1].data()};
    float* actual_ptr[kChannels] = {actual[0].data(), actual[1].data()};
    const pulp::audio::BufferView<const float> input_view(input_ptr, kChannels, kFrames);
    pulp::audio::BufferView<float> actual_view(actual_ptr, kChannels, kFrames);

    // A stopped worker must fail closed when the bounded ring is full; the
    // callback never waits for capacity and the drop is observable.
    pulp::examples::GpuNamCpuShadowWorker saturated(model, kChannels, kFrames, 1, 4);
    if (!saturated.prepare())
        return 11;
    for (std::uint64_t sequence = 0; sequence < 4; ++sequence)
        if (!saturated.submit(input_view, sequence))
            return 12;
    if (saturated.submit(input_view, 4) || saturated.dropped() != 1)
        return 13;

    // A future sequence cannot be mistaken for a ready fallback slot.
    if (worker.copy_fallback(1234, actual_view))
        return 3;
    // Submission order is part of the bounded ring contract. Rejecting a gap
    // keeps a stale slot from being published under a future sequence.
    if (worker.submit(input_view, 2) || worker.sequence_errors() != 1)
        return 9;
    for (std::uint32_t block = 0; block < kBlocks; ++block) {
        for (std::uint32_t channel = 0; channel < kChannels; ++channel)
            for (std::uint32_t i = 0; i < kFrames; ++i)
                input[channel][i] = .07f * std::sin(float(block * kFrames + i) *
                                                     (.013f + channel * .007f));
        const auto start = Clock::now();
        if (!worker.submit(input_view, block))
            return 4;
        if (!wait_for(block + 1, worker))
            return 5;
        for (std::uint32_t channel = 0; channel < kChannels; ++channel)
            oracle[channel].process(
                input[channel].data(),
                history.data() + (static_cast<std::size_t>(block) * kChannels + channel) * kFrames,
                kFrames);
        if (block >= kLead) {
            for (std::uint32_t channel = 0; channel < kChannels; ++channel)
                std::fill(actual[channel].begin(), actual[channel].end(), 0.f);
            if (!worker.copy_fallback(block - kLead, actual_view))
                return 6;
            for (std::uint32_t channel = 0; channel < kChannels; ++channel)
                for (std::uint32_t i = 0; i < kFrames; ++i) {
                    const auto expected = history[(static_cast<std::size_t>(block - kLead) *
                                                   kChannels + channel) * kFrames + i];
                    if (!std::isfinite(actual[channel][i]) ||
                        std::abs(actual[channel][i] - expected) > 1e-4f)
                        return 8;
                }
        }
        worker.retire(block);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start).count();
        (void)elapsed;
    }
    worker.stop();
    if (worker.submitted() != kBlocks || worker.processed() != kBlocks ||
        worker.dropped() != 0 || worker.sequence_errors() != 1)
        return 7;
    std::cout << "shadow_worker=1 channels=" << kChannels << " frames=" << kFrames
              << " lead=" << kLead << " blocks=" << kBlocks
              << " submitted=" << worker.submitted() << " processed=" << worker.processed()
              << " dropped=" << worker.dropped() << " sequence_errors="
              << worker.sequence_errors() << " ready_misses=" << worker.ready_misses()
              << " diagnostic_status=passed\n";
    return 0;
}
