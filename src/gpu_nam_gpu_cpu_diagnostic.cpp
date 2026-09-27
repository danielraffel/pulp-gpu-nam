#include "gpu_nam_shared_session_node.hpp"

#include <pulp/gpu_audio/gpu_audio_transport.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;

constexpr std::uint32_t kSampleRate = 48'000;


std::uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<Nanoseconds>(end - start).count());
}

bool close_enough(float actual, float expected) {
    const float tolerance = 2.0e-3f * (1.0f + std::abs(expected));
    return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

} // namespace

int main(int argc, char** argv) {
    std::uint32_t block_size = 32, lead_blocks = 1, blocks = 96; bool inject_error = false;
    std::string model_path = GPU_NAM_MODEL_PATH;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto parse = [&](const char* prefix, std::uint32_t& out) {
            if (arg.rfind(prefix, 0) == 0) { try { out = static_cast<std::uint32_t>(std::stoul(arg.substr(std::strlen(prefix)))); } catch (...) { out = 0; } return true; } return false;
        };
        parse("--block-size=", block_size) || parse("--lead-blocks=", lead_blocks);
        if (arg == "--inject-direct-output-error") inject_error = true;
        if (arg.rfind("--model-path=", 0) == 0) model_path = arg.substr(std::strlen("--model-path="));
    }
    if ((block_size != 32 && block_size != 64 && block_size != 128) ||
        (lead_blocks != 1 && lead_blocks != 2 && lead_blocks != 4 && lead_blocks != 8)) {
        std::cerr << "unsupported block/lead\n"; return 2;
    }
    pulp::examples::nam::NamModel model;
    std::string error;
    if (!pulp::examples::nam::load_nam(model_path, model, &error)) {
        std::cerr << "load_error=" << error << '\n';
        return 1;
    }

    // Build the reference before starting the cadence run.  It is the same
    // model and exact input sequence that the shared GPU adapter receives.
    pulp::examples::nam::NamModel oracle = model;
    oracle.prewarm_block_aligned(block_size);
    std::vector<std::vector<float>> inputs(blocks, std::vector<float>(block_size));
    std::vector<std::vector<float>> reference(blocks, std::vector<float>(block_size));
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t i = 0; i < block_size; ++i) {
            const auto sample = static_cast<float>(block * block_size + i);
            inputs[block][i] = 0.07f * std::sin(0.013f * sample)
                               + 0.02f * std::cos(0.037f * sample);
        }
        oracle.process(inputs[block].data(), reference[block].data(), block_size);
    }

    // Measure the callback-facing continuous fallback separately.  This is a
    // CPU cost measurement only, not a realtime claim; the transport run below
    // remains the matched GPU/shared-memory path.
    pulp::examples::GpuNamSharedSessionNode fallback_node(1, block_size, kSampleRate, &model, lead_blocks);
    if (!fallback_node.prepare()) {
        std::cerr << "fallback_prepare=0\n";
        return 1;
    }
    std::vector<float> fallback_output(block_size, 0.0f);
    const float* fallback_input_ptr[] = {inputs[0].data()};
    float* fallback_output_ptr[] = {fallback_output.data()};
    pulp::audio::BufferView<const float> fallback_input(fallback_input_ptr, 1, block_size);
    pulp::audio::BufferView<float> fallback_view(fallback_output_ptr, 1, block_size);
    std::uint64_t fallback_cpu_ns = 0;
    for (const auto& block : inputs) {
        fallback_input_ptr[0] = block.data();
        const auto start = Clock::now();
        fallback_node.prime_fallback(fallback_input, block_size);
        fallback_node.process_cpu_fallback(fallback_input, fallback_view, block_size);
        fallback_cpu_ns += elapsed_ns(start, Clock::now());
    }

    pulp::examples::GpuNamSharedSessionNode node(1, block_size, kSampleRate, &model, lead_blocks);
    if (!node.prepare()) {
        std::cerr << "shared_prepare=0\n";
        return 1;
    }
    const bool provider_available = node.gpu_available();
    std::cout << "provider_available=" << (provider_available ? 1 : 0)
              << " backend=" << node.backend() << '\n';
    if (!provider_available) {
        std::cout << "diagnostic_status=provider_unavailable\n";
        return 2;
    }

    // Isolate the adapter/model from transport PDC.  This direct sequence
    // should match the CPU oracle at the same block index; any discrepancy
    // here is a model/session issue rather than transport disposition.
    pulp::examples::GpuNamSharedSessionNode direct_node(1, block_size, kSampleRate, &model, lead_blocks);
    if (!direct_node.prepare()) {
        std::cerr << "direct_prepare=0\n";
        return 1;
    }
    std::vector<float> direct_output(block_size, 0.0f);
    const float* direct_input_ptr[] = {inputs[0].data()};
    float* direct_output_ptr[] = {direct_output.data()};
    pulp::audio::BufferView<const float> direct_input(direct_input_ptr, 1, block_size);
    pulp::audio::BufferView<float> direct_view(direct_output_ptr, 1, block_size);
    std::uint32_t direct_failures = 0;
    std::uint32_t direct_mismatch_blocks = 0;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        direct_input_ptr[0] = inputs[block].data();
        direct_node.process_block(direct_input, direct_view, block_size);
        if (inject_error && block == 0) direct_output[0] += 1.0f;
        bool block_mismatch = false;
        for (std::uint32_t i = 0; i < block_size; ++i)
            if (!close_enough(direct_output[i], reference[block][i])) {
                ++direct_failures;
                block_mismatch = true;
            }
        if (block_mismatch && direct_mismatch_blocks++ < 4)
            std::cout << "direct_mismatch_block=" << block
                      << " out0=" << direct_output[0]
                      << " ref0=" << reference[block][0] << '\n';
    }
    std::cout << "direct_model_parity_failures=" << direct_failures << '\n';

    pulp::gpu_audio::GpuAudioTransport transport;
    if (!transport.prepare(&node, {.ring_blocks = 16,
                                   .run_worker_thread = true,
                                   .wake_on_write = true})) {
        std::cerr << "transport_prepare=0\n";
        return 1;
    }

    std::vector<float> output(block_size, 0.0f);
    const float* input_ptr[] = {inputs[0].data()};
    float* output_ptr[] = {output.data()};
    pulp::audio::BufferView<const float> input_view(input_ptr, 1, block_size);
    pulp::audio::BufferView<float> output_view(output_ptr, 1, block_size);
    std::uint64_t callback_total_ns = 0;
    const auto process_cpu_start = std::clock();
    std::uint64_t callback_max_ns = 0;
    std::uint64_t late_total_ns = 0;
    std::uint32_t parity_failures = 0;
    std::uint32_t mismatch_blocks = 0;
    float max_error = 0.0f;
    const auto cadence_origin = Clock::now();
    for (std::uint32_t block = 0; block < blocks; ++block) {
        // Absolute scheduling: every deadline is derived from the origin.  A
        // slow callback delays only its own start; its processing time is not
        // added to the next sleep period.
        const auto deadline = cadence_origin
            + Nanoseconds(static_cast<std::int64_t>(block) * block_size * 1'000'000'000LL
                          / kSampleRate);
        std::this_thread::sleep_until(deadline);
        const auto callback_start = Clock::now();
        const auto lateness = callback_start > deadline ? elapsed_ns(deadline, callback_start) : 0;
        late_total_ns += lateness;

        input_ptr[0] = inputs[block].data();
        const auto start = Clock::now();
        transport.process(input_view, output_view, block_size);
        const auto cost = elapsed_ns(start, Clock::now());
        callback_total_ns += cost;
        callback_max_ns = std::max(callback_max_ns, cost);

        const auto expected_block = block < lead_blocks ? std::numeric_limits<std::uint32_t>::max()
                                                : block - lead_blocks;
        float block_max_error = 0.0f;
        bool block_failed = false;
        for (std::uint32_t i = 0; i < block_size; ++i) {
            const float expected = expected_block == std::numeric_limits<std::uint32_t>::max()
                                       ? 0.0f
                                       : reference[expected_block][i];
            block_max_error = std::max(block_max_error, std::abs(output[i] - expected));
            if (!close_enough(output[i], expected)) {
                ++parity_failures;
                block_failed = true;
            }
        }
        max_error = std::max(max_error, block_max_error);
        if (block_failed) {
            ++mismatch_blocks;
            if (mismatch_blocks <= 8)
                std::cout << "mismatch_block=" << block << " expected=" << expected_block
                          << " max_error=" << block_max_error << '\n';
        }
    }

    const auto stats_before_release = transport.stats();
    transport.release();
    const auto process_cpu_end = std::clock();
    if (process_cpu_start == std::clock_t(-1) || process_cpu_end == std::clock_t(-1) || process_cpu_end < process_cpu_start) return 4;
    const auto gpu_blocks = node.gpu_delivered_blocks();
    const auto fallback_blocks = node.cpu_fallback_blocks();
    const auto primed_blocks = node.fallback_prime_blocks();
    std::cout << "blocks=" << blocks
              << " input_blocks=" << blocks << " measured_blocks=" << blocks
              << " drain_blocks=" << lead_blocks
              << " provider_available=" << (provider_available ? 1 : 0)
              << " gpu_inner_completions=" << gpu_blocks
              << " cpu_fallback=" << fallback_blocks
              << " fallback_primed=" << primed_blocks
              << " produced=" << stats_before_release.produced_blocks
              << " transport_misses=" << stats_before_release.miss_blocks
              << " input_dropped=" << stats_before_release.input_dropped_frames
              << " parity_failures=" << parity_failures
              << " mismatch_blocks=" << mismatch_blocks
              << " max_error=" << max_error
              << " fallback_cpu_elapsed_ns=" << fallback_cpu_ns
              << " process_cpu_ticks=" << (process_cpu_end - process_cpu_start)
              << " process_cpu_seconds=" << (double(process_cpu_end - process_cpu_start) / CLOCKS_PER_SEC)
              << " process_cpu_ticks_per_second=" << CLOCKS_PER_SEC
              << " callback_total_ns=" << callback_total_ns
              << " callback_max_ns=" << callback_max_ns
              << " callback_late_total_ns=" << late_total_ns
              << " worker_avg_us=" << stats_before_release.avg_block_us
              << '\n';

    if (gpu_blocks == 0 || direct_failures != 0 || parity_failures != 0 || primed_blocks != blocks) {
        std::cout << "diagnostic_status=failed\n";
        return 3;
    }
    std::cout << "diagnostic_status=passed\n";
    return 0;
}
