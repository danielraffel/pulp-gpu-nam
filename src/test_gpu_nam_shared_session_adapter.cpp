#include "gpu_nam_shared_session_node.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

using pulp::examples::GpuNamSharedSessionNode;

TEST_CASE("experimental shared WaveNet adapter fails closed without a model",
          "[gpu_nam][gpu_audio][experimental]") {
    GpuNamSharedSessionNode node(2, 32, 48'000, nullptr);

    CHECK_FALSE(node.prepare());
    CHECK_FALSE(node.gpu_available());
    CHECK(node.backend().empty());

    const auto descriptor = node.descriptor();
    CHECK(descriptor.latency_blocks == 1);
    CHECK(descriptor.miss_policy == pulp::gpu_audio::MissPolicy::CpuFallback);
    CHECK(descriptor.supports_cpu_fallback);
}

TEST_CASE("experimental shared WaveNet adapter rejects an unbuilt model",
          "[gpu_nam][gpu_audio][experimental]") {
    pulp::examples::nam::NamModel model;
    GpuNamSharedSessionNode node(2, 32, 48'000, &model);

    CHECK_FALSE(node.prepare());
    CHECK_FALSE(node.gpu_available());
}

TEST_CASE("experimental shared WaveNet adapter keeps fallback across cold GPU dispatch",
          "[gpu_nam][gpu_audio][experimental][runtime]") {
    pulp::examples::nam::NamModel model;
    std::string error;
    REQUIRE(pulp::examples::nam::load_nam(GPU_NAM_MODEL_PATH, model, &error));

    GpuNamSharedSessionNode node(1, 32, 48'000, &model);
    REQUIRE(node.prepare());
    if (!node.gpu_available()) {
        WARN("shared provider unavailable: " << node.backend());
        return;
    }
    CHECK(node.backend() == "Dawn shared WaveNet (experimental)");

    std::vector<float> input(32, 0.0f);
    std::vector<float> output(32, 0.0f);
    input[0] = 0.25f;
    const float* input_channels[] = {input.data()};
    float* output_channels[] = {output.data()};
    pulp::audio::BufferView<const float> input_view(input_channels, 1, 32);
    pulp::audio::BufferView<float> output_view(output_channels, 1, 32);
    for (int block = 0; block < 8; ++block) {
        node.process_block(input_view, output_view, 32);
        for (float sample : output)
            CHECK(std::isfinite(sample));
        std::fill(input.begin(), input.end(), 0.0f);
    }

    // The adapter deliberately gives the non-realtime worker a bounded 2 ms
    // wait. A cold Dawn dispatch can exceed that budget even though later
    // blocks are delivered, so the contract is CPU fallback plus eventual
    // GPU delivery rather than an assertion that the first block is GPU-fast.
    CHECK(node.gpu_delivered_blocks() >= 1);
}

TEST_CASE("experimental shared WaveNet fallback stays history aligned across hits and misses",
          "[gpu_nam][gpu_audio][experimental][fallback]") {
    pulp::examples::nam::NamModel model;
    std::string error;
    REQUIRE(pulp::examples::nam::load_nam(GPU_NAM_MODEL_PATH, model, &error));

    GpuNamSharedSessionNode node(1, 32, 48'000, &model);
    REQUIRE(node.prepare());

    // Independent CPU oracle.  The first fallback slot corresponds to the
    // transport's one-block PDC and is therefore silence.
    pulp::examples::nam::NamModel oracle = model;
    oracle.prewarm();
    std::vector<float> input(32);
    std::vector<float> expected_current(32);
    std::vector<float> expected_due(32, 0.0f);
    std::vector<float> output(32, 0.0f);
    const float* input_channels[] = {input.data()};
    float* output_channels[] = {output.data()};
    pulp::audio::BufferView<const float> input_view(input_channels, 1, 32);
    pulp::audio::BufferView<float> output_view(output_channels, 1, 32);

    for (int block = 0; block < 8; ++block) {
        std::fill(input.begin(), input.end(), 0.0025f * static_cast<float>(block + 1));
        input[static_cast<std::size_t>(block % 7)] += 0.1f;
        oracle.process(input.data(), expected_current.data(), 32);

        // This is the callback-side operation performed for every block.
        node.prime_fallback(input_view, 32);
        if ((block & 1) == 0) {
            // Forced miss: consume the staged, one-block-delayed CPU result.
            std::fill(output.begin(), output.end(), -1.0f);
            node.process_cpu_fallback(input_view, output_view, 32);
            for (std::size_t i = 0; i < output.size(); ++i)
                CHECK(std::abs(output[i] - expected_due[i]) < 1.0e-6f);
        } else {
            // Simulated hit: the worker path may replace the output, but must
            // not advance the realtime fallback a second time.
            node.process_block(input_view, output_view, 32);
            for (float sample : output)
                CHECK(std::isfinite(sample));
        }
        expected_due = expected_current;
    }

    // Reprepare resets both the model state and the staged delay slot.  A
    // forced miss immediately after reprepare must therefore be clean silence.
    REQUIRE(node.prepare());
    std::fill(input.begin(), input.end(), 0.25f);
    node.prime_fallback(input_view, 32);
    std::fill(output.begin(), output.end(), -1.0f);
    node.process_cpu_fallback(input_view, output_view, 32);
    for (float sample : output)
        CHECK(sample == 0.0f);
}
