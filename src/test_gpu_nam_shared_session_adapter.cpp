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

TEST_CASE("experimental shared WaveNet adapter prepares the real example model",
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
    node.process_block(input_view, output_view, 32);
    CHECK(node.gpu_delivered_blocks() == 1);

    // A second block covers consecutive public-session dispositions rather
    // than a one-shot preparation path. CPU fallback cannot increment this.
    std::fill(input.begin(), input.end(), 0.0f);
    node.process_block(input_view, output_view, 32);
    CHECK(node.gpu_delivered_blocks() == 2);

    for (float sample : output)
        CHECK(std::isfinite(sample));
}
