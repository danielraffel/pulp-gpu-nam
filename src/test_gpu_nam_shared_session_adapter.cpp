#include "gpu_nam_shared_session_node.hpp"

#include <catch2/catch_test_macros.hpp>

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
