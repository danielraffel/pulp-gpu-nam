#include "gpu_nam/forge_adapter.hpp"

#include <pulp/host/signal_graph.hpp>
#include <pulp/signal/dc_blocker.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "nam_runtime.hpp"

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 128;
constexpr int kBlockCount = 32;

void configure_cpu_reference(pulp::state::StateStore& state) {
    using namespace pulp::examples;
    state.set_value(kInputGain, 0.0f);
    state.set_value(kOutputGain, 0.0f);
    state.set_value(kMix, 100.0f);
    state.set_value(kEngine, 0.0f);
    state.set_value(kBypass, 0.0f);
    state.set_value(kNoiseGateActive, 0.0f);
    state.set_value(kEQActive, 0.0f);
    state.set_value(kOutputMode, 0.0f);
}

std::vector<float> make_probe(const std::size_t frames) {
    std::vector<float> probe(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        const float t = static_cast<float>(i);
        probe[i] = 0.25f * (std::sin(0.031f * t) +
                            0.17f * std::sin(0.071f * t) +
                            0.05f * std::sin(0.113f * t));
    }
    return probe;
}

double best_alignment_correlation(const std::vector<float>& output,
                                  const std::vector<float>& oracle,
                                  const int max_lag,
                                  const std::size_t skip) {
    double best = -1.0;
    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        double xy = 0.0;
        double xx = 0.0;
        double yy = 0.0;
        std::size_t count = 0;
        for (std::size_t i = skip; i < oracle.size(); ++i) {
            const auto oi = static_cast<long long>(i) + lag;
            if (oi < 0 || oi >= static_cast<long long>(output.size())) continue;
            const double x = output[static_cast<std::size_t>(oi)];
            const double y = oracle[i];
            xy += x * y;
            xx += x * x;
            yy += y * y;
            ++count;
        }
        if (count != 0)
            best = std::max(best, xy / std::sqrt(xx * yy + 1.0e-30));
    }
    return best;
}

}  // namespace

TEST_CASE("GPU NAM Forge adapter returns the product Processor", "[forge][adapter]") {
    auto processor = pulp::examples::gpu_nam_forge_adapter::create_processor();
    REQUIRE(processor != nullptr);
    CHECK(processor->descriptor().name == "GPU NAM");
    CHECK(processor->descriptor().category == pulp::format::PluginCategory::Effect);
}

TEST_CASE("GPU NAM Forge adapter installs as a runtime graph ProcessorNode",
          "[forge][adapter]") {
    pulp::host::SignalGraph graph;
    const auto node = graph.add_processor_node(
        pulp::examples::gpu_nam_forge_adapter::create_processor(), "GPU NAM");
    CHECK(graph.is_processor_node(node));
}

TEST_CASE("GPU NAM Forge adapter renders through SignalGraph with the CPU oracle",
          "[forge][adapter][audio][oracle]") {
    // The graph fixture is a production-shaped route: AudioInput -> owned
    // ProcessorNode -> AudioOutput. The adapter must therefore prepare and run
    // through the same graph executor Forge uses, rather than calling Processor
    // directly from this test.
#ifndef GPU_NAM_DEFAULT_MODEL_PATH
#error "GPU_NAM_DEFAULT_MODEL_PATH is required for the graph audio fixture"
#endif
    REQUIRE(std::filesystem::exists(GPU_NAM_DEFAULT_MODEL_PATH));
    REQUIRE(::setenv("GPU_NAM_MODEL", GPU_NAM_DEFAULT_MODEL_PATH, 1) == 0);

    auto instance = pulp::format::ProcessorNodeInstance::create(
        pulp::examples::gpu_nam_forge_adapter::create_processor());
    REQUIRE(instance);
    configure_cpu_reference(instance->state_store());

    pulp::host::SignalGraph graph;
    const auto input = graph.add_input_node(2, "Forge input");
    const auto processor = graph.add_processor_node(instance, "GPU NAM");
    const auto output = graph.add_output_node(2, "Forge output");
    REQUIRE(input != 0);
    REQUIRE(processor != 0);
    REQUIRE(output != 0);
    REQUIRE(graph.connect(input, 0, processor, 0));
    REQUIRE(graph.connect(processor, 0, output, 0));
    REQUIRE(graph.connect(input, 1, processor, 1));
    REQUIRE(graph.connect(processor, 1, output, 1));
    graph.set_canonical_executor_routing_enabled(true);
    REQUIRE(graph.prepare(kSampleRate, kBlockSize));
    CHECK(graph.prepared_stats().node_count == 3);
    // node_latency_samples(processor) is the latency arriving at the processor
    // input. The intrinsic Processor latency is visible at the downstream output
    // and graph total, where Forge's PDC planner consumes it.
    CHECK(graph.node_latency_samples(processor) == 0);
    CHECK(graph.node_latency_samples(output) == instance->processor().latency_samples());
    CHECK(graph.latency_samples() == instance->processor().latency_samples());
    // A second prepare is the lifecycle path used when Forge renegotiates its
    // block size or sample rate. It must release and rebuild the owned Processor
    // node before the next graph render.
    REQUIRE(graph.prepare(kSampleRate, kBlockSize));

    const auto frames = static_cast<std::size_t>(kBlockSize * kBlockCount);
    const auto probe = make_probe(frames);
    std::vector<float> oracle(frames, 0.0f);
    std::string error;
    pulp::examples::nam::NamRuntime reference;
    REQUIRE(pulp::examples::nam::load_nam_runtime(
        GPU_NAM_DEFAULT_MODEL_PATH, reference, &error));
    reference.prewarm();
    reference.process(probe.data(), oracle.data(), static_cast<std::uint32_t>(frames));

    // The Processor adds this fixed DC blocker after the model. Reproduce that
    // small host-facing stage independently so the graph assertion compares
    // against the NAM runtime oracle rather than another Processor instance.
    pulp::signal::DcBlocker<float> blocker;
    blocker.set_pole(static_cast<float>(1.0 -
                                        2.0 * std::acos(-1.0) * 10.0 / kSampleRate));
    for (float& sample : oracle) sample = blocker.process(sample);

    std::vector<float> output_left(frames, 0.0f);
    std::vector<float> output_right(frames, 0.0f);
    std::vector<float> input_left(kBlockSize, 0.0f);
    std::vector<float> input_right(kBlockSize, 0.0f);
    std::vector<float> output_left_block(kBlockSize, 0.0f);
    std::vector<float> output_right_block(kBlockSize, 0.0f);
    const float* input_channels[] = {input_left.data(), input_right.data()};
    float* output_channels[] = {output_left_block.data(), output_right_block.data()};
    for (int block = 0; block < kBlockCount; ++block) {
        const auto offset = static_cast<std::size_t>(block * kBlockSize);
        std::copy_n(probe.data() + offset, kBlockSize, input_left.data());
        std::copy_n(probe.data() + offset, kBlockSize, input_right.data());
        std::fill(output_left_block.begin(), output_left_block.end(), 0.0f);
        std::fill(output_right_block.begin(), output_right_block.end(), 0.0f);
        pulp::audio::BufferView<const float> input_view(input_channels, 2, kBlockSize);
        pulp::audio::BufferView<float> output_view(output_channels, 2, kBlockSize);
        graph.process(output_view, input_view, kBlockSize);
        std::copy_n(output_left_block.data(), kBlockSize, output_left.data() + offset);
        std::copy_n(output_right_block.data(), kBlockSize, output_right.data() + offset);
    }

    double energy = 0.0;
    bool finite = true;
    for (const float sample : output_left) {
        finite = finite && std::isfinite(sample);
        energy += static_cast<double>(sample) * sample;
    }
    CHECK(finite);
    REQUIRE(energy > 1.0e-5);
    CHECK(best_alignment_correlation(output_left, oracle, 4096, 128) > 0.985);
    CHECK(best_alignment_correlation(output_right, oracle, 4096, 128) > 0.985);
}

TEST_CASE("GPU request is typed as CPU fallback when provider delivery is unavailable",
          "[forge][adapter][gpu][negative]") {
#ifndef GPU_NAM_DEFAULT_MODEL_PATH
#error "GPU_NAM_DEFAULT_MODEL_PATH is required for the provider negative"
#endif
    REQUIRE(std::filesystem::exists(GPU_NAM_DEFAULT_MODEL_PATH));
    REQUIRE(::setenv("GPU_NAM_MODEL", GPU_NAM_DEFAULT_MODEL_PATH, 1) == 0);

    auto instance = pulp::format::ProcessorNodeInstance::create(
        pulp::examples::gpu_nam_forge_adapter::create_processor());
    REQUIRE(instance);
    configure_cpu_reference(instance->state_store());
    instance->state_store().set_value(pulp::examples::kEngine, 1.0f);

    pulp::host::SignalGraph graph;
    const auto input = graph.add_input_node(2, "Forge input");
    const auto processor = graph.add_processor_node(instance, "GPU NAM");
    const auto output = graph.add_output_node(2, "Forge output");
    REQUIRE(graph.connect(input, 0, processor, 0));
    REQUIRE(graph.connect(processor, 0, output, 0));
    REQUIRE(graph.connect(input, 1, processor, 1));
    REQUIRE(graph.connect(processor, 1, output, 1));
    graph.set_canonical_executor_routing_enabled(true);
    REQUIRE(graph.prepare(kSampleRate, kBlockSize));

    auto* gpu_nam = dynamic_cast<pulp::examples::GpuNamProcessor*>(&instance->processor());
    REQUIRE(gpu_nam != nullptr);
    const auto status = gpu_nam->gpu_status();
    if (!gpu_nam->gpu_engine_active()) {
        // This is the expected typed negative on hosts without a provider: a
        // requested GPU engine remains CPU-effective and exposes no delivery
        // counts. Metadata or a worker probe must not be treated as GPU audio.
        CHECK(gpu_nam->requested_engine() == 1);
        CHECK(gpu_nam->effective_engine() == 0);
        CHECK_FALSE(status.active);
        CHECK(status.delivery.gpu_selected == 0);
        CHECK(status.delivery.worker_selected == 0);
    } else {
        // A provider-capable host must expose the authenticated capability report
        // and a delivery surface; numerical GPU acceptance is a separate Forge
        // receipt because it needs the host's independent audio oracle.
        CHECK(status.capability_report_available);
        CHECK(status.capability_ready);
        CHECK(status.fallback_available);
        CHECK(status.delivery.available);
    }
}
