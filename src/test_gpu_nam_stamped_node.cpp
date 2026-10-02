#include "gpu_nam_stamped_node.hpp"
#include "gpu_nam_stamped_paced.hpp"
#include <pulp/gpu_audio/gpu_audio_transport.hpp>
#include <pulp/runtime/trace_session.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    pulp::examples::GpuNamCompletionOptions completion;
    bool inject_output_error = false;
    pulp::examples::GpuNamPacedOptions paced;
    std::vector<std::string_view> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--inject-output-error") inject_output_error = true;
        else if (arg.starts_with("--")) {
            if (!paced.parse(arg) && !pulp::examples::parse_completion_option(arg, completion)) return 64;
        } else positional.push_back(arg);
    }
    if (!completion.valid() || positional.size() > 3) {
        std::cerr << "invalid completion configuration: worker wait must be 0..1000000ns; "
                     "positive wait requires timed-wait-any\n";
        return 64;
    }
    unsigned lead = 2;
    unsigned frames = 32;
    if (!positional.empty()) {
        const auto arg = positional[0];
        const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), lead);
        if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() ||
            (lead != 1 && lead != 2 && lead != 4 && lead != 8)) return 64;
    }
    if (positional.size() >= 2) {
        const auto arg = positional[1];
        const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), frames);
        if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() ||
            (frames != 32 && frames != 64 && frames != 128 && frames != 512)) return 64;
    }
    if (!paced.valid(frames, lead) || ((paced.cpu_only || paced.staged_gpu) &&
        (completion.policy != pulp::gpu_audio::GpuWaveNetCompletionPolicy::ProcessEvents ||
         completion.worker_wait_ns != 0))) return 64;
    paced.inject_error = inject_output_error;
    using namespace pulp;
    constexpr unsigned channels = 2, blocks = 48;
    examples::nam::NamModel model;
    std::string error;
    if (!examples::nam::load_nam(positional.size() == 3 ? std::string(positional[2]) : GPU_NAM_MODEL_PATH, model, &error)) return 1;
    if (paced.enabled) {
        // Plugin/host adapters own this lifecycle. The standalone validator
        // must start and stop it explicitly so --trace flushes a capture.
        const bool trace_started = paced.trace && pulp::runtime::Tracing::start();
        if (paced.trace && !trace_started) return 8;
        const int result = examples::run_stamped_paced(model, frames, lead, completion, paced);
        if (paced.trace && !pulp::runtime::Tracing::stop().ok) return 8;
        return result;
    }
    if (examples::GpuNamStampedNode::create(model, 65, frames, 48000, lead) ||
        examples::GpuNamStampedNode::create(model, channels, frames, 48000, 0)) return 2;
    auto node = examples::GpuNamStampedNode::create(model, channels, frames, 48000, lead, completion);
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
        if (inject_output_error && block == lead) output[0][0] += .25f;
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
    std::cout << "completion_policy=" << examples::completion_policy_name(completion.policy)
              << " worker_wait_ns=" << completion.worker_wait_ns << "\n";
    std::cout << "frames=" << frames << " lead=" << lead << " max_error=" << max_error
              << " cpu_model_calls=" << calls << " fallback_reads=" << misses
              << " gpu_callbacks=" << gpu_callbacks
              << " worker_produced=" << stats.produced_blocks << '\n';
    const bool passed = max_error <= 1e-4 && calls == (blocks + lead) * channels &&
                        misses > 0 && gpu_callbacks > 0;
    std::cout << "diagnostic_status=" << (passed ? "passed" : "failed") << '\n';
    return passed ? 0 : 7;
}
