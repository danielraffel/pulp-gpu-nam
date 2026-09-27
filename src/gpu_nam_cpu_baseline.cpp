#include "nam_model.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr unsigned input_blocks = 96;
struct Config {
    std::string model = GPU_NAM_MODEL_PATH;
    unsigned block = 32, lead = 1, verify_delay = 0;
    bool explicit_verify_delay = false;
};
bool number(std::string_view text, unsigned& value) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc{} && end == text.data() + text.size();
}
bool parse(int argc, char** argv, Config& c) {
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        const auto separator = arg.find('=');
        if (separator == std::string_view::npos) return false;
        const auto key = arg.substr(0, separator), value = arg.substr(separator + 1);
        if (key == "--model-path") {
            if (value.empty()) return false;
            c.model = value;
        } else if (key == "--block-size") {
            if (!number(value, c.block)) return false;
        } else if (key == "--lead-blocks") {
            if (!number(value, c.lead)) return false;
        } else if (key == "--verify-delay-blocks") {
            // Independent oracle policy; useful for a deliberately wrong-delay control.
            if (!number(value, c.verify_delay) || c.verify_delay > 8) return false;
            c.explicit_verify_delay = true;
        } else return false;
    }
    if (c.block != 32 && c.block != 64 && c.block != 128) return false;
    if (c.lead != 1 && c.lead != 2 && c.lead != 4 && c.lead != 8) return false;
    if (!c.explicit_verify_delay) c.verify_delay = c.lead;
    return true;
}
}

int main(int argc, char** argv) {
    Config c;
    if (!parse(argc, argv, c)) {
        std::cerr << "invalid arguments: --model-path=PATH --block-size=32|64|128 "
                     "--lead-blocks=1|2|4|8 [--verify-delay-blocks=0..8]\n";
        return 2;
    }
    pulp::examples::nam::NamModel model, reference;
    std::string error;
    if (!pulp::examples::nam::load_nam(c.model, model, &error) ||
        !pulp::examples::nam::load_nam(c.model, reference, &error)) {
        std::cerr << "load_error=" << error << '\n';
        return 2;
    }
    model.prewarm_block_aligned(c.block);
    reference.prewarm_block_aligned(c.block);
    const auto blocks = input_blocks + c.lead;
    const auto count = static_cast<std::size_t>(blocks) * c.block;
    std::vector<float> input(count, 0), raw_reference(count), actual(count);
    std::vector<float> processed(c.block), ring(static_cast<std::size_t>(c.lead) * c.block, 0);
    for (unsigned i = 0; i < input_blocks * c.block; ++i) {
        const auto sample = static_cast<float>(i);
        input[i] = 0.07f * std::sin(0.013f * sample) + 0.02f * std::cos(0.037f * sample);
    }
    // Generate the oracle before timing. Validation and stimulus generation are
    // excluded from the measured baseline. Extra zero blocks drain the lead ring.
    for (unsigned b = 0; b < blocks; ++b)
        reference.process(input.data() + b * c.block, raw_reference.data() + b * c.block, c.block);

    const auto cpu_start = std::clock();
    if (cpu_start == std::clock_t(-1)) {
        std::cerr << "process CPU clock unavailable\n";
        return 4;
    }
    const auto wall_start = Clock::now();
    std::uint64_t model_elapsed_ns = 0;
    unsigned ring_index = 0;
    for (unsigned b = 0; b < blocks; ++b) {
        const auto start = Clock::now();
        model.process(input.data() + b * c.block, processed.data(), c.block);
        model_elapsed_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
        auto* slot = ring.data() + static_cast<std::size_t>(ring_index) * c.block;
        for (unsigned i = 0; i < c.block; ++i) {
            actual[b * c.block + i] = slot[i];
            slot[i] = processed[i];
        }
        ring_index = (ring_index + 1) % c.lead;
    }
    const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - wall_start).count();
    const auto cpu_end = std::clock();
    if (cpu_end == std::clock_t(-1) || cpu_end < cpu_start) return 4;
    bool finite = true;
    double max_error = 0;
    const auto delay = static_cast<std::size_t>(c.verify_delay) * c.block;
    for (std::size_t i = 0; i < count; ++i) {
        const auto expected = i < delay ? 0.0f : raw_reference[i - delay];
        finite &= std::isfinite(actual[i]) && std::isfinite(raw_reference[i]);
        if (std::isfinite(actual[i]) && std::isfinite(expected))
            max_error = std::max(max_error, std::abs(double(actual[i]) - expected));
    }
    for (float sample : ring) finite &= std::isfinite(sample);
    const bool passed = finite && max_error <= 1e-6;
    std::cout << "diagnostic_status=" << (passed ? "passed" : "failed")
              << " block_size=" << c.block << " lead_blocks=" << c.lead
              << " input_blocks=" << input_blocks << " measured_blocks=" << blocks
              << " measured_samples=" << count << " elapsed_process_ns=" << model_elapsed_ns
              << " process_cpu_ticks=" << (cpu_end - cpu_start)
              << " process_cpu_ticks_per_second=" << CLOCKS_PER_SEC
              << " process_cpu_seconds=" << double(cpu_end - cpu_start) / CLOCKS_PER_SEC
              << " wall_ns=" << wall_ns << " output_finite=" << finite
              << " max_error=" << max_error << '\n';
    return passed ? 0 : 3;
}
