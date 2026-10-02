#pragma once
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

namespace pulp::examples {
struct GpuNamPacedOptions {
    bool enabled = false, cpu_only = false, inject_error = false;
    bool staged_gpu = false, force_fallback = false, inject_forward_failure = false;
    bool realtime_worker = false, audio_work_interval = false;
    bool trace = false;
    std::uint32_t seconds = 10, input_blocks = 0;
    // Transport capacity and logical worker concurrency are independent knobs.
    // Keep the defaults unchanged for existing validation invocations.
    std::uint32_t capacity = 16, max_inflight = 1;
    bool seconds_explicit = false;
    std::string sidecar;
    bool parse(std::string_view arg) {
        if (arg == "--staged-gpu") { staged_gpu = true; return true; }
        if (arg == "--force-fallback") { force_fallback = true; return true; }
        if (arg == "--inject-forward-failure") { inject_forward_failure = true; return true; }
        if (arg == "--trace") { trace = true; return true; }
        if (arg == "--realtime-worker") { realtime_worker = true; return true; }
        if (arg == "--audio-work-interval") { audio_work_interval = true; return true; }
        if (arg == "--paced") { enabled = true; return true; }
        if (arg == "--cpu-baseline") { cpu_only = true; return true; }
        if (arg.starts_with("--sidecar=")) { sidecar = arg.substr(10); return !sidecar.empty(); }
        auto number = [&](std::string_view prefix, std::uint32_t& destination) {
            if (!arg.starts_with(prefix)) return false;
            const auto text = arg.substr(prefix.size());
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), destination);
            return error == std::errc{} && end == text.data() + text.size() && destination != 0;
        };
        if (arg.starts_with("--duration-seconds=")) {
            seconds_explicit = true;
            return number("--duration-seconds=", seconds);
        }
        if (arg.starts_with("--capacity=")) {
            if (!number("--capacity=", capacity)) return false;
            return capacity == 2 || capacity == 4 || capacity == 8;
        }
        if (arg.starts_with("--max-inflight=")) {
            if (!number("--max-inflight=", max_inflight)) return false;
            return max_inflight == 1 || max_inflight == 2 ||
                   max_inflight == 4 || max_inflight == 8;
        }
        return number("--blocks=", input_blocks);
    }
    std::uint64_t blocks(std::uint32_t frames) const noexcept {
        return input_blocks ? input_blocks : (std::uint64_t(seconds) * 48000 + frames - 1) / frames;
    }
    bool valid(std::uint32_t frames, std::uint32_t lead) const noexcept {
        if ((cpu_only && (staged_gpu || force_fallback || inject_forward_failure)) ||
            (inject_forward_failure && (!staged_gpu || force_fallback))) return false;
        if (capacity <= lead || capacity > 64 || max_inflight == 0 || max_inflight > capacity - lead)
            return false;
        if (!enabled) return !staged_gpu && !force_fallback && !inject_forward_failure && !cpu_only && sidecar.empty() && !seconds_explicit && input_blocks == 0;
        if (sidecar.empty() || (seconds_explicit && input_blocks) || frames == 0) return false;
        const auto count = blocks(frames);
        // Two stereo float captures (input and output), capped before allocation.
        return count > 0 && count <= 1'000'000 &&
               (count + lead) * frames * 2 * sizeof(float) * 2 <= (1ULL << 30);
    }
};
inline std::uint64_t paced_offset_ns(std::uint64_t block, std::uint32_t frames) noexcept {
    return block * frames * 1'000'000'000ULL / 48000;
}
inline bool paced_sample_matches(float actual, float expected) noexcept {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(double(actual) - expected) <= 1e-4;
}
} // namespace pulp::examples
