#pragma once
#include <pulp/gpu_audio/gpu_wavenet.hpp>
#include <charconv>
#include <cstdint>
#include <string_view>

namespace pulp::examples {
struct GpuNamCompletionOptions {
    gpu_audio::GpuWaveNetCompletionPolicy policy = gpu_audio::GpuWaveNetCompletionPolicy::ProcessEvents;
    std::uint64_t worker_wait_ns = 0;
    bool valid() const noexcept {
        using P = gpu_audio::GpuWaveNetCompletionPolicy;
        return (policy == P::ProcessEvents || policy == P::WaitAny || policy == P::TimedWaitAny) &&
               worker_wait_ns <= 1'000'000 &&
               (worker_wait_ns == 0 || policy == P::TimedWaitAny);
    }
};
inline const char* completion_policy_name(gpu_audio::GpuWaveNetCompletionPolicy policy) noexcept {
    using P = gpu_audio::GpuWaveNetCompletionPolicy;
    switch (policy) {
    case P::ProcessEvents: return "process-events";
    case P::WaitAny: return "wait-any";
    case P::TimedWaitAny: return "timed-wait-any";
    }
    return "invalid";
}
// Parse only experiment switches. Call valid() after all options so flag order
// cannot silently alter which configurations are accepted.
inline bool parse_completion_option(std::string_view arg, GpuNamCompletionOptions& options) noexcept {
    constexpr std::string_view policy_prefix = "--completion-policy=";
    constexpr std::string_view wait_prefix = "--worker-wait-ns=";
    using P = gpu_audio::GpuWaveNetCompletionPolicy;
    if (arg.starts_with(policy_prefix)) {
        const auto value = arg.substr(policy_prefix.size());
        if (value == "process-events") options.policy = P::ProcessEvents;
        else if (value == "wait-any") options.policy = P::WaitAny;
        else if (value == "timed-wait-any") options.policy = P::TimedWaitAny;
        else return false;
        return true;
    }
    if (arg.starts_with(wait_prefix)) {
        const auto value = arg.substr(wait_prefix.size());
        std::uint64_t wait = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), wait);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) return false;
        options.worker_wait_ns = wait;
        return true;
    }
    return false;
}
} // namespace pulp::examples
