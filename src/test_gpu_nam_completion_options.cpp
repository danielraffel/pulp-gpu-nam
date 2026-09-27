#include "gpu_nam_completion_options.hpp"
#include <string>
#include <iostream>

int main() {
    using namespace pulp::examples;
    using P = pulp::gpu_audio::GpuWaveNetCompletionPolicy;
    unsigned checks = 0;
    const auto check = [&](bool condition) { ++checks; return condition; };
    GpuNamCompletionOptions defaults;
    if (!check(defaults.valid() && defaults.policy == P::ProcessEvents && defaults.worker_wait_ns == 0)) return 1;
    for (const auto policy : {P::ProcessEvents, P::WaitAny, P::TimedWaitAny}) {
        for (const auto wait : {0ULL, 1ULL, 1'000'000ULL, 1'000'001ULL}) {
            const GpuNamCompletionOptions options{policy, wait};
            const bool expected = wait <= 1'000'000 && (wait == 0 || policy == P::TimedWaitAny);
            if (!check(options.valid() == expected)) return 2;
        }
    }
    if (!check(!GpuNamCompletionOptions{static_cast<P>(99), 0}.valid())) return 3;
    for (const auto value : {"", "--worker-wait-ns=", "--worker-wait-ns=-1", "--worker-wait-ns=1x",
                             "--worker-wait-ns=18446744073709551616", "--completion-policy=unknown"}) {
        GpuNamCompletionOptions options;
        if (!check(!parse_completion_option(value, options))) return 4;
    }
    for (const auto policy : {"process-events", "wait-any", "timed-wait-any"}) {
        GpuNamCompletionOptions options;
        if (!check(parse_completion_option(std::string("--completion-policy=") + policy, options) &&
                   std::string_view(completion_policy_name(options.policy)) == policy && options.valid())) return 5;
    }
    GpuNamCompletionOptions wait_first;
    if (!check(parse_completion_option("--worker-wait-ns=1000000", wait_first) &&
               parse_completion_option("--completion-policy=timed-wait-any", wait_first) && wait_first.valid())) return 6;
    std::cout << checks << " completion option checks passed\n";
}
