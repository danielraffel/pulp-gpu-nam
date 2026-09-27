#pragma once
#include "gpu_nam_paced_options.hpp"
#include <algorithm>
#include <cstdio>

namespace pulp::examples {
// Filled only by the independent oracle replay after the timed callback loop.
struct PacedErrorSummary {
    std::uint64_t mismatches = 0, nonfinite_mismatches = 0;
    double max_finite_abs_error = 0;
    unsigned first_channel = 0, first_frame = 0;
    float first_actual = 0, first_expected = 0;

    void compare(float actual, float expected, unsigned channel, unsigned frame) noexcept {
        const bool finite = std::isfinite(actual) && std::isfinite(expected);
        if (finite) max_finite_abs_error = std::max(max_finite_abs_error, std::abs(double(actual)-expected));
        if (paced_sample_matches(actual, expected)) return;
        if (mismatches == 0) {
            first_channel=channel; first_frame=frame;
            first_actual=actual; first_expected=expected;
        }
        ++mismatches;
        nonfinite_mismatches += !finite;
    }
    const char* first_kind() const noexcept {
        if (!mismatches) return "none";
        if (!std::isfinite(first_actual) && !std::isfinite(first_expected)) return "both_nonfinite";
        if (!std::isfinite(first_actual)) return "actual_nonfinite";
        if (!std::isfinite(first_expected)) return "expected_nonfinite";
        return "finite";
    }
    void write_csv_fields(FILE* file) const {
        std::fprintf(file,",%llu,%llu,%.17g,",(unsigned long long)mismatches,
                     (unsigned long long)nonfinite_mismatches,max_finite_abs_error);
        if (mismatches) std::fprintf(file,"%u,%u,%.9g,%.9g,%s",first_channel,first_frame,
                                    double(first_actual),double(first_expected),first_kind());
        else std::fprintf(file,",,,,none");
    }
};
} // namespace pulp::examples
