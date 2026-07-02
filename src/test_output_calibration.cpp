// Unit tests for the Raw/Normalized/Calibrated output-level math. Pure math,
// no GPU/render stack — validates the make-up gain a NAM capture gets in each
// output mode.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>

#include "output_calibration.hpp"

using namespace pulp::examples::nam;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr float kFixedTarget = -18.0f;   // Normalized reference (dBFS)
constexpr float kMaxAbs = 24.0f;         // clamp on the retarget
float db_to_lin(float db) { return std::pow(10.0f, db / 20.0f); }
}  // namespace

TEST_CASE("output_mode_from_param maps stepped values and clamps out-of-range", "[nam][cal]") {
    REQUIRE(output_mode_from_param(0.0f) == OutputMode::Raw);
    REQUIRE(output_mode_from_param(1.0f) == OutputMode::Normalized);
    REQUIRE(output_mode_from_param(2.0f) == OutputMode::Calibrated);
    // Rounds to the nearest step.
    REQUIRE(output_mode_from_param(0.49f) == OutputMode::Raw);
    REQUIRE(output_mode_from_param(0.51f) == OutputMode::Normalized);
    REQUIRE(output_mode_from_param(1.6f) == OutputMode::Calibrated);
    // Out of range clamps to the ends, never undefined.
    REQUIRE(output_mode_from_param(-5.0f) == OutputMode::Raw);
    REQUIRE(output_mode_from_param(99.0f) == OutputMode::Calibrated);
}

TEST_CASE("Raw is always unity regardless of metadata", "[nam][cal]") {
    REQUIRE(output_mode_gain(OutputMode::Raw, true, -9.0, kFixedTarget, -12.0f, kMaxAbs) == 1.0f);
    REQUIRE(output_mode_gain(OutputMode::Raw, false, 0.0, kFixedTarget, -12.0f, kMaxAbs) == 1.0f);
}

TEST_CASE("A model without loudness metadata gets unity in every mode", "[nam][cal]") {
    REQUIRE(output_mode_gain(OutputMode::Normalized, false, 0.0, kFixedTarget, -12.0f, kMaxAbs) == 1.0f);
    REQUIRE(output_mode_gain(OutputMode::Calibrated, false, 0.0, kFixedTarget, -12.0f, kMaxAbs) == 1.0f);
}

TEST_CASE("Normalized retargets loudness to the fixed reference", "[nam][cal]") {
    // A model measured at -9 dBFS, normalized to -18, needs -9 dB of make-up.
    const float g = output_mode_gain(OutputMode::Normalized, true, -9.0, kFixedTarget, -12.0f, kMaxAbs);
    REQUIRE_THAT(g, WithinRel(db_to_lin(-18.0f - (-9.0f)), 1e-6f));  // = db_to_lin(-9)
    // A model already at the target gets ~unity.
    const float unity = output_mode_gain(OutputMode::Normalized, true, -18.0, kFixedTarget, -12.0f, kMaxAbs);
    REQUIRE_THAT(unity, WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("Calibrated retargets to the USER reference, not the fixed one", "[nam][cal]") {
    // Same -9 dBFS model, but the user's calibration reference is -12 dBFS.
    const float g = output_mode_gain(OutputMode::Calibrated, true, -9.0, kFixedTarget, -12.0f, kMaxAbs);
    REQUIRE_THAT(g, WithinRel(db_to_lin(-12.0f - (-9.0f)), 1e-6f));  // = db_to_lin(-3)
    // Moving the user reference changes the gain; the fixed target is ignored.
    const float hotter = output_mode_gain(OutputMode::Calibrated, true, -9.0, kFixedTarget, -6.0f, kMaxAbs);
    REQUIRE_THAT(hotter, WithinRel(db_to_lin(-6.0f - (-9.0f)), 1e-6f));  // = db_to_lin(+3)
    REQUIRE(hotter > g);  // a hotter reference is more gain
}

TEST_CASE("Retarget is clamped so an extreme loudness can't apply a wild gain", "[nam][cal]") {
    // A model that claims -60 dBFS would need +42 dB to hit -18 — clamp to +24.
    const float g = output_mode_gain(OutputMode::Normalized, true, -60.0, kFixedTarget, -12.0f, kMaxAbs);
    REQUIRE_THAT(g, WithinRel(db_to_lin(kMaxAbs), 1e-6f));
    // And a very hot model clamps the cut symmetrically.
    const float cut = output_mode_gain(OutputMode::Normalized, true, +30.0, kFixedTarget, -12.0f, kMaxAbs);
    REQUIRE_THAT(cut, WithinRel(db_to_lin(-kMaxAbs), 1e-6f));
}

TEST_CASE("Non-finite metadata or target degrades to unity, never NaN/Inf", "[nam][cal]") {
    const float nan = std::nan("");
    const float inf = std::numeric_limits<float>::infinity();
    REQUIRE(loudness_retarget_gain(true, nan, kFixedTarget, kMaxAbs) == 1.0f);
    REQUIRE(loudness_retarget_gain(true, -9.0, inf, kMaxAbs) == 1.0f);
    REQUIRE(loudness_retarget_gain(true, -9.0, kFixedTarget, nan) == 1.0f);
    // A finite result is always finite.
    REQUIRE(std::isfinite(output_mode_gain(OutputMode::Calibrated, true, -9.0, kFixedTarget, -12.0f, kMaxAbs)));
}

TEST_CASE("Old binary Normalize state maps to the new 3-way meaning", "[nam][cal]") {
    // The parameter widened from {0=off,1=on} to {0=Raw,1=Normalized,2=Calibrated}.
    // A saved session's 0 must still mean "no retarget" and 1 "normalize".
    REQUIRE(output_mode_from_param(0.0f) == OutputMode::Raw);          // was "off"
    REQUIRE(output_mode_gain(output_mode_from_param(0.0f), true, -9.0, kFixedTarget, -12.0f, kMaxAbs) == 1.0f);
    REQUIRE(output_mode_from_param(1.0f) == OutputMode::Normalized);   // was "on"
}
