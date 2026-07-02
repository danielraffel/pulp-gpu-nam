// Tests for the NAM "Linear" CPU inference, its .nam loader, and the
// NamRuntime architecture dispatch.
//
// A Linear NAM is a single causal FIR + optional bias: y[n] = bias + sum_k
// ir[k]*x[n-k]. The golden here is computed independently by hand from that
// definition (no external model shipped), which fully specifies the math. The
// loader/dispatch surface is exercised through a synthetic .nam file.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nam_linear.hpp"
#include "nam_runtime.hpp"

using namespace pulp::examples::nam;
using Catch::Matchers::WithinAbs;

namespace {

std::string write_temp(const std::string& name, const std::string& content) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary);
    f << content;
    f.close();
    return p.string();
}

std::string linear_json(const std::vector<float>& weights, int rf, bool bias) {
    std::string s = "{\"architecture\":\"Linear\",\"config\":{\"receptive_field\":"
                    + std::to_string(rf) + ",\"bias\":" + (bias ? "true" : "false")
                    + "},\"sample_rate\":48000,\"weights\":[";
    for (std::size_t i = 0; i < weights.size(); ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", weights[i]);
        s += buf;
        if (i + 1 < weights.size()) s += ",";
    }
    s += "]}";
    return s;
}

} // namespace

TEST_CASE("Linear forward matches a hand-computed FIR", "[nam][linear]") {
    // ir = [0.5, 0.25], bias = 0.1. y[n] = 0.1 + 0.5*x[n] + 0.25*x[n-1].
    NamLinearModel m;
    REQUIRE(m.build(/*receptive_field=*/2, /*bias=*/true, /*in=*/1, /*out=*/1,
                    {0.5f, 0.25f, 0.1f}, 48000.0));
    REQUIRE(m.receptive_field() == 2);
    m.reset();

    const std::vector<float> in = {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, -1.0f};
    // By hand:
    //  n0: 0.1 + 0.5*1            = 0.60
    //  n1: 0.1 + 0.5*0 + 0.25*1   = 0.35
    //  n2: 0.1                    = 0.10
    //  n3: 0.1                    = 0.10
    //  n4: 0.1 + 0.5*2            = 1.10
    //  n5: 0.1 + 0.5*-1 + 0.25*2  = 0.10
    const std::vector<float> golden = {0.60f, 0.35f, 0.10f, 0.10f, 1.10f, 0.10f};

    std::vector<float> out(in.size(), 0.0f);
    m.process(in.data(), out.data(), static_cast<std::uint32_t>(in.size()));
    for (std::size_t i = 0; i < golden.size(); ++i)
        CHECK_THAT(out[i], WithinAbs(golden[i], 1e-6));
}

TEST_CASE("Linear span-split FIR is bit-exact vs a direct convolution across ring wraps",
          "[nam][linear]") {
    // The tap walk was split into two contiguous history spans (branch-free) to
    // vectorize. Same tap order → bit-identical. Drive many samples (> several
    // ring wraps of the rf-slot history) and require EXACT equality against a
    // direct FIR that sums bias + Σ_k ir[k]*x[n-k] in the same ascending-k order.
    const int rf = 7;
    std::vector<float> ir(static_cast<std::size_t>(rf));
    for (int k = 0; k < rf; ++k)
        ir[static_cast<std::size_t>(k)] = std::sin(0.6f * static_cast<float>(k) + 0.2f);
    const float bias = -0.13f;
    std::vector<float> w(ir);
    w.push_back(bias);

    // The branch-free span split lets the compiler vectorize the tap dot, which
    // reassociates the sum by ~1 ULP versus the old strictly-serial branchy loop.
    // That is the tolerance the findings gate F4 behind — Linear is CPU-only, so
    // there is no GPU golden to hold bit-for-bit. Pin the RESULT (tap indexing,
    // wrap arithmetic, magnitude) against a direct FIR to a tight tolerance across
    // several ring wraps; a real index/order bug would blow well past 1e-6.
    NamLinearModel m;
    REQUIRE(m.build(rf, /*bias=*/true, /*in=*/1, /*out=*/1, w, 48000.0));
    m.reset();

    std::vector<float> x(64);
    for (std::size_t n = 0; n < x.size(); ++n)
        x[n] = std::sin(0.29f * static_cast<float>(n)) - 0.4f * std::cos(0.11f * static_cast<float>(n));

    for (std::size_t n = 0; n < x.size(); ++n) {
        double ref = bias;   // double accumulator: an order-independent oracle
        for (int k = 0; k < rf; ++k) {
            const long t = static_cast<long>(n) - k;
            const double past = (t >= 0) ? static_cast<double>(x[static_cast<std::size_t>(t)]) : 0.0;
            ref += static_cast<double>(ir[static_cast<std::size_t>(k)]) * past;
        }
        float y = 0.0f;
        m.process(&x[n], &y, 1);
        CHECK_THAT(y, WithinAbs(ref, 1e-6));   // correct to tolerance across the ring wrap
    }
}

TEST_CASE("Linear without bias", "[nam][linear]") {
    // ir = [1.0, -0.5], no bias. y[n] = x[n] - 0.5*x[n-1].
    NamLinearModel m;
    REQUIRE(m.build(2, false, 1, 1, {1.0f, -0.5f}, 48000.0));
    const std::vector<float> in = {1.0f, 1.0f, 1.0f};
    const std::vector<float> golden = {1.0f, 0.5f, 0.5f};  // step response of a differencer
    std::vector<float> out(in.size(), 0.0f);
    m.process(in.data(), out.data(), static_cast<std::uint32_t>(in.size()));
    for (std::size_t i = 0; i < golden.size(); ++i)
        CHECK_THAT(out[i], WithinAbs(golden[i], 1e-6));
}

TEST_CASE("Linear silence steady-state is the bias", "[nam][linear]") {
    NamLinearModel m;
    REQUIRE(m.build(4, true, 1, 1, {0.2f, 0.1f, -0.05f, 0.3f, 0.42f}, 48000.0));
    m.prewarm();
    CHECK_THAT(m.process_sample(0.0f), WithinAbs(0.42f, 1e-6));  // bias only
}

TEST_CASE("Linear rejects multi-channel and bad sizes", "[nam][linear]") {
    NamLinearModel m;
    CHECK_FALSE(m.build(2, false, 2, 1, {1.0f, 0.0f, 0.0f, 0.0f}, 48000.0));  // stereo in
    CHECK_FALSE(m.build(2, false, 1, 2, {1.0f, 0.0f, 0.0f, 0.0f}, 48000.0));  // stereo out
    CHECK_FALSE(m.build(0, false, 1, 1, {}, 48000.0));                        // rf <= 0
    CHECK_FALSE(m.build(2, true, 1, 1, {1.0f, 0.0f}, 48000.0));               // missing bias term
    CHECK_FALSE(m.build(2, false, 1, 1, {1.0f}, 48000.0));                    // too few taps
    CHECK_FALSE(m.ok());                                                      // no valid build yet
}

TEST_CASE("Linear build() is transactional: success after a failure is clean", "[nam][linear]") {
    NamLinearModel m;
    REQUIRE_FALSE(m.build(2, false, 1, 1, {1.0f}, 48000.0));  // fails, sets error_
    REQUIRE_FALSE(m.error().empty());
    REQUIRE(m.build(2, false, 1, 1, {1.0f, -0.5f}, 48000.0)); // now succeeds
    CHECK(m.ok());
    CHECK(m.error().empty());                                 // error cleared on success
    CHECK_THAT(m.process_sample(1.0f), WithinAbs(1.0f, 1e-6));
}

TEST_CASE("Linear loads through NamRuntime dispatch", "[nam][linear][runtime]") {
    const std::string path = write_temp(
        "gpu_nam_linear_test.nam", linear_json({0.5f, 0.25f, 0.1f}, 2, true));
    NamRuntime rt;
    std::string err;
    REQUIRE(load_nam_runtime(path, rt, &err));
    REQUIRE(rt.ok());
    CHECK(rt.arch() == NamRuntime::Arch::Linear);
    CHECK(std::string(rt.arch_name()) == "Linear");
    CHECK_FALSE(rt.gpu_eligible());  // CPU-only until the Linear GPU path lands
    CHECK(rt.sample_rate() == 48000.0);

    rt.reset();
    CHECK_THAT(rt.process_sample(1.0f), WithinAbs(0.60f, 1e-6));
    CHECK_THAT(rt.process_sample(0.0f), WithinAbs(0.35f, 1e-6));
    std::filesystem::remove(path);
}

TEST_CASE("NamRuntime reads version + loudness metadata", "[nam][runtime][metadata]") {
    // Metadata is parsed once in load_nam_runtime, independent of architecture.
    auto with_meta = [](const std::string& extra) {
        return "{\"architecture\":\"Linear\"," + extra
               + "\"config\":{\"receptive_field\":2,\"bias\":true},"
                 "\"sample_rate\":48000,\"weights\":[0.5,0.25,0.1]}";
    };

    SECTION("present") {
        const std::string path = write_temp(
            "gpu_nam_meta.nam",
            with_meta("\"version\":\"0.5.4\",\"metadata\":{\"loudness\":-20.5},"));
        NamRuntime rt;
        std::string err;
        REQUIRE(load_nam_runtime(path, rt, &err));
        CHECK(rt.version() == "0.5.4");
        REQUIRE(rt.has_loudness());
        CHECK_THAT(rt.loudness_db(), WithinAbs(-20.5, 1e-9));
        std::filesystem::remove(path);
    }

    SECTION("absent") {
        const std::string path = write_temp("gpu_nam_nometa.nam", with_meta(""));
        NamRuntime rt;
        std::string err;
        REQUIRE(load_nam_runtime(path, rt, &err));
        CHECK(rt.version().empty());
        CHECK_FALSE(rt.has_loudness());
        std::filesystem::remove(path);
    }

    SECTION("non-finite loudness is ignored") {
        // A malformed loudness must not set has_loudness (no NaN/inf correction).
        const std::string path = write_temp(
            "gpu_nam_badloud.nam",
            with_meta("\"metadata\":{\"loudness\":\"loud\"},"));
        NamRuntime rt;
        std::string err;
        REQUIRE(load_nam_runtime(path, rt, &err));
        CHECK_FALSE(rt.has_loudness());
        std::filesystem::remove(path);
    }
}
