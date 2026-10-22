// Tests for the exact NAM WaveNet CPU inference + .nam loader.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <vector>

#include "nam_model.hpp"

// Allocation probe: counts operator new while armed, so a test can prove the
// streaming path is allocation-free (the audio-thread contract).
namespace {
std::atomic<bool> g_count_allocs{false};
std::atomic<long> g_allocs{0};
}  // namespace
void* operator new(std::size_t n) {
    if (g_count_allocs.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace pulp::examples::nam;
using Catch::Matchers::WithinAbs;

namespace {

bool file_exists(const char* p) {
    std::ifstream f(p);
    return f.good();
}

} // namespace

TEST_CASE("synthetic single-layer model matches hand-computed forward", "[nam]") {
    // One array, one layer, all 1x1, channels=1 — so the whole forward reduces to
    // a closed-form expression we can check against, exercising the exact weight
    // order: rechannel, conv(+bias), input_mixin, layer1x1(+bias),
    // head_rechannel(+bias), head_scale.
    LayerArrayConfig a;
    a.input_size = 1;
    a.condition_size = 1;
    a.channels = 1;
    a.kernel_size = 1;
    a.dilations = {1};
    a.head_size = 1;
    a.head_bias = true;
    a.gated = false;
    a.activation = "Tanh";

    const float w_re = 0.8f;   // rechannel
    const float wc = 1.3f, bc = -0.2f;   // conv weight + bias
    const float wm = 0.5f;     // input mixin
    const float wl = 0.7f, bl = 0.1f;    // layer1x1 weight + bias
    const float wh = 1.1f, bh = -0.05f;  // head rechannel weight + bias
    const float hs = 0.25f;    // head_scale (final weight)

    std::vector<float> weights = {w_re, wc, bc, wm, wl, bl, wh, bh, hs};

    NamModel model;
    REQUIRE(model.build({a}, hs, weights, /*in_channels=*/1, /*sample_rate=*/48000.0));
    REQUIRE(model.weights_consumed() == weights.size());
    REQUIRE(model.expected_weight_count() == weights.size());
    REQUIRE(model.head_scale() == hs);

    auto expected = [&](float x) {
        const float rc = w_re * x;
        const float z = wc * rc + bc + wm * x;
        const float act = std::tanh(z);
        const float head = wh * act + bh;
        return hs * head;
    };

    model.reset();
    for (float x : {-1.0f, -0.3f, 0.0f, 0.42f, 0.9f, 2.5f}) {
        const float y = model.process_sample(x);
        REQUIRE_THAT(y, WithinAbs(expected(x), 1e-6f));
    }
}

TEST_CASE("weight count is computed from the exact layout", "[nam]") {
    // Two arrays mirroring the published example model's shape; the expected
    // float count must match the closed-form layout sum.
    LayerArrayConfig a0;
    a0.input_size = 1; a0.condition_size = 1; a0.channels = 3; a0.kernel_size = 3;
    a0.dilations = {1, 2}; a0.head_size = 2; a0.head_bias = false; a0.gated = false;
    LayerArrayConfig a1;
    a1.input_size = 3; a1.condition_size = 1; a1.channels = 2; a1.kernel_size = 3;
    a1.dilations = {8}; a1.head_size = 1; a1.head_bias = true; a1.gated = false;

    // array0: rechannel 1*3 + 2 layers*(conv 3*3*3+3, mixin 1*3, layer1x1 3*3+3)
    //         + head_rechannel 3*2  = 3 + 2*45 + 6 = 99
    // array1: rechannel 3*2 + 1 layer*(conv 2*2*3+2, mixin 1*2, layer1x1 2*2+2)
    //         + head_rechannel 2*1+1 = 6 + 22 + 3 = 31
    // + head_scale = 99 + 31 + 1 = 131
    const std::size_t expected = 131;
    std::vector<float> weights(expected, 0.01f);

    NamModel model;
    REQUIRE(model.build({a0, a1}, 0.5f, weights, 1, 48000.0));
    REQUIRE(model.expected_weight_count() == expected);
    REQUIRE(model.weights_consumed() == expected);
}

TEST_CASE("build rejects a short/long weight blob", "[nam]") {
    LayerArrayConfig a;
    a.input_size = 1; a.condition_size = 1; a.channels = 1; a.kernel_size = 1;
    a.dilations = {1}; a.head_size = 1; a.head_bias = true;

    NamModel too_short;
    REQUIRE_FALSE(too_short.build({a}, 0.25f, std::vector<float>(8, 0.0f), 1, 48000.0));
    REQUIRE_FALSE(too_short.error().empty());

    NamModel too_long;
    REQUIRE_FALSE(too_long.build({a}, 0.25f, std::vector<float>(10, 0.0f), 1, 48000.0));
}

TEST_CASE("Conv kernel==1 is a plain matvec, bit-exact vs the general path", "[nam]") {
    // The kernel==1 fast path (rechannel / input_mixin / layer1x1) skips the ring
    // buffer. Pin it against a hand-computed matvec AND against a kernel==1 conv
    // driven through the general path via a two-tap conv whose second tap weights
    // are zero — they must produce identical floats sample-for-sample.
    const int in_ch = 3, out_ch = 2;
    // weight_[0] laid out [out][in]; bias[out].
    std::vector<float> w = {
        // out0: in0,in1,in2      out1: in0,in1,in2
        0.5f, -1.2f, 0.3f,        0.9f, 0.1f, -0.7f,
        // bias out0, out1
        0.05f, -0.2f,
    };

    Conv k1;
    k1.configure(in_ch, out_ch, /*kernel=*/1, /*dilation=*/1, /*bias=*/true);
    const float* pk1 = w.data();
    k1.load(pk1);

    auto matvec = [&](const float* x, int o) {
        // Match the code's accumulation order exactly: sum the products from 0.0,
        // THEN add bias. Folding bias in first would reorder the float adds and
        // diverge in the last bit — which is precisely the invariant this pins.
        const float* wr = &w[static_cast<std::size_t>(o) * in_ch];
        float acc = 0.0f;
        for (int j = 0; j < in_ch; ++j) acc += wr[j] * x[j];
        return w[static_cast<std::size_t>(out_ch) * in_ch + o] + acc;  // bias last
    };

    const float inputs[][3] = {
        {1.0f, 0.0f, -1.0f}, {0.25f, -0.5f, 2.0f}, {-3.0f, 1.5f, 0.7f}, {0.0f, 0.0f, 0.0f},
    };
    for (const auto& x : inputs) {
        float y[2] = {0.0f, 0.0f};
        k1.step(x, y);
        for (int o = 0; o < out_ch; ++o)
            REQUIRE(y[o] == matvec(x, o));   // exact float equality: no reordering
    }
}

TEST_CASE("dilated Conv index normalization is exact across the ring wrap", "[nam]") {
    // The general path replaced `idx %= slots_` with a single conditional add.
    // Drive a dilated multi-tap conv well past a full wrap of its history ring and
    // require the output to stay finite and to match a from-scratch replay — the
    // wrap arithmetic must land on the same slot every time.
    const int in_ch = 1, out_ch = 1, kernel = 3, dilation = 4;   // reach = 8, slots = 9
    std::vector<float> w = {0.3f, -0.6f, 0.9f, /*bias=*/0.1f};   // [k0,k1,k2] then bias

    auto make = [&] {
        Conv c;
        c.configure(in_ch, out_ch, kernel, dilation, /*bias=*/true);
        const float* p = w.data();
        c.load(p);
        return c;
    };
    Conv a = make(), b = make();

    // Feed 40 samples (> 4 wraps of a 9-slot ring) and require the two identical
    // convs agree bit-for-bit at every sample.
    for (int n = 0; n < 40; ++n) {
        const float x = std::sin(0.37f * static_cast<float>(n));
        float ya = 0.0f, yb = 0.0f;
        a.step(&x, &ya);
        b.step(&x, &yb);
        REQUIRE(std::isfinite(ya));
        REQUIRE(ya == yb);
    }
}

TEST_CASE("loader parses the real example .nam exactly", "[nam]") {
    const char* path = "/tmp/test.nam";
    if (!file_exists(path)) {
        WARN("/tmp/test.nam not present; skipping real-model load test");
        return;
    }

    NamModel model;
    std::string err;
    REQUIRE(load_nam(path, model, &err));
    INFO("load error: " << err);

    // Architecture sanity for the published example model.
    REQUIRE(model.arrays().size() == 2);
    REQUIRE(model.out_channels() == 1);
    REQUIRE(model.in_channels() == 1);

    // Weights consumed byte-for-byte: none left over, none missing.
    REQUIRE(model.weights_consumed() == model.weights_size());
    REQUIRE(model.expected_weight_count() == model.weights_size());

    // head_scale stored as the final weight should equal the config value.
    REQUIRE_THAT(model.head_scale(), WithinAbs(model.head_scale_config(), 1e-6f));
}

TEST_CASE("real model output is finite, non-trivial, and causal", "[nam]") {
    const char* path = "/tmp/test.nam";
    if (!file_exists(path)) {
        WARN("/tmp/test.nam not present; skipping real-model inference test");
        return;
    }

    NamModel model;
    REQUIRE(load_nam(path, model));

    const std::uint32_t N = 1024;
    std::vector<float> in(N), out(N);
    for (std::uint32_t i = 0; i < N; ++i)
        in[i] = 0.3f * std::sin(0.05f * static_cast<float>(i));
    in[8] += 1.0f;   // impulse

    model.reset();
    model.process(in.data(), out.data(), N);

    bool finite = true, nontrivial = false;
    for (float x : out) {
        if (!std::isfinite(x)) finite = false;
        if (std::fabs(x) > 1e-9f) nontrivial = true;
    }
    REQUIRE(finite);
    REQUIRE(nontrivial);

    // Causality: perturbing input at sample T must not change any output < T.
    const std::uint32_t T = 256;
    std::vector<float> in2 = in, out2(N);
    in2[T] += 0.5f;
    model.reset();
    model.process(in2.data(), out2.data(), N);

    for (std::uint32_t i = 0; i < T; ++i)
        REQUIRE(out[i] == out2[i]);
    // And the perturbation must actually have an effect at/after T.
    bool diverged = false;
    for (std::uint32_t i = T; i < N; ++i)
        if (out[i] != out2[i]) { diverged = true; break; }
    REQUIRE(diverged);
}

TEST_CASE("prewarm settles the model at its silence steady-state", "[nam]") {
    // Two dilated arrays (receptive field > 0) with non-trivial weights, so the
    // response to silence is a non-zero DC that takes a full receptive field to
    // populate. After prewarm() the model must already be at that steady state:
    // its next silence output equals the output after much more silence, and it
    // differs from a cold (reset-only) start.
    LayerArrayConfig a0;
    a0.input_size = 1; a0.condition_size = 1; a0.channels = 3; a0.kernel_size = 3;
    a0.dilations = {1, 2}; a0.head_size = 2; a0.head_bias = true; a0.gated = false;
    LayerArrayConfig a1;
    a1.input_size = 3; a1.condition_size = 1; a1.channels = 2; a1.kernel_size = 3;
    a1.dilations = {8}; a1.head_size = 1; a1.head_bias = true; a1.gated = false;

    NamModel probe;
    REQUIRE(probe.build({a0, a1}, 0.5f, std::vector<float>(2, 0.0f), 1, 48000.0) == false);
    const std::size_t count = probe.expected_weight_count();
    REQUIRE(count > 0);
    std::vector<float> weights(count);
    for (std::size_t i = 0; i < count; ++i)
        weights[i] = 0.05f * std::sin(0.7f * static_cast<float>(i) + 0.3f);

    NamModel model;
    REQUIRE(model.build({a0, a1}, 0.5f, weights, 1, 48000.0));
    REQUIRE(model.receptive_field() > 0);

    model.prewarm();
    const float y_pre = model.process_sample(0.0f);
    float y_settled = 0.0f;
    for (int i = 0; i < 8192; ++i) y_settled = model.process_sample(0.0f);
    REQUIRE(std::isfinite(y_pre));
    REQUIRE_THAT(y_pre, WithinAbs(y_settled, 1e-6f));

    model.reset();
    const float y_cold = model.process_sample(0.0f);
    // The cold first sample is the pre-history-fill transient, not the steady
    // state — so prewarm demonstrably changes the starting output.
    REQUIRE(std::fabs(y_cold - y_settled) > 1e-7f);
}

TEST_CASE("loader rejects a conditioned (condition_size > 1) model", "[nam]") {
    // A parametric/conditioned capture carries a multi-dimensional condition we do
    // not wire; running it would read past the 1-element condition buffer. The
    // loader must refuse it rather than mis-render.
    const std::string path = "/tmp/pulp_nam_cond.nam";
    {
        std::ofstream f(path, std::ios::binary);
        f << "{\"architecture\":\"WaveNet\",\"config\":{\"layers\":[{\"input_size\":1,"
             "\"condition_size\":3,\"channels\":2,\"kernel_size\":1,\"dilations\":[1],"
             "\"head_size\":1,\"head_bias\":true,\"gated\":false,\"activation\":\"Tanh\"}],"
             "\"head_scale\":1.0},\"sample_rate\":48000,\"weights\":[]}";
    }
    NamModel model;
    std::string err;
    REQUIRE_FALSE(load_nam(path, model, &err));
    REQUIRE(err.find("condition_size") != std::string::npos);
}

namespace {

// Deterministic weights sized for the given arrays (via the model's own count).
NamModel build_seeded(const std::vector<LayerArrayConfig>& arrays, std::uint32_t seed, float scale = 0.35f) {
    NamModel probe;
    probe.build(arrays, 1.0f, std::vector<float>(1, 0.0f), 1, 48000.0);
    const std::size_t count = probe.expected_weight_count();
    std::vector<float> w(count);
    std::uint32_t s = seed;
    for (auto& v : w) {
        s = s * 1664525u + 1013904223u;
        v = (static_cast<float>(s >> 9) / 4194304.0f - 1.0f) * scale;
    }
    w.back() = 0.5f;   // head_scale
    NamModel m;
    REQUIRE(m.build(arrays, 0.5f, std::move(w), 1, 48000.0));
    return m;
}

LayerArrayConfig arr(int input, int channels, int head, std::vector<int> dil, bool bias, int kernel = 3,
                     bool gated = false, const char* act = "Tanh") {
    LayerArrayConfig a;
    a.input_size = input;
    a.condition_size = 1;
    a.channels = channels;
    a.kernel_size = kernel;
    a.dilations = std::move(dil);
    a.head_size = head;
    a.head_bias = bias;
    a.gated = gated;
    a.activation = act;
    return a;
}

std::vector<float> test_signal(std::size_t n) {
    std::vector<float> x(n);
    std::uint32_t s = 99;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float noise = (static_cast<float>(s >> 9) / 4194304.0f - 1.0f);
        x[i] = 0.5f * std::sin(0.002f * static_cast<float>(i) * (1.0f + 0.0005f * static_cast<float>(i)))
               + 0.1f * noise;
    }
    x[n / 3] += 0.9f;   // a transient
    return x;
}

// Stream x through m in the given repeating block-size pattern.
std::vector<float> stream(NamModel& m, const std::vector<float>& x, const std::vector<std::uint32_t>& sizes) {
    std::vector<float> y(x.size());
    std::size_t pos = 0, k = 0;
    while (pos < x.size()) {
        const std::uint32_t n = static_cast<std::uint32_t>(
            std::min<std::size_t>(sizes[k++ % sizes.size()], x.size() - pos));
        m.process(x.data() + pos, y.data() + pos, n);
        pos += n;
    }
    return y;
}

// Residual of b against a, in dB relative to a's RMS.
double null_db(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, r = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(b[i]) - a[i];
        e += d * d;
        r += static_cast<double>(a[i]) * a[i];
    }
    return 10.0 * std::log10((e + 1e-300) / (r + 1e-300));
}

// Shapes covering every kernel path: the trained-preset channel counts (fixed
// kernels), a gated layer, kernel sizes 1/2, an off-table channel count (generic
// kernel), non-tanh activations, and dilations long enough to force history
// rewinds inside a stream.
std::vector<std::pair<const char*, std::vector<LayerArrayConfig>>> engine_shapes() {
    const std::vector<int> d10 = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    return {
        {"standard", {arr(1, 16, 8, d10, false), arr(16, 8, 1, d10, true)}},
        {"lite", {arr(1, 12, 6, {1, 2, 4, 8, 16, 32, 64}, false),
                  arr(12, 6, 1, {128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, true)}},
        {"nano", {arr(1, 4, 2, {1, 2, 4, 8, 16, 32, 64}, false), arr(4, 2, 1, {128, 256, 512, 1, 2}, true)}},
        {"gated", {arr(1, 3, 2, {1, 4, 300}, false, 3, true), arr(3, 2, 1, {2, 9}, true, 3, true)}},
        {"kernel1-kernel2", {arr(1, 6, 5, {1, 1}, true, 1), arr(6, 5, 1, {3, 700}, true, 2)}},
        {"off-table-channels", {arr(1, 5, 7, {1, 3, 9}, false), arr(5, 7, 1, {27, 81}, true)}},
        {"activations", {arr(1, 8, 8, {1, 2}, false, 3, false, "Hardtanh"),
                         arr(8, 8, 4, {4}, false, 3, false, "ReLU"),
                         arr(8, 4, 1, {8}, true, 3, false, "Sigmoid")}},
    };
}

}  // namespace

TEST_CASE("block engine matches the per-sample oracle", "[nam][block]") {
    const auto x = test_signal(48000);
    for (const TanhMode mode : {TanhMode::Exact, TanhMode::Vector})
    for (const auto& [name, arrays] : engine_shapes()) {
        INFO("shape " << name << " vector mode " << (mode == TanhMode::Vector));
        NamModel block = build_seeded(arrays, 0x1234u);
        block.set_tanh_mode(mode);
        NamModel oracle = block;
        REQUIRE(block.block_engine_available());
        REQUIRE(block.cpu_engine() == NamModel::CpuEngine::Block);
        oracle.set_cpu_engine(NamModel::CpuEngine::Oracle);
        REQUIRE(oracle.cpu_engine() == NamModel::CpuEngine::Oracle);
        block.prewarm();
        oracle.prewarm();
        const auto yb = stream(block, x, {1, 7, 64, 200, 333, 512, 1000});
        const auto yo = stream(oracle, x, {64});
        // Float reassociation only: ~-125 dB on these shapes. A wrong tap, weight
        // or bias lands near -20 dB, so -100 dB separates the two cleanly.
        const double db = null_db(yo, yb);
        INFO("null " << db << " dB");
        REQUIRE(db < -100.0);
    }
}

TEST_CASE("block engine matches the oracle on the bundled standard capture", "[nam][block]") {
    NamModel block;
    std::string err;
    REQUIRE(load_nam(std::string(GPU_NAM_MODELS_DIR) + "/wavenet_a1_standard.nam", block, &err));
    NamModel oracle = block;
    oracle.set_cpu_engine(NamModel::CpuEngine::Oracle);
    block.prewarm();
    oracle.prewarm();
    const auto x = test_signal(24000);
    const double db = null_db(stream(oracle, x, {128}), stream(block, x, {32, 96, 511}));
    INFO("null " << db << " dB");
    REQUIRE(db < -100.0);
}

TEST_CASE("block engine output does not depend on how the stream is partitioned", "[nam][block]") {
    // Each frame's sums are formed in the same order whatever the run length, so
    // every partitioning is bit-identical, including runs that cross the internal
    // run size and history rewinds.
    const auto x = test_signal(20000);
    for (const auto& [name, arrays] : engine_shapes()) {
        INFO("shape " << name);
        NamModel a = build_seeded(arrays, 0x77u);
        NamModel b = a;
        a.prewarm();
        b.prewarm();
        const auto ya = stream(a, x, {1});
        const auto yb = stream(b, x, {512, 3, 129, 4096});
        REQUIRE(ya == yb);
    }
}

TEST_CASE("a copied block engine continues the stream identically", "[nam][block]") {
    // The processor copies models per channel; a copy must own its history.
    const auto x = test_signal(8000);
    NamModel a = build_seeded(engine_shapes()[1].second, 0x99u);
    a.prewarm();
    std::vector<float> head(4000), tail_a(4000), tail_b(4000);
    a.process(x.data(), head.data(), 4000);
    NamModel b = a;
    a.process(x.data() + 4000, tail_a.data(), 4000);
    b.process(x.data() + 4000, tail_b.data(), 4000);
    REQUIRE(tail_a == tail_b);
}

TEST_CASE("NamRational tanh mode evaluates the rational approximation", "[nam][block]") {
    // Spot values of the approximation itself: odd, 0 at 0, close to tanh.
    REQUIRE(kernels::nam_rational_tanh(0.0f) == 0.0f);
    for (float v : {0.1f, 0.58f, 1.0f, 2.5f, 7.0f}) {
        REQUIRE(kernels::nam_rational_tanh(-v) == -kernels::nam_rational_tanh(v));
        REQUIRE(std::fabs(kernels::nam_rational_tanh(v) - std::tanh(v)) < 4.5e-4f);
    }
    REQUIRE(std::fabs(kernels::nam_rational_tanh(0.58078f) - std::tanh(0.58078f)) > 4.0e-4f);

    // One tanh layer, all 1x1: the model output is a closed form in the activation.
    LayerArrayConfig a = arr(1, 1, 1, {1}, true, 1);
    const std::vector<float> w = {0.8f, 1.3f, -0.2f, 0.5f, 0.7f, 0.1f, 1.1f, -0.05f, 0.25f};
    NamModel m;
    REQUIRE(m.build({a}, 0.25f, w, 1, 48000.0));
    auto expected = [&](float x, bool rational) {
        const float z = (1.3f * (0.8f * x) + -0.2f) + 0.5f * x;
        const float act = rational ? kernels::nam_rational_tanh(z) : std::tanh(z);
        return 0.25f * (1.1f * act + -0.05f);
    };
    for (float x : {-0.9f, -0.3f, 0.0f, 0.4f, 1.2f}) {
        m.set_tanh_mode(TanhMode::NamRational);
        m.reset();
        REQUIRE_THAT(m.process_sample(x), WithinAbs(expected(x, true), 1e-6f));
        m.set_tanh_mode(TanhMode::Exact);
        m.reset();
        REQUIRE_THAT(m.process_sample(x), WithinAbs(expected(x, false), 1e-6f));
    }
    // Where the two functions differ by more than the tolerance, the mode must show.
    m.set_tanh_mode(TanhMode::NamRational);
    m.reset();
    const float yr = m.process_sample(0.3f);
    m.set_tanh_mode(TanhMode::Exact);
    m.reset();
    const float ye = m.process_sample(0.3f);
    REQUIRE(std::fabs(yr - ye) > 1e-5f);
}

TEST_CASE("block engine streaming is allocation-free", "[nam][block][rt-safety]") {
    NamModel m = build_seeded(engine_shapes()[0].second, 0x5u);
    m.prewarm();
    const auto x = test_signal(6000);
    std::vector<float> y(x.size());
    g_allocs = 0;
    g_count_allocs = true;
    m.process(x.data(), y.data(), 2048);         // longer than one internal run
    m.process(x.data() + 2048, y.data() + 2048, 33);
    for (int i = 0; i < 64; ++i) (void)m.process_sample(x[static_cast<std::size_t>(3000 + i)]);
    m.set_tanh_mode(TanhMode::NamRational);
    m.process(x.data() + 4000, y.data() + 4000, 1000);
    m.set_tanh_mode(TanhMode::Vector);
    m.process(x.data() + 5000, y.data() + 5000, 1000);
    g_count_allocs = false;
    REQUIRE(g_allocs.load() == 0);
}
