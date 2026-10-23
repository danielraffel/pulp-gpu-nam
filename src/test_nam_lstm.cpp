// Tests for the NAM "LSTM" CPU inference, its .nam loader, and the
// NamRuntime architecture dispatch.
//
// The correctness golden (synthetic single-layer LSTM below) was produced by an
// independent NumPy LSTM implementation, and the loader was separately validated
// bit-approximately (max abs diff 6.7e-8 over 64 samples) against NeuralAmpModeler
// Core's own reference output for a real LSTM capture. This test locks in both the
// forward math and the file/dispatch surface without shipping any external model.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <string>
#include <vector>

#include "nam_lstm.hpp"
#include "nam_runtime.hpp"

// Allocation probe: counts operator new while armed (audio-thread contract).
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

// Weights for a 1-layer, input_size=1, hidden_size=2 LSTM + linear head, laid out
// exactly as NamLstmModel consumes them: W[4h x (i+h)] row-major, b[4h], h0[h],
// c0[h], head W[1 x h], head b[1]. 39 floats.
const std::vector<float> kLstmWeights = {
    -0.508400f, 0.335900f, -0.073900f, 0.268200f, 0.573600f, 0.046200f, 0.001300f, -0.513500f,
    -0.277900f, -0.000100f, 0.215100f, 0.364500f, -0.142900f, -0.520900f, -0.254200f, 0.491500f,
    -0.343900f, -0.057500f, 0.517400f, -0.570100f, 0.120700f, 0.540200f, -0.323600f, 0.058200f,
    0.409100f, -0.366800f, 0.023400f, 0.250400f, 0.169000f, -0.032200f, -0.295200f, -0.009200f,
    -0.076600f, -0.013600f, -0.080500f, 0.202800f, 0.268600f, -0.186000f, 0.029100f};

// Independent NumPy reference: outputs for the input sequence below.
const std::vector<float> kLstmInput = {1.0f, -0.5f, 0.25f, 0.8f, -0.3f, 0.1f, 0.0f, -0.9f};
const std::vector<float> kLstmGolden = {
    -0.009121865f, 0.034539543f, 0.035764953f, 0.016620019f,
    0.040148773f,  0.043463078f, 0.046994925f, 0.062112819f};

std::string write_temp(const std::string& name, const std::string& content) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary);
    f << content;
    f.close();
    return p.string();
}

std::string lstm_json(const std::vector<float>& w, int in, int hidden, int layers) {
    std::string s = "{\"architecture\":\"LSTM\",\"config\":{\"input_size\":" + std::to_string(in)
                    + ",\"hidden_size\":" + std::to_string(hidden)
                    + ",\"num_layers\":" + std::to_string(layers) + "},\"sample_rate\":48000,\"weights\":[";
    for (std::size_t i = 0; i < w.size(); ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", w[i]);
        s += buf;
        if (i + 1 < w.size()) s += ",";
    }
    s += "]}";
    return s;
}

} // namespace

TEST_CASE("LSTM forward matches an independent reference", "[nam][lstm]") {
    NamLstmModel m;
    REQUIRE(m.build(1, 2, 1, 1, kLstmWeights, 48000.0));
    REQUIRE(m.weights_consumed() == kLstmWeights.size());
    REQUIRE(m.expected_weight_count() == kLstmWeights.size());
    m.reset();

    std::vector<float> out(kLstmInput.size(), 0.0f);
    m.process(kLstmInput.data(), out.data(), static_cast<std::uint32_t>(kLstmInput.size()));
    for (std::size_t i = 0; i < kLstmGolden.size(); ++i)
        CHECK_THAT(out[i], WithinAbs(kLstmGolden[i], 1e-5));
}

TEST_CASE("LSTM reset restores stored initial state (streaming repeatability)", "[nam][lstm]") {
    NamLstmModel m;
    REQUIRE(m.build(1, 2, 1, 1, kLstmWeights, 48000.0));

    std::vector<float> a(kLstmInput.size()), b(kLstmInput.size());
    m.reset();
    m.process(kLstmInput.data(), a.data(), static_cast<std::uint32_t>(kLstmInput.size()));
    m.reset();
    m.process(kLstmInput.data(), b.data(), static_cast<std::uint32_t>(kLstmInput.size()));
    for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i] == b[i]);
}

TEST_CASE("LSTM per-sample and block processing agree", "[nam][lstm]") {
    NamLstmModel block, sample;
    REQUIRE(block.build(1, 2, 1, 1, kLstmWeights, 48000.0));
    REQUIRE(sample.build(1, 2, 1, 1, kLstmWeights, 48000.0));
    block.reset();
    sample.reset();

    std::vector<float> bo(kLstmInput.size());
    block.process(kLstmInput.data(), bo.data(), static_cast<std::uint32_t>(kLstmInput.size()));
    for (std::size_t i = 0; i < kLstmInput.size(); ++i)
        CHECK(sample.process_sample(kLstmInput[i]) == bo[i]);
}

TEST_CASE("LSTM loader parses a .nam and rejects a bad weight count", "[nam][lstm]") {
    const std::string good = write_temp("pulp_lstm_ok.nam", lstm_json(kLstmWeights, 1, 2, 1));
    NamLstmModel m;
    std::string err;
    REQUIRE(load_nam_lstm(good, m, &err));
    CHECK(m.hidden_size() == 2);
    CHECK(m.num_layers() == 1);

    std::vector<float> short_w(kLstmWeights.begin(), kLstmWeights.end() - 3);
    const std::string bad = write_temp("pulp_lstm_bad.nam", lstm_json(short_w, 1, 2, 1));
    NamLstmModel m2;
    std::string err2;
    CHECK_FALSE(load_nam_lstm(bad, m2, &err2));
    CHECK_FALSE(err2.empty());
}

TEST_CASE("LSTM build rejects a non-mono input_size", "[nam][lstm]") {
    // The runtime feeds a 1-wide input; a first layer expecting >1 channel would
    // read past that buffer, so build() must refuse it.
    NamLstmModel m;
    // 40 weights for input_size=2: cell W[8 x 4]=32, b[8]=8 ... but build should
    // reject on the shape before it even checks the count.
    std::vector<float> w(64, 0.1f);
    std::string err;
    CHECK_FALSE(m.build(2, 2, 1, 1, w, 48000.0));
    CHECK(err.empty());  // build() sets its own error_, not this local
    CHECK(m.error().find("input_size") != std::string::npos);
}

TEST_CASE("NamRuntime dispatches on the architecture field", "[nam][runtime]") {
    // LSTM file → LSTM engine, faithful output.
    const std::string lpath = write_temp("pulp_rt_lstm.nam", lstm_json(kLstmWeights, 1, 2, 1));
    NamRuntime rt;
    std::string err;
    REQUIRE(load_nam_runtime(lpath, rt, &err));
    CHECK(rt.arch() == NamRuntime::Arch::Lstm);
    CHECK(std::string(rt.arch_name()) == "LSTM");
    CHECK_FALSE(rt.gpu_eligible());          // LSTM is CPU-only
    CHECK(rt.wavenet() == nullptr);
    rt.reset();
    for (std::size_t i = 0; i < kLstmGolden.size(); ++i)
        CHECK_THAT(rt.process_sample(kLstmInput[i]), WithinAbs(kLstmGolden[i], 1e-5));

    // A minimal WaveNet file → WaveNet engine, GPU-eligible.
    const std::string wave =
        "{\"architecture\":\"WaveNet\",\"config\":{\"layers\":[{\"input_size\":1,\"condition_size\":1,"
        "\"channels\":1,\"kernel_size\":1,\"dilations\":[1],\"head_size\":1,\"head_bias\":true,"
        "\"gated\":false,\"activation\":\"Tanh\"}],\"head_scale\":0.25},\"sample_rate\":48000,"
        "\"weights\":[0.8,1.3,-0.2,0.5,0.7,0.1,1.1,-0.05,0.25]}";
    const std::string wpath = write_temp("pulp_rt_wave.nam", wave);
    NamRuntime rw;
    REQUIRE(load_nam_runtime(wpath, rw, &err));
    CHECK(rw.arch() == NamRuntime::Arch::WaveNet);
    CHECK(rw.gpu_eligible());
    CHECK(rw.wavenet() != nullptr);

    // An unsupported architecture → clean failure, Arch::None.
    const std::string cpath = write_temp("pulp_rt_conv.nam",
        "{\"architecture\":\"ConvNet\",\"config\":{},\"weights\":[]}");
    NamRuntime rc;
    std::string cerr;
    CHECK_FALSE(load_nam_runtime(cpath, rc, &cerr));
    CHECK(rc.arch() == NamRuntime::Arch::None);
    CHECK_FALSE(cerr.empty());
}

namespace {

NamLstmModel seeded_lstm(int hidden, int layers, std::uint32_t seed) {
    std::size_t count = 0;
    for (int l = 0; l < layers; ++l) {
        const std::size_t in = l == 0 ? 1u : static_cast<std::size_t>(hidden);
        const std::size_t h = static_cast<std::size_t>(hidden);
        count += 4u * h * (in + h) + 4u * h + 2u * h;
    }
    count += static_cast<std::size_t>(hidden) + 1u;
    std::vector<float> w(count);
    std::uint32_t s = seed;
    for (auto& v : w) {
        s = s * 1664525u + 1013904223u;
        v = (static_cast<float>(s >> 9) / 4194304.0f - 1.0f) * 0.4f;
    }
    NamLstmModel m;
    REQUIRE(m.build(1, hidden, layers, 1, std::move(w), 48000.0));
    return m;
}

std::vector<float> lstm_signal(std::size_t n) {
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 0.6f * std::sin(0.013f * static_cast<float>(i)) + 0.2f * std::sin(0.31f * static_cast<float>(i));
    x[n / 2] += 0.8f;
    return x;
}

double lstm_null_db(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, r = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(b[i]) - a[i];
        e += d * d;
        r += static_cast<double>(a[i]) * a[i];
    }
    return 10.0 * std::log10((e + 1e-300) / (r + 1e-300));
}

// Scalar LSTM using the rational tanh/sigmoid, written independently of
// LstmCell, for the NamRational mode check.
std::vector<float> rational_reference(const std::vector<float>& w, int hidden, int layers,
                                      const std::vector<float>& x) {
    struct Cell { std::vector<float> W, b, h, c; int in; };
    std::vector<Cell> cells;
    std::size_t p = 0;
    for (int l = 0; l < layers; ++l) {
        Cell c;
        c.in = l == 0 ? 1 : hidden;
        const int rows = 4 * hidden, cols = c.in + hidden;
        c.W.assign(w.begin() + static_cast<long>(p), w.begin() + static_cast<long>(p) + rows * cols);
        p += static_cast<std::size_t>(rows * cols);
        c.b.assign(w.begin() + static_cast<long>(p), w.begin() + static_cast<long>(p) + rows);
        p += static_cast<std::size_t>(rows);
        c.h.assign(w.begin() + static_cast<long>(p), w.begin() + static_cast<long>(p) + hidden);
        p += static_cast<std::size_t>(hidden);
        c.c.assign(w.begin() + static_cast<long>(p), w.begin() + static_cast<long>(p) + hidden);
        p += static_cast<std::size_t>(hidden);
        cells.push_back(std::move(c));
    }
    const std::vector<float> hw(w.begin() + static_cast<long>(p), w.begin() + static_cast<long>(p) + hidden);
    const float hb = w[p + static_cast<std::size_t>(hidden)];
    auto sig = [](float v) { return kernels::nam_rational_sigmoid(v); };
    std::vector<float> y;
    for (float xv : x) {
        std::vector<float> v = {xv};
        for (Cell& c : cells) {
            std::vector<float> xh = v;
            xh.insert(xh.end(), c.h.begin(), c.h.end());
            std::vector<float> z(static_cast<std::size_t>(4 * hidden));
            for (int r = 0; r < 4 * hidden; ++r) {
                double acc = c.b[static_cast<std::size_t>(r)];
                for (std::size_t j = 0; j < xh.size(); ++j) acc += static_cast<double>(c.W[r * xh.size() + j]) * xh[j];
                z[static_cast<std::size_t>(r)] = static_cast<float>(acc);
            }
            for (int j = 0; j < hidden; ++j) {
                const auto J = static_cast<std::size_t>(j), H = static_cast<std::size_t>(hidden);
                c.c[J] = sig(z[H + J]) * c.c[J] + sig(z[J]) * kernels::nam_rational_tanh(z[2 * H + J]);
                c.h[J] = sig(z[3 * H + J]) * kernels::nam_rational_tanh(c.c[J]);
            }
            v = c.h;
        }
        float out = hb;
        for (int j = 0; j < hidden; ++j) out += hw[static_cast<std::size_t>(j)] * v[static_cast<std::size_t>(j)];
        y.push_back(out);
    }
    return y;
}

}  // namespace

TEST_CASE("LSTM fast engine matches the scalar oracle", "[nam][lstm]") {
    const auto x = lstm_signal(24000);
    for (const TanhMode mode : {TanhMode::Exact, TanhMode::Vector})
    for (int hidden : {2, 8, 24, 40}) {
        INFO("hidden " << hidden << " vector mode " << (mode == TanhMode::Vector));
        NamLstmModel fast = seeded_lstm(hidden, 3, 0xABCu + static_cast<std::uint32_t>(hidden));
        fast.set_tanh_mode(mode);
        NamLstmModel oracle = fast;
        REQUIRE(fast.cpu_engine() == NamLstmModel::CpuEngine::Fast);
        oracle.set_cpu_engine(NamLstmModel::CpuEngine::Oracle);
        fast.reset();
        oracle.reset();
        std::vector<float> yf(x.size()), yo(x.size());
        fast.process(x.data(), yf.data(), static_cast<std::uint32_t>(x.size()));
        oracle.process(x.data(), yo.data(), static_cast<std::uint32_t>(x.size()));
        const double db = lstm_null_db(yo, yf);
        INFO("null " << db << " dB");
        REQUIRE(db < -100.0);
    }
}

TEST_CASE("LSTM fast engine matches the oracle on the bundled capture", "[nam][lstm]") {
    NamLstmModel fast;
    std::string err;
    REQUIRE(load_nam_lstm(std::string(GPU_NAM_MODELS_DIR) + "/lstm.nam", fast, &err));
    NamLstmModel oracle = fast;
    oracle.set_cpu_engine(NamLstmModel::CpuEngine::Oracle);
    fast.prewarm();
    oracle.prewarm();
    const auto x = lstm_signal(24000);
    std::vector<float> yf(x.size()), yo(x.size());
    fast.process(x.data(), yf.data(), static_cast<std::uint32_t>(x.size()));
    oracle.process(x.data(), yo.data(), static_cast<std::uint32_t>(x.size()));
    REQUIRE(lstm_null_db(yo, yf) < -100.0);
}

TEST_CASE("LSTM NamRational mode reproduces the rational recurrence", "[nam][lstm]") {
    const int hidden = 6, layers = 2;
    NamLstmModel m = seeded_lstm(hidden, layers, 0x31u);
    // Rebuild the same weights for the independent reference.
    std::size_t count = 0;
    for (int l = 0; l < layers; ++l) {
        const std::size_t in = l == 0 ? 1u : static_cast<std::size_t>(hidden);
        count += 4u * hidden * (in + hidden) + 4u * hidden + 2u * hidden;
    }
    count += static_cast<std::size_t>(hidden) + 1u;
    std::vector<float> w(count);
    std::uint32_t s = 0x31u;
    for (auto& v : w) {
        s = s * 1664525u + 1013904223u;
        v = (static_cast<float>(s >> 9) / 4194304.0f - 1.0f) * 0.4f;
    }
    const auto x = lstm_signal(4000);
    const auto ref = rational_reference(w, hidden, layers, x);

    m.set_tanh_mode(TanhMode::NamRational);
    m.reset();
    std::vector<float> y(x.size());
    m.process(x.data(), y.data(), static_cast<std::uint32_t>(x.size()));
    const double db_rational = lstm_null_db(ref, y);
    INFO("rational mode vs rational reference " << db_rational << " dB");
    REQUIRE(db_rational < -100.0);

    // The exact mode is a measurably different function on the same input.
    m.set_tanh_mode(TanhMode::Exact);
    m.reset();
    m.process(x.data(), y.data(), static_cast<std::uint32_t>(x.size()));
    REQUIRE(lstm_null_db(ref, y) > -90.0);
}

TEST_CASE("LSTM streaming is allocation-free in every tanh mode", "[nam][lstm][rt-safety]") {
    NamLstmModel m = seeded_lstm(24, 3, 0x77u);
    m.prewarm();
    const auto x = lstm_signal(4096);
    std::vector<float> y(x.size());
    g_allocs = 0;
    g_count_allocs = true;
    m.process(x.data(), y.data(), 2048);
    m.set_tanh_mode(TanhMode::NamRational);
    m.process(x.data() + 2048, y.data() + 2048, 1024);
    m.set_tanh_mode(TanhMode::Vector);
    m.process(x.data() + 3072, y.data() + 3072, 1024);
    g_count_allocs = false;
    REQUIRE(g_allocs.load() == 0);
}

TEST_CASE("LSTM run processing is bit-identical to per-sample processing", "[nam][lstm]") {
    // process() runs layer-major over runs of frames (the input half batched);
    // process_sample() steps every layer per sample. Every term is added in the
    // same order, so the outputs and the carried state must match exactly, in
    // every tanh mode and across run boundaries.
    const auto x = lstm_signal(3000);
    // Hidden sizes cover every recurrent-product tile width (4H = 12, 32, 48, 64,
    // 80, 96, 128) and both gate paths (fused NamRational, array-wide passes).
    for (const TanhMode mode : {TanhMode::Exact, TanhMode::Vector, TanhMode::NamRational})
    for (int hidden : {3, 8, 12, 16, 20, 24, 32}) {
        INFO("hidden " << hidden);
        NamLstmModel runm = seeded_lstm(hidden, 3, 0x51u + static_cast<std::uint32_t>(hidden));
        NamLstmModel step = runm;
        runm.set_tanh_mode(mode);
        step.set_tanh_mode(mode);
        runm.reset();
        step.reset();
        std::vector<float> yr(x.size()), ys(x.size());
        std::size_t pos = 0;
        for (std::uint32_t n : {1u, 127u, 128u, 129u, 500u, 1024u, 1091u}) {
            runm.process(x.data() + pos, yr.data() + pos, n);
            pos += n;
        }
        REQUIRE(pos == x.size());
        for (std::size_t i = 0; i < x.size(); ++i) ys[i] = step.process_sample(x[i]);
        REQUIRE(yr == ys);
        // State carries over: interleaving the two entry points stays identical.
        float a = runm.process_sample(0.25f), b = step.process_sample(0.25f);
        REQUIRE(a == b);
    }
}
