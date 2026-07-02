// GPU NAM plugin processor tests: the inline CPU engine produces finite,
// non-trivial amped audio and bypass passes the dry signal; the opt-in GPU engine
// activates (with a device), produces blocks, populates gpu_status, and reproduces
// the CPU engine within tolerance; the Engine switch is live and stays finite at a
// fixed latency; and load_model() rebuilds the engines without NaNs. GPU cases
// skip cleanly with no device (Metal is present in the dev environment, so they run).

#include <catch2/catch_test_macros.hpp>

#include <pulp/audio/audio_file.hpp>
#include <pulp/format/headless.hpp>
#include <pulp/midi/buffer.hpp>
#include <pulp/state/store.hpp>

#include <filesystem>

#include "gpu_nam_processor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>
#include <vector>

using namespace pulp::examples;

namespace {

// Render nblocks of `block` samples through the host, with the FIRST block's
// channel 0/1 supplied by `first_block` (rest zero), and collect channel 0.
std::vector<float> render(pulp::format::HeadlessHost& host, std::size_t block,
                          int nblocks, const std::vector<float>& first_block) {
    std::vector<float> out_all;
    std::vector<float> in_l(block, 0.0f), in_r(block, 0.0f), out_l(block), out_r(block);
    for (int b = 0; b < nblocks; ++b) {
        std::fill(in_l.begin(), in_l.end(), 0.0f);
        std::fill(in_r.begin(), in_r.end(), 0.0f);
        if (b == 0)
            for (std::size_t i = 0; i < block && i < first_block.size(); ++i)
                in_l[i] = in_r[i] = first_block[i];
        const float* ip[2] = {in_l.data(), in_r.data()};
        float* op[2] = {out_l.data(), out_r.data()};
        pulp::audio::BufferView<const float> iv(ip, 2, block);
        pulp::audio::BufferView<float> ov(op, 2, block);
        host.process(ov, iv);
        out_all.insert(out_all.end(), out_l.begin(), out_l.end());
    }
    return out_all;
}

std::vector<float> sine(std::size_t n, float freq = 0.06f, float amp = 0.5f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = amp * std::sin(freq * static_cast<float>(i));
    return v;
}

std::size_t find_dry_lag(const std::vector<float>& out, const std::vector<float>& probe) {
    std::size_t best_d = 0;
    double best_err = 1e300;
    for (std::size_t d = 0; d + probe.size() <= out.size(); ++d) {
        double e = 0;
        for (std::size_t i = 0; i < probe.size(); ++i) {
            const double diff = static_cast<double>(out[d + i]) - probe[i];
            e += diff * diff;
        }
        if (e < best_err) { best_err = e; best_d = d; }
    }
    return best_d;
}

// Best normalized cross-correlation of `a` against `b` over a steady-state window
// [skip, skip+len), searching b's offset across ±max_lag samples. A startup
// transport miss can globally shift one stream by a whole block relative to the
// other; the lag search finds that uniform shift so a genuine per-sample match
// scores high regardless of the constant offset.
double best_lag_xcorr(const std::vector<float>& a, const std::vector<float>& b,
                      std::size_t skip, std::size_t len, int max_lag) {
    double best = -1.0;
    for (int d = -max_lag; d <= max_lag; ++d) {
        double sxy = 0, sxx = 0, syy = 0;
        std::size_t scored = 0;
        for (std::size_t i = 0; i < len; ++i) {
            const long ia = static_cast<long>(skip + i);
            const long ib = ia + d;
            if (ia < 0 || ib < 0 || ia >= static_cast<long>(a.size())
                || ib >= static_cast<long>(b.size()))
                continue;
            const double x = a[static_cast<std::size_t>(ia)], y = b[static_cast<std::size_t>(ib)];
            sxy += x * y; sxx += x * x; syy += y * y; ++scored;
        }
        if (scored > 0) best = std::max(best, sxy / std::sqrt(sxx * syy + 1e-30));
    }
    return best;
}

// Drive the processor directly (so we can read gpu_engine_active()).
struct Driver {
    GpuNamProcessor& proc;
    std::size_t block;
    pulp::midi::MidiBuffer mi, mo;
    pulp::format::ProcessContext pctx;
    std::vector<float> in_l, in_r, out_l, out_r;
    Driver(GpuNamProcessor& p, std::size_t b, double sr)
        : proc(p), block(b), in_l(b, 0.0f), in_r(b, 0.0f), out_l(b), out_r(b) {
        pctx.sample_rate = sr;
        pctx.num_samples = static_cast<int>(b);
    }
    std::vector<float> block_io(const std::vector<float>& in, int sleep_ms) {
        for (std::size_t i = 0; i < block; ++i) in_l[i] = in_r[i] = i < in.size() ? in[i] : 0.0f;
        const float* ip[2] = {in_l.data(), in_r.data()};
        float* op[2] = {out_l.data(), out_r.data()};
        pulp::audio::BufferView<const float> iv(ip, 2, block);
        pulp::audio::BufferView<float> ov(op, 2, block);
        proc.process(ov, iv, mi, mo, pctx);
        if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        return {out_l.begin(), out_l.end()};
    }
    void pump(int nblocks, int sleep_ms) {
        static const std::vector<float> z;
        for (int b = 0; b < nblocks; ++b) block_io(z, sleep_ms);
    }
};

template <class F>
bool pump_until(Driver& d, int max_iter, int sleep_ms, F pred) {
    for (int i = 0; i < max_iter && !pred(); ++i) d.pump(1, sleep_ms);
    return pred();
}

void prepare_proc(GpuNamProcessor& proc, pulp::state::StateStore& store, double sr,
                  std::size_t block, float engine) {
    proc.set_state_store(&store);
    proc.define_parameters(store);
    store.set_value(kInputGain, 0.0f);
    store.set_value(kOutputGain, 0.0f);
    store.set_value(kMix, 100.0f);
    store.set_value(kBypass, 0.0f);
    store.set_value(kEngine, engine);
    pulp::format::PrepareContext ctx;
    ctx.sample_rate = sr;
    ctx.max_buffer_size = static_cast<int>(block);
    ctx.input_channels = 2;
    ctx.output_channels = 2;
    proc.prepare(ctx);
}

// Run `nblocks` of the per-block signal `sig` through a fresh CPU-engine
// processor, applying `setup(store)` on top of the neutral defaults, and return
// the concatenated channel-0 output. Isolates a single DSP stage under test.
template <class Setup>
std::vector<float> run_cpu(double sr, std::size_t block, int nblocks,
                           const std::vector<float>& sig, Setup setup) {
    GpuNamProcessor proc;
    pulp::state::StateStore store;
    proc.set_state_store(&store);
    proc.define_parameters(store);
    store.set_value(kInputGain, 0.0f);
    store.set_value(kOutputGain, 0.0f);
    store.set_value(kMix, 100.0f);
    store.set_value(kBypass, 0.0f);
    store.set_value(kEngine, 0.0f);
    setup(store);
    pulp::format::PrepareContext ctx;
    ctx.sample_rate = sr;
    ctx.max_buffer_size = static_cast<int>(block);
    ctx.input_channels = 2;
    ctx.output_channels = 2;
    proc.prepare(ctx);
    Driver d(proc, block, sr);
    std::vector<float> out;
    for (int b = 0; b < nblocks; ++b) {
        const auto o = d.block_io(sig, 0);
        out.insert(out.end(), o.begin(), o.end());
    }
    proc.release();
    return out;
}

// Stream a continuous signal through a fresh CPU-engine processor in host blocks
// of `hblock` samples (final chunk = remainder), each process() call using the
// chunk's real width, and return channel-0 output with the processor's reported
// latency trimmed so different host-block schedules line up sample-for-sample.
// The point: exercise the re-block FIFO with host blocks that are NOT the
// internal block size, the case a 512-aligned test never hits.
std::vector<float> run_stream_cpu(double sr, std::size_t hblock,
                                  const std::vector<float>& sig) {
    GpuNamProcessor proc;
    pulp::state::StateStore store;
    prepare_proc(proc, store, sr, hblock, /*engine=*/0.0f);
    std::vector<float> out;
    out.reserve(sig.size());
    std::vector<float> in_l(hblock), in_r(hblock), out_l(hblock), out_r(hblock);
    pulp::midi::MidiBuffer mi, mo;
    pulp::format::ProcessContext pctx;
    pctx.sample_rate = sr;
    for (std::size_t off = 0; off < sig.size(); off += hblock) {
        const std::size_t n = std::min(hblock, sig.size() - off);
        for (std::size_t i = 0; i < n; ++i) in_l[i] = in_r[i] = sig[off + i];
        const float* ip[2] = {in_l.data(), in_r.data()};
        float* op[2] = {out_l.data(), out_r.data()};
        pulp::audio::BufferView<const float> iv(ip, 2, n);
        pulp::audio::BufferView<float> ov(op, 2, n);
        pctx.num_samples = static_cast<int>(n);
        proc.process(ov, iv, mi, mo, pctx);
        for (std::size_t i = 0; i < n; ++i) out.push_back(out_l[i]);
    }
    const int lat = proc.latency_samples();
    proc.release();
    if (lat > 0 && static_cast<std::size_t>(lat) < out.size())
        out.erase(out.begin(), out.begin() + lat);
    return out;
}

// Write a mono IR WAV so load_ir() has a real file to decode.
void write_ir_wav(const std::string& path, const std::vector<float>& samples,
                  std::uint32_t sr = 48000) {
    pulp::audio::AudioFileData d;
    d.sample_rate = sr;
    d.channels = {samples};
    REQUIRE(pulp::audio::write_wav_file(path, d));
}

// Like run_stream_cpu but loads `ir_path` before prepare() (so the IR is built
// synchronously and the first block has it). Empty path = no IR.
std::vector<float> run_stream_cpu_ir(double sr, std::size_t hblock,
                                     const std::vector<float>& sig,
                                     const std::string& ir_path) {
    GpuNamProcessor proc;
    pulp::state::StateStore store;
    proc.set_state_store(&store);
    proc.define_parameters(store);
    store.set_value(kInputGain, 0.0f);
    store.set_value(kOutputGain, 0.0f);
    store.set_value(kMix, 100.0f);
    store.set_value(kBypass, 0.0f);
    store.set_value(kEngine, 0.0f);
    if (!ir_path.empty()) proc.load_ir(ir_path);
    pulp::format::PrepareContext ctx;
    ctx.sample_rate = sr;
    ctx.max_buffer_size = static_cast<int>(hblock);
    ctx.input_channels = 2;
    ctx.output_channels = 2;
    proc.prepare(ctx);
    std::vector<float> out;
    out.reserve(sig.size());
    std::vector<float> in_l(hblock), in_r(hblock), out_l(hblock), out_r(hblock);
    pulp::midi::MidiBuffer mi, mo;
    pulp::format::ProcessContext pctx;
    pctx.sample_rate = sr;
    for (std::size_t off = 0; off < sig.size(); off += hblock) {
        const std::size_t n = std::min(hblock, sig.size() - off);
        for (std::size_t i = 0; i < n; ++i) in_l[i] = in_r[i] = sig[off + i];
        const float* ip[2] = {in_l.data(), in_r.data()};
        float* op[2] = {out_l.data(), out_r.data()};
        pulp::audio::BufferView<const float> iv(ip, 2, n);
        pulp::audio::BufferView<float> ov(op, 2, n);
        pctx.num_samples = static_cast<int>(n);
        proc.process(ov, iv, mi, mo, pctx);
        for (std::size_t i = 0; i < n; ++i) out.push_back(out_l[i]);
    }
    const int lat = proc.latency_samples();
    proc.release();
    if (lat > 0 && static_cast<std::size_t>(lat) < out.size())
        out.erase(out.begin(), out.begin() + lat);
    return out;
}

// Load `model_path` (and optionally `ir_path`) BEFORE prepare() so both are built
// synchronously and active on the first block, then stream `sig` through the full
// CPU-engine processor with the noise gate OFF (so the check measures the model +
// IR path, not gating). Returns channel-0 output with reported latency trimmed.
std::vector<float> run_stream_cpu_model_ir(double sr, std::size_t hblock,
                                           const std::vector<float>& sig,
                                           const std::string& model_path,
                                           const std::string& ir_path) {
    GpuNamProcessor proc;
    pulp::state::StateStore store;
    proc.set_state_store(&store);
    proc.define_parameters(store);
    store.set_value(kInputGain, 0.0f);
    store.set_value(kOutputGain, 0.0f);
    store.set_value(kMix, 100.0f);
    store.set_value(kBypass, 0.0f);
    store.set_value(kEngine, 0.0f);
    store.set_value(kNoiseGateActive, 0.0f);
    if (!model_path.empty()) proc.load_model(model_path);
    if (!ir_path.empty()) proc.load_ir(ir_path);
    pulp::format::PrepareContext ctx;
    ctx.sample_rate = sr;
    ctx.max_buffer_size = static_cast<int>(hblock);
    ctx.input_channels = 2;
    ctx.output_channels = 2;
    proc.prepare(ctx);
    std::vector<float> out;
    out.reserve(sig.size());
    std::vector<float> in_l(hblock), in_r(hblock), out_l(hblock), out_r(hblock);
    pulp::midi::MidiBuffer mi, mo;
    pulp::format::ProcessContext pctx;
    pctx.sample_rate = sr;
    for (std::size_t off = 0; off < sig.size(); off += hblock) {
        const std::size_t n = std::min(hblock, sig.size() - off);
        for (std::size_t i = 0; i < n; ++i) in_l[i] = in_r[i] = sig[off + i];
        const float* ip[2] = {in_l.data(), in_r.data()};
        float* op[2] = {out_l.data(), out_r.data()};
        pulp::audio::BufferView<const float> iv(ip, 2, n);
        pulp::audio::BufferView<float> ov(op, 2, n);
        pctx.num_samples = static_cast<int>(n);
        proc.process(ov, iv, mi, mo, pctx);
        for (std::size_t i = 0; i < n; ++i) out.push_back(out_l[i]);
    }
    const int lat = proc.latency_samples();
    proc.release();
    if (lat > 0 && static_cast<std::size_t>(lat) < out.size())
        out.erase(out.begin(), out.begin() + lat);
    return out;
}

double signal_rms(const std::vector<float>& v) {
    if (v.empty()) return 0.0;
    double acc = 0.0;
    for (float x : v) acc += static_cast<double>(x) * static_cast<double>(x);
    return std::sqrt(acc / static_cast<double>(v.size()));
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// A guitar-like drive: a 220 Hz tone with light 2nd/3rd harmonics at ~0.4 peak, so
// a real amp model has something to shape and the (disabled) gate is irrelevant.
std::vector<float> drive_signal(double sr, std::size_t n) {
    std::vector<float> s(n);
    const double w = 2.0 * 3.14159265358979323846 * 220.0 / sr;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        s[i] = static_cast<float>(0.4 * (std::sin(w * t)
                                         + 0.2 * std::sin(2.0 * w * t)
                                         + 0.1 * std::sin(3.0 * w * t)));
    }
    return s;
}

}  // namespace

TEST_CASE("GPU NAM CPU engine produces finite non-trivial amped output", "[nam]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;

    pulp::format::HeadlessHost host(create_gpu_nam);
    host.state().set_value(kInputGain, 6.0f);   // drive the amp
    host.state().set_value(kMix, 100.0f);
    host.state().set_value(kEngine, 0.0f);      // inline CPU engine
    host.state().set_value(kBypass, 0.0f);
    host.prepare(SR, static_cast<int>(BLOCK));

    const auto probe = sine(BLOCK);
    const auto out = render(host, BLOCK, 8, probe);

    double energy = 0.0;
    for (float v : out) { REQUIRE(std::isfinite(v)); energy += static_cast<double>(v) * v; }
    REQUIRE(energy > 1e-4);   // the amp produced real output
}

TEST_CASE("GPU NAM bypass passes the dry signal", "[nam]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;

    pulp::format::HeadlessHost host(create_gpu_nam);
    host.state().set_value(kBypass, 1.0f);
    host.state().set_value(kMix, 100.0f);
    host.state().set_value(kInputGain, 6.0f);   // ignored on the dry path
    host.state().set_value(kEngine, 0.0f);
    host.prepare(SR, static_cast<int>(BLOCK));

    const auto probe = sine(BLOCK);
    const auto out = render(host, BLOCK, 8, probe);
    const std::size_t lag = find_dry_lag(out, probe);
    for (std::size_t i = 0; i < BLOCK; ++i)
        REQUIRE(std::abs(out[lag + i] - probe[i]) < 1e-6f);
}

TEST_CASE("GPU NAM GPU engine reproduces the CPU engine", "[nam][gpu]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;
    const int nblocks = 32;

    std::vector<float> in;
    for (int b = 0; b < nblocks; ++b) {
        for (std::size_t i = 0; i < BLOCK; ++i)
            in.push_back(0.5f * std::sin(0.06f * static_cast<float>(b * BLOCK + i)));
    }

    // Reference: the inline CPU engine.
    std::vector<float> cpu_out;
    {
        GpuNamProcessor proc;
        pulp::state::StateStore store;
        prepare_proc(proc, store, SR, BLOCK, /*engine=*/0.0f);
        Driver d(proc, BLOCK, SR);
        for (int b = 0; b < nblocks; ++b) {
            std::vector<float> blk(in.begin() + b * BLOCK, in.begin() + (b + 1) * BLOCK);
            const auto o = d.block_io(blk, 0);
            cpu_out.insert(cpu_out.end(), o.begin(), o.end());
        }
        proc.release();
    }

    GpuNamProcessor proc;
    pulp::state::StateStore store;
    prepare_proc(proc, store, SR, BLOCK, /*engine=*/1.0f);
    if (!proc.gpu_engine_active()) {
        WARN("GPU engine unavailable — skipping GPU-vs-CPU test (CPU path still covered).");
        proc.release();
        return;
    }
    REQUIRE(proc.latency_samples() > static_cast<int>(BLOCK));  // re-block + transport delay

    std::vector<float> gpu_out;
    Driver d(proc, BLOCK, SR);
    for (int b = 0; b < nblocks; ++b) {
        std::vector<float> blk(in.begin() + b * BLOCK, in.begin() + (b + 1) * BLOCK);
        const auto o = d.block_io(blk, 12);  // let the worker produce the delayed block
        gpu_out.insert(gpu_out.end(), o.begin(), o.end());
    }

    const auto status = proc.gpu_status();
    const auto stats = proc.gpu_block_stats();
    const auto us = proc.gpu_block_us();
    INFO("blocks=" << stats.first << " misses=" << stats.second
         << " avg_us=" << us.second << " budget_us=" << status.budget_us
         << " rt%=" << status.rt_percent);
    REQUIRE(status.active);
    REQUIRE(stats.first > 0);
    REQUIRE(us.second > 0.0);
    REQUIRE(status.blocks == stats.first);
    REQUIRE(status.budget_us > 0.0);
    REQUIRE(status.rt_percent > 0.0);

    for (float v : gpu_out) REQUIRE(std::isfinite(v));
    double energy = 0.0;
    for (float v : gpu_out) energy += static_cast<double>(v) * v;
    REQUIRE(energy > 1e-4);

    // Both engines run the same model on the same input → the GPU reproduces the
    // CPU output. Score a steady-state window past the priming transient with a
    // ±4-block lag search (a startup miss can globally shift one stream by a
    // block); the bar is the GPU f32 path's 0.99.
    const double xc = best_lag_xcorr(gpu_out, cpu_out,
                                     /*skip=*/12 * BLOCK, /*len=*/14 * BLOCK,
                                     /*max_lag=*/4 * static_cast<int>(BLOCK));
    INFO("GPU-vs-CPU xcorr=" << xc);
    REQUIRE(xc > 0.99);
    proc.release();
}

TEST_CASE("GPU NAM switches Engine CPU->GPU->CPU live at fixed latency", "[nam][gpu]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;

    GpuNamProcessor proc;
    pulp::state::StateStore store;
    prepare_proc(proc, store, SR, BLOCK, /*engine=*/0.0f);
    REQUIRE_FALSE(proc.gpu_engine_active());
    const int latency = proc.latency_samples();

    Driver d(proc, BLOCK, SR);
    auto drive = [&](int nblocks, int sleep_ms) {
        double energy = 0.0;
        for (int b = 0; b < nblocks; ++b) {
            std::vector<float> in(BLOCK);
            for (std::size_t i = 0; i < BLOCK; ++i)
                in[i] = 0.5f * std::sin(0.06f * static_cast<float>(b * BLOCK + i));
            const auto o = d.block_io(in, sleep_ms);
            for (float v : o) { REQUIRE(std::isfinite(v)); energy += static_cast<double>(v) * v; }
        }
        return energy;
    };
    REQUIRE(drive(8, 0) > 1e-4);   // CPU produces audio

    store.set_value(kEngine, 1.0f);
    if (!pump_until(d, 150, 8, [&] { return proc.gpu_engine_active(); })) {
        WARN("GPU engine unavailable — skipping live-switch test.");
        proc.release();
        return;
    }
    REQUIRE(proc.latency_samples() == latency);   // FIXED across the switch
    REQUIRE(drive(24, 8) > 1e-4);                 // GPU produces audio, stays finite
    REQUIRE(proc.gpu_block_stats().first > 0);

    store.set_value(kEngine, 0.0f);
    REQUIRE(pump_until(d, 100, 5, [&] { return !proc.gpu_engine_active(); }));
    REQUIRE(proc.latency_samples() == latency);   // still fixed
    REQUIRE(drive(8, 0) > 1e-4);                  // CPU resumed, stays finite
    proc.release();
}

TEST_CASE("GPU NAM Engine=Auto resolves once at prepare and the choice takes effect",
          "[nam][engine]") {
    // Auto (Engine==2) benchmarks the CPU at prepare and picks CPU or GPU. The
    // resolved engine must actually drive the live path — gpu_engine_active() has
    // to agree with the resolution, which proves the selection wired through.
    {
        GpuNamProcessor proc;
        pulp::state::StateStore store;
        prepare_proc(proc, store, 48000.0, 128, /*engine=*/2.0f);
        const int resolved = proc.auto_resolved_engine();
        CHECK((resolved == 0 || resolved == 1));
        CHECK(proc.gpu_engine_active() == (resolved == 1));
        proc.release();
    }
    // Manual CPU (Engine==0) always stays on the CPU, independent of any device.
    {
        GpuNamProcessor proc;
        pulp::state::StateStore store;
        prepare_proc(proc, store, 48000.0, 128, /*engine=*/0.0f);
        CHECK_FALSE(proc.gpu_engine_active());
        proc.release();
    }
}

TEST_CASE("GPU NAM dual-mono fast path stays bit-exact through channel divergence",
          "[nam]") {
    // When both channels carry identical drive (mono panned to stereo) the CPU
    // engine runs ONE NAM inference and feeds both channels. The moment the drives
    // differ it resyncs channel 1's NAM state from channel 0 and runs both. This
    // proves the whole channel-1 path — fast phase, the divergence resync, and the
    // diverged phase — is bit-identical to always inferring channel 1 independently.
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;
    const int mono_blocks = 6;
    const int div_blocks  = 6;
    const int nblocks = mono_blocks + div_blocks;

    std::vector<float> Lseq(nblocks * BLOCK), Rseq(nblocks * BLOCK);
    for (int b = 0; b < nblocks; ++b) {
        for (std::size_t i = 0; i < BLOCK; ++i) {
            const int n = b * static_cast<int>(BLOCK) + static_cast<int>(i);
            const float fn = static_cast<float>(n);
            if (b < mono_blocks) {                       // identical drive → fast path
                Lseq[n] = Rseq[n] = 0.4f * std::sin(0.05f * fn);
            } else {                                     // channels diverge
                Lseq[n] = 0.3f * std::cos(0.07f * fn);
                Rseq[n] = 0.35f * std::sin(0.031f * fn) + 0.1f;
            }
        }
    }

    // Feed distinct L/R through a fresh CPU-engine processor, concatenating both
    // output channels.
    auto run_stereo = [&](const std::vector<float>& L, const std::vector<float>& R,
                          std::vector<float>& outL, std::vector<float>& outR) {
        GpuNamProcessor proc;
        pulp::state::StateStore store;
        prepare_proc(proc, store, SR, BLOCK, /*engine=*/0.0f);
        REQUIRE_FALSE(proc.gpu_engine_active());
        std::vector<float> il(BLOCK), ir(BLOCK), ol(BLOCK), orr(BLOCK);
        pulp::midi::MidiBuffer mi, mo;
        pulp::format::ProcessContext pctx;
        pctx.sample_rate = SR;
        pctx.num_samples = static_cast<int>(BLOCK);
        const int nb = static_cast<int>(L.size() / BLOCK);
        for (int b = 0; b < nb; ++b) {
            for (std::size_t i = 0; i < BLOCK; ++i) {
                il[i] = L[b * BLOCK + i];
                ir[i] = R[b * BLOCK + i];
            }
            const float* ip[2] = {il.data(), ir.data()};
            float* op[2] = {ol.data(), orr.data()};
            pulp::audio::BufferView<const float> iv(ip, 2, BLOCK);
            pulp::audio::BufferView<float> ov(op, 2, BLOCK);
            proc.process(ov, iv, mi, mo, pctx);
            outL.insert(outL.end(), ol.begin(), ol.end());
            outR.insert(outR.end(), orr.begin(), orr.end());
        }
        proc.release();
    };

    std::vector<float> realL, realR;
    run_stereo(Lseq, Rseq, realL, realR);

    // Reference: feed channel 1's full input as mono. Channel 0 always infers, so
    // its output is the ground truth for what channel 1 must produce.
    std::vector<float> refL, refR;
    run_stereo(Rseq, Rseq, refL, refR);

    REQUIRE(realR.size() == refL.size());
    REQUIRE(realR.size() == static_cast<std::size_t>(nblocks) * BLOCK);

    // Channel 1 (fast path + resync + diverged inference) is bit-identical to the
    // always-inferred reference.
    for (std::size_t i = 0; i < realR.size(); ++i) {
        INFO("sample " << i);
        REQUIRE(realR[i] == refL[i]);
    }

    // Sanity: in the mono phase both output channels agree (one shared inference);
    // after divergence they genuinely differ.
    for (std::size_t i = 0; i < static_cast<std::size_t>(mono_blocks) * BLOCK; ++i)
        REQUIRE(realL[i] == realR[i]);
    double diverged = 0.0;
    for (std::size_t i = realR.size() - 2 * BLOCK; i < realR.size(); ++i)
        diverged += std::abs(realL[i] - realR[i]);
    REQUIRE(diverged > 1e-3);
}

TEST_CASE("GPU NAM noise gate attenuates a sub-threshold signal", "[nam]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;
    // ~ -29 dBFS sustained tone — well below a -12 dB gate threshold.
    const auto quiet = sine(BLOCK, 0.06f, 0.05f);

    const auto open = run_cpu(SR, BLOCK, 24, quiet,
                              [](auto& s) { s.set_value(kNoiseGateActive, 0.0f); });
    const auto gated = run_cpu(SR, BLOCK, 24, quiet, [](auto& s) {
        s.set_value(kNoiseGateActive, 1.0f);
        s.set_value(kNoiseGateThreshold, -12.0f);
    });

    double e_open = 0.0, e_gated = 0.0;
    for (float v : open)  { REQUIRE(std::isfinite(v)); e_open  += static_cast<double>(v) * v; }
    for (float v : gated) { REQUIRE(std::isfinite(v)); e_gated += static_cast<double>(v) * v; }
    INFO("open=" << e_open << " gated=" << e_gated);
    REQUIRE(e_open > 1e-8);               // the ungated amp produced real output
    REQUIRE(e_gated < e_open * 0.05);     // the gate cut the sub-threshold signal hard
}

TEST_CASE("GPU NAM EQ is transparent when flat and shapes tone when boosted", "[nam]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;
    // Two-tone drive: a low (~150 Hz) and a high (~6.9 kHz) partial.
    std::vector<float> sig(BLOCK);
    for (std::size_t i = 0; i < BLOCK; ++i)
        sig[i] = 0.3f * std::sin(0.02f * static_cast<float>(i))
               + 0.3f * std::sin(0.90f * static_cast<float>(i));

    const auto eq_off = run_cpu(SR, BLOCK, 24, sig,
                                [](auto& s) { s.set_value(kEQActive, 0.0f); });
    const auto eq_flat = run_cpu(SR, BLOCK, 24, sig, [](auto& s) {
        s.set_value(kEQActive, 1.0f);
        s.set_value(kToneBass, 5.0f); s.set_value(kToneMiddle, 5.0f);
        s.set_value(kToneTreble, 5.0f);
    });
    const auto eq_treble = run_cpu(SR, BLOCK, 24, sig, [](auto& s) {
        s.set_value(kEQActive, 1.0f);
        s.set_value(kToneBass, 5.0f); s.set_value(kToneMiddle, 5.0f);
        s.set_value(kToneTreble, 10.0f);
    });

    // A flat tone stack (5/5/5) is unity gain — the output matches EQ-off.
    REQUIRE(eq_off.size() == eq_flat.size());
    double num = 0, da = 0, db = 0;
    for (std::size_t i = 0; i < eq_off.size(); ++i) {
        num += static_cast<double>(eq_off[i]) * eq_flat[i];
        da  += static_cast<double>(eq_off[i]) * eq_off[i];
        db  += static_cast<double>(eq_flat[i]) * eq_flat[i];
    }
    const double xc = num / std::sqrt(da * db + 1e-30);
    INFO("flat-vs-off xcorr=" << xc);
    REQUIRE(xc > 0.999);

    // A +12 dB treble (high-shelf) boost lifts the high partial's energy.
    double e_flat = 0.0, e_treble = 0.0;
    for (float v : eq_flat)   { REQUIRE(std::isfinite(v)); e_flat   += static_cast<double>(v) * v; }
    for (float v : eq_treble) { REQUIRE(std::isfinite(v)); e_treble += static_cast<double>(v) * v; }
    INFO("flat_energy=" << e_flat << " treble_energy=" << e_treble);
    REQUIRE(e_treble > e_flat * 1.05);
}

TEST_CASE("GPU NAM Normalize retargets output to the model loudness", "[nam][normalize]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;

    // The make-up gain Normalize should apply is derived from the bundled model's
    // loudness metadata, retargeted to kNormalizeTargetDb and clamped.
    nam::NamRuntime ref;
    std::string err;
    REQUIRE(nam::load_nam_runtime(GPU_NAM_DEFAULT_MODEL_PATH, ref, &err));
    REQUIRE(ref.has_loudness());
    float exp_db = kNormalizeTargetDb - static_cast<float>(ref.loudness_db());
    exp_db = std::clamp(exp_db, -kNormalizeMaxAbsDb, kNormalizeMaxAbsDb);
    const double exp_ratio = std::pow(10.0f, exp_db / 20.0f);

    std::vector<float> sig(BLOCK);
    for (std::size_t i = 0; i < BLOCK; ++i)
        sig[i] = 0.4f * std::sin(0.05f * static_cast<float>(i));

    // Everything else identical — only Normalize toggles — so the per-sample
    // output ratio is exactly the make-up gain and the RMS ratio matches it.
    const auto off = run_cpu(SR, BLOCK, 24, sig,
                             [](auto& s) { s.set_value(kOutputMode, 0.0f); });
    const auto on = run_cpu(SR, BLOCK, 24, sig,
                            [](auto& s) { s.set_value(kOutputMode, 1.0f); });
    REQUIRE(off.size() == on.size());

    double e_on = 0.0, e_off = 0.0;
    for (std::size_t i = 0; i < off.size(); ++i) {
        e_on  += static_cast<double>(on[i]) * on[i];
        e_off += static_cast<double>(off[i]) * off[i];
    }
    REQUIRE(e_off > 1e-9);
    const double ratio = std::sqrt(e_on / e_off);
    INFO("ratio=" << ratio << " expected=" << exp_ratio);
    CHECK(std::abs(ratio - exp_ratio) < 0.01 * exp_ratio);
    CHECK(exp_ratio > 1.0);  // this capture is quieter than the target, so it boosts
}

TEST_CASE("GPU NAM Calibrated retargets to the user reference level", "[nam][normalize][calibrated]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;

    // Calibrated mode retargets the model's loudness to the USER's Cal Level
    // (kCalibrationLevel), not the fixed Normalized reference. Pick a hotter
    // reference than kNormalizeTargetDb and prove the end-to-end make-up matches
    // it — and is louder than Normalized would have been.
    constexpr float kCalLevel = -12.0f;  // hotter than the -18 Normalized target
    nam::NamRuntime ref;
    std::string err;
    REQUIRE(nam::load_nam_runtime(GPU_NAM_DEFAULT_MODEL_PATH, ref, &err));
    REQUIRE(ref.has_loudness());
    float exp_db = kCalLevel - static_cast<float>(ref.loudness_db());
    exp_db = std::clamp(exp_db, -kNormalizeMaxAbsDb, kNormalizeMaxAbsDb);
    const double exp_ratio = std::pow(10.0f, exp_db / 20.0f);

    std::vector<float> sig(BLOCK);
    for (std::size_t i = 0; i < BLOCK; ++i)
        sig[i] = 0.4f * std::sin(0.05f * static_cast<float>(i));

    // Raw baseline vs Calibrated (mode 2) at the user Cal Level.
    const auto off = run_cpu(SR, BLOCK, 24, sig,
                             [](auto& s) { s.set_value(kOutputMode, 0.0f); });
    const auto cal = run_cpu(SR, BLOCK, 24, sig, [](auto& s) {
        s.set_value(kOutputMode, 2.0f);
        s.set_value(kCalibrationLevel, kCalLevel);
    });
    REQUIRE(off.size() == cal.size());

    double e_cal = 0.0, e_off = 0.0;
    for (std::size_t i = 0; i < off.size(); ++i) {
        e_cal += static_cast<double>(cal[i]) * cal[i];
        e_off += static_cast<double>(off[i]) * off[i];
    }
    REQUIRE(e_off > 1e-9);
    const double ratio = std::sqrt(e_cal / e_off);
    INFO("cal ratio=" << ratio << " expected=" << exp_ratio);
    CHECK(std::abs(ratio - exp_ratio) < 0.01 * exp_ratio);
    // A -12 dBFS reference boosts more than the -18 Normalized target would.
    const double norm_ratio =
        std::pow(10.0f, std::clamp(kNormalizeTargetDb - static_cast<float>(ref.loudness_db()),
                                   -kNormalizeMaxAbsDb, kNormalizeMaxAbsDb) / 20.0f);
    CHECK(ratio > norm_ratio);
}

TEST_CASE("GPU NAM resamples around an off-rate host", "[nam][resample]") {
    // The bundled model is captured at some rate MSR. Driving it at MSR (matched,
    // no resampling) vs 2*MSR (resampling engaged around the model) with the same
    // tone must yield the same model response — proving the host<->model rate
    // conversion is wired correctly, not pitch-shifting or diverging.
    nam::NamRuntime probe;
    std::string err;
    REQUIRE(nam::load_nam_runtime(GPU_NAM_DEFAULT_MODEL_PATH, probe, &err));
    const double MSR = probe.sample_rate();
    REQUIRE(MSR > 0.0);

    const double freq = 300.0, dur = 0.25;
    auto sine = [&](double sr, std::size_t n) {
        std::vector<float> s(n);
        for (std::size_t i = 0; i < n; ++i)
            s[i] = 0.3f * std::sin(2.0 * M_PI * freq * static_cast<double>(i) / sr);
        return s;
    };

    // Reference: host == model rate (matched path, no resampling).
    const auto out_ref = run_stream_cpu(MSR, 512, sine(MSR, static_cast<std::size_t>(MSR * dur)));

    // Off-rate: host = 2x model rate — resampling engages around the model.
    const double host2 = 2.0 * MSR;
    const auto out2 = run_stream_cpu(host2, 512, sine(host2, static_cast<std::size_t>(host2 * dur)));

    for (float v : out2) REQUIRE(std::isfinite(v));
    for (float v : out2) REQUIRE(std::abs(v) < 10.0f);   // no resampler blow-up

    // Downsample the 2x output back to MSR to compare like-for-like.
    pulp::signal::Resampler down;
    down.prepare(host2, MSR, 1, out2.size());
    std::vector<float> out2d(down.max_output_for(out2.size()), 0.0f);
    out2d.resize(down.process_block_mono(out2.data(), out2.size(), out2d.data(), out2d.size()));

    REQUIRE(out_ref.size() > 3000);
    REQUIRE(out2d.size() > 3000);

    // Best-lag normalized cross-correlation over a mid window (skips warm-up and
    // the tail so the two latency-trimmed streams line up).
    auto mid = [](const std::vector<float>& v) {
        return std::vector<float>(v.begin() + v.size() / 4, v.begin() + 3 * v.size() / 4);
    };
    const auto a = mid(out_ref);
    const auto b = mid(out2d);
    const std::size_t len = std::min(a.size(), b.size());
    double best = -1.0;
    const int max_lag = 400;
    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        double num = 0, ea = 0, eb = 0;
        for (std::size_t i = max_lag; i + max_lag < len; ++i) {
            const float av = a[i];
            const float bv = b[static_cast<std::size_t>(static_cast<int>(i) + lag)];
            num += av * bv; ea += av * av; eb += bv * bv;
        }
        best = std::max(best, num / std::sqrt(ea * eb + 1e-20));
    }
    INFO("resample round-trip best xcorr = " << best);
    CHECK(best > 0.9);
}

TEST_CASE("GPU NAM cabinet swap is click-free", "[nam][ir]") {
    // Swapping cabinets while audio flows must not step-discontinuity the output:
    // the swapper hands the new IR to the running convolver in place, preserving
    // its overlap history. A hard engine swap (zeroed overlap) would spike the
    // sample-to-sample delta at the change.
    constexpr double SR = 48000.0;
    constexpr std::size_t BLOCK = 512;
    const auto dir = std::filesystem::temp_directory_path();
    const std::string irA = (dir / "gpu_nam_cab_a.wav").string();
    const std::string irB = (dir / "gpu_nam_cab_b.wav").string();
    // Two distinct cabinets: different early reflections + decay.
    std::vector<float> a(400, 0.0f), b(400, 0.0f);
    a[0] = 1.0f; a[37] = 0.6f; for (std::size_t i = 0; i < a.size(); ++i) a[i] += 0.3f * std::exp(-0.01f * i) * std::sin(0.2f * i);
    b[0] = 0.8f; b[91] = -0.7f; for (std::size_t i = 0; i < b.size(); ++i) b[i] += 0.4f * std::exp(-0.004f * i) * std::sin(0.05f * i);
    write_ir_wav(irA, a);
    write_ir_wav(irB, b);

    GpuNamProcessor proc;
    pulp::state::StateStore store;
    proc.set_state_store(&store);
    proc.define_parameters(store);
    store.set_value(kInputGain, 3.0f);
    store.set_value(kMix, 100.0f);
    store.set_value(kBypass, 0.0f);
    store.set_value(kEngine, 0.0f);
    proc.load_ir(irA);
    pulp::format::PrepareContext ctx;
    ctx.sample_rate = SR; ctx.max_buffer_size = static_cast<int>(BLOCK);
    ctx.input_channels = 2; ctx.output_channels = 2;
    proc.prepare(ctx);

    pulp::midi::MidiBuffer mi, mo;
    pulp::format::ProcessContext pctx; pctx.sample_rate = SR; pctx.num_samples = static_cast<int>(BLOCK);
    std::vector<float> l(BLOCK), r(BLOCK), ol(BLOCK), orr(BLOCK);
    std::vector<float> out;
    const int nblocks = 60;
    std::size_t swap_at = 0;
    double phase = 0.0;
    const double dp = 2.0 * M_PI * 330.0 / SR;
    for (int blk = 0; blk < nblocks; ++blk) {
        for (std::size_t i = 0; i < BLOCK; ++i) { l[i] = r[i] = 0.3f * std::sin(phase); phase += dp; }
        if (blk == nblocks / 2) {
            proc.load_ir(irB);                                   // audition a new cabinet
            std::this_thread::sleep_for(std::chrono::milliseconds(60));  // worker stages it
            swap_at = out.size();
        }
        const float* ip[2] = {l.data(), r.data()};
        float* op[2] = {ol.data(), orr.data()};
        pulp::audio::BufferView<const float> iv(ip, 2, BLOCK);
        pulp::audio::BufferView<float> ov(op, 2, BLOCK);
        proc.process(ov, iv, mi, mo, pctx);
        for (std::size_t i = 0; i < BLOCK; ++i) out.push_back(ol[i]);
    }
    proc.release();

    // Baseline max |Δ| over a steady pre-swap window; assert the post-swap stream
    // never exceeds a small multiple of it (a hard-reset click would spike far
    // higher). Skip the initial warm-up.
    auto max_delta = [&](std::size_t lo, std::size_t hi) {
        double m = 0.0;
        for (std::size_t i = lo + 1; i < hi && i < out.size(); ++i)
            m = std::max(m, std::abs(static_cast<double>(out[i]) - out[i - 1]));
        return m;
    };
    REQUIRE(swap_at > BLOCK * 4);
    const double baseline = max_delta(BLOCK * 2, swap_at - BLOCK);
    const double post = max_delta(swap_at, out.size());
    for (float v : out) REQUIRE(std::isfinite(v));
    INFO("baseline maxΔ=" << baseline << " post-swap maxΔ=" << post);
    REQUIRE(baseline > 0.0);
    CHECK(post < 4.0 * baseline);   // no step discontinuity at the cabinet swap

    std::filesystem::remove(irA);
    std::filesystem::remove(irB);
}

TEST_CASE("GPU NAM amps identically under any host block size", "[nam]") {
    // A real host feeds variable, often-smaller-than-internal blocks. The re-block
    // FIFO must make the amped output independent of that chunking; a fixed-block
    // processor that silently dry-passes or desyncs on an unaligned block would
    // pass every 512-aligned test yet be wrong in a DAW. Prove it: the same signal
    // streamed at awkward block sizes matches the internal-block reference
    // sample-for-sample, and is genuinely amped (not silently the dry signal).
    constexpr double SR = 48000.0;
    const std::size_t IB = GpuNamProcessor::kInternalBlock;
    const auto sig = sine(IB * 12);  // long enough for a steady window past latency

    const auto ref = run_stream_cpu(SR, IB, sig);       // internal-block schedule
    REQUIRE(ref.size() > IB * 3);

    // The reference is a real amped signal, not a dry/silent pass-through.
    double e_ref = 0.0;
    for (float v : ref) { REQUIRE(std::isfinite(v)); e_ref += static_cast<double>(v) * v; }
    REQUIRE(e_ref > 1e-3);

    for (std::size_t hb : {std::size_t{1}, std::size_t{32}, std::size_t{96},
                           std::size_t{300}, std::size_t{480}, std::size_t{777}}) {
        const auto got = run_stream_cpu(SR, hb, sig);
        const std::size_t len = std::min(ref.size(), got.size());
        REQUIRE(len > IB * 2);
        // Compare a steady window past the FIFO priming region.
        double max_abs = 0.0;
        for (std::size_t i = IB; i < len; ++i)
            max_abs = std::max(max_abs,
                               std::abs(static_cast<double>(got[i]) - ref[i]));
        INFO("host_block=" << hb << " max_abs_diff=" << max_abs);
        REQUIRE(max_abs < 1e-4);   // chunk-schedule-independent to the sample
    }
}

TEST_CASE("GPU NAM applies a loaded cabinet IR", "[nam]") {
    constexpr double SR = 48000.0;
    const std::size_t IB = GpuNamProcessor::kInternalBlock;
    const auto sig = sine(IB * 8);

    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path();
    const auto id_path = (dir / "gpu_nam_ir_identity.wav").string();
    const auto lp_path = (dir / "gpu_nam_ir_decay.wav").string();
    write_ir_wav(id_path, {1.0f});                          // identity (stays [1] after norm)
    write_ir_wav(lp_path, {0.5f, 0.5f, 0.25f, 0.12f});      // short decaying kernel

    const auto no_ir = run_stream_cpu_ir(SR, IB, sig, "");
    const auto ident = run_stream_cpu_ir(SR, IB, sig, id_path);
    const auto decay = run_stream_cpu_ir(SR, IB, sig, lp_path);
    REQUIRE(no_ir.size() > IB * 2);
    REQUIRE(ident.size() == no_ir.size());
    REQUIRE(decay.size() == no_ir.size());

    // A unit-impulse IR is transparent (0-latency overlap-save): the wet chain is
    // unchanged. Proves the convolver is wired correctly, not just "does something".
    double max_id = 0.0;
    for (std::size_t i = IB; i < no_ir.size(); ++i)
        max_id = std::max(max_id, std::abs(static_cast<double>(ident[i]) - no_ir[i]));
    INFO("identity-IR max_abs_diff=" << max_id);
    REQUIRE(max_id < 1e-3);

    // A real (multi-tap) IR is actually convolved in: the output changes and
    // stays finite.
    double diff_e = 0.0, ref_e = 0.0;
    for (std::size_t i = IB; i < no_ir.size(); ++i) {
        REQUIRE(std::isfinite(decay[i]));
        const double d = static_cast<double>(decay[i]) - no_ir[i];
        diff_e += d * d;
        ref_e += static_cast<double>(no_ir[i]) * no_ir[i];
    }
    INFO("decay-IR diff/ref energy=" << diff_e / (ref_e + 1e-30));
    REQUIRE(diff_e > ref_e * 1e-4);

    // A bad / missing IR path fails gracefully: no crash, the IR is simply not
    // applied, so the output matches the no-IR path. (Guards the worker/prepare
    // exception firewall around the decode.)
    const auto bad = run_stream_cpu_ir(SR, IB, sig, "/nonexistent/pulp/no_such_ir.wav");
    REQUIRE(bad.size() == no_ir.size());
    double max_bad = 0.0;
    for (std::size_t i = IB; i < no_ir.size(); ++i)
        max_bad = std::max(max_bad, std::abs(static_cast<double>(bad[i]) - no_ir[i]));
    INFO("bad-IR-path max_abs_diff=" << max_bad);
    REQUIRE(max_bad < 1e-3);
}

TEST_CASE("GPU NAM load_model rebuilds without NaNs", "[nam]") {
    constexpr std::size_t BLOCK = GpuNamProcessor::kInternalBlock;
    constexpr double SR = 48000.0;

    GpuNamProcessor proc;
    pulp::state::StateStore store;
    prepare_proc(proc, store, SR, BLOCK, /*engine=*/0.0f);

    Driver d(proc, BLOCK, SR);
    // Reload the same default model and let the worker rebuild the engines.
    proc.load_model("");   // empty → default model path
    d.pump(20, 5);

    double energy = 0.0;
    for (int b = 0; b < 8; ++b) {
        std::vector<float> in(BLOCK);
        for (std::size_t i = 0; i < BLOCK; ++i)
            in[i] = 0.5f * std::sin(0.06f * static_cast<float>(b * BLOCK + i));
        const auto o = d.block_io(in, 0);
        for (float v : o) { REQUIRE(std::isfinite(v)); energy += static_cast<double>(v) * v; }
    }
    REQUIRE(energy > 1e-4);
    proc.release();
}

// The wet path carries the model's DC (a capture's steady response to any input is
// a nonzero DC from its biases/asymmetry). The processor's output DC blocker must
// remove it. Premise (measured from the engine) + outcome (measured at the output).
TEST_CASE("Wet output DC offset is removed", "[gpu-nam][plugin][dc]") {
    constexpr double SR = 48000.0;
    constexpr std::size_t BLOCK = 512;
    constexpr float kDcIn = 0.3f;

    // Premise: the model's steady response to a constant input is itself a nonzero DC.
    nam::NamRuntime rt;
    std::string err;
    REQUIRE(nam::load_nam_runtime(gpu_nam_default_model_path(), rt, &err));
    rt.prewarm();
    for (int i = 0; i < 8000; ++i) rt.process_sample(kDcIn);   // reach steady state
    double raw = 0.0;
    const int meas = 4000;
    for (int i = 0; i < meas; ++i) raw += rt.process_sample(kDcIn);
    raw /= meas;
    REQUIRE(std::abs(raw) > 1e-3);   // the model genuinely carries DC to remove

    // Outcome: full wet, gate off, constant input -> the output settles to ~0 mean.
    const std::vector<float> dc(BLOCK, kDcIn);
    const auto out = run_cpu(SR, BLOCK, 40, dc,
                             [](auto& s) { s.set_value(kNoiseGateActive, 0.0f); });
    double mean = 0.0;
    const std::size_t tail = out.size() / 4;                   // past prewarm + HP settle
    for (std::size_t i = out.size() - tail; i < out.size(); ++i) mean += out[i];
    mean /= static_cast<double>(tail);
    CHECK(std::abs(mean) < 1e-3);                              // DC removed
    CHECK(std::abs(mean) < 0.2 * std::abs(raw));               // << the raw model DC
}

// The file-slot prev/next browse controls (and the clear glyph) are backed by the
// pure path helpers in gpu_nam_paths.hpp. Exercise them directly with a temp
// directory so the model/IR cycling UX is proven without a live window.
TEST_CASE("GPU NAM file browse: list_files filters by extension, sorted", "[nam][browse]") {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "gpu-nam-browse-list";
    fs::remove_all(dir);
    fs::create_directories(dir);
    for (const char* n : {"b.nam", "a.nam", "c.NAM", "note.txt", "cab.wav"})
        std::ofstream(dir / n) << "x";

    const auto nams = gpu_nam_list_files(dir.string(), {"nam"});
    REQUIRE(nams.size() == 3);                                  // .txt/.wav excluded
    CHECK(gpu_nam_basename(nams[0]) == "a.nam");                // sorted by name
    CHECK(gpu_nam_basename(nams[1]) == "b.nam");
    CHECK(gpu_nam_basename(nams[2]) == "c.NAM");                // case-insensitive ext

    CHECK(gpu_nam_list_files(dir.string(), {"wav", "aiff"}).size() == 1);
    CHECK(gpu_nam_list_files((dir / "does-not-exist").string(), {"nam"}).empty());
    fs::remove_all(dir);
}

TEST_CASE("GPU NAM file browse: neighbor cycles and wraps", "[nam][browse]") {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "gpu-nam-browse-neighbor";
    fs::remove_all(dir);
    fs::create_directories(dir);
    for (const char* n : {"a.nam", "b.nam", "c.nam"})
        std::ofstream(dir / n) << "x";
    const std::string a = (dir / "a.nam").string();
    const std::string b = (dir / "b.nam").string();
    const std::string c = (dir / "c.nam").string();

    CHECK(gpu_nam_neighbor_file(a, {"nam"}, +1) == b);          // next
    CHECK(gpu_nam_neighbor_file(b, {"nam"}, +1) == c);
    CHECK(gpu_nam_neighbor_file(c, {"nam"}, +1) == a);          // wraps forward
    CHECK(gpu_nam_neighbor_file(a, {"nam"}, -1) == c);          // wraps backward
    CHECK(gpu_nam_neighbor_file(b, {"nam"}, -1) == a);          // prev

    // A path not in the listing (stale selection) snaps to the first match.
    CHECK(gpu_nam_neighbor_file((dir / "gone.nam").string(), {"nam"}, +1) == a);

    // Single-file directory returns the same file; empty directory returns "".
    const auto solo = fs::temp_directory_path() / "gpu-nam-browse-solo";
    fs::remove_all(solo);
    fs::create_directories(solo);
    const std::string only = (solo / "only.nam").string();
    std::ofstream(only) << "x";
    CHECK(gpu_nam_neighbor_file(only, {"nam"}, +1) == only);
    CHECK(gpu_nam_neighbor_file(only, {"wav"}, +1).empty());    // no match at all

    fs::remove_all(dir);
    fs::remove_all(solo);
}

// clear_model()/clear_ir() are thin wrappers that reset the slot to its
// no-user-selection state; verify the observable UI-facing flags flip without a
// live audio thread (load_* only latches the request; no prepare() needed here).
TEST_CASE("GPU NAM clear resets the model + IR slots", "[nam][browse]") {
    GpuNamProcessor proc;
    // Model: a user selection marks user_model_loaded(); clear reverts it.
    proc.load_model("/nonexistent/custom.nam");
    CHECK(proc.user_model_loaded());
    proc.clear_model();
    CHECK_FALSE(proc.user_model_loaded());                      // back to bundled default

    // IR: user_ir_loaded() only becomes true once an IR is actually built on the
    // audio path, so with no prepare()/process() it stays false; clear_ir() must
    // still leave it false and not crash.
    proc.clear_ir();
    CHECK_FALSE(proc.user_ir_loaded());
}

// End-to-end proof that each real model capture in src/models/ loads through the
// full processor and passes audio: finite, non-silent, and materially reshaped vs
// the dry input. This is the regression guard that a loader-hardening change never
// silently rejects (or mis-loads) a legitimate real .nam file.
TEST_CASE("GPU NAM streams real model captures end-to-end", "[nam][e2e]") {
    namespace fs = std::filesystem;
    constexpr double SR = 48000.0;
    const std::size_t N = 4096;
    const auto dry = drive_signal(SR, N);
    const double dry_rms = signal_rms(dry);

    struct Fixture { const char* file; nam::NamRuntime::Arch arch; const char* name; };
    const Fixture fixtures[] = {
        {"example.nam", nam::NamRuntime::Arch::WaveNet, "WaveNet"},
        {"wavenet_a1_standard.nam", nam::NamRuntime::Arch::WaveNet, "WaveNet"},
        {"lstm.nam", nam::NamRuntime::Arch::Lstm, "LSTM"},
    };

    for (const auto& fx : fixtures) {
        const std::string path = (fs::path(GPU_NAM_MODELS_DIR) / fx.file).string();
        if (!fs::exists(path)) {
            WARN("missing real fixture " << path << "; skipping");
            continue;
        }
        INFO("model = " << fx.file);

        // The loader must accept it and classify the architecture we expect.
        nam::NamRuntime probe;
        std::string err;
        REQUIRE(nam::load_nam_runtime(path, probe, &err));
        CHECK(probe.arch() == fx.arch);

        // Full-processor audio must come out finite, audible, and changed by the amp.
        const auto wet = run_stream_cpu_model_ir(SR, 512, dry, path, /*ir=*/std::string());
        REQUIRE(all_finite(wet));
        REQUIRE(wet.size() > N / 2);
        const double wet_rms = signal_rms(wet);
        CHECK(wet_rms > 1e-4);                                  // not silent
        // The amp reshapes the tone: the trimmed output is not a bit-copy of the dry
        // input over the overlapping region.
        bool differs = false;
        const std::size_t cmp = std::min(wet.size(), dry.size());
        for (std::size_t i = 0; i < cmp; ++i)
            if (std::abs(wet[i] - dry[i]) > 1e-5f) { differs = true; break; }
        CHECK(differs);
        INFO("dry_rms=" << dry_rms << " wet_rms=" << wet_rms);
    }
}

// End-to-end proof that a real impulse response (src/models/cabinet.wav) loads via
// the plugin's IR reader and colors the amped signal: with the cab engaged the
// output stays finite and non-silent and differs from the no-IR amped tone.
TEST_CASE("GPU NAM streams a real IR end-to-end", "[nam][e2e][ir]") {
    namespace fs = std::filesystem;
    constexpr double SR = 48000.0;
    const std::size_t N = 4096;

    const std::string model = (fs::path(GPU_NAM_MODELS_DIR) / "example.nam").string();
    const std::string ir = (fs::path(GPU_NAM_MODELS_DIR) / "cabinet.wav").string();
    if (!fs::exists(model) || !fs::exists(ir)) {
        WARN("missing real model/IR fixture; skipping");
        return;
    }

    const auto dry = drive_signal(SR, N);
    const auto no_ir = run_stream_cpu_model_ir(SR, 512, dry, model, std::string());
    const auto with_ir = run_stream_cpu_model_ir(SR, 512, dry, model, ir);

    REQUIRE(all_finite(with_ir));
    REQUIRE(with_ir.size() > N / 2);
    CHECK(signal_rms(with_ir) > 1e-4);                         // cab output is audible

    // The cabinet must actually change the tone relative to the amp-only signal.
    const std::size_t cmp = std::min(no_ir.size(), with_ir.size());
    REQUIRE(cmp > 0);
    double diff = 0.0;
    for (std::size_t i = 0; i < cmp; ++i)
        diff += std::abs(static_cast<double>(with_ir[i]) - static_cast<double>(no_ir[i]));
    CHECK(diff / static_cast<double>(cmp) > 1e-4);             // IR colors the signal
}

// The processor's CPU engine handles every supported architecture, not just the
// two we ship real captures for. Build a valid ConvNet and a valid Linear capture,
// stream them through the full processor, and require finite, non-silent, reshaped
// audio — so a loader change to any arch is caught end-to-end, not just in a unit.
TEST_CASE("GPU NAM streams ConvNet and Linear captures end-to-end", "[nam][e2e]") {
    namespace fs = std::filesystem;
    constexpr double SR = 48000.0;
    const std::size_t N = 4096;
    const auto dry = drive_signal(SR, N);

    auto stream_and_check = [&](const std::string& json, const char* tag,
                                nam::NamRuntime::Arch arch) {
        INFO("arch = " << tag);
        const auto path = (fs::temp_directory_path() / (std::string("gpu_nam_e2e_") + tag + ".nam")).string();
        std::ofstream(path, std::ios::binary) << json;
        nam::NamRuntime probe;
        std::string err;
        REQUIRE(nam::load_nam_runtime(path, probe, &err));
        CHECK(probe.arch() == arch);
        const auto wet = run_stream_cpu_model_ir(SR, 512, dry, path, std::string());
        REQUIRE(all_finite(wet));
        CHECK(signal_rms(wet) > 1e-4);
        bool differs = false;
        const std::size_t cmp = std::min(wet.size(), dry.size());
        for (std::size_t i = 0; i < cmp; ++i)
            if (std::abs(wet[i] - dry[i]) > 1e-5f) { differs = true; break; }
        CHECK(differs);
        fs::remove(path);
    };

    // ConvNet: 1 channel, one dilation-1 block (ReLU), 1x1 head. Weights are
    // conv(1*1*2 + bias) then head(1*1 + bias) = 5 floats.
    const std::string convnet =
        "{\"architecture\":\"ConvNet\",\"sample_rate\":48000,\"config\":{"
        "\"channels\":1,\"dilations\":[1],\"batchnorm\":false,\"activation\":\"ReLU\","
        "\"in_channels\":1,\"out_channels\":1},\"weights\":[1.0,2.0,0.5,1.0,0.0]}";
    stream_and_check(convnet, "convnet", nam::NamRuntime::Arch::ConvNet);

    // Linear: a 1-tap FIR with bias — output = 0.1 + 0.5*x. Weights are rf(1)+bias(1).
    const std::string linear =
        "{\"architecture\":\"Linear\",\"sample_rate\":48000,\"config\":{"
        "\"receptive_field\":1,\"bias\":true,\"in_channels\":1,\"out_channels\":1},"
        "\"weights\":[0.5,0.1]}";
    stream_and_check(linear, "linear", nam::NamRuntime::Arch::Linear);
}
