#pragma once

// GPU NAM — a Neural Amp Modeler player that runs open-source `.nam` WaveNet
// captures through Pulp's GPU audio runtime.
//
// The live audio path defaults to the RT-safe inline CPU engine: the exact NAM
// WaveNet oracle (nam_model.hpp), applied per channel sample-by-sample. The input
// is re-blocked into fixed kInternalBlock chunks so the inline CPU engine and the
// GPU engine share one re-blocking FIFO and one fixed reported latency.
//
// An optional, default-OFF GPU engine (the Engine knob) routes the same fixed
// blocks through the real GPU audio runtime: a GpuNamCloudNode (one fused GPU
// `wavenet_forward` per channel) driven by gpu_audio::GpuAudioTransport on a non-RT
// worker. The GPU forward blocks on the device readback, so it runs only on the
// transport worker; the audio thread calls the lock-free transport.process() and,
// on a worker miss, the node's CpuFallback runs the exact CPU oracle. If no GPU
// device exists the processor stays on the inline CPU engine and always works.
//
// Engine (CPU<->GPU) and the loaded model are switchable LIVE without a reload:
// a background worker builds the requested engine stack off-thread and publishes
// it through atomic pointers (gpu_active_ for the GPU transport, cpu_active_ for
// the inline CPU engine) that the audio thread loads each block. The previously
// active stack is retired and freed one rebuild later, so the audio thread never
// holds a stack as it is freed — it never allocates or frees. Reported latency is
// FIXED for the prepared lifetime (kInternalBlock plus the GPU transport's delay
// when a device exists), applied to both engines so a live switch keeps the
// host's PDC correct and dry/wet phase-aligned. See gpu_engine_active().
//
// The native GPU front-end (input→output transfer curve + gain/mix/engine
// controls, rendered through canvas/Skia/Dawn) is in gpu_nam_ui.hpp.

#include "gpu_nam.hpp"
#include "gpu_nam_cloud_node.hpp"
#include "nam_model.hpp"
#include "nam_retire_list.hpp"
#include "gpu_nam_license.hpp"
#include "nam_runtime.hpp"
#include "output_calibration.hpp"

// The GPU NAM plugin's engine is WaveNet-specific (the fused GPU forward uploads a
// NamModel). A build that compiles WaveNet out of the inference substrate cannot
// build the plugin — fail fast with a clear message instead of a confusing missing
// -symbol/void* error. LSTM/Linear are optional; they run on the generic CPU path.
#if !GPU_NAM_WITH_WAVENET
#error "GpuNamProcessor requires WaveNet (GPU_NAM_WITH_WAVENET). Build the CPU inference substrate (nam_runtime.hpp) directly for non-WaveNet-only configurations."
#endif

// Plugin version, injected as a compile define from the single GPU_NAM_VERSION
// source in src/CMakeLists.txt so the descriptor can't drift from the bundle
// plists. Falls back for header-only / non-CMake builds.
#ifndef GPU_NAM_VERSION_STRING
#define GPU_NAM_VERSION_STRING "0.0.0-dev"
#endif

#include <pulp/format/processor.hpp>
#include <pulp/gpu_audio/gpu_audio_transport.hpp>
#include <pulp/runtime/log.hpp>
#include <pulp/audio/impulse_response.hpp>
#include <pulp/signal/biquad.hpp>
#include <pulp/signal/convolver.hpp>
#include <pulp/signal/dc_blocker.hpp>
#include <pulp/signal/noise_gate.hpp>
#include <pulp/signal/resampler.hpp>
#include <pulp/signal/scoped_flush_denormals.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gpu_nam_paths.hpp"

namespace pulp::view { class View; }

namespace pulp::examples {

enum GpuNamParams : state::ParamID {
    kInputGain  = 1,  // drive into the model, dB
    kOutputGain = 2,  // output gain, dB
    kMix        = 3,  // dry/wet, % — retained for state compat; face UI is 100% wet
    kEngine     = 4,  // 0 = CPU (default), 1 = GPU
    kBypass     = 5,
    // NAM-faithful control set. The noise gate runs on the drive before the
    // model; the Bass/Middle/Treble tone stack runs on the model output.
    kNoiseGateThreshold = 6,  // gate threshold, dB
    kToneBass           = 7,  // low shelf, 0..10 (5 = flat)
    kToneMiddle         = 8,  // mid peak,  0..10 (5 = flat)
    kToneTreble         = 9,  // high shelf,0..10 (5 = flat)
    kNoiseGateActive    = 10, // 0 = off, 1 = on
    kEQActive           = 11, // 0 = off (tone stack bypassed), 1 = on
    // Output level handling. Widened from the old binary Normalize (0 = off,
    // 1 = on): 0 still means Raw (no retarget) and 1 still means Normalized, so a
    // saved session keeps its meaning; 2 adds Calibrated (retarget the model's
    // loudness to the user's kCalibrationLevel instead of the fixed reference).
    kOutputMode         = 12,
    kCalibrationLevel   = 13, // Calibrated-mode target loudness, dBFS
};

// Target output loudness (dBFS) the Normalize mode aims for, and the maximum
// correction it will apply in either direction (models without loudness metadata,
// or with an extreme value, get no more than this much gain/cut).
inline constexpr float kNormalizeTargetDb = -18.0f;
inline constexpr float kNormalizeMaxAbsDb = 24.0f;

// Static input→output transfer ("amp character") curve. kCurvePoints output
// levels for inputs swept linearly across [-kCurveRange, +kCurveRange].
inline constexpr int kCurvePoints = 97;
inline constexpr float kCurveRange = 1.0f;
using TransferCurve = std::array<float, kCurvePoints>;

// Path / resource resolution (default model, installed sample content, UI assets)
// lives in gpu_nam_paths.hpp, shared with the editor's file choosers.

class GpuNamProcessor : public format::Processor {
public:
    // Fixed engine block. The GPU NAM forward is prepared at this size, so an
    // internal re-blocking FIFO chunks the host stream (variable, often smaller)
    // into kInternalBlock blocks for both engines. 512 samples ≈ 10.7 ms at
    // 48 kHz; the re-block FIFO adds this much latency, reported for host PDC.
    static constexpr std::size_t kInternalBlock = 512;

    // Engine=Auto offloads to the GPU only when the measured CPU inference exceeds
    // this fraction of one internal block's real-time budget.
    static constexpr double kAutoGpuBudgetFraction = 0.5;

    ~GpuNamProcessor() override { stop_worker(); }

    format::PluginDescriptor descriptor() const override {
        return {
            .name = "GPU NAM",
            .manufacturer = "Pulp",
            .bundle_id = "com.pulp.gpunam",
            .version = GPU_NAM_VERSION_STRING,
            .category = format::PluginCategory::Effect,
            .input_buses = {{"Audio In", 2}},
            .output_buses = {{"Audio Out", 2}},
        };
    }

    void define_parameters(state::StateStore& store) override {
        // Ranges mirror NeuralAmpModelerPlugin's face controls.
        store.add_parameter({.id = kInputGain,  .name = "Input",  .unit = "dB",
                             .range = {-20.0f, 20.0f, 0.0f, 0.0f}});
        store.add_parameter({.id = kOutputGain, .name = "Output", .unit = "dB",
                             .range = {-40.0f, 40.0f, 0.0f, 0.0f}});
        store.add_parameter({.id = kMix, .name = "Mix", .unit = "%",
                             .range = {0.0f, 100.0f, 100.0f, 0.1f}});
        // Engine: 0 = inline CPU oracle (default), 1 = GPU runtime, 2 = Auto.
        // CPU is the default — the live GPU path stays opt-in. Auto benchmarks the
        // CPU cost once at prepare and offloads to the GPU only when the CPU can't
        // comfortably meet the block's real-time budget (see resolve_auto_engine).
        store.add_parameter({.id = kEngine, .name = "Engine", .unit = "",
                             .range = {0.0f, 2.0f, 0.0f, 1.0f}});
        store.add_parameter({.id = kBypass, .name = "Bypass", .unit = "",
                             .range = {0.0f, 1.0f, 0.0f, 1.0f}});
        store.add_parameter({.id = kNoiseGateThreshold, .name = "Gate", .unit = "dB",
                             .range = {-100.0f, 0.0f, -80.0f, 0.0f}});
        store.add_parameter({.id = kToneBass,   .name = "Bass",   .unit = "",
                             .range = {0.0f, 10.0f, 5.0f, 0.0f}});
        store.add_parameter({.id = kToneMiddle, .name = "Middle", .unit = "",
                             .range = {0.0f, 10.0f, 5.0f, 0.0f}});
        store.add_parameter({.id = kToneTreble, .name = "Treble", .unit = "",
                             .range = {0.0f, 10.0f, 5.0f, 0.0f}});
        store.add_parameter({.id = kNoiseGateActive, .name = "Noise Gate", .unit = "",
                             .range = {0.0f, 1.0f, 1.0f, 1.0f}});
        store.add_parameter({.id = kEQActive, .name = "EQ", .unit = "",
                             .range = {0.0f, 1.0f, 1.0f, 1.0f}});
        // Output mode: 0 = Raw (default; the shipped level is exactly the model's),
        // 1 = Normalized (retarget the model's loudness to the fixed reference),
        // 2 = Calibrated (retarget to the user's Cal Level). Stepped 0..2; 0/1 keep
        // the old binary Normalize meaning so saved sessions are preserved.
        store.add_parameter({.id = kOutputMode, .name = "Output Mode", .unit = "",
                             .range = {0.0f, 2.0f, 0.0f, 1.0f}});
        // Calibrated-mode reference loudness (dBFS); only affects Calibrated mode.
        store.add_parameter({.id = kCalibrationLevel, .name = "Cal Level", .unit = "dB",
                             .range = {-36.0f, 0.0f, kNormalizeTargetDb, 0.0f}});
    }

    // Total latency, FIXED at prepare() for the prepared lifetime: the re-block
    // FIFO (kInternalBlock) plus, whenever a GPU device exists, the GPU
    // transport's fixed delay — reported for BOTH engines so a live Engine switch
    // never moves the reported latency (the host's PDC stays correct and dry/wet
    // stay phase-aligned). When no GPU device exists the latency is kInternalBlock.
    int latency_samples() const override { return latency_samples_; }

    /// Set the .nam model loaded for a fresh path at runtime. Thread-safe; the
    /// background worker rebuilds both engines off the audio thread (no audio-
    /// thread alloc/free). Returns immediately; the swap lands a few worker ticks
    /// later. Pass an empty path to (re)load the default model.
    void load_model(const std::string& path) {
        std::lock_guard<std::mutex> lock(model_req_mutex_);
        requested_model_path_ = path.empty() ? gpu_nam_default_model_path() : path;
        // A non-empty path is a real user selection; empty resets to the bundled
        // default. The file-slot UI shows the prompt until a user picks a model.
        user_model_loaded_.store(!path.empty(), std::memory_order_release);
        model_req_generation_.fetch_add(1, std::memory_order_release);
    }

    /// True once a user has selected a model via load_model(); false for the
    /// bundled default. UI-thread only. Drives the file-slot prompt vs name.
    bool user_model_loaded() const {
        return user_model_loaded_.load(std::memory_order_acquire);
    }

    /// Browse to the next (`delta` >= 0) / previous (`delta` < 0) `.nam` in the
    /// current model's folder, sorted by name and wrapping. With no user model yet,
    /// starts from the installed sample Models folder. No-op when nothing matches.
    /// UI/main-thread only; the swap itself is RT-safe via load_model().
    void cycle_model(int delta) {
        const bool user_loaded = user_model_loaded_.load(std::memory_order_acquire);
        std::string current;
        {
            std::lock_guard<std::mutex> lock(model_req_mutex_);
            current = requested_model_path_;
        }
        std::string next;
        if (user_loaded && !current.empty()) {
            next = gpu_nam_neighbor_file(current, {"nam"}, delta);
        } else {
            const auto files = gpu_nam_list_files(gpu_nam_content_subdir("Models"), {"nam"});
            if (!files.empty()) next = delta >= 0 ? files.front() : files.back();
        }
        if (!next.empty()) load_model(next);
    }
    /// Drop the user's model selection and revert to the bundled default.
    void clear_model() { load_model(std::string()); }

    /// Load (or clear, with an empty path) a cabinet impulse response. The IR is
    /// decoded + built off the audio thread and swapped in RT-safely; it sits
    /// after the tone stack in the wet chain, matching the reference's order.
    void load_ir(const std::string& path) {
        std::lock_guard<std::mutex> lock(ir_req_mutex_);
        requested_ir_path_ = path;
        ir_req_generation_.fetch_add(1, std::memory_order_release);
        // The zero-latency CPU path carries no convolver (an IR would reintroduce
        // the re-block latency it exists to avoid). A cabinet requested while it is
        // active is staged and engages on the next prepare (reopen / rate change).
        if (direct_cpu_ && !path.empty())
            runtime::log_info("GPU NAM: cabinet IR staged; it engages after the "
                              "plugin is reopened while the zero-latency CPU path is active.");
    }
    /// True when an IR is actually applied (built + published), NOT merely
    /// requested — so a pending or failed load reads as "no IR", matching what
    /// the audio path is doing. UI-thread only.
    bool user_ir_loaded() const {
        return ir_active_.load(std::memory_order_acquire) != nullptr;
    }
    /// Display name of the loaded IR (filename), or "" when none. UI-thread only.
    std::string ir_name() const {
        std::lock_guard<std::mutex> lock(ir_mutex_);
        return ir_name_;
    }

    /// Browse to the next (`delta` >= 0) / previous (`delta` < 0) cabinet IR in the
    /// current IR's folder, sorted by name and wrapping. With no IR loaded yet,
    /// starts from the installed sample Cabinets folder. No-op when nothing matches.
    /// UI/main-thread only; the swap itself is click-free + RT-safe via load_ir().
    void cycle_ir(int delta) {
        std::string current;
        {
            std::lock_guard<std::mutex> lock(ir_req_mutex_);
            current = requested_ir_path_;
        }
        static const std::vector<std::string> kIrExts{"wav", "aiff", "aif", "flac"};
        std::string next;
        if (!current.empty()) {
            next = gpu_nam_neighbor_file(current, kIrExts, delta);
        } else {
            const auto files = gpu_nam_list_files(gpu_nam_content_subdir("Cabinets"), kIrExts);
            if (!files.empty()) next = delta >= 0 ? files.front() : files.back();
        }
        if (!next.empty()) load_ir(next);
    }
    /// Remove the current cabinet IR (reverts to the dry, IR-off path).
    void clear_ir() { load_ir(std::string()); }

    /// True when the live audio path is actually the GPU engine (Engine=GPU is
    /// requested AND a GPU device is available AND the transport is published).
    bool gpu_engine_active() const {
        return gpu_engine_active_.load(std::memory_order_acquire);
    }

    /// The engine that Engine=Auto resolved to at prepare(): 0 = CPU, 1 = GPU.
    /// Meaningful only while the Engine parameter is Auto. UI/main-thread only.
    int auto_resolved_engine() const {
        return auto_resolved_engine_.load(std::memory_order_relaxed);
    }

    /// The live GPU backend ("Metal"/"D3D12"/"Vulkan") when the GPU engine is
    /// active, else "". UI/main-thread only (takes the worker mutex).
    std::string gpu_backend() const {
        std::lock_guard<std::mutex> lock(stack_mutex_);
        if (!gpu_engine_active() || !current_stack_ || !current_stack_->node)
            return std::string();
        return current_stack_->node->backend();
    }

    /// Live {GPU blocks produced, blocks missed (CPU-filled)}. UI/main-thread only.
    std::pair<std::uint64_t, std::uint64_t> gpu_block_stats() const {
        std::lock_guard<std::mutex> lock(stack_mutex_);
        if (!gpu_engine_active() || !current_stack_ || !current_stack_->transport)
            return {0, 0};
        const auto s = current_stack_->transport->stats();
        return {s.produced_blocks, s.miss_blocks};
    }

    /// Live GPU cost: {last, average} wall-clock microseconds per block.
    std::pair<double, double> gpu_block_us() const {
        std::lock_guard<std::mutex> lock(stack_mutex_);
        if (!gpu_engine_active() || !current_stack_ || !current_stack_->transport)
            return {0.0, 0.0};
        const auto s = current_stack_->transport->stats();
        return {s.last_block_us, s.avg_block_us};
    }

    /// One coherent snapshot of the live GPU engine for the UI status line, taken
    /// under a SINGLE lock so the fields can't disagree across a repaint.
    /// `budget_us` is one GPU block's real-time budget on THIS device + sample
    /// rate; `rt_percent` is the measured average cost as a percentage of it.
    struct GpuStatus {
        bool active = false;
        std::string backend;
        std::uint64_t blocks = 0;
        std::uint64_t misses = 0;
        double avg_us = 0.0;
        double budget_us = 0.0;
        double rt_percent = 0.0;
    };
    GpuStatus gpu_status() const {
        std::lock_guard<std::mutex> lock(stack_mutex_);
        GpuStatus g;
        g.active = gpu_engine_active();
        if (!g.active || !current_stack_) return g;
        if (current_stack_->node) g.backend = current_stack_->node->backend();
        if (current_stack_->transport) {
            const auto s = current_stack_->transport->stats();
            g.blocks = s.produced_blocks;
            g.misses = s.miss_blocks;
            g.avg_us = s.avg_block_us;
        }
        if (sample_rate_ > 0.0) {
            g.budget_us = static_cast<double>(kInternalBlock) / sample_rate_ * 1e6;
            if (g.budget_us > 0.0) g.rt_percent = g.avg_us / g.budget_us * 100.0;
        }
        return g;
    }

    /// Display name of the loaded model (its filename). UI/main-thread only.
    std::string model_name() const {
        std::lock_guard<std::mutex> lock(model_mutex_);
        return model_name_;
    }

    /// Architecture of the loaded model ("WaveNet", "LSTM", or "none"), for the
    /// UI to surface which .nam family is running. UI/main-thread only.
    std::string model_arch() const {
        std::lock_guard<std::mutex> lock(model_mutex_);
        return model_arch_;
    }

    /// Licensing status for the editor's About panel. "Free (MIT) build" for the
    /// shipped demo; a GPU_NAM_WITH_LICENSE build reports Registered / Unregistered.
    std::string license_status() const { return license_.status_text(); }

    /// Snapshot of the input→output transfer ("amp character") curve. UI-thread
    /// only — never the audio thread.
    TransferCurve transfer_curve_snapshot() const {
        std::lock_guard<std::mutex> lock(model_mutex_);
        return transfer_curve_;
    }
    /// Monotonic counter bumped each time the model (and its curve) changes.
    std::uint32_t model_generation() const {
        return model_generation_.load(std::memory_order_relaxed);
    }

    /// Live {input, output} meter levels in dBFS (peak with fast-attack/
    /// slow-release ballistics). Published from the audio thread; UI reads.
    /// −120 dB is the noise floor / silence sentinel.
    std::pair<float, float> meter_levels_db() const {
        return {in_level_db_.load(std::memory_order_relaxed),
                out_level_db_.load(std::memory_order_relaxed)};
    }

    /// Native GPU front-end.
    std::unique_ptr<view::View> create_view() override;

    format::ViewSize view_size() const override {
        // NeuralAmpModelerPlugin's window aspect (its Background art is 600×400).
        return format::view_size_from_design(600, 400);
    }

    void prepare(const format::PrepareContext& ctx) override {
        stop_worker();
        gpu_active_.store(nullptr, std::memory_order_release);
        gpu_engine_active_.store(false, std::memory_order_release);
        cpu_active_.store(nullptr, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(stack_mutex_);
            current_stack_.reset();
        }
        current_cpu_.reset();
        retirees_.clear();                                   // worker stopped — safe
        audio_epoch_.store(0, std::memory_order_relaxed);

        sample_rate_ = ctx.sample_rate;
        const std::size_t max_block =
            ctx.max_buffer_size > 0 ? static_cast<std::size_t>(ctx.max_buffer_size) : 512;
        // The re-blocking FIFOs are sized below, once the model's rate (and thus the
        // internal pipeline rate) is known — the FIFOs carry internal-rate samples.

        // Noise gate + tone-stack DSP (RT-safe; no alloc after this). Reset filter
        // state and invalidate the cached coefficients so process() retunes on its
        // first block from the live parameter values.
        // ~10 Hz high-pass pole for the wet-output DC blocker: pole = 1 - 2*pi*fc/sr.
        const float dc_pole = sample_rate_ > 0.0
            ? 1.0f - static_cast<float>(2.0 * M_PI * 10.0 / sample_rate_) : 0.999f;
        for (std::size_t ch = 0; ch < kChannels; ++ch) {
            gate_[ch].set_sample_rate(static_cast<float>(sample_rate_));
            gate_[ch].reset();
            for (auto& b : tone_[ch]) b.reset();
            dc_[ch].set_pole(dc_pole);
            dc_[ch].reset();
        }
        cached_gate_threshold_ = std::nanf("");
        cached_bass_ = cached_middle_ = cached_treble_ = std::nanf("");

        // Resolve + load the model synchronously so the first block is correct.
        const std::string path = current_requested_path();
        nam::NamRuntime model;
        std::string err;
        const bool ok = nam::load_nam_runtime(path, model, &err);
        if (!ok) {
            runtime::log_error("GPU NAM: failed to load model '{}' ({}); passing dry.",
                               path, err);
        }
        loaded_ok_ = ok;
        publish_model_loudness(model, ok);

        // Pin the internal pipeline to the model's rate and resample the drive in /
        // wet out when the host runs at a different rate. NAM captures are trained
        // at a fixed rate (usually 48 kHz); running the network off-rate shifts its
        // response. Matched rates (the common case) skip resampling entirely.
        {
            const double msr = ok ? model.sample_rate() : 0.0;
            internal_rate_ = (msr > 0.0) ? msr : sample_rate_;
            resample_active_ =
                sample_rate_ > 0.0 && std::abs(internal_rate_ - sample_rate_) > 0.01 * sample_rate_;
            if (resample_active_) {
                for (std::size_t ch = 0; ch < kChannels; ++ch) {
                    in_rs_[ch].prepare(sample_rate_, internal_rate_, 1, max_block);
                    out_rs_[ch].prepare(internal_rate_, sample_rate_, 1,
                                        in_rs_[ch].max_output_for(max_block));
                }
                runtime::log_info(
                    "GPU NAM: model '{}' captured at {} Hz; host at {} Hz — resampling "
                    "around the model (CPU engine).",
                    gpu_nam_basename(path), internal_rate_, sample_rate_);
            }
        }

        // Size the re-block FIFOs for internal-rate samples. A host block of
        // max_block samples yields up to ceil(max_block * internal/host) internal
        // samples; the output staging FIFO carries host-rate wet.
        const double up = resample_active_ ? internal_rate_ / sample_rate_ : 1.0;
        const std::size_t max_internal =
            static_cast<std::size_t>(std::ceil(static_cast<double>(max_block) * up)) + 8;
        const std::size_t cap = max_internal + 4 * kInternalBlock;
        const std::size_t host_cap = max_block + 4 * kInternalBlock;
        for (std::size_t ch = 0; ch < kChannels; ++ch) {
            in_buf_[ch].assign(cap, 0.0f);   in_len_[ch] = 0;
            out_buf_[ch].assign(cap, 0.0f);  out_len_[ch] = kInternalBlock;  // primed: B zeros
            gpu_wet_[ch].assign(kInternalBlock, 0.0f);
            out_host_[ch].assign(host_cap + kInternalBlock, 0.0f);
            out_host_len_[ch] = 0;
            in_rs_[ch].reset();
            out_rs_[ch].reset();
        }
        wet_.assign(kInternalBlock, 0.0f);
        nam_share_.assign(kInternalBlock, 0.0f);
        rs_drive_.assign(max_block, 0.0f);
        direct_wet_.assign(max_block, 0.0f);
        direct_drive1_.assign(max_block, 0.0f);
        direct_wet1_.assign(max_block, 0.0f);
        {
            std::lock_guard<std::mutex> lock(model_mutex_);
            model_ = model;
            model_name_ = ok ? gpu_nam_basename(path) : std::string("(no model)");
            model_arch_ = ok ? model.arch_name() : "none";
        }
        recompute_transfer_curve();

        // Inline CPU engine: per-channel copies of the model. Published for the
        // audio thread via cpu_active_ (built/freed only off the audio thread).
        {
            auto cpu = std::make_unique<CpuEngine>();
            for (std::size_t ch = 0; ch < kChannels; ++ch) {
                cpu->model[ch] = model;
                cpu->model[ch].prewarm();   // silence steady-state — no cold-start thump
            }
            current_cpu_ = std::move(cpu);
            cpu_active_.store(current_cpu_.get(), std::memory_order_seq_cst);
            // Re-arm the dual-mono fast path for the new engine. The audio-thread
            // guard keys on the engine pointer; clearing it here makes the re-arm
            // deterministic even if the allocator hands back the prior address.
            fill_cpu_seen_ = nullptr;
            nam_diverged_ = false;
        }

        // Learn the GPU transport latency by pre-building the stack once when a
        // device exists. The stack stays built even when Engine=CPU; gpu_active_
        // is only published when Engine=GPU.
        // GPU availability (and the transport's extra latency) is decided ONCE here
        // and fixed for the prepared lifetime, because the reported latency must not
        // move while streaming. A consequence: if prepare() loads a CPU-only model
        // (e.g. an LSTM), the GPU engine stays unavailable for this activation even
        // if a WaveNet is loaded later — reopen/re-prepare to re-probe. The default
        // bundled model is WaveNet, so normally the device is probed here.
        // Configure the licensing gate. Free-tier (unlocked) unless a
        // GPU_NAM_WITH_LICENSE build supplies a public key + license key, in which
        // case an invalid/absent license locks the GPU path to the CPU engine.
#if GPU_NAM_WITH_LICENSE
        {
            const char* key = std::getenv("GPU_NAM_LICENSE_KEY");
            license_.configure_rsa(GPU_NAM_LICENSE_PUBKEY, key ? key : "");
        }
#endif

        // The opt-in GPU engine runs the model at the host rate; when resampling is
        // active the model runs at the internal rate on the CPU engine, so the GPU
        // stack is not built (device stays unavailable and the engine is CPU). The
        // license gate can also hold the GPU path to the free-tier CPU engine.
        gpu_extra_ = 0;
        device_available_ = false;
        if (ok && model.gpu_eligible() && !resample_active_ && license_.unlocked()) {
            auto stack = build_gpu_stack(*model.wavenet());
            if (stack) {
                device_available_ = true;
                gpu_extra_ = static_cast<std::size_t>(stack->transport->latency_samples());
                std::lock_guard<std::mutex> lock(stack_mutex_);
                current_stack_ = std::move(stack);
            }
        }

        // Zero-latency direct path: no GPU device (so no CPU↔GPU switch is possible
        // for this activation), rates match (no resampler group delay), and no IR is
        // requested (the partitioned convolver needs fixed kInternalBlock blocks —
        // an IR keeps the FIFO). The CPU engines are per-sample streaming, so a
        // host-sized block runs directly with zero added latency. Decided ONCE here
        // and fixed for the prepared lifetime, exactly like gpu_extra_.
        direct_cpu_ = !device_available_ && !resample_active_ &&
                      current_requested_ir_path().empty();

        // Reported latency is at the HOST rate. The re-block FIFO + GPU-transport
        // delay are internal-rate sample counts; when resampling, convert them to
        // host samples and add each resampler's group delay (≈ (taps-1)/2 at its
        // input rate). Matched rates reduce to kInternalBlock + gpu_extra_; the
        // direct path reports zero.
        if (direct_cpu_) {
            latency_samples_ = 0;
        } else if (resample_active_) {
            const double to_host = sample_rate_ / internal_rate_;
            const double in_delay = 0.5 * static_cast<double>(in_rs_[0].taps_per_phase() - 1);
            const double out_delay =
                0.5 * static_cast<double>(out_rs_[0].taps_per_phase() - 1) * to_host;
            latency_samples_ = static_cast<int>(std::llround(
                static_cast<double>(kInternalBlock) * to_host + in_delay + out_delay));
        } else {
            latency_samples_ = static_cast<int>(kInternalBlock + gpu_extra_);
        }
        const std::size_t dry_delay = static_cast<std::size_t>(latency_samples_);
        for (std::size_t ch = 0; ch < kChannels; ++ch) {
            dry_ring_[ch].assign(dry_delay, 0.0f);
            dry_pos_[ch] = 0;
            cpu_extra_ring_[ch].assign(gpu_extra_, 0.0f);
            cpu_extra_pos_[ch] = 0;
        }

        // Rebuild a previously-selected IR at the (possibly new) sample rate,
        // synchronously so the first block has it. Empty = no IR = pass-through.
        ir_prev_.store(nullptr, std::memory_order_release);
        prev_ir_.reset();
        ir_fade_pos_.store(0, std::memory_order_relaxed);
        current_ir_.reset();
        ir_active_.store(nullptr, std::memory_order_release);
        if (const std::string ipath = current_requested_ir_path(); !ipath.empty())
            build_and_publish_ir(ipath);

        // Engine=Auto picks CPU vs GPU from a one-time CPU benchmark on the
        // prewarmed model (needs current_cpu_ + device_available_, both ready here).
        if (state().get_value(kEngine) >= 1.5f) resolve_auto_engine();
        requested_engine_.store(resolve_engine_selection(), std::memory_order_relaxed);
        if (requested_engine_.load(std::memory_order_relaxed) == 1) {
            std::lock_guard<std::mutex> lock(stack_mutex_);
            if (current_stack_) {
                gpu_active_.store(current_stack_->transport.get(), std::memory_order_seq_cst);
                gpu_engine_active_.store(true, std::memory_order_release);
            }
        }

        start_worker();
    }

    void release() override {
        stop_worker();
        gpu_active_.store(nullptr, std::memory_order_release);
        gpu_engine_active_.store(false, std::memory_order_release);
        cpu_active_.store(nullptr, std::memory_order_release);
        ir_active_.store(nullptr, std::memory_order_release);
        ir_prev_.store(nullptr, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(stack_mutex_);
            current_stack_.reset();
        }
        current_cpu_.reset();
        current_ir_.reset();
        prev_ir_.reset();
        retirees_.clear();   // worker stopped — safe to free all retirees now
    }

    void process(audio::BufferView<float>& output,
                 const audio::BufferView<const float>& input,
                 midi::MidiBuffer&, midi::MidiBuffer&,
                 const format::ProcessContext&) override {
        // NAM's decaying state (WaveNet dilation history, LSTM cell tails) generates
        // denormals; flushing them to zero for the whole callback avoids the CPU
        // stalls that show up as xruns in a real DAW. RAII-restored on return.
        signal::ScopedFlushDenormals flush_denormals;

        const std::size_t n = output.num_samples();
        const std::size_t ch_count = output.num_channels();

        // Monotonic per-callback epoch: the worker frees a retired engine only once
        // this has advanced past the retirement, so an engine the audio thread is
        // still using is never freed. seq_cst so the bump and the engine-pointer
        // loads below share one total order with the worker's unpublish + stamp read
        // — the retirement grace is then airtight, not timing-dependent. One atomic
        // op per block (not per sample): negligible on the audio thread.
        audio_epoch_.fetch_add(1, std::memory_order_seq_cst);

        const bool bypass = state().get_value(kBypass) >= 0.5f;
        const float mix = bypass ? 0.0f : state().get_value(kMix) / 100.0f;
        const float in_gain = std::pow(10.0f, state().get_value(kInputGain) / 20.0f);
        // Output make-up gain: the user Output knob, times the output-mode retarget
        // (Raw = unity; Normalized = retarget the model's loudness to the fixed
        // reference; Calibrated = retarget to the user's Cal Level). Unity whenever
        // the model carries no loudness metadata. Derived once per block (a single
        // pow), never per sample.
        const nam::OutputMode out_mode =
            nam::output_mode_from_param(state().get_value(kOutputMode));
        // Acquire the has-flag first so a true flag pairs with the level published
        // before it (release in publish_model_loudness) — never has=true vs a stale
        // level.
        const bool has_loud = model_has_loudness_.load(std::memory_order_acquire);
        const float loud_db = model_loudness_db_.load(std::memory_order_relaxed);
        const float norm = nam::output_mode_gain(out_mode, has_loud, loud_db,
                                                 kNormalizeTargetDb,
                                                 state().get_value(kCalibrationLevel),
                                                 kNormalizeMaxAbsDb);
        const float out_gain = std::pow(10.0f, state().get_value(kOutputGain) / 20.0f) * norm;

        requested_engine_.store(resolve_engine_selection(), std::memory_order_relaxed);

        // Retune the gate + tone stack from the live parameters (only when they
        // change, at this block boundary — the RT-safe way to move IIR/gate coeffs).
        gate_active_ = state().get_value(kNoiseGateActive) >= 0.5f;
        eq_active_ = state().get_value(kEQActive) >= 0.5f;
        update_gate(state().get_value(kNoiseGateThreshold));
        update_tone(state().get_value(kToneBass), state().get_value(kToneMiddle),
                    state().get_value(kToneTreble));

        // Zero-latency CPU path: no FIFO, host-sized blocks straight through the
        // per-sample engine, dry/wet aligned with no delay. Same per-sample math as
        // the re-block path, minus the kInternalBlock latency.
        if (direct_cpu_) {
            fill_direct_cpu(input, output, mix, in_gain, out_gain);
            publish_meters(input, output, n);
            return;
        }

        // Append the drive signal (input × input-gain, then noise gate) to the
        // re-block FIFO; the active engine transforms drive → wet (model output).
        // The gate runs at the host rate on the input; when resampling, the gated
        // drive is converted host→internal before entering the internal-rate FIFO.
        for (std::size_t ch = 0; ch < kChannels; ++ch) {
            const float* in = ch < input.num_channels() ? input.channel(ch).data() : nullptr;
            if (!resample_active_) {
                float* dst = in_buf_[ch].data() + in_len_[ch];
                for (std::size_t i = 0; i < n; ++i) {
                    float drive = in ? in[i] * in_gain : 0.0f;
                    if (gate_active_) drive = gate_[ch].process(drive);
                    dst[i] = drive;
                }
                in_len_[ch] += n;
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    float drive = in ? in[i] * in_gain : 0.0f;
                    if (gate_active_) drive = gate_[ch].process(drive);
                    rs_drive_[i] = drive;
                }
                const std::size_t room = in_buf_[ch].size() - in_len_[ch];
                const std::size_t produced = in_rs_[ch].process_block_mono(
                    rs_drive_.data(), n, in_buf_[ch].data() + in_len_[ch], room);
                in_len_[ch] += produced;
            }
        }

        gpu_audio::GpuAudioTransport* tp = gpu_active_.load(std::memory_order_seq_cst);
        if (tp) fill_wet_gpu(tp);
        else    fill_wet_cpu();

        // When resampling, convert the internal-rate model output (out_buf_) back to
        // the host rate into out_host_, consuming out_buf_. The emit then reads
        // host-rate wet from out_host_; otherwise it reads out_buf_ directly (the
        // matched-rate path is byte-for-byte unchanged).
        if (resample_active_) {
            for (std::size_t ch = 0; ch < kChannels; ++ch) {
                const std::size_t room = out_host_[ch].size() - out_host_len_[ch];
                const auto res = out_rs_[ch].process_block_mono_detailed(
                    out_buf_[ch].data(), out_len_[ch],
                    out_host_[ch].data() + out_host_len_[ch], room);
                out_host_len_[ch] += res.output_frames;
                if (res.input_frames_consumed > 0) {
                    std::memmove(out_buf_[ch].data(),
                                 out_buf_[ch].data() + res.input_frames_consumed,
                                 (out_len_[ch] - res.input_frames_consumed) * sizeof(float));
                    out_len_[ch] -= res.input_frames_consumed;
                }
            }
        }

        // Emit n samples per channel: wet from the primed output FIFO, dry delayed
        // by the fixed total latency so they stay aligned under either engine.
        for (std::size_t ch = 0; ch < ch_count && ch < kChannels; ++ch) {
            const float* in = ch < input.num_channels() ? input.channel(ch).data() : nullptr;
            float* out = output.channel(ch).data();
            const float* wet_src = resample_active_ ? out_host_[ch].data() : out_buf_[ch].data();
            const std::size_t avail = resample_active_ ? out_host_len_[ch] : out_len_[ch];
            const std::size_t delay = dry_ring_[ch].size();
            for (std::size_t i = 0; i < n; ++i) {
                // DC-block the wet stream once per sample, in order, before it reaches
                // the mix/meters — covers both the CPU and GPU engines.
                const float wet_i = dc_[ch].process(i < avail ? wet_src[i] : 0.0f);
                const float dry_i = delay > 0 ? dry_ring_[ch][dry_pos_[ch]] : (in ? in[i] : 0.0f);
                if (delay > 0) {
                    dry_ring_[ch][dry_pos_[ch]] = in ? in[i] : 0.0f;
                    dry_pos_[ch] = (dry_pos_[ch] + 1) % delay;
                }
                out[i] = (1.0f - mix) * dry_i + mix * out_gain * wet_i;
            }
            const std::size_t consumed = n < avail ? n : avail;
            if (resample_active_) {
                std::memmove(out_host_[ch].data(), out_host_[ch].data() + consumed,
                             (out_host_len_[ch] - consumed) * sizeof(float));
                out_host_len_[ch] -= consumed;
            } else {
                std::memmove(out_buf_[ch].data(), out_buf_[ch].data() + consumed,
                             (out_len_[ch] - consumed) * sizeof(float));
                out_len_[ch] -= consumed;
            }
        }

        publish_meters(input, output, n);
    }

private:
    static constexpr std::size_t kChannels = 2;

    // The inline CPU engine: one NAM oracle per channel (WaveNet or LSTM — the
    // NamRuntime hides which). Heap-owned, built/freed only off the audio thread;
    // the audio thread loads a pointer to it.
    struct CpuEngine {
        std::array<nam::NamRuntime, kChannels> model{};
    };

    // Cabinet IR: one partitioned convolver per channel over the fixed internal
    // block. Heap-owned, built/freed only off the audio thread; the audio thread
    // loads a pointer to it (nullptr = no IR = pass-through). Auditioning a new
    // cabinet builds a fresh engine and the audio thread crossfades its output from
    // the outgoing engine to the incoming one (see apply_ir), so switching cabinets
    // is click-free — a hard in-place swap would step-discontinuity the output
    // because the new IR's response starts mid-signal.
    struct IrEngine {
        std::array<signal::PartitionedConvolver, kChannels> conv{};
    };
    // Cabinet IRs are short; cap the decode + convolution cost.
    static constexpr double kMaxIrSeconds = 2.0;

    // One GPU engine: the node (per-channel GPU NAM forward + CPU fallback) plus
    // the transport. The stack owns its OWN model copy so the node's model pointer
    // stays valid across a live model reload (the processor's authoritative model_
    // is a separate copy). Member destruction is reverse-of-declaration: transport
    // (last) is destroyed before node — required, as the transport points into the
    // node.
    struct GpuStack {
        std::unique_ptr<nam::NamModel> model;
        std::unique_ptr<GpuNamCloudNode> node;
        std::unique_ptr<gpu_audio::GpuAudioTransport> transport;
    };

    // Map the raw Engine parameter (0=CPU, 1=GPU, 2=Auto) to the effective engine
    // (0=CPU, 1=GPU). Auto returns the choice benchmarked at prepare().
    int resolve_engine_selection() const {
        const float v = state().get_value(kEngine);
        if (v >= 1.5f) return auto_resolved_engine_.load(std::memory_order_relaxed);
        return v >= 0.5f ? 1 : 0;
    }

    // Engine=Auto: benchmark the CPU inference cost once (off the audio thread, on
    // the prewarmed model copy) and offload to the GPU only when the CPU would eat
    // more than kAutoGpuBudgetFraction of one internal block's real-time budget.
    // Small models (the common case) stay on the CPU, avoiding the GPU's fixed
    // submit/readback overhead; heavy models move to the GPU. With no GPU device
    // the answer is always CPU. Runs at prepare(), never on the audio thread.
    void resolve_auto_engine() {
        int choice = 0;  // CPU
        CpuEngine* cpu = current_cpu_.get();
        if (device_available_ && cpu) {
            std::vector<float> in(kInternalBlock, 0.25f), out(kInternalBlock, 0.0f);
            const auto once = [&] {
                cpu->model[0].process(in.data(), out.data(),
                                      static_cast<std::uint32_t>(kInternalBlock));
            };
            once();  // warm caches / branch predictors
            double best_us = std::numeric_limits<double>::max();
            for (int i = 0; i < 8; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                once();
                const auto t1 = std::chrono::steady_clock::now();
                best_us = std::min(best_us,
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            const double rate = resample_active_ ? internal_rate_ : sample_rate_;
            const double budget_us =
                rate > 0.0 ? static_cast<double>(kInternalBlock) / rate * 1e6 : 0.0;
            if (budget_us > 0.0 && best_us > budget_us * kAutoGpuBudgetFraction)
                choice = 1;  // CPU too heavy for the RT budget → offload to the GPU
            // The benchmark advanced channel 0's history; restore it to match the
            // other channels' freshly-prewarmed state before audio starts.
            cpu->model[0].reset();
            cpu->model[0].prewarm();
        }
        auto_resolved_engine_.store(choice, std::memory_order_relaxed);
    }

    // CPU engine: re-block the drive FIFO into fixed blocks, run the inline NAM
    // oracle per channel, delay by the GPU transport's fixed extra latency so the
    // two engines align, and push to the output FIFO. RT-safe (no alloc/free).
    void fill_wet_cpu() {
        static_assert(kChannels == 2, "dual-mono fast path assumes stereo");
        CpuEngine* cpu = cpu_active_.load(std::memory_order_seq_cst);
        // A fresh engine (prepare or a live model swap) publishes two identical
        // prewarmed channels, so the fast path is eligible again — reset the latch
        // on the audio thread when the engine pointer changes (no cross-thread race).
        if (cpu != fill_cpu_seen_) { nam_diverged_ = false; fill_cpu_seen_ = cpu; }

        const std::size_t bytes = kInternalBlock * sizeof(float);
        // Both channels are fed the same host-sample count, so their FIFOs advance
        // in lockstep. When a mono source is panned to stereo the two drives are
        // bit-identical, so one NAM inference feeds both (the dominant cost); tone,
        // IR, and delay still run per channel. The first block where the drives
        // differ resyncs channel 1's NAM state from channel 0 (identical shared
        // history) and retires the fast path for the engine's lifetime — keeping
        // the diverged output bit-exact against always running both engines.
        while (in_len_[0] >= kInternalBlock && in_len_[1] >= kInternalBlock) {
            const bool mono = cpu && !nam_diverged_ &&
                std::memcmp(in_buf_[0].data(), in_buf_[1].data(), bytes) == 0;
            if (cpu && !nam_diverged_ && !mono) {
                // Channels just diverged: copy channel 0's NAM state (which carries
                // the shared history up to here) into channel 1 before either runs,
                // then latch. Same-shape runtimes reuse their storage on assign, so
                // this is alloc-free and happens at most once per engine.
                cpu->model[1] = cpu->model[0];
                nam_diverged_ = true;
            }

            // Channel 0 — always inferred.
            if (cpu) cpu->model[0].process(in_buf_[0].data(), wet_.data(),
                                           static_cast<std::uint32_t>(kInternalBlock));
            else std::memset(wet_.data(), 0, bytes);
            if (mono) std::memcpy(nam_share_.data(), wet_.data(), bytes);
            std::memmove(in_buf_[0].data(), in_buf_[0].data() + kInternalBlock,
                         (in_len_[0] - kInternalBlock) * sizeof(float));
            in_len_[0] -= kInternalBlock;
            apply_tone(0, wet_.data());
            apply_ir(0, wet_.data());
            delay_cpu_wet(0);
            std::memcpy(out_buf_[0].data() + out_len_[0], wet_.data(), bytes);
            out_len_[0] += kInternalBlock;

            // Channel 1 — reuse channel 0's NAM output on the mono fast path,
            // otherwise infer independently. Its own tone/IR/delay always run.
            if (mono) std::memcpy(wet_.data(), nam_share_.data(), bytes);
            else if (cpu) cpu->model[1].process(in_buf_[1].data(), wet_.data(),
                                                static_cast<std::uint32_t>(kInternalBlock));
            else std::memset(wet_.data(), 0, bytes);
            std::memmove(in_buf_[1].data(), in_buf_[1].data() + kInternalBlock,
                         (in_len_[1] - kInternalBlock) * sizeof(float));
            in_len_[1] -= kInternalBlock;
            apply_tone(1, wet_.data());
            apply_ir(1, wet_.data());
            delay_cpu_wet(1);
            std::memcpy(out_buf_[1].data() + out_len_[1], wet_.data(), bytes);
            out_len_[1] += kInternalBlock;
        }
    }

    // Zero-latency path: run the CPU engine on the whole host block, no FIFO. The
    // engine (gate → NAM → tone → DC) is per-sample streaming, so a variable-width
    // block produces the same samples as the re-block path would — just emitted
    // kInternalBlock samples earlier. No IR here (direct_cpu_ requires none). Dry
    // and wet are aligned at zero delay. RT-safe: rs_drive_/direct_wet_ are
    // preallocated to the host-max block. Runs on the audio thread.
    void fill_direct_cpu(const audio::BufferView<const float>& input,
                         audio::BufferView<float>& output,
                         float mix, float in_gain, float out_gain) {
        CpuEngine* cpu = cpu_active_.load(std::memory_order_seq_cst);
        // Reset the dual-mono latch on a fresh engine (shared with fill_wet_cpu;
        // only one of the two paths runs for a prepared lifetime).
        if (cpu != fill_cpu_seen_) { nam_diverged_ = false; fill_cpu_seen_ = cpu; }
        const std::size_t n = output.num_samples();
        const std::size_t ch_count = output.num_channels();
        const std::size_t bytes = n * sizeof(float);

        // Drive: input × gain, then the gate — matching the re-block path exactly.
        auto build_drive = [&](std::size_t ch, const float* in, float* dst) {
            for (std::size_t i = 0; i < n; ++i) {
                float drive = in ? in[i] * in_gain : 0.0f;
                if (gate_active_) drive = gate_[ch].process(drive);
                dst[i] = drive;
            }
        };
        // Emit one channel: tone → DC-block → mix with the un-delayed dry.
        auto emit = [&](std::size_t ch, float* wet, const float* in, float* out) {
            apply_tone_n(ch, wet, n);
            for (std::size_t i = 0; i < n; ++i) {
                const float wet_i = dc_[ch].process(wet[i]);
                const float dry_i = in ? in[i] : 0.0f;
                out[i] = (1.0f - mix) * dry_i + mix * out_gain * wet_i;
            }
        };

        const float* in0 = (ch_count > 0 && input.num_channels() > 0)
                               ? input.channel(0).data() : nullptr;
        const float* in1 = (ch_count > 1 && input.num_channels() > 1)
                               ? input.channel(1).data() : nullptr;

        // Dual-mono fast path: a mono source panned to stereo drives both channels
        // identically, so run ONE NAM inference and feed both — the dominant cost for
        // the CPU-only users this path targets. tone/DC still run per channel. On the
        // first block the drives differ, resync channel 1's NAM state from channel 0
        // (which carried the shared history) and retire the shortcut for the engine's
        // life, keeping the diverged output bit-exact. Mirrors fill_wet_cpu.
        if (cpu && ch_count >= 2 && in0 && in1) {
            build_drive(0, in0, rs_drive_.data());
            build_drive(1, in1, direct_drive1_.data());
            const bool mono = !nam_diverged_ &&
                std::memcmp(rs_drive_.data(), direct_drive1_.data(), bytes) == 0;
            if (!nam_diverged_ && !mono) {
                cpu->model[1] = cpu->model[0];
                nam_diverged_ = true;
            }
            cpu->model[0].process(rs_drive_.data(), direct_wet_.data(),
                                  static_cast<std::uint32_t>(n));
            if (mono) std::memcpy(direct_wet1_.data(), direct_wet_.data(), bytes);
            else cpu->model[1].process(direct_drive1_.data(), direct_wet1_.data(),
                                       static_cast<std::uint32_t>(n));
            emit(0, direct_wet_.data(), in0, output.channel(0).data());
            emit(1, direct_wet1_.data(), in1, output.channel(1).data());
            return;
        }

        // General path (mono output, a missing input channel, or no engine).
        for (std::size_t ch = 0; ch < ch_count && ch < kChannels; ++ch) {
            const float* in = ch < input.num_channels() ? input.channel(ch).data() : nullptr;
            build_drive(ch, in, rs_drive_.data());
            if (cpu) cpu->model[ch].process(rs_drive_.data(), direct_wet_.data(),
                                            static_cast<std::uint32_t>(n));
            else std::fill_n(direct_wet_.data(), n, 0.0f);
            emit(ch, direct_wet_.data(), in, output.channel(ch).data());
        }
    }

    // Push one internal block of CPU wet through the per-channel extra-delay ring
    // (length gpu_extra_), in place. RT-safe: the ring is preallocated.
    void delay_cpu_wet(std::size_t ch) {
        const std::size_t d = gpu_extra_;
        if (d == 0) return;
        auto& ring = cpu_extra_ring_[ch];
        std::size_t& pos = cpu_extra_pos_[ch];
        for (std::size_t i = 0; i < kInternalBlock; ++i) {
            const float in = wet_[i];
            wet_[i] = ring[pos];
            ring[pos] = in;
            pos = (pos + 1) % d;
        }
    }

    // Retune the per-channel noise gate on a threshold change (block boundary).
    // Fast, alloc-free — safe to call every block; the cache skips the no-op case.
    void update_gate(float threshold_db) {
        if (threshold_db == cached_gate_threshold_) return;
        cached_gate_threshold_ = threshold_db;
        signal::NoiseGate::Params p;
        p.threshold_db = threshold_db;
        p.ratio = 10.0f;       // hard downward expander ≈ gate
        p.attack_ms = 1.0f;
        p.release_ms = 50.0f;
        p.range_db = -80.0f;
        for (std::size_t ch = 0; ch < kChannels; ++ch) gate_[ch].set_params(p);
    }

    // Retune the per-channel Bass/Middle/Treble tone stack. 0..10 maps linearly to
    // ±kToneRangeDb with 5 = flat: low shelf (Bass), mid peak (Middle), high shelf
    // (Treble). Only recomputes coefficients when a value actually changes.
    // The tone stack runs on the model output inside the internal-rate pipeline,
    // so its coefficients use the internal rate (== host rate when not resampling).
    void update_tone(float bass, float middle, float treble) {
        if (bass == cached_bass_ && middle == cached_middle_ && treble == cached_treble_)
            return;
        cached_bass_ = bass;
        cached_middle_ = middle;
        cached_treble_ = treble;
        const float sr = static_cast<float>(internal_rate_);
        const float bass_db   = (bass   - 5.0f) / 5.0f * kToneRangeDb;
        const float mid_db    = (middle - 5.0f) / 5.0f * kToneRangeDb;
        const float treble_db = (treble - 5.0f) / 5.0f * kToneRangeDb;
        for (std::size_t ch = 0; ch < kChannels; ++ch) {
            tone_[ch][0].set_coefficients(signal::Biquad::Type::low_shelf, 120.0f, 0.707f, sr, bass_db);
            tone_[ch][1].set_coefficients(signal::Biquad::Type::peaking, 750.0f, 0.7f, sr, mid_db);
            tone_[ch][2].set_coefficients(signal::Biquad::Type::high_shelf, 3000.0f, 0.707f, sr, treble_db);
        }
    }

    // Apply one channel's tone stack in place over an internal block (RT-safe).
    // A no-op when EQ is off, so the model output passes through unfiltered.
    void apply_tone(std::size_t ch, float* buf) { apply_tone_n(ch, buf, kInternalBlock); }
    void apply_tone_n(std::size_t ch, float* buf, std::size_t count) {
        if (!eq_active_) return;
        auto& t = tone_[ch];
        for (std::size_t i = 0; i < count; ++i)
            buf[i] = t[2].process(t[1].process(t[0].process(buf[i])));
    }

    // Apply the cabinet IR (if loaded) in place over one internal block. RT-safe:
    // the convolver is preallocated; nullptr = no IR = pass-through. In-place is
    // safe — the convolver reads the whole input block before writing output.
    //
    // Changing cabinets publishes a fresh engine (ir_active_) while the outgoing one
    // stays live in ir_prev_; here the audio thread runs BOTH on the same input and
    // linearly blends prev→cur across kIrFadeSamples so the switch is click-free (a
    // hard swap would step-discontinuity the output — the new IR's response begins
    // mid-signal, and a one-sided fade would trade the up-step for a down-step).
    // Both channels use the same fade position within a frame; the last channel
    // advances it. The worker retires the outgoing engine once the fade completes.
    // Adding the first IR / removing the last go straight through ir_active_ (an
    // expected timbral change, no crossfade).
    void apply_ir(std::size_t ch, float* buf) {
        // Load cur (the flag) BEFORE prev (the data): the worker publishes prev then
        // cur, so reading cur first guarantees that observing the post-swap cur also
        // observes the non-null prev — otherwise a callback could see the new engine
        // with prev still null and run it at full gain for one block (the very click
        // the crossfade removes).
        IrEngine* cur  = ir_active_.load(std::memory_order_seq_cst);
        IrEngine* prev = ir_prev_.load(std::memory_order_seq_cst);
        if (!prev) {                       // common path — no cabinet crossfade in flight
            if (cur) cur->conv[ch].process(buf, buf, kInternalBlock);
            return;
        }
        const int pos = ir_fade_pos_.load(std::memory_order_relaxed);
        float old_out[kInternalBlock];
        std::copy_n(buf, kInternalBlock, old_out);
        if (cur) cur->conv[ch].process(buf, buf, kInternalBlock);
        else     std::fill_n(buf, kInternalBlock, 0.0f);
        prev->conv[ch].process(old_out, old_out, kInternalBlock);
        const float inv = 1.0f / static_cast<float>(kIrFadeSamples);
        for (std::size_t i = 0; i < kInternalBlock; ++i) {
            float a = static_cast<float>(pos + static_cast<int>(i)) * inv;
            if (a > 1.0f) a = 1.0f;
            buf[i] = a * buf[i] + (1.0f - a) * old_out[i];
        }
        if (ch == kChannels - 1)
            ir_fade_pos_.store(pos + static_cast<int>(kInternalBlock),
                               std::memory_order_relaxed);
    }

    // GPU engine: same fixed re-blocking, each B-block processed as ONE stereo
    // block through the GPU transport (RT-safe by contract — the GPU forward runs
    // on the transport's non-RT worker; a worker miss runs the node's CpuFallback).
    // Both channels advance in lockstep, so in_len_[0] gates the drain.
    void fill_wet_gpu(gpu_audio::GpuAudioTransport* tp) {
        while (in_len_[0] >= kInternalBlock && in_len_[1] >= kInternalBlock) {
            const float* in_ptrs[kChannels] = {in_buf_[0].data(), in_buf_[1].data()};
            float* out_ptrs[kChannels] = {gpu_wet_[0].data(), gpu_wet_[1].data()};
            audio::BufferView<const float> iv(in_ptrs, kChannels, kInternalBlock);
            audio::BufferView<float> ov(out_ptrs, kChannels, kInternalBlock);
            tp->process(iv, ov, static_cast<std::uint32_t>(kInternalBlock));
            for (std::size_t ch = 0; ch < kChannels; ++ch) {
                std::memmove(in_buf_[ch].data(), in_buf_[ch].data() + kInternalBlock,
                             (in_len_[ch] - kInternalBlock) * sizeof(float));
                in_len_[ch] -= kInternalBlock;
                apply_tone(ch, gpu_wet_[ch].data());
                apply_ir(ch, gpu_wet_[ch].data());
                std::memcpy(out_buf_[ch].data() + out_len_[ch], gpu_wet_[ch].data(),
                            kInternalBlock * sizeof(float));
                out_len_[ch] += kInternalBlock;
            }
        }
    }

    // Build a self-contained GPU stack (model copy + node + transport) off the
    // audio thread. Returns nullptr on any failure (no GPU device, unsupported
    // shape) — the caller then routes the CPU engine. Non-RT.
    std::unique_ptr<GpuStack> build_gpu_stack(const nam::NamModel& model) {
        auto stack = std::make_unique<GpuStack>();
        stack->model = std::make_unique<nam::NamModel>(model);
        stack->node = std::make_unique<GpuNamCloudNode>(
            static_cast<std::uint32_t>(kChannels),
            static_cast<std::uint32_t>(kInternalBlock),
            static_cast<std::uint32_t>(sample_rate_), stack->model.get());
        if (!stack->node->prepare() || !stack->node->gpu_available()) return nullptr;

        stack->transport = std::make_unique<gpu_audio::GpuAudioTransport>();
        gpu_audio::GpuAudioTransport::Config cfg;
        cfg.ring_blocks = 8;
        cfg.run_worker_thread = true;
        if (!stack->transport->prepare(stack->node.get(), cfg)) return nullptr;
        return stack;
    }

    // Worker-thread only. Build + publish a fresh GPU stack for `model`. The
    // current stack is RETIRED and freed on the NEXT rebuild — never freed while
    // the audio thread might still be loading its pointer. A CPU-only model (LSTM)
    // tears the GPU stack down and returns: the audio path falls back to the CPU
    // engine, whose fixed extra-latency ring keeps the reported latency unchanged.
    void rebuild_gpu_stack(const nam::NamRuntime& model) {
        gpu_active_.store(nullptr, std::memory_order_seq_cst);   // unpublish before retiring
        gpu_engine_active_.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(stack_mutex_);
            retire_engine(std::shared_ptr<void>(std::move(current_stack_)));
        }
        if (!model.gpu_eligible()) {
            runtime::log_info("GPU NAM: '{}' is CPU-only; GPU engine disabled for this model.",
                              model.arch_name());
            return;
        }
        auto fresh = build_gpu_stack(*model.wavenet());
        if (!fresh) {
            runtime::log_info("GPU NAM: GPU stack rebuild failed; routing the CPU engine.");
            return;
        }
        gpu_audio::GpuAudioTransport* tp = fresh->transport.get();
        {
            std::lock_guard<std::mutex> lock(stack_mutex_);
            current_stack_ = std::move(fresh);
        }
        if (requested_engine_.load(std::memory_order_relaxed) == 1) {
            gpu_active_.store(tp, std::memory_order_seq_cst);   // reader loads this, then it retires
            gpu_engine_active_.store(true, std::memory_order_release);
        }
    }

    // Loader-thread only. Publish the model's loudness metadata for the audio
    // thread; the per-mode output make-up gain (Normalized against the fixed
    // reference, Calibrated against the user's Cal Level) is derived from it once
    // per block in process(). Models without loudness metadata (and any failed
    // load) publish has_loudness=false, so every output mode is a no-op rather than
    // a guess. Publishes the level before the has-flag so the audio thread never
    // reads has=true against a stale level.
    void publish_model_loudness(const nam::NamRuntime& model, bool ok) {
        const bool has = ok && model.has_loudness();
        model_loudness_db_.store(has ? static_cast<float>(model.loudness_db()) : 0.0f,
                                 std::memory_order_relaxed);
        model_has_loudness_.store(has, std::memory_order_release);
    }

    // Worker-thread only. Publish a fresh inline CPU engine for `model`, retiring
    // the old one (freed one rebuild later).
    void rebuild_cpu_engine(const nam::NamRuntime& model) {
        cpu_active_.store(nullptr, std::memory_order_seq_cst);   // unpublish before retiring
        retire_engine(std::shared_ptr<void>(std::move(current_cpu_)));
        auto cpu = std::make_unique<CpuEngine>();
        for (std::size_t ch = 0; ch < kChannels; ++ch) {
            cpu->model[ch] = model;
            cpu->model[ch].prewarm();   // silence steady-state — no cold-start thump
        }
        current_cpu_ = std::move(cpu);
        cpu_active_.store(current_cpu_.get(), std::memory_order_seq_cst);
    }

    // Compute the static input→output transfer curve on a fresh model copy: for
    // each swept input level, settle a reset model on that constant and take the
    // last output. Publishes under model_mutex_ + bumps model_generation_. UI/
    // worker-thread only (never the audio thread).
    void recompute_transfer_curve() {
        nam::NamRuntime m;
        {
            std::lock_guard<std::mutex> lock(model_mutex_);
            m = model_;
        }
        TransferCurve curve{};
        if (loaded_ok_ && m.ok()) {
            constexpr int settle = 1024;
            for (int p = 0; p < kCurvePoints; ++p) {
                const float x = -kCurveRange + 2.0f * kCurveRange * p / (kCurvePoints - 1);
                m.reset();
                float y = 0.0f;
                for (int i = 0; i < settle; ++i) y = m.process_sample(x);
                curve[static_cast<std::size_t>(p)] = std::isfinite(y) ? y : 0.0f;
            }
        }
        {
            std::lock_guard<std::mutex> lock(model_mutex_);
            transfer_curve_ = curve;
        }
        model_generation_.fetch_add(1, std::memory_order_relaxed);
    }

    // Peak meter with fast-attack / slow-release ballistics, per block. Input is
    // the raw (pre-gain) signal; output is the post-mix result. Audio-thread only.
    void publish_meters(const audio::BufferView<const float>& input,
                        const audio::BufferView<float>& output, std::size_t n) {
        float in_peak = 0.0f, out_peak = 0.0f;
        for (std::size_t ch = 0; ch < input.num_channels(); ++ch) {
            const float* p = input.channel(ch).data();
            for (std::size_t i = 0; i < n; ++i) in_peak = std::max(in_peak, std::abs(p[i]));
        }
        for (std::size_t ch = 0; ch < output.num_channels(); ++ch) {
            const float* p = output.channel(ch).data();
            for (std::size_t i = 0; i < n; ++i) out_peak = std::max(out_peak, std::abs(p[i]));
        }
        const float rel = 0.10f;  // slow release toward the new peak
        in_env_  = in_peak  > in_env_  ? in_peak  : in_env_  + rel * (in_peak  - in_env_);
        out_env_ = out_peak > out_env_ ? out_peak : out_env_ + rel * (out_peak - out_env_);
        auto to_db = [](float lin) {
            return lin > 1e-6f ? 20.0f * std::log10(lin) : -120.0f;
        };
        in_level_db_.store(to_db(in_env_), std::memory_order_relaxed);
        out_level_db_.store(to_db(out_env_), std::memory_order_relaxed);
    }


    std::string current_requested_path() {
        std::lock_guard<std::mutex> lock(model_req_mutex_);
        if (requested_model_path_.empty())
            requested_model_path_ = gpu_nam_default_model_path();
        return requested_model_path_;
    }

    std::string current_requested_ir_path() {
        std::lock_guard<std::mutex> lock(ir_req_mutex_);
        return requested_ir_path_;
    }

    // Build per-channel convolvers from an IR file at the session rate and
    // publish them for the audio thread, retiring the previous engine (freed one
    // rebuild later). Worker / prepare thread only. On failure, leaves the
    // current IR untouched.
    void build_and_publish_ir(const std::string& path) {
        // read_impulse_response decodes the whole file + allocates FFT/resampler
        // state — any exception (bad_alloc on a huge/corrupt file, decoder throw)
        // must never escape onto the worker thread (→ std::terminate) or out of
        // prepare(). Keep the current IR on failure.
        std::optional<std::vector<float>> ir;
        try {
            ir = audio::read_impulse_response(
                path, internal_rate_, {.max_seconds = kMaxIrSeconds, .normalize_unit_energy = true});
        } catch (const std::exception& e) {
            runtime::log_error("GPU NAM: load_ir('{}') threw ({}); keeping current IR.",
                               path, e.what());
            return;
        } catch (...) {
            runtime::log_error("GPU NAM: load_ir('{}') threw; keeping current IR.", path);
            return;
        }
        if (!ir || ir->empty()) {
            runtime::log_error("GPU NAM: load_ir('{}') failed; keeping current IR.", path);
            return;
        }
        // Build the new IR into a fresh engine (allocates + FFTs here, off the audio
        // thread). The first block sees it convolving from silence.
        auto eng = std::make_unique<IrEngine>();
        for (std::size_t ch = 0; ch < kChannels; ++ch)
            eng->conv[ch].load_ir(ir->data(), ir->size(), kInternalBlock);

        if (ir_active_.load(std::memory_order_acquire) && !direct_cpu_) {
            // Cabinet change while an IR is live: demote the current engine to the
            // outgoing slot and publish the new one, so the audio thread crossfades
            // prev→cur over kIrFadeSamples (apply_ir) — click-free. The worker only
            // reaches here when no fade is in flight (gated in worker_loop), so the
            // 2-engine invariant holds: prev_ir_ is empty on entry. The outgoing
            // engine is retired once the fade completes.
            ir_fade_pos_.store(0, std::memory_order_relaxed);
            prev_ir_ = std::move(current_ir_);                        // keep it alive...
            ir_prev_.store(prev_ir_.get(), std::memory_order_seq_cst); // ...crossfade from it
            current_ir_ = std::move(eng);
            ir_active_.store(current_ir_.get(), std::memory_order_seq_cst);
        } else {
            // First IR (silence → IR is an expected timbral change), OR the
            // zero-latency direct CPU path — which carries no convolver, so apply_ir
            // never runs and could never advance a crossfade (leaving a fade stuck
            // and blocking every later swap). Publish directly and retire any engine
            // we displace through the epoch list so the audio thread never frees it.
            std::unique_ptr<IrEngine> displaced = std::move(current_ir_);
            current_ir_ = std::move(eng);
            ir_active_.store(current_ir_.get(), std::memory_order_seq_cst);
            if (displaced) retire_engine(std::shared_ptr<void>(std::move(displaced)));
        }
        std::lock_guard<std::mutex> lock(ir_mutex_);
        ir_name_ = gpu_nam_basename(path);
    }

    void unpublish_ir() {
        ir_active_.store(nullptr, std::memory_order_seq_cst);   // unpublish before retiring
        retire_engine(std::shared_ptr<void>(std::move(current_ir_)));
        std::lock_guard<std::mutex> lock(ir_mutex_);
        ir_name_.clear();
    }

    // Record a just-unpublished engine for grace-period reclamation. The epoch is
    // read AFTER the atomic pointer was set to null / the new engine published, so
    // any audio callback that could still hold this engine has an epoch <= it and
    // the RetireList frees it only once the audio thread has cycled past. Worker /
    // prepare thread only (the audio thread is never here).
    void retire_engine(std::shared_ptr<void> engine) {
        // seq_cst stamp read, ordered after the caller's seq_cst unpublish/publish
        // store: any audio callback that could still hold this engine has an epoch
        // <= the stamp, so RetireList frees it only once the audio thread cycled past.
        retirees_.retire(std::move(engine), audio_epoch_.load(std::memory_order_seq_cst));
    }

    void start_worker() {
        worker_run_.store(true, std::memory_order_release);
        last_seen_model_gen_ = model_req_generation_.load(std::memory_order_acquire);
        last_seen_ir_gen_ = ir_req_generation_.load(std::memory_order_acquire);
        worker_ = std::thread([this] { worker_loop(); });
    }

    void stop_worker() {
        worker_run_.store(false, std::memory_order_release);
        if (worker_.joinable()) worker_.join();
    }

    // Background thread: reconcile the published GPU path with the requested
    // Engine, and rebuild both engines on a model reload. Never touches the audio
    // thread's buffers or frees a stack the audio thread might still hold.
    void worker_loop() {
        using namespace std::chrono_literals;
        while (worker_run_.load(std::memory_order_acquire)) {
            // Free engines retired long enough ago that the audio thread has
            // provably cycled past them (grace-period reclamation).
            retirees_.reclaim(audio_epoch_.load(std::memory_order_acquire));

            // Cabinet crossfade complete? Once the audio thread has ramped fully to
            // the new cabinet (apply_ir), drop the outgoing engine and reclaim it via
            // the epoch list — never freed while the audio thread might still hold it.
            if (prev_ir_ &&
                ir_fade_pos_.load(std::memory_order_acquire) >= kIrFadeSamples) {
                ir_prev_.store(nullptr, std::memory_order_seq_cst);
                retire_engine(std::shared_ptr<void>(std::move(prev_ir_)));
            }

            // Model reload?
            const std::uint32_t gen = model_req_generation_.load(std::memory_order_acquire);
            if (gen != last_seen_model_gen_) {
                last_seen_model_gen_ = gen;
                const std::string path = current_requested_path();
                nam::NamRuntime m;
                std::string err;
                if (nam::load_nam_runtime(path, m, &err)) {
                    loaded_ok_ = true;
                    publish_model_loudness(m, true);
                    // The internal pipeline rate is pinned at prepare(); a runtime
                    // swap to a model captured at a different rate runs at the pinned
                    // rate (response shifted). Reopen the plugin to re-pin. Matched-
                    // rate swaps (the norm) are exact.
                    const double msr = m.sample_rate();
                    if (msr > 0.0 && std::abs(msr - internal_rate_) > 0.01 * internal_rate_)
                        runtime::log_warn(
                            "GPU NAM: model '{}' was captured at {} Hz but the pipeline is pinned "
                            "to {} Hz; its response is shifted. Reopen the plugin to re-pin.",
                            gpu_nam_basename(path), msr, internal_rate_);
                    {
                        std::lock_guard<std::mutex> lock(model_mutex_);
                        model_ = m;
                        model_name_ = gpu_nam_basename(path);
                        model_arch_ = m.arch_name();
                    }
                    rebuild_cpu_engine(m);
                    // A CPU-only model (LSTM) still calls rebuild_gpu_stack, which
                    // tears down the GPU stack and returns — the fixed latency ring
                    // keeps dry/wet aligned on the CPU path.
                    if (device_available_) rebuild_gpu_stack(m);
                    recompute_transfer_curve();
                } else {
                    runtime::log_error("GPU NAM: load_model('{}') failed ({}).", path, err);
                }
            }

            // IR reload / clear? Defer while a cabinet crossfade is still draining so
            // only one fade is ever in flight (the 2-engine invariant) — the pending
            // request is picked up on a later tick once prev_ir_ has been retired.
            const std::uint32_t ir_gen = ir_req_generation_.load(std::memory_order_acquire);
            if (ir_gen != last_seen_ir_gen_ && !prev_ir_) {
                last_seen_ir_gen_ = ir_gen;
                const std::string ipath = current_requested_ir_path();
                if (ipath.empty()) unpublish_ir();
                else               build_and_publish_ir(ipath);
            }

            // Engine toggle (GPU publish/unpublish; the stack stays built).
            if (device_available_) {
                const int want = requested_engine_.load(std::memory_order_relaxed);
                if (want == 1 && gpu_active_.load(std::memory_order_relaxed) == nullptr) {
                    gpu_audio::GpuAudioTransport* tp = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(stack_mutex_);
                        if (current_stack_) tp = current_stack_->transport.get();
                    }
                    if (tp) {
                        gpu_active_.store(tp, std::memory_order_seq_cst);
                        gpu_engine_active_.store(true, std::memory_order_release);
                    }
                } else if (want == 0 && gpu_active_.load(std::memory_order_relaxed) != nullptr) {
                    gpu_active_.store(nullptr, std::memory_order_release);
                    gpu_engine_active_.store(false, std::memory_order_release);
                }
            }
            std::this_thread::sleep_for(5ms);
        }
    }

    double sample_rate_ = 48000.0;
    bool loaded_ok_ = false;

    // Optional licensing gate (a reference for paid Pulp plugins). Free-tier by
    // default, so the shipped MIT demo is unrestricted; a GPU_NAM_WITH_LICENSE
    // build gates the opt-in GPU acceleration behind a valid license.
    LicenseGate license_;

    // Re-blocking FIFO state (audio thread only). in_buf_ carries the drive.
    std::array<std::vector<float>, kChannels> in_buf_{};
    std::array<std::vector<float>, kChannels> out_buf_{};
    std::array<std::size_t, kChannels> in_len_{};
    std::array<std::size_t, kChannels> out_len_{};
    std::array<std::vector<float>, kChannels> dry_ring_{};   // dry delay (total latency)
    std::array<std::size_t, kChannels> dry_pos_{};
    std::array<std::vector<float>, kChannels> cpu_extra_ring_{};  // CPU wet → GPU-latency align
    std::array<std::size_t, kChannels> cpu_extra_pos_{};
    std::array<std::vector<float>, kChannels> gpu_wet_{};
    std::vector<float> wet_;                                  // internal-block scratch
    // Dual-mono fast path (audio-thread only): when both channels' drive is
    // identical, one NAM inference feeds both. nam_share_ holds channel 0's NAM
    // output for reuse; nam_diverged_ latches once the channels differ (retiring
    // the fast path for the engine's lifetime); fill_cpu_seen_ detects an engine
    // swap so the latch resets when a fresh (re-synced) engine is published.
    std::vector<float> nam_share_;
    bool nam_diverged_ = false;
    CpuEngine* fill_cpu_seen_ = nullptr;
    // Zero-latency direct path (matched-rate, CPU-only, no IR — decided once at
    // prepare, fixed for the prepared lifetime). Host-block-sized scratch: rs_drive_
    // (unused when not resampling) + direct_drive1_ carry each channel's drive;
    // direct_wet_ + direct_wet1_ hold each channel's NAM output. The dual-mono fast
    // path (shared with the FIFO path via nam_diverged_/fill_cpu_seen_) runs ONE
    // inference when both channels' drive is identical.
    std::vector<float> direct_wet_;
    std::vector<float> direct_drive1_;
    std::vector<float> direct_wet1_;

    // Sample-rate conversion around the model. NAM captures are trained at a fixed
    // rate (usually 48 kHz); running the network at a different rate shifts its
    // response. When the host rate differs from the model rate we pin the internal
    // pipeline (re-block FIFO, model, tone stack, IR) to the MODEL rate and
    // resample the drive in and the wet out. `internal_rate_` == `sample_rate_`
    // (and `resample_active_` is false) in the common matched case, so that path
    // is byte-for-byte the non-resampling pipeline. Resampling forces the CPU
    // engine (the opt-in GPU path stays for the matched case).
    double internal_rate_ = 48000.0;
    bool resample_active_ = false;
    std::array<signal::Resampler, kChannels> in_rs_{};    // host -> internal (drive)
    std::array<signal::Resampler, kChannels> out_rs_{};   // internal -> host (wet)
    std::vector<float> rs_drive_;                          // host-rate gated drive scratch
    std::array<std::vector<float>, kChannels> out_host_{};// host-rate wet staging FIFO
    std::array<std::size_t, kChannels> out_host_len_{};

    // Noise gate (on the drive) + Bass/Middle/Treble tone stack (on the model
    // output). Audio-thread only; retuned at block boundaries via the caches.
    static constexpr float kToneRangeDb = 12.0f;
    std::array<signal::NoiseGate, kChannels> gate_{};
    std::array<std::array<signal::Biquad, 3>, kChannels> tone_{};
    // A NAM capture's response to silence is a nonzero DC (network biases; prewarm
    // settles to a DC steady state), and gate/mix/engine changes step that offset.
    // A one-pole high-pass on the wet output removes it so it never reaches the mix,
    // the meters, or the spectrum, and there are no thumps on those transitions.
    std::array<signal::DcBlocker<float>, kChannels> dc_{};
    bool gate_active_ = true;
    bool eq_active_ = true;
    float cached_gate_threshold_ = 0.0f;
    float cached_bass_ = 0.0f;
    float cached_middle_ = 0.0f;
    float cached_treble_ = 0.0f;

    // Monotonic audio-callback epoch (audio thread bumps, worker reads) and the
    // grace-period retirement list every live engine swap parks its predecessor in,
    // so the audio thread never frees or uses-after-free across a swap.
    std::atomic<std::uint64_t> audio_epoch_{0};
    nam::RetireList retirees_;   // worker / prepare thread only (audio thread never here)

    // Inline CPU engine, published lock-free for the audio thread.
    std::unique_ptr<CpuEngine> current_cpu_;
    std::atomic<CpuEngine*> cpu_active_{nullptr};

    // Cabinet IR engine (default none), published lock-free; built off the audio
    // thread and retired through retirees_ so the audio thread never frees. A
    // cabinet change publishes a fresh current_ir_ and keeps the outgoing engine in
    // prev_ir_/ir_prev_ so the audio thread can crossfade prev→cur (apply_ir); the
    // worker retires the outgoing engine once ir_fade_pos_ reaches kIrFadeSamples.
    std::unique_ptr<IrEngine> current_ir_;
    std::atomic<IrEngine*> ir_active_{nullptr};
    std::unique_ptr<IrEngine> prev_ir_;                 // outgoing engine (worker-owned)
    std::atomic<IrEngine*> ir_prev_{nullptr};           // audio reads it during a crossfade
    std::atomic<int> ir_fade_pos_{0};                   // samples elapsed in the active fade
    // Cabinet-swap crossfade length. ~43 ms @ 48 kHz — long enough to declick a
    // hard cabinet change, short enough to feel immediate.
    static constexpr int kIrFadeSamples = 2048;

    // Optional GPU engine (default OFF), switchable live.
    std::unique_ptr<GpuStack> current_stack_;
    std::atomic<gpu_audio::GpuAudioTransport*> gpu_active_{nullptr};
    mutable std::mutex stack_mutex_;
    std::atomic<bool> gpu_engine_active_{false};
    bool device_available_ = false;
    // Latched at prepare: run the CPU engine directly on host-sized blocks with no
    // re-block FIFO, when no GPU device exists (permanently CPU), rates match, and
    // no IR is loaded. Removes the kInternalBlock (10.7 ms @ 48 kHz) FIFO latency a
    // CPU-only host would otherwise pay for parity with an absent GPU engine.
    bool direct_cpu_ = false;
    std::size_t gpu_extra_ = 0;
    int latency_samples_ = static_cast<int>(kInternalBlock);

    // Worker / live-reload state.
    std::thread worker_;
    std::atomic<bool> worker_run_{false};
    std::atomic<int> requested_engine_{0};
    // Engine=Auto resolves to this (0 = CPU, 1 = GPU), decided at prepare().
    std::atomic<int> auto_resolved_engine_{0};
    std::mutex model_req_mutex_;
    std::string requested_model_path_;
    std::atomic<std::uint32_t> model_req_generation_{0};
    std::uint32_t last_seen_model_gen_ = 0;

    // IR live-load request + display state (UI/worker only).
    std::mutex ir_req_mutex_;
    std::string requested_ir_path_;
    std::atomic<std::uint32_t> ir_req_generation_{0};
    std::uint32_t last_seen_ir_gen_ = 0;
    mutable std::mutex ir_mutex_;
    std::string ir_name_;

    // Authoritative model + display (UI/worker only; never the audio thread).
    mutable std::mutex model_mutex_;
    nam::NamRuntime model_;
    std::string model_name_ = "(no model)";
    std::string model_arch_ = "none";
    std::atomic<bool> user_model_loaded_{false};
    TransferCurve transfer_curve_{};
    std::atomic<std::uint32_t> model_generation_{0};

    // Live peak meters (audio thread writes, UI reads). in_env_/out_env_ are
    // audio-thread-only envelope state; the dB atomics are the published values.
    float in_env_ = 0.0f, out_env_ = 0.0f;
    std::atomic<float> in_level_db_{-120.0f};
    std::atomic<float> out_level_db_{-120.0f};

    // The loaded model's loudness metadata, published from the loader thread and
    // read on the audio thread; the per-mode output make-up gain is derived from
    // these in process() (see output_calibration.hpp). has_loudness=false means the
    // model carried no loudness metadata → every output mode is a no-op.
    std::atomic<bool>  model_has_loudness_{false};
    std::atomic<float> model_loudness_db_{0.0f};
};

inline std::unique_ptr<format::Processor> create_gpu_nam() {
    return std::make_unique<GpuNamProcessor>();
}

} // namespace pulp::examples
