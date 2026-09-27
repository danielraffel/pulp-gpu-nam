#include "gpu_nam_stamped_paced.hpp"
#include "gpu_nam_stamped_node.hpp"
#include <pulp/gpu_audio/gpu_audio_transport.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <thread>
#include <unistd.h>

namespace pulp::examples {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t nanoseconds(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
struct Row {
    std::uint64_t scheduled = 0, start = 0, end = 0, deadline = 0;
    const char* selected = "priming";
};
}
int run_stamped_paced(const nam::NamModel& model, unsigned frames, unsigned lead,
                     GpuNamCompletionOptions completion, const GpuNamPacedOptions& options) {
    constexpr unsigned channels = 2;
    if (!options.valid(frames, lead) || !completion.valid()) return 64;
    const auto inputs_count = options.blocks(frames), blocks = inputs_count + lead;
    const auto stride = std::size_t(channels) * frames;
    // Refuse to overwrite a prior raw receipt. Nothing is serialized while timed.
    const int fd = open(options.sidecar.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) { std::cerr << "sidecar_create_failed\n"; return 65; }
    FILE* sidecar = fdopen(fd, "w");
    if (!sidecar) { close(fd); return 65; }
    struct Close { FILE* f; ~Close() { if (f) fclose(f); } } file{sidecar};
    std::vector<float> input(blocks * stride), actual(blocks * stride);
    std::vector<Row> rows(blocks);
    for (std::uint64_t b = 0; b < inputs_count; ++b)
        for (unsigned ch = 0; ch < channels; ++ch)
            for (unsigned i = 0; i < frames; ++i) {
                const double sample = double(b * frames + i);
                input[b * stride + ch * frames + i] =
                    float(.07 * std::sin((.013 + ch * .007) * sample) +
                          .02 * std::cos((.037 + ch * .011) * sample));
            }
    std::array<nam::NamModel, channels> cpu{model, model};
    std::vector<float> delay(std::size_t(lead) * stride);
    std::unique_ptr<GpuNamStampedNode> node;
    gpu_audio::GpuAudioTransport transport;
    if (options.cpu_only) {
        for (auto& c : cpu) c.prewarm_block_aligned(frames);
    } else {
        node = GpuNamStampedNode::create(model, channels, frames, 48000, lead, completion);
        if (!node || !node->prepare() ||
            !transport.prepare(node.get(), {.ring_blocks=16, .run_worker_thread=true, .wake_on_write=true})) {
            std::cerr << "shared_prepare_failed=1 cause=unclassified\n"; return 8;
        }
        const auto capability = transport.capability_report();
        if (capability.path != gpu_audio::GpuAudioExecutionPath::SharedMemory ||
            capability.provider != gpu_audio::GpuAudioProvider::Dawn ||
            capability.prepared_lead_blocks != lead) return 8;
    }
    const float* ins[channels]{};
    float* outs[channels]{};
    audio::BufferView<const float> in(ins, channels, frames);
    audio::BufferView<float> out(outs, channels, frames);
    const auto cpu_start = std::clock();
    if (cpu_start == std::clock_t(-1)) return 66;
    const auto origin = Clock::now();
    const auto origin_ns = nanoseconds(origin);
    unsigned cursor = 0;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        for (unsigned ch = 0; ch < channels; ++ch) {
            ins[ch] = input.data() + b * stride + ch * frames;
            outs[ch] = actual.data() + b * stride + ch * frames;
        }
        auto& row = rows[b];
        row.scheduled = origin_ns + paced_offset_ns(b, frames);
        row.deadline = origin_ns + paced_offset_ns(b + 1, frames);
        std::this_thread::sleep_until(origin + std::chrono::nanoseconds(paced_offset_ns(b, frames)));
        const auto fallback_before = node ? node->fallback_reads() : 0;
        row.start = nanoseconds(Clock::now());
        if (options.cpu_only) {
            auto* slot = delay.data() + cursor * stride;
            for (unsigned ch = 0; ch < channels; ++ch) {
                std::copy_n(slot + ch * frames, frames, outs[ch]);
                cpu[ch].process(ins[ch], slot + ch * frames, frames);
            }
            cursor = (cursor + 1) % lead;
        } else transport.process(in, out, frames);
        row.end = nanoseconds(Clock::now());
        row.selected = b < lead ? "priming" : options.cpu_only ? "cpu_baseline" :
            node->fallback_reads() == fallback_before ? "gpu_delivered" : "cpu_fallback";
    }
    const auto loop_end_ns = nanoseconds(Clock::now());
    const auto cpu_loop_end = std::clock();
    const auto stats = transport.stats();
    const bool fenced_before_release = node && node->fenced();
    transport.release();
    const auto cpu_calls = node ? node->cpu_model_calls() : blocks * channels;
    const bool drained = !node || node->release();
    const auto cpu_drain_end = std::clock();
    if (cpu_loop_end == std::clock_t(-1) || cpu_drain_end == std::clock_t(-1) ||
        cpu_loop_end < cpu_start || cpu_drain_end < cpu_loop_end) return 66;
    if (options.inject_error) actual[lead * stride] += .25f;
    // Replay only after the measured callback loop and physical GPU retirement.
    // Independent CPU state supplies the same model and input history. Compare
    // against each output's declared delayed block, not against worker counters.
    std::array<nam::NamModel, channels> oracle{model, model};
    for (auto& c : oracle) c.prewarm_block_aligned(frames);
    std::vector<float> expected(stride);
    std::uint64_t mismatches = 0;
    double max_error = 0;
    auto compare = [&](float a, float e) {
        if (!paced_sample_matches(a,e)) ++mismatches;
        if (std::isfinite(a) && std::isfinite(e)) max_error=std::max(max_error,std::abs(double(a)-e));
    };
    for (std::size_t i=0;i<lead*stride;++i) compare(actual[i],0.f);
    for (std::uint64_t b=0;b<inputs_count;++b) {
        for (unsigned ch=0;ch<channels;++ch)
            oracle[ch].process(input.data()+b*stride+ch*frames,expected.data()+ch*frames,frames);
        for (std::size_t i=0;i<stride;++i) compare(actual[(b+lead)*stride+i],expected[i]);
    }
    std::uint64_t gpu=0,fallback=0,late_starts=0,deadline_misses=0,callback_ns=0,max_callback=0;
    fprintf(sidecar,"block,scheduled_ns,start_ns,end_ns,deadline_ns,selected,callback_ns,start_lateness_ns,deadline_missed\n");
    for (std::uint64_t b=0;b<blocks;++b) {
        const auto& row=rows[b]; const auto cost=row.end-row.start;
        const auto lateness=row.start>row.scheduled ? row.start-row.scheduled : 0;
        gpu += std::string_view(row.selected)=="gpu_delivered";
        fallback += std::string_view(row.selected)=="cpu_fallback";
        late_starts += lateness!=0; deadline_misses += row.end>row.deadline;
        callback_ns+=cost; max_callback=std::max(max_callback,cost);
        fprintf(sidecar,"%llu,%llu,%llu,%llu,%llu,%s,%llu,%llu,%d\n",
            (unsigned long long)b,(unsigned long long)row.scheduled,(unsigned long long)row.start,
            (unsigned long long)row.end,(unsigned long long)row.deadline,row.selected,
            (unsigned long long)cost,(unsigned long long)lateness,int(row.end>row.deadline));
    }
    const bool io_ok=fflush(sidecar)==0 && !ferror(sidecar);
    const bool passed=drained && io_ok && mismatches==0 && cpu_calls==blocks*channels;
    std::cout << "paced=1 engine=" << (options.cpu_only?"cpu":"stamped")
        << " scheduling=ordinary_os_thread hard_realtime=0 sample_rate=48000 channels=2 frames=" << frames << " lead=" << lead
        << " input_blocks=" << inputs_count << " drain_blocks=" << lead << " measured_blocks=" << blocks
        << " completion_policy=" << completion_policy_name(completion.policy)
        << " worker_wait_ns=" << completion.worker_wait_ns
        << " wake_on_write=" << (!options.cpu_only)
        << " worker_poll_interval_us=" << (options.cpu_only ? 0 : std::clamp(frames * 1'000'000u / 48000u / 4u, 50u, 2000u))
        << " cpu_model_calls=" << cpu_calls << " gpu_callbacks=" << gpu << " fallback_callbacks=" << fallback
        << " worker_produced=" << stats.produced_blocks << " transport_misses=" << stats.miss_blocks
        << " fenced_before_release=" << fenced_before_release
        << " input_dropped_frames=" << stats.input_dropped_frames << " resynced_blocks=" << stats.resynced_blocks
        << " callback_total_ns=" << callback_ns << " callback_max_ns=" << max_callback
        << " late_starts=" << late_starts << " callback_deadline_misses=" << deadline_misses
        << " wall_loop_ns=" << loop_end_ns-origin_ns
        << " process_cpu_ticks_per_second=" << CLOCKS_PER_SEC
        << " process_cpu_loop_seconds=" << double(cpu_loop_end-cpu_start)/CLOCKS_PER_SEC
        << " process_cpu_drain_seconds=" << double(cpu_drain_end-cpu_loop_end)/CLOCKS_PER_SEC
        << " max_error=" << max_error << " mismatched_samples=" << mismatches
        << " full_cpu_shadow=" << (!options.cpu_only) << " gpu_timestamps_available=0"
        << " phase_timestamps_available=callback_only\n";
    std::cout << "diagnostic_status=" << (passed?"passed":"failed") << '\n';
    // Correct fallback-only output is valid audio but is NOT useful GPU evidence.
    std::cout << "gpu_delivery_observed=" << (gpu>0) << '\n';
    return passed ? 0 : 7;
}
} // namespace pulp::examples
