#pragma once
#include "gpu_nam_host_probe.hpp"
#include <atomic>
namespace pulp::examples::host_probe {
inline std::atomic<unsigned> instances{0};
inline std::atomic<GpuNamProcessor*> instance{nullptr};
class Processor final : public GpuNamProcessor {
public:
    Processor() { if (instances.fetch_add(1)==0) instance.store(this); }
    ~Processor() override {
        GpuNamProcessor* expected=this;
        instance.compare_exchange_strong(expected,nullptr);
        instances.fetch_sub(1);
    }
};
inline std::unique_ptr<format::Processor> create() { return std::make_unique<Processor>(); }
}
extern "C" __attribute__((visibility("default")))
int gpu_nam_host_probe_v1(GpuNamHostProbeSnapshot* out) {
    using namespace pulp::examples::host_probe;
    if (!out || out->size!=sizeof(*out) || out->version!=1) return 1;
    auto* processor=instance.load();
    if (instances.load()!=1 || !processor) return 3;
    out->active=processor->gpu_engine_active();
    out->requested_engine=processor->requested_engine();
    out->prepared_engine=processor->prepared_engine();
    out->latency_samples=processor->latency_samples();
    const auto delivery=processor->gpu_delivery_snapshot();
    out->gpu_delivered=delivery.gpu_blocks;
    out->cpu_fallback=delivery.cpu_fallback_blocks;
    out->priming=delivery.priming_blocks;
    out->other=delivery.worker_output_blocks+delivery.silence_blocks+
               delivery.passthrough_blocks+delivery.invalid_blocks;
    return 0;
}
