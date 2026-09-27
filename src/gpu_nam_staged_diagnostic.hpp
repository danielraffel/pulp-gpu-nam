#pragma once
#include "gpu_nam.hpp"
#include <pulp/gpu_audio/gpu_audio_node.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>

namespace pulp::examples {
// Diagnostic only: old staged primitive with the same delayed full CPU shadow
// as the stamped adapter. No worker-side CPU substitution is permitted.
class GpuNamStagedDiagnostic final : public gpu_audio::GpuAudioNode {
public:
    GpuNamStagedDiagnostic(const nam::NamModel& model, unsigned frames, unsigned lead,
                           bool inject_failure)
        : model_(model), frames_(frames), lead_(lead), inject_failure_(inject_failure) {}
    gpu_audio::GpuAudioNodeDescriptor descriptor() const override {
        return {"StagedNamDiagnostic", 2, 2, frames_, 48000, lead_,
                gpu_audio::MissPolicy::CpuFallback, true};
    }
    bool prepare() override {
        device_ = render::GpuCompute::create();
        if (!device_ || !device_->initialize_standalone()) return false;
        std::vector<float> zeros(frames_), scratch(frames_);
        for (unsigned ch=0;ch<2;++ch) {
            if (!gpu_[ch].prepare_with(*device_,model_,frames_,ch)) return false;
            for (std::uint64_t b=0;b<model_.prewarm_block_count(frames_);++b)
                if (!gpu_[ch].forward(zeros.data(),scratch.data(),frames_)) return false;
            cpu_[ch]=model_;
            cpu_[ch].prewarm_block_aligned(frames_);
            ring_[ch].assign(std::size_t(lead_)*frames_,0.f);
            due_[ch].assign(frames_,0.f);
        }
        cursor_=0; cpu_calls_=fallback_reads_=0; failed_forwards_=0;
        return true;
    }
    void prime_fallback(const audio::BufferView<const float>& input, unsigned n) noexcept override {
        if (n!=frames_) return;
        for (unsigned ch=0;ch<2;++ch) {
            auto* slot=ring_[ch].data()+std::size_t(cursor_)*frames_;
            std::copy_n(slot,frames_,due_[ch].data());
            cpu_[ch].process(input.channel_ptr(ch),slot,frames_);
            ++cpu_calls_;
        }
        cursor_=(cursor_+1)%lead_;
    }
    void process_cpu_fallback(const audio::BufferView<const float>&,
                              audio::BufferView<float>& output, unsigned n) noexcept override {
        ++fallback_reads_;
        for (unsigned ch=0;ch<2;++ch) std::copy_n(due_[ch].data(),n,output.channel_ptr(ch));
    }
    void process_block(const audio::BufferView<const float>& input,
                       audio::BufferView<float>& output, unsigned n) override {
        for (unsigned ch=0;ch<2;++ch) {
            if (inject_failure_ || !gpu_[ch].forward(input.channel_ptr(ch),output.channel_ptr(ch),n)) {
                failed_forwards_.fetch_add(1,std::memory_order_relaxed);
                // The legacy void API cannot reject a result. Poison it visibly;
                // the diagnostic refuses both numerical acceptance and GPU credit.
                std::fill_n(output.channel_ptr(ch),n,std::numeric_limits<float>::quiet_NaN());
            }
        }
    }
    std::uint64_t fallback_reads() const { return fallback_reads_; }
    std::uint64_t cpu_model_calls() const { return cpu_calls_; }
    std::uint64_t failed_forwards() const { return failed_forwards_.load(std::memory_order_relaxed); }
    std::string backend() const { return device_ ? device_->capabilities().backend : ""; }
private:
    nam::NamModel model_;
    unsigned frames_,lead_,cursor_=0;
    bool inject_failure_;
    std::unique_ptr<render::GpuCompute> device_;
    std::array<nam::GpuNam,2> gpu_;
    std::array<nam::NamModel,2> cpu_;
    std::array<std::vector<float>,2> ring_,due_;
    std::uint64_t cpu_calls_=0,fallback_reads_=0;
    std::atomic<std::uint64_t> failed_forwards_{0};
};
}
