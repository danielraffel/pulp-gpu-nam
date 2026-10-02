#pragma once

#include "nam_model.hpp"
#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include <pulp/audio/buffer.hpp>

namespace pulp::examples {

// Experimental measurement seam for a GPU node whose CPU fallback must remain
// stateful but should not execute on the audio callback. The callback only
// copies into a preallocated input ring, publishes a sequence, and retires the
// corresponding output position. A worker advances an independent NamModel
// state and publishes latency-aligned fallback blocks. This is deliberately
// separate from GpuNamStampedNode until its readiness and miss behavior have
// been measured under real transport pressure.
class GpuNamCpuShadowWorker final {
  public:
    static constexpr std::uint64_t kNoSequence = UINT64_MAX;

    GpuNamCpuShadowWorker(const nam::NamModel& model, std::uint32_t channels,
                          std::uint32_t frames, std::uint32_t lead,
                          std::uint32_t capacity = 32)
        : model_(model), channels_(channels), frames_(frames), lead_(lead),
          capacity_(capacity), wake_(0) {}
    ~GpuNamCpuShadowWorker() { stop(); }

    GpuNamCpuShadowWorker(const GpuNamCpuShadowWorker&) = delete;
    GpuNamCpuShadowWorker& operator=(const GpuNamCpuShadowWorker&) = delete;

    bool prepare() {
        // Re-preparation is a lifecycle boundary; never resize or reset rings
        // while a prior worker can still access them.
        if (worker_.joinable())
            stop();
        if (channels_ == 0 || channels_ > 64 || frames_ == 0 || lead_ == 0 ||
            capacity_ > 64 || lead_ >= capacity_ || model_.arrays().empty())
            return false;
        input_.assign(static_cast<std::size_t>(capacity_) * channels_ * frames_, 0.f);
        output_.assign(input_.size(), 0.f);
        input_sequence_.assign(capacity_, kNoSequence);
        ready_sequence_ = std::make_unique<std::atomic<std::uint64_t>[]>(capacity_);
        for (std::uint32_t i = 0; i < capacity_; ++i)
            ready_sequence_[i].store(kNoSequence, std::memory_order_relaxed);
        for (std::uint32_t channel = 0; channel < channels_; ++channel) {
            cpu_[channel] = model_;
            cpu_[channel].prewarm_block_aligned(frames_);
        }
        write_sequence_.store(0, std::memory_order_relaxed);
        read_sequence_.store(0, std::memory_order_relaxed);
        retired_sequence_.store(0, std::memory_order_relaxed);
        submitted_.store(0, std::memory_order_relaxed);
        processed_.store(0, std::memory_order_relaxed);
        dropped_.store(0, std::memory_order_relaxed);
        ready_misses_.store(0, std::memory_order_relaxed);
        sequence_errors_.store(0, std::memory_order_relaxed);
        stop_requested_.store(false, std::memory_order_relaxed);
        prepared_ = true;
        return true;
    }

    bool start() {
        if (!prepared_ || worker_.joinable())
            return false;
        stop_requested_.store(false, std::memory_order_release);
        worker_ = std::thread([this] { run(); });
        return true;
    }

    void stop() noexcept {
        stop_requested_.store(true, std::memory_order_release);
        wake_.release();
        if (worker_.joinable())
            worker_.join();
    }

    // Audio-callback operation. No allocation, locking, waiting, or model
    // execution occurs here. The sequence must be monotonically contiguous.
    bool submit(const audio::BufferView<const float>& input,
                std::uint64_t sequence) noexcept {
        if (!prepared_ || sequence == kNoSequence)
            return false;
        const auto write = write_sequence_.load(std::memory_order_relaxed);
        const auto read = read_sequence_.load(std::memory_order_acquire);
        if (write - read >= capacity_) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (sequence != write) {
            sequence_errors_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const auto slot = static_cast<std::size_t>(write % capacity_);
        input_sequence_[slot] = sequence;
        ready_sequence_[slot].store(kNoSequence, std::memory_order_relaxed);
        for (std::uint32_t channel = 0; channel < channels_; ++channel) {
            auto* destination = input_.data() + slot * channels_ * frames_ + channel * frames_;
            const auto* source = channel < input.num_channels() &&
                                         input.num_samples() >= frames_ &&
                                         input.channel_ptr(channel) != nullptr
                                     ? input.channel_ptr(channel)
                                     : nullptr;
            if (source)
                std::copy_n(source, frames_, destination);
            else
                std::fill_n(destination, frames_, 0.f);
        }
        write_sequence_.store(write + 1, std::memory_order_release);
        submitted_.fetch_add(1, std::memory_order_relaxed);
        wake_.release();
        return true;
    }

    // Audio-callback operation. Retiring every callback position allows the
    // worker to reuse output slots without racing a fallback copy.
    void retire(std::uint64_t sequence) noexcept {
        const auto next = sequence == kNoSequence ? 0 : sequence + 1;
        auto current = retired_sequence_.load(std::memory_order_relaxed);
        while (next > current &&
               !retired_sequence_.compare_exchange_weak(
                   current, next, std::memory_order_release,
                   std::memory_order_relaxed)) {}
    }

    // Audio-callback operation. Returns false when the exact delayed sequence
    // is not ready; callers must use their bounded miss policy in that case.
    bool copy_fallback(std::uint64_t sequence, audio::BufferView<float>& output) noexcept {
        if (!prepared_ || sequence == kNoSequence || output.num_samples() < frames_) {
            ready_misses_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const auto slot = static_cast<std::size_t>(sequence % capacity_);
        if (ready_sequence_[slot].load(std::memory_order_acquire) != sequence) {
            ready_misses_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        for (std::uint32_t channel = 0; channel < channels_ &&
                                      channel < output.num_channels(); ++channel)
            std::copy_n(output_.data() + slot * channels_ * frames_ + channel * frames_,
                        frames_, output.channel_ptr(channel));
        return true;
    }

    std::uint64_t submitted() const noexcept { return submitted_.load(); }
    std::uint64_t processed() const noexcept { return processed_.load(); }
    std::uint64_t dropped() const noexcept { return dropped_.load(); }
    std::uint64_t ready_misses() const noexcept { return ready_misses_.load(); }
    std::uint64_t sequence_errors() const noexcept { return sequence_errors_.load(); }

  private:
    void run() noexcept {
        while (!stop_requested_.load(std::memory_order_acquire)) {
            wake_.acquire();
            for (;;) {
                const auto read = read_sequence_.load(std::memory_order_relaxed);
                const auto write = write_sequence_.load(std::memory_order_acquire);
                if (read >= write)
                    break;
                const auto slot = static_cast<std::size_t>(read % capacity_);
                const auto sequence = input_sequence_[slot];
                if (sequence != read)
                    sequence_errors_.fetch_add(1, std::memory_order_relaxed);
                // Do not overwrite a fallback slot until the audio callback has
                // retired that timeline position. This is the key safety gate.
                while (sequence >= retired_sequence_.load(std::memory_order_acquire) +
                                      capacity_ &&
                       !stop_requested_.load(std::memory_order_acquire))
                    std::this_thread::yield();
                for (std::uint32_t channel = 0; channel < channels_; ++channel)
                    cpu_[channel].process(
                        input_.data() + slot * channels_ * frames_ + channel * frames_,
                        output_.data() + slot * channels_ * frames_ + channel * frames_, frames_);
                ready_sequence_[slot].store(sequence, std::memory_order_release);
                processed_.fetch_add(1, std::memory_order_relaxed);
                read_sequence_.store(read + 1, std::memory_order_release);
            }
        }
    }

    nam::NamModel model_;
    std::array<nam::NamModel, 64> cpu_;
    std::uint32_t channels_, frames_, lead_, capacity_;
    std::vector<float> input_, output_;
    std::vector<std::uint64_t> input_sequence_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> ready_sequence_;
    std::atomic<std::uint64_t> write_sequence_{0}, read_sequence_{0}, retired_sequence_{0};
    std::atomic<std::uint64_t> submitted_{0}, processed_{0}, dropped_{0}, ready_misses_{0};
    std::atomic<std::uint64_t> sequence_errors_{0};
    std::counting_semaphore<64> wake_;
    std::atomic<bool> stop_requested_{false};
    std::thread worker_;
    bool prepared_ = false;
};

} // namespace pulp::examples
