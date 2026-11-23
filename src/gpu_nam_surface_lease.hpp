#pragma once

#include <pulp/render/gpu_surface.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace pulp::examples {

struct GpuNamSurfaceLeaseState {
    std::atomic<render::GpuSurface*> surface{nullptr};
    std::atomic<std::uint64_t> generation{0};
};

class GpuNamSurfaceLease {
public:
    GpuNamSurfaceLease() = default;

    render::GpuSurface* get() const noexcept {
        auto state = state_.lock();
        if (!state || state->generation.load(std::memory_order_acquire) != generation_)
            return nullptr;
        auto* surface = state->surface.load(std::memory_order_acquire);
        return state->generation.load(std::memory_order_acquire) == generation_
                   ? surface
                   : nullptr;
    }

    std::uint64_t generation() const noexcept { return generation_; }
    explicit operator bool() const noexcept { return get() != nullptr; }

private:
    friend class GpuNamSurfaceLeaseSource;
    GpuNamSurfaceLease(std::weak_ptr<GpuNamSurfaceLeaseState> state,
                       std::uint64_t generation)
        : state_(std::move(state)), generation_(generation) {}

    std::weak_ptr<GpuNamSurfaceLeaseState> state_;
    std::uint64_t generation_ = 0;
};

class GpuNamSurfaceLeaseSource {
public:
    GpuNamSurfaceLeaseSource()
        : state_(std::make_shared<GpuNamSurfaceLeaseState>()) {}

    GpuNamSurfaceLease lease() const noexcept {
        return GpuNamSurfaceLease(state_, state_->generation.load(std::memory_order_acquire));
    }

    void publish(render::GpuSurface* surface) noexcept {
        if (!surface) {
            invalidate();
            return;
        }
        state_->surface.store(nullptr, std::memory_order_release);
        const auto generation = state_->generation.fetch_add(1, std::memory_order_acq_rel) + 1;
        (void)generation;
        state_->surface.store(surface, std::memory_order_release);
    }

    void invalidate() noexcept {
        state_->surface.store(nullptr, std::memory_order_release);
        state_->generation.fetch_add(1, std::memory_order_acq_rel);
    }

private:
    std::shared_ptr<GpuNamSurfaceLeaseState> state_;
};

} // namespace pulp::examples
