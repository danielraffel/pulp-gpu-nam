#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace pulp::examples {

// A generation-checked, non-owning handle to a host-owned surface. The
// templates take the surface type as a parameter so this header needs no GPU
// declarations: whichever namespace an SDK defines its GpuSurface in, the
// consumer names it once in gpu_nam_surface_lease.hpp.
template <class Surface>
struct SurfaceLeaseState {
    std::atomic<Surface*> surface{nullptr};
    std::atomic<std::uint64_t> generation{0};
};

template <class Surface>
class BasicSurfaceLeaseSource;

template <class Surface>
class BasicSurfaceLease {
public:
    BasicSurfaceLease() = default;

    Surface* get() const noexcept {
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
    friend class BasicSurfaceLeaseSource<Surface>;
    BasicSurfaceLease(std::weak_ptr<SurfaceLeaseState<Surface>> state,
                      std::uint64_t generation)
        : state_(std::move(state)), generation_(generation) {}

    std::weak_ptr<SurfaceLeaseState<Surface>> state_;
    std::uint64_t generation_ = 0;
};

template <class Surface>
class BasicSurfaceLeaseSource {
public:
    BasicSurfaceLeaseSource()
        : state_(std::make_shared<SurfaceLeaseState<Surface>>()) {}

    BasicSurfaceLease<Surface> lease() const noexcept {
        return BasicSurfaceLease<Surface>(
            state_, state_->generation.load(std::memory_order_acquire));
    }

    void publish(Surface* surface) noexcept {
        if (!surface) {
            invalidate();
            return;
        }
        state_->surface.store(nullptr, std::memory_order_release);
        state_->generation.fetch_add(1, std::memory_order_acq_rel);
        state_->surface.store(surface, std::memory_order_release);
    }

    void invalidate() noexcept {
        state_->surface.store(nullptr, std::memory_order_release);
        state_->generation.fetch_add(1, std::memory_order_acq_rel);
    }

private:
    std::shared_ptr<SurfaceLeaseState<Surface>> state_;
};

} // namespace pulp::examples
