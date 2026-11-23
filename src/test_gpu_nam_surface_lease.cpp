#include "gpu_nam_surface_lease.hpp"

#include <catch2/catch_test_macros.hpp>

namespace {

using pulp::examples::GpuNamSurfaceLeaseSource;

TEST_CASE("surface lease rejects an invalidated generation", "[nam][host]") {
    GpuNamSurfaceLeaseSource source;
    auto* surface = reinterpret_cast<pulp::render::GpuSurface*>(0x1);

    source.publish(surface);
    auto lease = source.lease();
    REQUIRE(lease.get() == surface);

    source.invalidate();
    REQUIRE(lease.get() == nullptr);
    REQUIRE(source.lease().get() == nullptr);
}

TEST_CASE("surface lease rejects a replaced generation", "[nam][host]") {
    GpuNamSurfaceLeaseSource source;
    auto* first = reinterpret_cast<pulp::render::GpuSurface*>(0x1);
    auto* second = reinterpret_cast<pulp::render::GpuSurface*>(0x2);

    source.publish(first);
    auto old_lease = source.lease();
    source.publish(second);

    REQUIRE(old_lease.get() == nullptr);
    REQUIRE(source.lease().get() == second);
}

} // namespace
