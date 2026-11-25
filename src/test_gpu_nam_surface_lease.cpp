#include "surface_lease.hpp"

#include <catch2/catch_test_macros.hpp>

namespace {

struct FakeSurface {};
using LeaseSource = pulp::examples::BasicSurfaceLeaseSource<FakeSurface>;

TEST_CASE("surface lease rejects an invalidated generation", "[nam][host]") {
    LeaseSource source;
    FakeSurface surface;

    source.publish(&surface);
    auto lease = source.lease();
    REQUIRE(lease.get() == &surface);

    source.invalidate();
    REQUIRE(lease.get() == nullptr);
    REQUIRE(source.lease().get() == nullptr);
}

TEST_CASE("surface lease rejects a replaced generation", "[nam][host]") {
    LeaseSource source;
    FakeSurface first;
    FakeSurface second;

    source.publish(&first);
    auto old_lease = source.lease();
    source.publish(&second);

    REQUIRE(old_lease.get() == nullptr);
    REQUIRE(source.lease().get() == &second);
}

} // namespace
