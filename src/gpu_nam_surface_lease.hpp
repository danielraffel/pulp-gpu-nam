#pragma once

#include "surface_lease.hpp"

#include <pulp/render/gpu_surface.hpp>

namespace pulp::examples {

// The SDK's GpuSurface is named through its own header rather than forward
// declared: an SDK may define it in another namespace and alias it into
// pulp::render, and a forward declaration would conflict with that alias.
using GpuNamSurfaceLease = BasicSurfaceLease<render::GpuSurface>;
using GpuNamSurfaceLeaseSource = BasicSurfaceLeaseSource<render::GpuSurface>;

} // namespace pulp::examples
