#pragma once

// Optional in-process consumer seam for Forge and other Pulp graph hosts.
// GPU-NAM remains the authority for its NAM model, GPU provider, controls,
// fallback policy, and delivery accounting. This adapter only exposes the
// existing Processor factory; it does not duplicate any DSP implementation.

#include "gpu_nam_processor.hpp"

namespace pulp::examples::gpu_nam_forge_adapter {

inline std::unique_ptr<format::Processor> create_processor() {
    return create_gpu_nam();
}

}  // namespace pulp::examples::gpu_nam_forge_adapter
