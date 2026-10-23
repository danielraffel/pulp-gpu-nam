#pragma once

// Shape limits every .nam loader enforces before it allocates anything sized by
// the file. A capture is untrusted input: its config alone (a kernel size times a
// dilation, or a channel count) can otherwise ask for gigabytes of history or
// weights, or a prewarm that runs for minutes, before any weight-count check.
// Real captures sit orders of magnitude below every limit here.

#include <cstddef>

namespace pulp::examples::nam {

// Sum over layers of (kernel - 1) * dilation, in samples: the oldest input a
// model output can depend on. Matches the GPU WaveNet path's 2^20-sample bound.
inline constexpr long kMaxReceptiveField = 1L << 20;

// Largest kernel size a layer may declare (the A2 loader's existing bound).
inline constexpr long kMaxKernelSize = 1L << 12;

// Largest channel / head width a layer may declare.
inline constexpr long kMaxChannels = 1L << 12;

// Upper bound on the floats a model may allocate for its weights plus its
// per-layer input history (about 256 MB).
inline constexpr std::size_t kMaxModelFloats = std::size_t{1} << 26;

}  // namespace pulp::examples::nam
