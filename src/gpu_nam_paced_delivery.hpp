#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace pulp::examples {

// Keep output selection distinct from numerical correctness and worker activity.
enum class PacedSelection : std::uint8_t {
    GpuDelivered, WorkerOutput, CpuFallback, Silence, Passthrough, Priming,
    InvalidRejected, CpuBaseline, AccountingError
};
using PacedDeliveryCounts = std::array<std::uint64_t, 7>;

template <class Snapshot>
PacedDeliveryCounts paced_delivery_counts(const Snapshot& value) noexcept {
    return {value.gpu_blocks, value.worker_output_blocks, value.cpu_fallback_blocks,
            value.silence_blocks, value.passthrough_blocks, value.priming_blocks,
            value.invalid_blocks};
}

inline const char* paced_selection_name(PacedSelection value) noexcept {
    switch (value) {
    case PacedSelection::GpuDelivered: return "gpu_delivered";
    case PacedSelection::WorkerOutput: return "worker_output";
    case PacedSelection::CpuFallback: return "cpu_fallback";
    case PacedSelection::Silence: return "silence";
    case PacedSelection::Passthrough: return "passthrough";
    case PacedSelection::Priming: return "priming";
    case PacedSelection::InvalidRejected: return "invalid_rejected";
    case PacedSelection::CpuBaseline: return "cpu_baseline";
    case PacedSelection::AccountingError: return "accounting_error";
    }
    return "accounting_error";
}

// The paced runner is the sole process() caller. Delivery counters are written
// by that caller, so exactly one increment must occur between these snapshots.
// Zero, multiple, reset or wrapped deltas are measurement errors, never GPU proof.
inline PacedSelection paced_delivery_selection(const PacedDeliveryCounts& before,
                                               const PacedDeliveryCounts& after) noexcept {
    PacedSelection selected = PacedSelection::AccountingError;
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (after[i] < before[i] || after[i] - before[i] > 1)
            return PacedSelection::AccountingError;
        if (after[i] != before[i]) {
            if (selected != PacedSelection::AccountingError)
                return PacedSelection::AccountingError;
            selected = static_cast<PacedSelection>(i);
        }
    }
    return selected;
}

inline bool paced_record_delivery(PacedDeliveryCounts& counts,
                                  PacedSelection selected) noexcept {
    const auto index = static_cast<std::size_t>(selected);
    if (index >= counts.size()) return false;
    ++counts[index];
    return true;
}

inline bool paced_delivery_reconciles(const PacedDeliveryCounts& initial,
                                     const PacedDeliveryCounts& stopped,
                                     const PacedDeliveryCounts& rows,
                                     std::uint64_t expected_rows) noexcept {
    auto remaining = expected_rows;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (stopped[i] < initial[i] || stopped[i] - initial[i] != rows[i] ||
            rows[i] > remaining)
            return false;
        remaining -= rows[i];
    }
    return remaining == 0;
}

} // namespace pulp::examples
