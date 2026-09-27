#pragma once

namespace gpu_nam_diagnostic {
// Control-thread ordering for a stopped physical epoch. Callback completion
// notification alone cannot authorize reading or releasing callback owners.
enum class EpochStopResult { Complete, AdmissionNotDrained, SnapshotRejected };
template<class Stop, class Snapshot, class Deactivate>
EpochStopResult stop_epoch(Stop&& stop, Snapshot&& snapshot, Deactivate&& deactivate) {
    if (!stop()) return EpochStopResult::AdmissionNotDrained;
    if (!snapshot()) return EpochStopResult::SnapshotRejected;
    deactivate();
    return EpochStopResult::Complete;
}
}
