#pragma once

// Output level handling for the NAM player: Raw / Normalized / Calibrated.
//
// A NAM capture's absolute output level is arbitrary (it depends on how hot the
// reamp was), so three ways to present it:
//   - Raw        — the model's native level, unity make-up.
//   - Normalized — retarget the model's measured loudness to a FIXED reference
//                  (kNormalizeTargetDb), so any two models A/B at equal loudness.
//   - Calibrated — retarget to a USER-set reference level, so a calibrated rig
//                  reproduces one consistent monitoring level across model swaps.
//
// All three reuse the model's `metadata.loudness` (dBFS); a model without that
// metadata gets unity in every mode (no guessing). The retarget is clamped so an
// extreme or bogus loudness value can never apply a wild gain.
//
// Pure and dependency-free (only <algorithm>/<cmath>) so the gain math is unit-
// tested without the GPU/render stack.

#include <algorithm>
#include <cmath>

namespace pulp::examples::nam {

enum class OutputMode { Raw = 0, Normalized = 1, Calibrated = 2 };

// Map the 3-way parameter value (0..2, stepped) to the enum. Values below 0
// clamp to Raw and above 2 clamp to Calibrated so a stale/out-of-range automation
// value is still well-defined. Rounds to the nearest step.
inline OutputMode output_mode_from_param(float v) {
    const int m = static_cast<int>(std::lround(v));
    if (m <= 0) return OutputMode::Raw;
    if (m >= 2) return OutputMode::Calibrated;
    return OutputMode::Normalized;
}

// Linear make-up gain that retargets a model's measured loudness (dBFS) to
// `target_db`, clamped to ±`max_abs_db`. Returns unity when the model carries no
// loudness metadata (so the mode is a no-op rather than a guess) or when any
// input is non-finite.
inline float loudness_retarget_gain(bool has_loudness, double loudness_db,
                                    float target_db, float max_abs_db) {
    if (!has_loudness) return 1.0f;
    if (!std::isfinite(loudness_db) || !std::isfinite(target_db) ||
        !std::isfinite(max_abs_db)) {
        return 1.0f;
    }
    const float cap = std::abs(max_abs_db);
    float db = target_db - static_cast<float>(loudness_db);
    db = std::clamp(db, -cap, cap);
    return std::pow(10.0f, db / 20.0f);
}

// The output make-up factor for a given mode. `fixed_target_db` is the
// Normalized reference; `cal_target_db` is the user's Calibrated reference.
inline float output_mode_gain(OutputMode mode, bool has_loudness, double loudness_db,
                              float fixed_target_db, float cal_target_db,
                              float max_abs_db) {
    switch (mode) {
        case OutputMode::Raw:
            return 1.0f;
        case OutputMode::Normalized:
            return loudness_retarget_gain(has_loudness, loudness_db, fixed_target_db,
                                          max_abs_db);
        case OutputMode::Calibrated:
            return loudness_retarget_gain(has_loudness, loudness_db, cal_target_db,
                                          max_abs_db);
    }
    return 1.0f;
}

}  // namespace pulp::examples::nam
