#pragma once

// Backend-neutral preparation metadata for the GPU NAM WaveNet path.
//
// This is intentionally a declaration only.  It contains no Dawn/Metal
// handles, queues, rings, callbacks, or executable provider state.  The
// descriptor lets a host inspect what was prepared and which CPU fallback is
// ready without mistaking the existing staged transport for shared-memory
// execution.

#include <cstdint>

#include <pulp/gpu_audio/gpu_audio_capability.hpp>

namespace pulp::examples::nam {

enum class GpuNamProgramError : std::uint8_t {
  None = 0,
  InvalidShape,
  MissingLead,
  MissingPipeline,
  MissingProviderSlots,
  MissingCpuFallback,
  CapabilityUnavailable,
  CapabilityMismatch,
};

enum class GpuNamProgramKind : std::uint8_t {
  Unknown = 0,
  WaveNet = 1,
};

struct GpuNamPreparedProgram {
  static constexpr std::uint32_t kPreparedLeadBlocks = 1;
  static constexpr std::uint32_t kPreparedPipelineDepth = 2;

  // WaveNet is the only Pulp GPU-NAM execution family today.  Keeping the
  // family typed prevents a future ConvNet/LSTM descriptor from being
  // accepted by the WaveNet GPU node accidentally.
  GpuNamProgramKind kind = GpuNamProgramKind::Unknown;
  std::uint32_t channels = 0;
  std::uint32_t block_size = 0;
  std::uint32_t sample_rate = 0;
  std::uint32_t model_layers = 0;
  std::uint32_t model_weights = 0;
  std::uint32_t receptive_field = 0;
  std::uint32_t algorithmic_lead_blocks = 0;
  std::uint32_t pipeline_depth = 0;
  std::uint32_t provider_slots = 0;
  gpu_audio::GpuAudioExecutionPath path =
      gpu_audio::GpuAudioExecutionPath::Unavailable;
  gpu_audio::GpuAudioProvider provider = gpu_audio::GpuAudioProvider::Unknown;
  gpu_audio::MissPolicy miss_policy = gpu_audio::MissPolicy::Silence;
  bool provider_owned_resources = false;
  bool cpu_fallback_prepared = false;
};

constexpr GpuNamProgramError
validate_gpu_nam_program(const GpuNamPreparedProgram &program) noexcept {
  if (program.kind != GpuNamProgramKind::WaveNet || program.channels == 0 ||
      program.block_size == 0 || program.sample_rate == 0 ||
      program.model_layers == 0 || program.model_weights == 0)
    return GpuNamProgramError::InvalidShape;
  if (program.path == gpu_audio::GpuAudioExecutionPath::Unavailable)
    return GpuNamProgramError::CapabilityUnavailable;
  if (program.algorithmic_lead_blocks == 0)
    return GpuNamProgramError::MissingLead;
  if (program.pipeline_depth <= program.algorithmic_lead_blocks)
    return GpuNamProgramError::MissingPipeline;
  if (program.provider_slots == 0)
    return GpuNamProgramError::MissingProviderSlots;
  if (program.miss_policy == gpu_audio::MissPolicy::CpuFallback &&
      !program.cpu_fallback_prepared)
    return GpuNamProgramError::MissingCpuFallback;
  return GpuNamProgramError::None;
}

constexpr GpuNamProgramError validate_gpu_nam_program(
    const GpuNamPreparedProgram &program,
    const gpu_audio::GpuAudioCapabilityReport &capability) noexcept {
  const auto descriptor = validate_gpu_nam_program(program);
  if (descriptor != GpuNamProgramError::None)
    return descriptor;
  if (!capability.prepared ||
      capability.eligibility != gpu_audio::GpuAudioEligibility::Eligible)
    return GpuNamProgramError::CapabilityUnavailable;
  if (capability.path != program.path ||
      capability.fallback_policy != program.miss_policy ||
      capability.prepared_lead_blocks != program.algorithmic_lead_blocks)
    return GpuNamProgramError::CapabilityMismatch;
  // A shared-memory program must be authenticated against the same provider
  // identity that prepared its resources.  Staged consumers may leave the
  // provider unspecified, but a SharedRequired WaveNet descriptor cannot
  // silently accept a different backend (or a descriptor without provider
  // owned resources) and then report that shared execution is available.
  if (program.path == gpu_audio::GpuAudioExecutionPath::SharedMemory &&
      (program.provider == gpu_audio::GpuAudioProvider::Unknown ||
       capability.provider != program.provider || !program.provider_owned_resources))
    return GpuNamProgramError::CapabilityMismatch;
  if (program.miss_policy == gpu_audio::MissPolicy::CpuFallback &&
      !capability.fallback_available)
    return GpuNamProgramError::CapabilityMismatch;
  return GpuNamProgramError::None;
}

} // namespace pulp::examples::nam
