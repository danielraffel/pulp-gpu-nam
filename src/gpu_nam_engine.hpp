#pragma once

#include "gpu_nam_prepared_program.hpp"
#include "nam_model.hpp"
#if defined(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET) &&                           \
    defined(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
#error "Select only one experimental GPU NAM adapter"
#endif
#if defined(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
#include "gpu_nam_stamped_node.hpp"
#elif defined(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
#include "gpu_nam_shared_session_node.hpp"
#else
#include "gpu_nam_cloud_node.hpp"
#endif
#include <limits>
#include <memory>
#include <string>

namespace pulp::examples::nam {

enum class GpuNamEngineRoute { Cloud, SharedSession, Stamped };
#if defined(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
inline constexpr auto kGpuNamEngineRoute = GpuNamEngineRoute::Stamped;
#elif defined(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
inline constexpr auto kGpuNamEngineRoute = GpuNamEngineRoute::SharedSession;
#else
inline constexpr auto kGpuNamEngineRoute = GpuNamEngineRoute::Cloud;
#endif
inline constexpr bool kGpuNamPreparationBoundEngine =
    kGpuNamEngineRoute == GpuNamEngineRoute::Stamped;
inline constexpr bool kGpuNamAutoMayOffload =
    kGpuNamEngineRoute == GpuNamEngineRoute::Cloud;

// Preserve the plugin's existing delay. Longer lead remains an explicit future
// stopped-preparation choice, never an unreported live engine change.
inline constexpr std::uint32_t kGpuNamPluginLead = 1;

struct PreparedGpuNamEngine {
  std::unique_ptr<gpu_audio::GpuAudioNode> node;
  std::string backend;
  GpuNamPreparedProgram program;
};

// The capability report describes the prepared adapter, not accepted GPU audio.
// In particular the old shared session still has a staged outer transport.
inline GpuNamProgramError bind_gpu_nam_engine_program(
    const GpuNamPreparedProgram &program,
    const gpu_audio::GpuAudioCapabilityReport &capability) noexcept {
  const auto error = validate_gpu_nam_program(program, capability);
  if (error != GpuNamProgramError::None)
    return error;
  if (program.provider != capability.provider ||
      (program.path == gpu_audio::GpuAudioExecutionPath::SharedMemory &&
       (!program.provider_owned_resources ||
        capability.provider == gpu_audio::GpuAudioProvider::Unknown)))
    return GpuNamProgramError::CapabilityMismatch;
  return GpuNamProgramError::None;
}

inline PreparedGpuNamEngine prepare_gpu_nam_engine(const NamModel *model,
                                                   std::uint32_t channels,
                                                   std::uint32_t frames,
                                                   std::uint32_t sample_rate) {
  PreparedGpuNamEngine result;
  if (!model || channels == 0 || channels > 64 || frames == 0 ||
      sample_rate == 0 || model->arrays().empty() ||
      model->weights_size() > std::numeric_limits<std::uint32_t>::max() ||
      model->arrays().size() > std::numeric_limits<std::uint32_t>::max())
    return result;
  const auto receptive_field = model->receptive_field();
  if (receptive_field < 0 || static_cast<std::uint64_t>(receptive_field) >
                                 std::numeric_limits<std::uint32_t>::max())
    return result;
#if defined(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
  auto node = GpuNamStampedNode::create(*model, channels, frames, sample_rate,
                                        kGpuNamPluginLead);
  if (!node || !node->prepare() || node->fenced())
    return result;
  result.backend = "Dawn stamped shared WaveNet (experimental)";
#elif defined(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
  auto node = std::make_unique<GpuNamSharedSessionNode>(
      channels, frames, sample_rate, model, kGpuNamPluginLead);
  if (!node->prepare() || !node->gpu_available())
    return result;
  result.backend = node->backend();
#else
  auto node =
      std::make_unique<GpuNamCloudNode>(channels, frames, sample_rate, model);
  if (!node->prepare() || !node->gpu_available())
    return result;
  result.backend = node->backend();
  result.program = node->prepared_program();
#endif
#if defined(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET) ||                           \
    defined(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
  auto &program = result.program;
  program.kind = GpuNamProgramKind::WaveNet;
  program.channels = channels;
  program.block_size = frames;
  program.sample_rate = sample_rate;
  program.model_layers = static_cast<std::uint32_t>(model->arrays().size());
  program.model_weights = static_cast<std::uint32_t>(model->weights_size());
  program.receptive_field = static_cast<std::uint32_t>(receptive_field);
  program.algorithmic_lead_blocks = kGpuNamPluginLead;
  program.miss_policy = gpu_audio::MissPolicy::CpuFallback;
  program.cpu_fallback_prepared = true;
#if defined(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
  // These are the immutable settings in GpuNamStampedNode::create: sixteen
  // stamped queue records and sixteen provider slots per mono session. Queue
  // capacity is not simultaneous GPU concurrency (one group is in flight).
  program.pipeline_depth = 16;
  program.provider_slots = 16 * channels;
  program.path = gpu_audio::GpuAudioExecutionPath::SharedMemory;
  program.provider = gpu_audio::GpuAudioProvider::Dawn;
  program.provider_owned_resources = true;
#else
  program.pipeline_depth = 2;
  program.provider_slots = 2 * channels;
  program.path = gpu_audio::GpuAudioExecutionPath::Staged;
  program.provider = gpu_audio::GpuAudioProvider::Unknown;
  program.provider_owned_resources = false;
#endif
#endif
  result.node = std::move(node);
  return result;
}
} // namespace pulp::examples::nam
