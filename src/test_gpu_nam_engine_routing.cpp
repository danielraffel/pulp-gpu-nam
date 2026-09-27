#include "gpu_nam_engine.hpp"
#include <iostream>
using namespace pulp::examples::nam;
using namespace pulp::gpu_audio;
int main() {
#if defined(GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET)
  static_assert(kGpuNamEngineRoute == GpuNamEngineRoute::Stamped);
#elif defined(GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION)
  static_assert(kGpuNamEngineRoute == GpuNamEngineRoute::SharedSession);
#else
  static_assert(kGpuNamEngineRoute == GpuNamEngineRoute::Cloud);
#endif
  static_assert(kGpuNamPluginLead == 1);
  GpuNamPreparedProgram program;
  program.kind = GpuNamProgramKind::WaveNet;
  program.channels = 2;
  program.block_size = 512;
  program.sample_rate = 48000;
  program.model_layers = 1;
  program.model_weights = 9;
  program.algorithmic_lead_blocks = 1;
  program.pipeline_depth = 16;
  program.provider_slots = 4;
  program.path = GpuAudioExecutionPath::SharedMemory;
  program.provider = GpuAudioProvider::Dawn;
  program.miss_policy = MissPolicy::CpuFallback;
  program.provider_owned_resources = true;
  program.cpu_fallback_prepared = true;
  GpuAudioCapabilityReport report;
  report.path = program.path;
  report.provider = program.provider;
  report.eligibility = GpuAudioEligibility::Eligible;
  report.fallback_policy = program.miss_policy;
  report.prepared_lead_blocks = 1;
  report.prepared = true;
  report.fallback_available = true;
  if (bind_gpu_nam_engine_program(program, report) != GpuNamProgramError::None)
    return 1;
  for (unsigned mutation = 0; mutation < 6; ++mutation) {
    auto bad = report;
    if (mutation == 0)
      bad.path = GpuAudioExecutionPath::Staged;
    if (mutation == 1)
      bad.provider = GpuAudioProvider::Unknown;
    if (mutation == 2)
      bad.prepared_lead_blocks = 2;
    if (mutation == 3)
      bad.fallback_available = false;
    if (mutation == 4)
      bad.prepared = false;
    if (mutation == 5)
      bad.provider = GpuAudioProvider::Metal;
    if (bind_gpu_nam_engine_program(program, bad) == GpuNamProgramError::None)
      return 2 + mutation;
  }
  auto absent = program;
  absent.provider_owned_resources = false;
  if (bind_gpu_nam_engine_program(absent, report) == GpuNamProgramError::None)
    return 8;
  // An inner shared-memory session cannot upgrade its staged outer transport.
  program.path = GpuAudioExecutionPath::Staged;
  program.provider = GpuAudioProvider::Unknown;
  program.provider_owned_resources = false;
  report.path = program.path;
  report.provider = program.provider;
  if (bind_gpu_nam_engine_program(program, report) != GpuNamProgramError::None)
    return 9;
  std::cout
      << "routing_identity_checks=9 plugin_lead=1 staged_remains_staged=1\n";
}
