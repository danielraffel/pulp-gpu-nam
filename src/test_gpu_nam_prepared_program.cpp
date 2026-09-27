#include "gpu_nam_prepared_program.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pulp::examples::nam;

TEST_CASE("GPU NAM prepared WaveNet descriptor is typed and fail-closed",
          "[nam][gpu][prepared]") {
  GpuNamPreparedProgram program;
  REQUIRE(validate_gpu_nam_program(program) ==
          GpuNamProgramError::InvalidShape);

  program.channels = 2;
  program.kind = GpuNamProgramKind::WaveNet;
  program.block_size = 256;
  program.sample_rate = 48000;
  program.model_layers = 3;
  program.model_weights = 128;
  program.path = pulp::gpu_audio::GpuAudioExecutionPath::Staged;
  program.miss_policy = pulp::gpu_audio::MissPolicy::CpuFallback;
  program.algorithmic_lead_blocks = GpuNamPreparedProgram::kPreparedLeadBlocks;
  program.pipeline_depth = GpuNamPreparedProgram::kPreparedPipelineDepth;
  program.provider_slots = 2;
  REQUIRE(validate_gpu_nam_program(program) ==
          GpuNamProgramError::MissingCpuFallback);

  program.cpu_fallback_prepared = true;
  REQUIRE(validate_gpu_nam_program(program) == GpuNamProgramError::None);

  pulp::gpu_audio::GpuAudioCapabilityReport capability;
  capability.path = pulp::gpu_audio::GpuAudioExecutionPath::Staged;
  capability.eligibility = pulp::gpu_audio::GpuAudioEligibility::Eligible;
  capability.fallback_policy = pulp::gpu_audio::MissPolicy::CpuFallback;
  capability.prepared_lead_blocks = GpuNamPreparedProgram::kPreparedLeadBlocks;
  capability.prepared = true;
  capability.fallback_available = true;
  REQUIRE(validate_gpu_nam_program(program, capability) ==
          GpuNamProgramError::None);

  capability.path = pulp::gpu_audio::GpuAudioExecutionPath::SharedMemory;
  REQUIRE(validate_gpu_nam_program(program, capability) ==
          GpuNamProgramError::CapabilityMismatch);
}
