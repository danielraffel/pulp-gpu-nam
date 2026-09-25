#include "gpu_nam_prepared_program.hpp"

#include <catch2/catch_test_macros.hpp>

#if __has_include(<pulp/gpu_audio/gpu_audio_program.hpp>)
#include <pulp/gpu_audio/gpu_audio_program.hpp>
#define GPU_NAM_HAS_TYPED_SDK_PROGRAM_CONTRACT 1
#else
#define GPU_NAM_HAS_TYPED_SDK_PROGRAM_CONTRACT 0
#endif

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

TEST_CASE("GPU NAM shared-provider admission is authenticated and fail-closed",
          "[nam][gpu][prepared][shared]") {
#if GPU_NAM_HAS_TYPED_SDK_PROGRAM_CONTRACT
  // This is a consumer contract fixture only.  No Dawn/Metal provider is
  // constructed here.  It proves that a future shared provider can be admitted
  // only when the SDK capability snapshot authenticates the same path/provider
  // identity and that the generic Pulp program validator agrees with GPU NAM's
  // local preparation metadata.
  pulp::gpu_audio::GpuAudioProgramDescriptor sdk_program;
  sdk_program.kind = pulp::gpu_audio::GpuAudioProgramKind::Neural;
  sdk_program.path = pulp::gpu_audio::GpuAudioExecutionPath::SharedMemory;
  sdk_program.provider = pulp::gpu_audio::GpuAudioProvider::Dawn;
  sdk_program.miss_policy = pulp::gpu_audio::MissPolicy::CpuFallback;
  sdk_program.channels = 2;
  sdk_program.block_size = 256;
  sdk_program.sample_rate = 48000;
  sdk_program.algorithmic_lead_blocks = 1;
  sdk_program.pipeline_depth = 2;
  sdk_program.provider_slots = 2;
  sdk_program.provider_owned_resources = true;
  sdk_program.cpu_fallback_prepared = true;

  pulp::gpu_audio::GpuAudioCapabilityReport capability;
  capability.path = pulp::gpu_audio::GpuAudioExecutionPath::SharedMemory;
  capability.provider = pulp::gpu_audio::GpuAudioProvider::Dawn;
  capability.eligibility = pulp::gpu_audio::GpuAudioEligibility::Eligible;
  capability.fallback_policy = pulp::gpu_audio::MissPolicy::CpuFallback;
  capability.prepared_lead_blocks = 1;
  capability.prepared = true;
  capability.fallback_available = true;

  const auto sdk_validation =
      pulp::gpu_audio::validate_gpu_audio_program(sdk_program, capability);
  REQUIRE(sdk_validation.accepted());

  GpuNamPreparedProgram nam_program;
  nam_program.kind = GpuNamProgramKind::WaveNet;
  nam_program.channels = sdk_program.channels;
  nam_program.block_size = sdk_program.block_size;
  nam_program.sample_rate = sdk_program.sample_rate;
  nam_program.model_layers = 1;
  nam_program.model_weights = 1;
  nam_program.algorithmic_lead_blocks = sdk_program.algorithmic_lead_blocks;
  nam_program.pipeline_depth = sdk_program.pipeline_depth;
  nam_program.provider_slots = sdk_program.provider_slots;
  nam_program.path = sdk_program.path;
  nam_program.provider = sdk_program.provider;
  nam_program.miss_policy = sdk_program.miss_policy;
  nam_program.provider_owned_resources = sdk_program.provider_owned_resources;
  nam_program.cpu_fallback_prepared = sdk_program.cpu_fallback_prepared;
  REQUIRE(validate_gpu_nam_program(nam_program, capability) ==
          GpuNamProgramError::None);

  // A staged capability or a different provider identity cannot satisfy a
  // SharedRequired consumer, even if all other metadata remains valid.
  capability.path = pulp::gpu_audio::GpuAudioExecutionPath::Staged;
  REQUIRE_FALSE(
      pulp::gpu_audio::validate_gpu_audio_program(sdk_program, capability).accepted());
  REQUIRE(validate_gpu_nam_program(nam_program, capability) !=
          GpuNamProgramError::None);

  capability.path = pulp::gpu_audio::GpuAudioExecutionPath::SharedMemory;
  capability.provider = pulp::gpu_audio::GpuAudioProvider::Metal;
  REQUIRE_FALSE(
      pulp::gpu_audio::validate_gpu_audio_program(sdk_program, capability).accepted());
  REQUIRE(validate_gpu_nam_program(nam_program, capability) !=
          GpuNamProgramError::None);
#else
  // The current release pin predates the typed program contract.  Keep this
  // test explicit rather than pretending that a staged transport is shared;
  // the same fixture runs when an installed SDK or submodule exposes the API.
  SKIP("Pulp SDK has no gpu_audio_program.hpp; shared admission remains unavailable");
#endif
}
