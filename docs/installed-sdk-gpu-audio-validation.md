# Installed SDK GPU-audio validation

This is the handoff for validating GPU NAM against an exact installed Pulp SDK
once a Pulp #8848 SDK prefix is available. It checks the public consumer
surface, not an invented WaveNet provider.

Run from the GPU-NAM repository root:

```sh
scripts/validate-installed-pulp-gpu-audio.sh \
  --sdk-prefix /absolute/path/to/pulp-sdk \
  --sdk-commit a29961a987e59dff1441726c80d35f652f571f5c \
  --build-dir /tmp/gpu-nam-sdk-8848 \
  --receipt /tmp/gpu-nam-sdk-8848/receipt.json \
  --jobs 4
```

`--sdk-commit` is required so the receipt cannot silently describe an unknown
or stale SDK. A release manifest or the exact Pulp source SHA may be supplied;
the script does not put credentials or receipt contents in process arguments
beyond that public provenance identifier.

The script performs these checks in order:

| Step | Check | Evidence |
| --- | --- | --- |
| 1 | Installed `PulpConfig.cmake`, `gpu_audio_capability.hpp`, `gpu_audio_program.hpp`, and `gpu_convolver.hpp` exist. | Prefix/header paths and exact SDK identifier are recorded. |
| 2 | `GpuConvolver::ProviderPolicy::SharedRequired` and `GpuAudioProgramDescriptor` compile with C++20. | A temporary syntax-only probe proves the #8848/#8843 public API without constructing a provider. |
| 3 | GPU NAM configures with `GPU_NAM_USE_INSTALLED_PULP=ON`. | CMake cache and configure output show the installed SDK path. |
| 4 | `gpu-nam-prepared-program-test` and `gpu-nam-gpu-audio-capability-probe` build and pass. | CTest output proves the consumer metadata and fail-closed capability behavior. |

The receipt is JSON with schema
`gpu-nam.installed-sdk-gpu-audio-validation.v1`. It records the GPU-NAM source
head, SDK identifier and prefix, CMake build directory, bounded job count, and
one of these dispositions for each check:

- `pass` means the public API or consumer validation was proven.
- `blocked_provider_not_implemented` is the required result for shared WaveNet
  execution at this stage. Pulp #8848's `SharedIoProgramSession` is a private
  lifecycle seam, and Pulp currently authenticates shared execution only for
  its own `GpuConvolver` adapter. GPU NAM must remain staged with CPU fallback
  until a Pulp-owned WaveNet provider and numerical/terminal receipts exist.

This command matrix does not run `GpuConvolver::ProviderPolicy::SharedRequired`
as a substitute for WaveNet. That runtime gate belongs to Pulp's exact-provider
tests; reusing it here would falsely turn a convolver receipt into GPU NAM
evidence.
