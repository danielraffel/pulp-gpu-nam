# GPU NAM validation against the published Pulp v0.896.0 SDK

This receipt records consumer validation after updating the `pulp` submodule
from v0.890.2 to the published v0.896.0 SDK source.

## Source and SDK identity

- GPU NAM base: `ccf5947fb7b5c433468fb4435b282834727ffe53`
- Pulp submodule: `v0.896.0`
- Pulp source SHA: `d7336ebf678f4271f7240a098c542f7fdb3cb3a8`
- SDK artifact: `pulp-sdk-darwin-arm64.tar.gz`
- SDK artifact SHA-256: `815d8e07d309cc33bc43f3bedf5980f7da247871f29c09d2bf701c1cf0abafaf`
- SDK provenance: `kind: release`, `profile: official-release`,
  `distribution_eligible: true`, `platform: darwin-arm64`,
  `sdk_version: 0.896.0`, `source_git_sha: d7336ebf678f4271f7240a098c542f7fdb3cb3a8`.

## Configure and focused build

The installed SDK consumer path configured successfully with the exact source
SHA contract:

```text
cmake -S . -B build-sdk0896 -DCMAKE_BUILD_TYPE=Release \
  -DGPU_NAM_USE_INSTALLED_PULP=ON \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/pulp-sdk \
  -DGPU_NAM_EXPECTED_PULP_SOURCE_SHA=d7336ebf678f4271f7240a098c542f7fdb3cb3a8 \
  -DGPU_NAM_BUILD_TESTS=ON -DGPU_NAM_BUILD_GPU_AUDIO_CAPABILITY_PROBE=ON
cmake --build build-sdk0896 --parallel 4
```

The Release build produced CLAP, VST3, AU, Standalone, GPU capability, and
consumer test targets. Bundle relocatability checks passed. The existing
Standalone asset-path warning remains present and is unrelated to this SDK pin.

## Tests

The focused SDK/GPU suite passed:

```text
gpu-nam-sdk-contract                 Passed
gpu-nam-gpu-test                     Passed
gpu-nam-gpu-audio-capability-probe   Passed
gpu-nam-gpu-cpu-diagnostic-options   Passed
gpu-nam-retire-test                  Passed
gpu-nam-prepared-program-test        Passed
gpu-nam-delivery-status               Passed
```

The complete 55-test run reached 52 completed tests before the AU validator
timed out. Two unrelated timing-sensitive plugin tests also failed in that
loaded run (`cabinet swap is click-free` and `applies a loaded cabinet IR`),
while the GPU parity, streaming, lifecycle, status, format, and model-family
tests passed. This is consumer compatibility evidence, not a realtime deadline
claim. The physical-device lifecycle probe remains a separate manual gate.
