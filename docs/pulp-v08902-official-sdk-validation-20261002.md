# GPU NAM validation against the published Pulp v0.890.2 SDK

This receipt records the exact published ARM64 SDK consumer validation for
`chore/pin-pulp-sdk-0.890.2-20261002`.

## Source and SDK identity

- GPU NAM validation commit: `c48c461e98b2d12c8c40157712829c64e3e1dc03`
- Pulp submodule tag: `v0.890.2`
- Pulp source SHA: `209c0c92274f55e33b4192137564946e8e81e55a`
- SDK artifact: `pulp-sdk-darwin-arm64.tar.gz`
- SDK artifact SHA-256: `87ea4f48ca45d9f256a4978109f6ae7590ddb107b03e4c2b93623fc15da648e8`
- SDK provenance: `sdk-provenance.json` reports `kind: release`,
  `profile: official-release`, `distribution_eligible: true`,
  `platform: darwin-arm64`, `sdk_version: 0.890.2`, and
  `source_git_sha: 209c0c92274f55e33b4192137564946e8e81e55a`.
- Build configuration: `Release`, arm64, installed SDK mode, expected source SHA
  enforced by `GPU_NAM_EXPECTED_PULP_SOURCE_SHA`.

## Build

The published SDK was extracted to a private temporary directory and consumed
without modifying the active plugin installation:

```text
cmake -S /private/tmp/gpu-nam-sdk-9213-pin-20261002 \
  -B /private/tmp/gpu-nam-sdk-9213-official-release-20261002 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/private/tmp/pulp-sdk-v0.890.2-official-arm64-20261002/pulp-sdk \
  -DGPU_NAM_USE_INSTALLED_PULP=ON \
  -DGPU_NAM_EXPECTED_PULP_SOURCE_SHA=209c0c92274f55e33b4192137564946e8e81e55a \
  -DGPU_NAM_BUILD_TESTS=ON \
  -DGPU_NAM_BUILD_GPU_AUDIO_CAPABILITY_PROBE=ON
ninja -C /private/tmp/gpu-nam-sdk-9213-official-release-20261002 -j8
```

Build result: `187/187` Ninja steps completed successfully. CLAP, VST3, AU,
Standalone, GPU capability probe, GPU engine tests, and consumer test binaries
were produced.

The build emitted a pre-existing warning that the Standalone target references
`src/assets/nam` from the source tree while also passing the bundle relocatable
check. This is recorded as a packaging follow-up; it does not invalidate the
SDK or test result.

## Tests

```text
ctest --test-dir /private/tmp/gpu-nam-sdk-9213-official-release-20261002 \
  --output-on-failure --parallel 4

100% tests passed, 0 tests failed out of 55
Total Test time (real) = 8.52 sec
```

The 55 tests include the CPU and GPU WaveNet parity tests, stereo shared-device
independence, live CPU/GPU switching, all model-family tests, editor tests,
GPU-audio capability probe, delivery-status tests, CLAP loading, AU validation,
VST3 `pluginval`, and lifecycle/watchdog contract tests.

These are correctness, integration, and packaging checks. They do not establish
hard realtime scheduling or deadline reliability.
