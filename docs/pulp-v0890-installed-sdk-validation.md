# Pulp v0.890.0 installed SDK validation

This note records a downstream consumer check against the official arm64 SDK
release, not a source-tree rebuild of Pulp.

## Inputs

- Pulp release: `v0.890.0`
- Pulp source SHA: `a883a45af7de9fdc7cd5f8cd2767ae891c6fa7d4`
- SDK: `pulp-sdk-darwin-arm64.tar.gz`
- SDK SHA-256: `9af69df239c53a5d47f3b7d8f62eeaafb9d057fda58c59c7ab506b681f2f512b`
- SDK provenance: `sdk-provenance.json` reports `exact_provider_proof=true`,
  `shared_provider=true`, and `shared_convolver=true`.

## Validation

A clean arm64 Release consumer configure used
`GPU_NAM_USE_INSTALLED_PULP=ON`, `GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET=ON`,
and `CMAKE_PREFIX_PATH` set to the extracted official SDK. The 16-test stamped
matrix passed for 32, 64, 128, and 512 frame blocks at lead 1, 2, 4, and 8.

The 1,000-block paced check at 48 kHz, 32 frames, lead 8 passed with:

- 1,000 GPU deliveries
- zero transport misses
- zero callback deadline misses
- zero numerical mismatches
- zero failed forwards

That check reports `hard_realtime=0`, `gpu_timestamps_available=0`, and
`phase_timestamps_available=callback_only`. It proves installed-SDK consumer
compatibility and numerical/lifecycle correctness. It does not prove a
hard-realtime guarantee, GPU scheduling determinism, or a speedup over CPU.

The corresponding 100,000-block loaded-host sample is recorded in the Pulp
planning research note. The remaining acceptance work is a quiet-host
million-block run, contention matrices, and authentic GPU phase timestamps.
