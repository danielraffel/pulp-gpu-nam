# GPU NAM documentation

## Using and building

- [nam-support.md](nam-support.md): supported capture architectures, the CPU and
  GPU engines, tanh modes, signal chain, output level, Slim size
- [how-it-works.md](how-it-works.md): signal chain, CPU block engine, GPU engine,
  how the plugin sits on Pulp, repository layout
- [performance.md](performance.md): CPU engine cost and output parity, and how they
  were measured
- [building.md](building.md): build, smaller builds, packaging, tests
- [PULP_SUBMODULE.md](PULP_SUBMODULE.md): the pinned Pulp commit and how to update it
- [../src/docs/comparison.md](../src/docs/comparison.md): feature comparison with
  the reference NAM player

## GPU audio experiments and validation records

These record specific experiments and validation runs against named Pulp SDK
versions. They describe what was true for that SDK and are kept as history.

- [production-gpu-nam-receipt.md](production-gpu-nam-receipt.md): production
  consumer receipt and its validation tool
- [cpu-baseline-diagnostic.md](cpu-baseline-diagnostic.md): CPU baseline diagnostic
  for the shared-GPU comparison
- [matched-staged-diagnostic.md](matched-staged-diagnostic.md),
  [realtime-worker-experiment.md](realtime-worker-experiment.md),
  [audio-work-interval-experiment.md](audio-work-interval-experiment.md),
  [paced-delivery-evidence.md](paced-delivery-evidence.md),
  [paced-numerical-sidecar.md](paced-numerical-sidecar.md): GPU delivery and
  scheduling experiments
- [silent-device-callback-diagnostic.md](silent-device-callback-diagnostic.md),
  [physical-lifecycle-acceptance.md](physical-lifecycle-acceptance.md): physical
  device lifecycle checks
- [stamped-wavenet-validation.md](stamped-wavenet-validation.md),
  [stamped-wavenet-trace.md](stamped-wavenet-trace.md): SDK-owned WaveNet transport
- [installed-sdk-gpu-audio.md](installed-sdk-gpu-audio.md),
  [installed-stamped-acceptance.md](installed-stamped-acceptance.md),
  [pulp-v08850-build-receipt.md](pulp-v08850-build-receipt.md),
  [pulp-v0890-installed-sdk-validation.md](pulp-v0890-installed-sdk-validation.md),
  [pulp-v08902-official-sdk-validation-20261002.md](pulp-v08902-official-sdk-validation-20261002.md),
  [pulp-v08902-official-stamped-longrun-20261002.md](pulp-v08902-official-stamped-longrun-20261002.md),
  [pulp-v08960-installed-sdk-validation-20261003.md](pulp-v08960-installed-sdk-validation-20261003.md):
  installed-SDK validation against specific Pulp releases

Raw artifacts from those runs are under [../receipts/](../receipts/).
