# GPU-NAM multiflight development-SDK receipt

This is provisional development evidence. It is not official Pulp SDK 0.893.0 evidence.

- GPU-NAM consumer: `codex/gpu-nam-max-inflight-20261002` at `155fd963e`
- Pulp SDK source: `codex/wavenet-multiflight-20261002` at `dab6cfad20309319c83b2cb67eea4f411f1fc118`
- Installed prefix: `/tmp/pulp-sdk-multiflight-dab6cfad-20261002`
- SDK reports version `0.894.0`; the source SHA is bound by the checkout above.
- Build: Release, installed-SDK consumer, real Dawn provider path
- Workload: stereo, 48 kHz, 32 frames, 100 input blocks, lead 1, capacity 16, ProcessEvents completion policy
- Rows: max_inflight 1, 2, 4, 8

Every row passed numerical parity and observed GPU delivery. This short run is a provider/consumer correctness and fallback-accounting check, not a realtime-suitability result. GPU timestamps were unavailable and the worker used an ordinary OS thread.
