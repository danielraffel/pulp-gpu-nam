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

A second 1,000-block matrix was run with the same settings. It remained numerically correct, but ordinary-thread scheduling produced callback deadline misses (2, 0, 0, and 11 for max_inflight 1, 2, 4, and 8 respectively). These rows reinforce that the seam is useful for controlled comparison, but are not evidence that higher in-flight depth improves realtime reliability.

SDK artifact hashes:

- `gpu_wavenet_realtime_node.hpp`: `2ffd35c47a6cc12d25e106fe4e10328aa9f29551342482cc505befa56e0f86df`
- `libpulp-gpu-audio.a`: `4d779e621ceb67c0fecdd64294b9ee9549d9fca881782abfcf06ed68cf6074db`
- SDK source checkout was clean at the stated commit; CMake build type was Release.
