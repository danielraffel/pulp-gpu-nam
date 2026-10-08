# GPU-NAM Forge consumer receipt — merged Pulp SDK

- Date: 2026-10-08
- Consumer head: `6f96d1786ae0a2c07aba4d9dc1a5105bfbeb3b5c`
- Pulp SDK source head: `78b214f871f73a9842f80e140a887aff2de24609`
- SDK prefix: `/Volumes/Workshop/Code/pulp-sdk-merged-78b214f871f73a9842f80e140a887aff2de24609`
- Build: Release, GPU SDK configured, Forge adapter targets built with the governed build wrapper.

The owner adapter is the `GpuNam::ForgeAdapter` interface target and exposes
the existing `GpuNamProcessor` through
`pulp::examples::gpu_nam_forge_adapter::create_processor()`. It carries no
second DSP, model registry, provider, or control authority. The graph host owns
only node placement and routing; GPU-NAM remains the authority for the model,
controls, provider status, and CPU fallback policy.

Command:

```sh
ctest --test-dir build-dspx-merged-pulp-r2 \
  -R 'GPU NAM Forge adapter|GPU request is typed' --output-on-failure
```

Result: **4/4 passed**.

The locally built graph-test executable was
`build-dspx-merged-pulp-r2/src/gpu-nam-forge-adapter-test` with SHA-256
`1092300e4031f113a249f4303f343519a8b41e83d8df180b7b886699befde794`.

The four cases prove product Processor creation, runtime SignalGraph
ProcessorNode installation, SignalGraph rendering against the independent CPU
oracle, and typed CPU fallback when GPU provider delivery is unavailable. The
separate production GPU-NAM provider receipt remains the evidence for
hardware GPU delivery and records its own consumer source revision, model,
device, provider, parity, and corruption negative; this Forge adapter receipt
does not relabel that run as a Forge production GPU run.

`GpuNamProcessor::define_parameters` remains the unified control authority;
the Forge adapter has no CLI or MCP surface of its own. The test's factory and
host command is the CTest command above.
