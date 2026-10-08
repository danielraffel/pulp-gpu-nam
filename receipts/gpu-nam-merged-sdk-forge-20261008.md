# GPU-NAM Forge consumer receipt — merged Pulp SDK

- Date: 2026-10-08
- Consumer head: `79aedb6c4363a165080bc31a69c02ed7dcf52730`
- Pulp SDK source head: `78b214f871f73a9842f80e140a887aff2de24609`
- SDK prefix: `/Volumes/Workshop/Code/pulp-sdk-merged-78b214f871f73a9842f80e140a887aff2de24609`
- Build: Release, GPU SDK configured, Forge adapter targets built with the governed build wrapper.

Command:

```sh
ctest --test-dir build-dspx-merged-pulp-r2 \
  -R 'GPU NAM Forge adapter|GPU request is typed' --output-on-failure
```

Result: **4/4 passed**.

The four cases prove product Processor creation, runtime SignalGraph
ProcessorNode installation, SignalGraph rendering against the independent CPU
oracle, and typed CPU fallback when GPU provider delivery is unavailable.
