# Pulp submodule pin

GPU NAM builds against the Pulp SDK vendored at `./pulp`. The submodule is pinned
to a specific Pulp commit so a clone always builds against a known framework
version.

## Updating the pin

```bash
cd pulp
git fetch origin
git checkout <pulp-commit-or-tag>
cd ..
git add pulp
git commit -m "chore: bump Pulp submodule to <commit>"
```

Pick a Pulp commit on `main` that includes the generalized GPU WaveNet inference
primitive (`pulp::render::GpuCompute::prepare_wavenet` / `wavenet_forward`). The
plugin's GPU engine will not compile against an older Pulp that still exposes the
pre-generalization `prepare_nam` / `nam_forward` names.

## Provisional shared-session adapter

The opt-in
`GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION=ON` build uses the provisional
`pulp::gpu_audio::GpuWaveNetSession` API from Pulp #8885. The released submodule
pin intentionally remains unchanged until that API lands. To experiment locally,
check out the exact reviewed Pulp head in `./pulp` before configuring; do not
replace the committed pin with a moving PR branch. This adapter is a validation
path only and is not part of the default GPU engine.

## Integration boundary

The plugin depends only on Pulp's public targets — `pulp::render`,
`pulp::gpu-audio`, `pulp::signal`, `pulp::view`, `pulp::canvas`, `pulp::runtime` —
and its CMake helpers (`pulp_add_plugin`, `PulpBundleRelocatable`). It does not
reach into Pulp's internals. Keeping this boundary thin is deliberate: it lets the
same plugin later consume Pulp via an installed SDK instead of a submodule (see
the installed-SDK plan in the Pulp planning repo).
