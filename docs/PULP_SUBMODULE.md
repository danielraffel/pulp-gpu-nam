# Pulp submodule pin

GPU NAM builds against the Pulp SDK vendored at `./pulp`. The submodule is pinned
to a specific Pulp commit so a clone always builds against a known framework
version.

The current pin is Pulp `main` at
`09f38fa957dcc652537911b93bfdb1195b782ecd` (Pulp 0.997.1). It includes the
saturating WGSL tanh from Generous-Corp/pulp#10635, which keeps the GPU WaveNet
engine finite on captures with large pre-activations.

The earlier installed-SDK and physical-lifecycle validation records in `docs/`
name the SDK versions they were run against; they were not repeated for this pin.

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

## Shared-session adapter

The opt-in
`GPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION=ON` build uses the
`pulp::gpu_audio::GpuWaveNetSession` API from Pulp #8885. The released submodule
pin now contains that API. This adapter remains a validation path only and is
not part of the default GPU engine. Keep experiments on an exact reviewed Pulp
commit; do not replace the committed pin with a moving PR branch.

## Integration boundary

The plugin depends only on Pulp's public targets — `pulp::render`,
`pulp::gpu-audio`, `pulp::signal`, `pulp::view`, `pulp::canvas`, `pulp::runtime` —
and its CMake helpers (`pulp_add_plugin`, `PulpBundleRelocatable`). It does not
reach into Pulp's internals. Keeping this boundary thin is deliberate: it lets the
same plugin later consume Pulp via an installed SDK instead of a submodule (see
the installed-SDK plan in the Pulp planning repo).
