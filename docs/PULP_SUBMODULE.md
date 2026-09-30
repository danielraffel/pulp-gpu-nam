# Pulp submodule pin

GPU NAM builds against the Pulp SDK vendored at `./pulp`. The submodule is pinned
to a specific Pulp commit so a clone always builds against a known framework
version.

The current pin is Pulp SDK v0.881.2, source commit
`167a475d3ad6c777451ca8b4a45a1b981052fba2`. The matching Darwin arm64 SDK
artifact has SHA-256
`77cb4155f6f55bba88b59851ab9420e7d359af6ae7d7f29d99530100b6ca68b4`.

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
