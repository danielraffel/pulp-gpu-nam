# Pulp submodule pin

GPU NAM builds against the Pulp SDK vendored at `./pulp`. The submodule is pinned
to a specific Pulp commit so a clone always builds against a known framework
version.

The current pin is Pulp SDK v0.896.0, source commit
`d7336ebf678f4271f7240a098c542f7fdb3cb3a8`. The matching Darwin arm64 SDK
artifact has SHA-256
`815d8e07d309cc33bc43f3bedf5980f7da247871f29c09d2bf701c1cf0abafaf`.

The normal installed-SDK build and consumer tests have been validated against
the published v0.896.0 SDK. The separate manual physical-lifecycle probe stays
restricted to its previously reviewed SDK pin until that hardware gate is
repeated for this release.

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
