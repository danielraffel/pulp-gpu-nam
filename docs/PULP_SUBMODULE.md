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

## Integration boundary

The plugin depends only on Pulp's public targets — `pulp::render`,
`pulp::gpu-audio`, `pulp::signal`, `pulp::view`, `pulp::canvas`, `pulp::runtime` —
and its CMake helpers (`pulp_add_plugin`, `PulpBundleRelocatable`). It does not
reach into Pulp's internals. Keeping this boundary thin is deliberate: it lets the
same plugin later consume Pulp via an installed SDK instead of a submodule (see
the installed-SDK plan in the Pulp planning repo).
