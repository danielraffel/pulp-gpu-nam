# Building and testing GPU NAM

## Build

GPU NAM builds against the Pulp SDK, vendored here as a git submodule at `./pulp`.

**Requirements:** macOS (Apple Silicon), CMake ≥ 3.24, a C++20 toolchain (Xcode
command-line tools). The GPU stack (Dawn + Skia Graphite) is fetched/prebuilt by
Pulp's own build.

```bash
# 1. Clone with the Pulp submodule
git clone https://github.com/danielraffel/pulp-gpu-nam.git
cd pulp-gpu-nam
git submodule update --init --recursive

# Already have a checkout? Make extra working copies with
#   scripts/new-worktree.sh <name> [<ref>] [-b <branch>]
# instead of cloning again: it reuses the Pulp objects already on this machine,
# keeps GitHub as the submodule origin, and refuses /tmp.

# 2. (Optional) Fetch the VST3 + AU SDKs. CLAP and the Standalone app build
#    without any external SDK; run this only if you want the VST3 and/or AU
#    formats. They're developer-supplied and not vendored here.
./scripts/fetch-sdks.sh

# 3. Configure (Release; the GPU stack is required)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 4. Build the plugin + the standalone app (drop the formats you didn't fetch
#    an SDK for)
cmake --build build --target GpuNam_Standalone GpuNam_CLAP GpuNam_VST3 GpuNam_AU -j8
```

Pulp is built lean from the submodule (its own examples, tests, and CLIs are
skipped when it's consumed this way); the first build compiles the Pulp SDK, so
it takes a while. CLAP + Standalone need no external SDK.

Build products land under `build/` (`CLAP/GpuNam.clap`, `VST3/GpuNam.vst3`,
`AU/GpuNam.component`, and the standalone app). The bundles are self-contained and
relocatable — the build fails if any bundle bakes a build-tree dylib path.

## Smaller builds

Each capture family is a build toggle — drop the ones you don't need:

```bash
cmake -S . -B build -DGPU_NAM_WITH_LSTM=OFF -DGPU_NAM_WITH_KERAS=OFF
```

Toggles: `GPU_NAM_WITH_WAVENET` (A1), `_A2`, `_CONVNET`, `_LSTM`, `_LINEAR`,
`_KERAS`. The GPU engine is WaveNet-specific, so a build that turns WaveNet off
targets the CPU inference layer only.

## Package a signed installer

`src/package.sh` drives the signed + notarized combined installer (it calls
Pulp's `ship` tooling from the submodule). See the script header for the
signing-identity and notary-key environment it expects.

## Tests

GPU NAM ships its own test suite (it fetches a private Catch2; it does not turn on
Pulp's whole test tree):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_NAM_BUILD_TESTS=ON
cmake --build build -j8
ctest --test-dir build -R gpu-nam --output-on-failure
```

For an exact installed-SDK/provider/model receipt with a typed corruption
control, see [`docs/production-gpu-nam-receipt.md`](docs/production-gpu-nam-receipt.md)
and run `tools/validation/gpu_nam_production_receipt.py`.
The receipt parser and its positive/negative contract are also available as
the `gpu-nam-production-receipt-contract` CTest; that test uses a fake driver
and does not replace the real provider receipt.

- `gpu-nam-gpu-test` — the fused GPU WaveNet reproduces the CPU oracle (single
  block, streaming, and wins at scale).
- `gpu-nam-plugin-test` — CPU/GPU engines produce finite amped audio, bypass is
  the dry signal, the Engine switch is live at fixed latency, the noise gate
  attenuates sub-threshold signal, the tone stack is transparent when flat.
- `gpu-nam-{model,a2,convnet,lstm,linear,keras}-test` — each capture family loads
  and runs.
- `gpu-nam-ui-test` — the editor renders non-blank and asset-composited, and
  pointer input drives real parameters.

The CPU engines are also covered by `gpu-nam-model-test` and `gpu-nam-lstm-test`:
the block engines against the per-sample oracle in every tanh mode, identical
output for any block partitioning, allocation-free streaming, and the
NAM-compatible rational tanh against an independent implementation.

Use a bounded job count (`-j8` here) rather than every core on a shared machine.

### CPU engine CI and output identity

`tools/cpu-engine` builds the CPU inference engine on its own, with no Pulp SDK
and no GPU. CI (`.github/workflows/cpu-engine.yml`) runs it on Linux:

```bash
cmake -S tools/cpu-engine -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j4
ctest --test-dir build-cpu --output-on-failure
python3 tools/cpu-engine/render_hashes.py --render build-cpu/gpu-nam-golden-render
```

The last step renders 8 captures in the default and exact tanh modes at block
sizes 37, 64, 128 and 512. It then checks that every output file has the
SHA-256 stored in `tools/cpu-engine/render-hashes.json`. A performance change
must keep the output byte-identical. If a change is meant to alter the output,
re-record the hashes with `--update` and say why in the PR. On a mismatch the
check prints the new hashes for its platform. A manual run of the workflow with
`update` set re-records the Linux hashes and uploads the file as an artifact,
so a runner toolchain change can be re-recorded without a Linux machine. CI pins
GCC 13, the compiler the stored hashes were recorded with.

Hashes are stored per platform, and each platform is compared only with itself.
`linux-x86_64` (CI, GCC) and `darwin-arm64` (Apple clang) share no hashes. arm64
fuses multiply-adds that x86-64 without FMA rounds separately, and the exact
mode calls each platform's libm. Running the check on a Mac uses the
`darwin-arm64` hashes.
