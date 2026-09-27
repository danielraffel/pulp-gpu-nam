# Installed SDK and native CLAP acceptance

Use a fresh Release consumer build after the SDK owner confirms the exact prefix
is reconciled, installed, stopped and provenance-stamped. Do not start another
SDK build. Set `SDK_SHA` to that owner-confirmed full commit, not a branch name or
an assumed latest release. The consumer checks it against the installed SDK's
canonical provenance variable. This supplements the producer's artifact integrity
receipt; source-SHA equality is not itself archive authentication.

```sh
SDK_PREFIX=/tmp/pulp-shared-gpu-sdk-prefix-20260927
SDK_SHA=<owner-confirmed-40-character-source-sha>
NAM_SOURCE=/tmp/gpu-nam-installed-host-controls-20260927
NAM_BUILD=/tmp/gpu-nam-installed-sdk-full-plugin-20260927
cmake -S "$NAM_SOURCE" -B "$NAM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DGPU_NAM_USE_INSTALLED_PULP=ON \
  -DPulp_DIR="$SDK_PREFIX/lib/cmake/Pulp" \
  -DGPU_NAM_EXPECTED_PULP_SOURCE_SHA="$SDK_SHA" \
  -DGPU_NAM_EXPERIMENTAL_STAMPED_WAVENET=ON \
  -DGPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION=OFF \
  -DGPU_NAM_NATIVE_HOST_PROBE=ON -DGPU_NAM_BUILD_TESTS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
/Users/danielraffel/Code/pulp/tools/ci/governed-build.sh \
  cmake --build "$NAM_BUILD" --target GpuNam_CLAP GpuNam_Standalone \
  gpu-nam-clap-host-test gpu-nam-plugin-test gpu-nam-stamped-validation \
  gpu-nam-completion-options-test gpu-nam-paced-options-test
ctest --test-dir "$NAM_BUILD" --output-on-failure \
  -R '^(gpu-nam-sdk-contract|gpu-nam-completion-options|gpu-nam-paced-options|clap-dlopen-GpuNam|gpu-nam-clap-host-acceptance)$'
"$NAM_BUILD/src/gpu-nam-plugin-test"
```

Coordinate the GPU slot before executing tests. Preserve the SDK producer receipt,
consumer source SHA, CMake cache, compile/link commands, binary/model hashes and
complete logs. Check link commands actually use this prefix without source object
overlays. A missing required format target is a failure; do not silently drop it.
The stamped consumer refuses a package that does not declare its compiled Dawn
shared-I/O provider. The legacy compatibility mode remains usable without that
provider or an expected-SHA pin.

## Evidence boundaries

- `gpu-nam-plugin-test` compiles the real processor and editor source, exercises
  HeadlessHost, output, PDC, Engine state/restore/reprepare, and GPU selection.
  Explicit stamped tests fail if GPU preparation is unavailable; legacy
  CPU-compatible tests retain their existing behavior.
- `clap-dlopen-GpuNam` proves only loadability.
- `gpu-nam-clap-host-test` loads the actual built CLAP bundle, applies Engine via
  inactive `params.flush` before activation, runs distinct stereo audio at 128
  frames/48 kHz, checks fixed 1024-sample PDC, and stops/deactivates/reactivates.
  It compares two GPU epochs against a CPU epoch. Actual internal GPU blocks
  remain 512 frames with one block of lead; host128 is not GPU128.
- The host test requires an explicit diagnostic delivery snapshot. Correct audio,
  capability, Engine=GPU, PDC, or worker-produced counts cannot prove accepted GPU
  output. The host queries only after stop_processing and before deactivate.
  Exactly one instance must exist; a second-instance control must be rejected.
  Null, malformed-size and unsupported-version queries must also be rejected.
  CPU epochs must have zero transport counts; all four counters are logged per epoch.

The diagnostic C export `gpu_nam_host_probe_v1` exists only with the explicit
`GPU_NAM_NATIVE_HOST_PROBE` build option, OFF by default. These builds are not for
distribution. Normal CLAP builds export no probe. This is a consumer test ABI, not
an extension to the public SDK or CLAP standard.

The probe reads the SDK's public `GpuAudioTransport::delivery_snapshot()` after
processing stops. Only `gpu_blocks` establishes accepted GPU delivery. Generic
worker output, silence, passthrough and invalid calls are grouped as unexpected
outcomes; priming and CPU fallback remain separate. A CPU engine epoch returns
zero transport counts. The host requires positive GPU delivery in both GPU epochs
and accounts for every internal block. The private P4 trial observer is not used.

Do not substitute `pulp audio render --initial-param 4=1` for this test. That
command applies initial values after prepare; a preparation-bound Engine request
then remains pending until the next activation. It can render correct CPU audio
without ever exercising the requested stamped GPU engine.
