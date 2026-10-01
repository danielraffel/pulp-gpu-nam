# Pulp v0.885.0 GPU-NAM validation receipt

This receipt records the exact local validation of the Pulp v0.885.0 SDK pin in
this branch. It is a correctness and provenance result, not a realtime
performance claim.

- GPU-NAM source: `c23e9de711e051f403605ac6df447e36455a626c`
- Pulp source: `b46afb476b0e640cd3146f34a15912c1a52887da`
- SDK archive SHA-256: `7be3387d213f3bfe52f60c578ad548edc58a3d1ceb5a18b19cefcdc516665f06`
- Validation executable SHA-256: `5504b1661fd9d02c086652872bca45e47c6514553e6318934e4bde1308ee2b69`
- Model SHA-256: `ceb53469a19ce278e2235da982ae676cb8d5451a8de22a7ecc7a2617d07224d1`
- Host: Apple Silicon macOS, Release/Ninja, Dawn-enabled Pulp build

Configure and build:

```sh
cmake -S . -B build-v08850 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGPU_NAM_EXPERIMENTAL_STAMPED_WAVENET=ON
cmake --build build-v08850 --target gpu-nam-stamped-validation -j2
```

Focused validation:

```sh
ctest --test-dir build-v08850 -R 'gpu-nam-stamped' --output-on-failure
```

Result: **16/16 tests passed in 13.52 seconds** across 32/64/128/512-frame
blocks and 1/2/4/8-block lead values.

The tests establish current-SDK build and stamped consumer correctness,
including output parity and fallback/lifecycle checks. They do not establish a
GPU speedup or hard-realtime behavior. The consumer exposes callback-side
phases only; GPU admission, execution, and completion timestamps remain
unavailable. The long-duration matched GPU/CPU campaign therefore remains a
separate, quiet-host experiment.

## Product bundle smoke

The same Release build also produced the CLAP and standalone product bundles.
Both completed Pulp's relocatability/shipping scans, and the CLAP dynamic-load
test passed. After explicitly building the two product test targets, their
direct Catch2 runs passed:

- `gpu-nam-plugin-test`: 550,934 assertions, 29 cases
- `gpu-nam-ui-test`: 19 assertions, 5 cases
- `clap-dlopen-GpuNam`: passed

The bundle directory hashes were `fb600f9c3018df020e83fdb209dfad78e7c2cdb8f525b1d1e02a05905141d894`
for `GpuNam.clap` and `90673f4320f353a9ff4d9707c61293b362274b3760d327489bf9d493a16cb57c`
for `GpuNam.app`. These are unsigned local build artifacts; `codesign
--verify` therefore does not pass, and no notarized release claim follows.
