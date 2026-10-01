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
