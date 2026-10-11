# GPU-NAM production consumer receipt

GPU-NAM's shipped CPU/GPU processor is a real Pulp consumer. This receipt lane
binds one run to the exact model, consumer source, installed SDK provenance,
diagnostic executable, provider/backend, and host identity. It also runs a
typed corruption control so a numerical or delivery failure cannot be hidden by
the CPU fallback.

## Run it

Configure GPU-NAM against an official, distribution-eligible Pulp SDK. The
shared-session option below is the public provider consumer path exercised by
this receipt; it is opt-in so ordinary CPU/staged builds keep their existing
compatibility surface, and it keeps a continuously prepared CPU fallback:

```sh
cmake -S . -B build-production -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_NAM_USE_INSTALLED_PULP=ON \
  -DPulp_DIR=/absolute/pulp-sdk/lib/cmake/Pulp \
  -DGPU_NAM_EXPECTED_PULP_SOURCE_SHA=EXACT_SDK_SOURCE_SHA \
  -DGPU_NAM_EXPERIMENTAL_SHARED_WAVENET_SESSION=ON \
  -DGPU_NAM_BUILD_TESTS=ON
# Use the Pulp checkout's governed wrapper (set PULP_ROOT to that checkout).
export PULP_ROOT=/absolute/path/to/pulp
"$PULP_ROOT/tools/ci/governed-build.sh" \
  cmake --build build-production --target gpu-nam-gpu-cpu-diagnostic
python3 tools/validation/gpu_nam_production_receipt.py \
  --diagnostic build-production/src/gpu-nam-gpu-cpu-diagnostic \
  --model src/models/example.nam \
  --sdk-prefix /absolute/pulp-sdk \
  --source-root . \
  --output receipts/gpu-nam-production.json
```

The receipt has `status: pass` only when all of these are true:

* the SDK has canonical `PulpConfig.cmake`, valid provenance, and
  `distribution_eligible: true`;
* the provenance includes the authenticated GPU-audio capability receipt and
  complete coherence/archive integrity members, all matching the installed
  files;
* the real bundled WaveNet model loads and the provider reports an available
  backend/device;
* the GPU path produces at least one block, the stateful CPU oracle has zero
  parity failures, and dropped input is zero;
* the typed `--inject-direct-output-error` run exits non-zero, reports a direct
  model mismatch, and still reports zero transport parity failures.

The wrapper also fails closed before reporting a positive status when the
consumer checkout is not an exact clean Git tree, the installed SDK does not
carry official-release provenance with matching `pulp.sdk-integrity.v1` file
hashes, or the positive diagnostic reports dropped input, an unprimed CPU
fallback, a non-finite error, an error above `1e-3`, or no GPU completion.
These checks protect the bounded consumer receipt from being mistaken for a
product qualification result.

The receipt records `gpu_inner_completions`, selected CPU fallback, transport
misses, input drops, maximum oracle error, model SHA-256, executable SHA-256,
SDK provenance SHA-256, source revisions, and host/device labels. These fields
are evidence for that exact model/provider/device combination. They do not
generalize to another GPU or establish a hard realtime or speedup claim.

The `provider.device` labels are diagnostic host metadata from the consumer
process. They are not a source-bound host-preflight or authenticated physical
GPU receipt. P2/P6 acceptance must join this consumer evidence to a fresh
source-bound Pulp host-preflight and provider observation with matching source,
executable, SDK, model, and host hashes; this v1 wrapper does not invent that
missing evidence.

The parser and its positive/typed-negative fake-driver controls run as the
`gpu-nam-production-receipt-contract` CTest. That gate does not substitute for
the hardware-backed receipt: the latter must be generated from the real
diagnostic and a distribution-eligible SDK, then committed under `receipts/`.

## Unified controls

`GpuNamProcessor::define_parameters` is the control authority. The receipt
copies its stable IDs and names for the shared control/docs surface:

| ID | Control | Values |
| ---: | --- | --- |
| 1 | Input | dB |
| 2 | Output | dB |
| 3 | Mix | percent |
| 4 | Engine | CPU, GPU, Auto |
| 5 | Bypass | boolean |
| 6 | Gate | dB |
| 7–9 | Bass, Middle, Treble | 0..10 |
| 10–11 | Noise Gate, EQ | boolean |
| 12 | Output Mode | Raw, Normalized, Calibrated |
| 13 | Cal Level | dB |
| 14 | Slim | 0..1 |

The native UI drives the same `StateStore` parameters. The diagnostic is the
CLI proof surface. GPU-NAM does not ship an MCP executor; the receipt records
that typed unsupported surface instead of implying one exists. Forge catalog
registration remains a separate product-owner integration: this receipt is the
named GPU-NAM consumer/provider proof and must not be copied into a Forge
catalog without a Forge factory and host acceptance test.

The fake-driver tests in
`tools/validation/test_gpu_nam_production_receipt.py` exercise both the positive
receipt and provider-unavailable fail-closed behavior without requiring a GPU.
