# Installed Pulp GPU-audio consumer proof

GPU NAM can be checked against a packaged Pulp GPU SDK without rebuilding the
vendored Pulp submodule. The receipt command configures the installed-SDK path,
builds the existing GPU NAM plugin and public capability probe, and records
source, SDK, model, executable, and log hashes.

From the GPU NAM checkout, run:

```sh
python3 tools/validation/installed_sdk_gpu_audio_receipt.py \
  --sdk-prefix /path/to/pulp-sdk \
  --output /tmp/gpu-nam-installed-sdk-receipt.json
```

The receipt requires the canonical installed config at
`<sdk-prefix>/lib/cmake/Pulp/PulpConfig.cmake`. It uses CTest's JSON inventory
to require these exact tests:

* `GPU NAM GPU engine reproduces the CPU engine`
* `GPU NAM GPU engine keeps stereo channels independent on the shared device`
* `GPU NAM switches Engine CPU->GPU->CPU live at fixed latency`
* `gpu-nam-gpu-audio-capability-probe`

It then runs CTest verbosely and fails if the inventory is empty/incomplete, a
selected test is absent from the execution log, or a GPU-unavailable skip is
reported. A caller-supplied build directory must be empty so an old executable
cannot be mistaken for the current SDK proof.

The receipt's `installed_sdk_consumer` result is positive only when configure,
build, and those exact test executions all succeed. The capability probe is
intentionally fail-closed before preparation. A passing receipt proves that GPU
NAM consumes the public installed SDK and preserves its existing GPU/CPU
lifecycle and fallback tests. It does not prove adoption of Pulp's private
shared-memory provider. That requires a future authenticated public provider
contract.
