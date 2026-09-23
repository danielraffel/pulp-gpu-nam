# Installed Pulp GPU-audio consumer proof

GPU NAM can be checked against a packaged Pulp GPU SDK without rebuilding the
vendored Pulp submodule. The receipt command configures the installed-SDK path,
builds the existing GPU NAM plugin and public capability probe, runs both tests,
and records source, SDK, model, executable, and log hashes.

From the GPU NAM checkout, run:

```sh
python3 tools/validation/installed_sdk_gpu_audio_receipt.py \
  --sdk-prefix /path/to/pulp-sdk \
  --output /tmp/gpu-nam-installed-sdk-receipt.json
```

The receipt's `installed_sdk_consumer` result is positive only when configure,
build, and test execution all succeed. The capability probe is intentionally
fail-closed before preparation. A passing receipt proves that GPU NAM consumes
the public installed SDK and preserves its existing GPU/CPU lifecycle and
fallback tests. It does not prove adoption of Pulp's private shared-memory
provider. That requires a future authenticated public provider contract.
