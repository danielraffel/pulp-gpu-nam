# Official v0.890.2 stamped GPU-NAM validation

This is a supplemental receipt for the published Pulp SDK, separate from the
55-test consumer gate.

## Exact identity

- GPU-NAM validation head: `74999e372a0670b5d4454d5fabe0dbf69b892ed0`
- Pulp: `v0.890.2`, source `209c0c92274f55e33b4192137564946e8e81e55a`
- SDK archive SHA-256: `87ea4f48ca45d9f256a4978109f6ae7590ddb107b03e4c2b93623fc15da648e8`
- SDK provenance: official release, distribution eligible, `darwin-arm64`.

## Build and bounded suite

The installed SDK configuration used `GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET=ON`,
with `GPU_NAM_EXPECTED_PULP_SOURCE_SHA` set to the exact Pulp source SHA. The
stamped validation target built successfully. The 17 tests covering paced options
and frames 32/64/128/512 at lead 1/2/4/8 passed 17/17.

Representative verbose output at frames 32, lead 8:

```text
max_error=9.12696e-08 cpu_model_calls=112 fallback_reads=16
 gpu_callbacks=32 worker_produced=32 diagnostic_status=passed
```

This is a bounded functional/lifecycle result, not a hard-realtime claim.

## Paced runs and host-load qualification

The following runs used the real `wavenet_a1_standard.nam` model, two channels,
32-frame blocks, `process-events` completion, and ordinary OS-thread scheduling:

| input blocks | lead | GPU callbacks | CPU fallback callbacks | transport misses | max error | deadline misses |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 2 | 1,000 | 0 | 0 | 2.17017e-05 | 0 |
| 1,000 | 4 | 1,000 | 0 | 0 | 2.17017e-05 | 0 |
| 1,000 | 8 | 144 | 856 | 856 | 1.63298e-05 | 0 |
| 10,000 | 2 | 12 | 9,988 | 9,988 | 1.12243e-05 | 0 |
| 10,000 | 4 | 25 | 9,975 | 9,975 | 1.12243e-05 | 3 |
| 10,000 | 8 | 65 | 9,935 | 9,935 | 1.63298e-05 | 1 |
| 100,000 | 4 | 9 | 99,991 | 99,991 | 1.12243e-05 | 181 |

All runs reported zero numerical mismatches and passed the diagnostic's
correctness checks. The 10,000- and 100,000-block results were captured on a
heavily loaded M5 host (18 logical CPUs; load averages approximately
8.02/10.44/15.48). At observation time, unrelated Python, WindowServer,
WebKit GPU, Shipyard, and Codex processes were active. Therefore these runs
are evidence of correctness and fallback behavior under contention, not a
clean-machine regression verdict.

Raw files are retained in `/private/tmp` on the validation host:

- `gpu-nam-official-v08902-lead{2,4,8}-1000-20261002.log`
- `gpu-nam-official-v08902-lead{2,4,8}-10000-20261002.log`
- `gpu-nam-official-v08902-lead4-100k-20261002.log`
- matching `.csv` sidecars

A quiet M1/M3/M5s rerun is required before attributing this long-run fallback
rate to the SDK/provider rather than host contention.

The 100,000-block raw receipt hashes are:

```text
25b0f3bf2313b51e91a90c6791c01bd45f849b006ebf271b7af7cffcf495ee23  gpu-nam-official-v08902-lead4-100k-20261002.csv
446d093bf2d83db83b6a76695e56f48abe4debbcfa7e3bba60179fbd0f4b1c60  gpu-nam-official-v08902-lead4-100k-20261002.log
```
