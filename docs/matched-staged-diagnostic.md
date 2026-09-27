# Matched staged GPU diagnostic

The paced validation executable accepts `--staged-gpu` to compare the original
GpuCompute WaveNet upload/readback primitive against stamped shared memory.
This changes only the diagnostic; the plugin engine defaults are unchanged.

```sh
gpu-nam-stamped-validation 4 128 src/models/wavenet_a1_standard.nam --paced --duration-seconds=2 --sidecar=staged.csv --staged-gpu
gpu-nam-stamped-validation 4 128 src/models/wavenet_a1_standard.nam --paced --duration-seconds=2 --sidecar=shared.csv
```

Both paths use stereo 48 kHz, the same samples, block-aligned silence prewarm,
a continuously advanced CPU model, four-block delayed fallback, absolute callback
pacing, and post-run delayed sample comparison. Neither is a hard-realtime host
callback. Retain raw CSV, executable/source/library hashes and host load.

This staged adapter intentionally differs from the shipping Cloud node: it
prepares continuous delayed fallback and permits the requested lead. It still
uses one shared compute device, channel-specific plans, synchronous per-channel
forwards, upload, GPU readback copies, and blocking MapAsync completion servicing.
The stamped provider uses separate channel sessions and asynchronous retirement.
Results therefore compare architectures, not the isolated cost of copies.
Staged completion is the existing bounded-yield readback loop (up to 4 ms before
1 ms backoff), not the stamped provider completion policy. Non-default stamped
completion options are rejected with `--staged-gpu`.

Controls:

- `--force-fallback`: suppress the transport worker. Audio must still match the
  delayed CPU reference and no due block should count as a GPU delivery.
- `--inject-output-error`: corrupt one output after the run. The sample oracle
  must fail with exit 7.
- `--staged-gpu --inject-forward-failure`: simulate failure of every live staged
  forward. Worker output is poisoned and failed forwards counted; there is no
  hidden CPU replacement. Numerical acceptance must fail with exit 7. Nonfinite
  selected results retain their exact transport disposition (`worker_output` for
  a generic staged node). `failed_gpu_forwards` and the numerical oracle report
  forward failure independently; worker output is never labeled GPU delivery.

Preparation failure exits 8. A failed or unsupported positive run gives no valid
speedup comparison. Numerically correct fallback-only audio also gives no proof
of GPU execution. Whole-process CPU includes full CPU shadow; GPU selection is
not evidence that CPU work was saved.
