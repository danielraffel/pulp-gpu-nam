# Stamped WaveNet Perfetto trace validation

The stamped GPU-NAM validation executable can opt into Pulp's existing
`GpuWaveNetRealtimeNode` trace surface with `--trace`:

```sh
PULP_TRACE_PATH=/tmp/gpu-nam-stamped.pftrace \
  ./build/src/gpu-nam-stamped-validation 4 32 /absolute/model.nam \
  --trace --paced --blocks=1000 \
  --sidecar=/tmp/gpu-nam-stamped.csv
```

This option is deliberately separate from ordinary performance runs. It calls
`configure_trace` before `prepare()` and enables admission records, callback
timing, and stride-one success records. The SDK then emits the canonical Pulp
Perfetto records, including the shared stream identity, admission, terminal GPU
outcome, audio delivery, fallback reason, and lifecycle counters. GPU-NAM does
not define a second trace schema. Because this executable is a standalone
consumer rather than a plugin host, it also starts and stops the process-global
Pulp tracing session around the paced run. Plugin hosts keep their existing
adapter-owned session lifecycle.

The executable must be linked against a Pulp SDK built with tracing enabled
(`PULP_TRACING=ON`, exporting `Pulp::tracing`). The ordinary v0.890.0 SDK
consumer build is intentionally not a tracing build, so setting
`PULP_TRACE_PATH` alone is insufficient and `--trace` fails closed. Build or
install a tracing-enabled SDK/profile first, then verify the capture with
Pulp's `trace-analysis` and `trace-sql` tools.

For a trace smoke, check that the capture contains one session marker, that
each admitted `(generation, sequence)` has exactly one terminal disposition,
and that admission, terminal, and lifecycle drop/invalid counters are zero.
Keep this bounded trace run separate from long-tail timing receipts because
trace recording changes the workload. The direct `GpuNamSharedSessionNode`
adapter is not covered by this mode; it bypasses the stamped node and has no
equivalent public observer yet.
