# SDK-owned WaveNet transport validation

`GPU_NAM_EXPERIMENTAL_STAMPED_WAVENET=ON` builds a separate consumer of
`GpuWaveNetRealtimeNode`. It requires the new Pulp SDK header and implementation.
The existing session diagnostic remains available for comparison; the shipping
plugin factory is unchanged until this consumer passes installed-SDK tests.

The SDK owns GPU sessions and stamped admission/completion. GPU-NAM translates
the model once and maintains one CPU model per audio channel for fallback. There
is no worker-side CPU model. The callback still computes the entire CPU model,
so removing the worker duplicate is not a claim of CPU savings over CPU-only NAM.

Build target `gpu-nam-stamped-validation`, then run CTest matching
`gpu-nam-stamped-`. Each case checks the configured channel count (the functional matrix remains stereo), block-aligned model prewarm,
32/64/128/512-frame blocks, fixed 1/2/4/8-block delay, exactly one callback CPU-model invocation per channel
per block, and parity after worker servicing stops. It requires at least one
post-priming callback without CPU fallback and correct nonzero output. Worker
production counts are reported separately from callback selection. Failed
preparation is a failure with an unclassified cause, not a hardware skip.

The executable accepts `[lead [frames [model_path]]]` so the same test can use
the standard WaveNet model as well as the bundled tiny model. Invalid arguments
fail before creating a provider.
Append `--inject-output-error` to alter an actual delivered sample. The oracle
must report exit 7 and a residual near 0.25. This is a failure-detector control,
not a passing GPU case. The 512-frame case matches the plugin's internal quantum.

Paced diagnostics also accept independent slot and worker-flight controls: `--capacity=2|4|8|16` sets the persistent provider/transport slot capacity, while `--max-inflight=1|2|4|8` sets the maximum number of submissions retained by the serialized worker. They are recorded separately in stdout receipts. The matrix runner's `receipt.json` also records the requested/effective capacity, p50/p99/p99.9/p99.99/max start lateness, actual deadline-miss rate, and terminal dispositions derived from every validated sidecar row. The multi-flight option requires an SDK exporting `GpuWaveNetRealtimeNode::Config::max_inflight`; older installed SDKs fail closed for values greater than one rather than silently running a single-flight experiment.

The SDK-owned adapter is bounded to 64 independent channels, matching `GpuWaveNetRealtimeNode`; the current GPU-NAM plugin descriptor remains stereo. `gpu-nam-stamped-multichannel-4096` is a separate four-channel, 4,096-block paced lifecycle validation. It checks CPU-oracle parity, zero misses on the quiet validation host, sustained GPU delivery, continuously advanced fallback accounting, and successful transport/session retirement.

The synthetic worker receives 10 ms between blocks. This checks correctness and
selection, not realtime reliability. Installed-SDK execution, paced performance,
model replacement, native host lifecycle, and packaging remain open.

Initial source compilation used the new node header with the existing development
SDK include dependencies. Both consumer translation units passed syntax checking.
This is not link, runtime, or authenticated installed-SDK proof.

## Backend changes in the experimental plugin

The stamped plugin applies Engine selection when the host prepares a stopped
plugin. CPU and GPU model histories stop advancing when their respective engine
is inactive, so resuming an old engine during playback would use stale state.
A live Engine edit or state restore therefore changes the requested backend, not
the active backend. The settings panel shows the active path and the pending
request, with "next activation" text. Reactivate the plugin through the host to
apply it. Engine is not advertised as automatable in this experimental build.

A stopped reprepare starts a freshly warmed history with the requested backend.
The one-block GPU lead stays fixed during an activation. Initial CPU/Auto
preparation still probes and warms the GPU stack, preserving the existing device
availability and fixed-PDC contract; avoiding that setup would require a separate
latency-policy change. Auto stays on CPU because this experiment retains the full
CPU shadow. Explicit GPU selection remains available.

Model reload remains an explicit new-model/history transition. It uses the
backend selected for the current preparation and does not apply a pending Engine
request as a side effect. It is not a sample-continuous handoff between inactive
models. Default and legacy plugin builds retain their previous Engine behavior.

The `stamped-selection` test compares independent stereo samples against an
uninterrupted control before and after live requests in both directions,
including listener-silent state restoration. It also compares stopped reprepare
against a fresh instance, verifies non-automatable metadata and pending UI text,
and keeps PDC constant. These source-linked checks do not prove installed-SDK or
DAW behavior by themselves.

## Completion service experiments

The stamped factory accepts optional `GpuNamCompletionOptions`. Defaults remain
ProcessEvents and a zero worker wait budget. The standalone consumer accepts:

```sh
gpu-nam-stamped-validation 2 64 /absolute/model.nam \
  --completion-policy=timed-wait-any --worker-wait-ns=100000
scripts/run_gpu_nam_matrix.sh /absolute/gpu-nam-stamped-validation /new/results \
  --model /absolute/model.nam --stamped \
  --completion-policy=timed-wait-any --worker-wait-ns=100000
```

Policies are `process-events`, `wait-any`, and `timed-wait-any`. Zero preserves
nonblocking worker servicing for every policy. A positive relative budget is
accepted only for TimedWaitAny and must not exceed 1,000,000 ns. ProcessEvents
and ordinary WaitAny do not honor a positive timeout, so such combinations fail
explicitly. The SDK recomputes the deadline for each worker pump and shares that
budget across channels; it is neither an absolute audio deadline nor a callback
wait. The timed provider wait is capped to the same requested budget. Unsupported
provider preparation remains a failed experiment, never a fallback pass.

Use separate output directories for matched policies with identical model,
frames and lead. Receipts record the requested policy/budget, model and binary
hashes, and all twelve 32/64/128 x 1/2/4/8 results. The consumer still uses its
10 ms functional pacing and end-of-run fallback control. These switches do not
turn that test into a performance measurement. Installed-SDK runtime comparison
and actual scheduling effects remain pending.

## Paced CPU and stamped comparison

`--paced` replaces the functional 10 ms pumping loop with an absolute 48 kHz
callback schedule and a separate non-realtime transport worker. It defaults to
10 seconds of input, then drains the declared lead with zero input. Run each
engine in a separate process on the same quiet machine and installed SDK:

```sh
gpu-nam-stamped-validation 4 64 /absolute/model.nam --paced \
  --cpu-baseline --duration-seconds=10 --sidecar=/new/cpu.csv
gpu-nam-stamped-validation 4 64 /absolute/model.nam --paced \
  --duration-seconds=10 --sidecar=/new/gpu.csv \
  --completion-policy=timed-wait-any --worker-wait-ns=100000
```

Both runs use the same stereo stimulus, block-aligned model warmup, callback
count and fixed delayed output contract. The CPU baseline evaluates one model
per channel and delays its output. The stamped callback evaluates the same full
CPU shadow and may select GPU output instead. It cannot claim CPU DSP savings
merely because GPU results were selected. Repeat with reversed run order before
attributing small differences to a backend; record host load, thermal/power state
and graphics activity beside the receipts. Do not run on an occupied CI host.

The worker uses `wake_on_write=true`; the report records its computed polling
interval. The transport still waits after every pump and the node services
completion before admitting its next input. This is not fully event-driven
completion servicing. TimedWaitAny alone does not eliminate that inter-pump gap.

The CSV keeps every callback's scheduled/start/end/deadline timestamps and
selected output (`priming`, `cpu_baseline`, `gpu_delivered`, or `cpu_fallback`).
`block` is the callback index. `delivered_input_sequence` is blank during
priming and then equals `block - lead`; a selected result belongs to that older
input, not to the input newly admitted by this callback. The public API supplies
no source epoch, so this is a capture-local sequence mapping, not an invented
SDK stream epoch. The matrix validator checks this relationship for every row.
A callback deadline is the next block boundary. Start lateness and callback
execution cost remain separate. GPU selection and fallback counters describe
callback output, not inner completion counts. The transport's produced, missed,
dropped and resynchronized counts are reported separately.

Stimulus generation, allocation, model/provider preparation, independent delayed
sample validation and CSV serialization are outside the measured loop. Process
CPU time covers the entire loop, including the GPU worker, wakeups and capture
bookkeeping. Drain CPU time is reported separately. The raw captures and rows
are preallocated; no file I/O or reference-model evaluation occurs in a measured
callback. The independent oracle runs after worker retirement. An injected
`--inject-output-error` must fail even when all ordinary callbacks would have
fallen back correctly.

Only callback timestamps are available from this installed public consumer API.
The existing private SDK phase trace observers are not re-declared here. The
report explicitly marks GPU and internal admission/submit/retirement timestamps
unavailable; callback time is not GPU execution time. A correct fallback-only
run is valid audio but `gpu_delivery_observed=0` is not evidence of useful GPU
participation. Deadline misses and zero GPU delivery remain data, not errors to
hide by changing the schedule.

The existing matrix runner accepts `--stamped --paced`, plus `--cpu-baseline`
for the matched baseline. It hashes the CSV and checks complete sequential rows
as well as the log. Use `--blocks=N` instead of duration for fixed-count runs,
up to one million input blocks. Input and output capture storage is capped at
1 GiB before allocation; the million-block 32-frame case is supported but larger
captures may be rejected. Long runs remain a separate experiment. Initial work
has only syntax/parser/matrix validation; installed runtime and performance
conclusions are still pending.
