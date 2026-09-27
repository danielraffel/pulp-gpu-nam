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
`gpu-nam-stamped-`. Each case checks stereo, block-aligned model prewarm,
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
