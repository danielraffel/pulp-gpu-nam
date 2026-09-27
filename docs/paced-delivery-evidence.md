# Paced benchmark output selection

The paced runner records the selected output from `GpuAudioTransport` delivery
counters immediately before and after its one `process()` call. Exactly one
counter must advance by one. Zero, multiple, reset, or wrapped deltas are an
accounting error and fail the receipt. The stopped final totals must equal the
sum of all recorded dispositions. No other caller may process or reprepare the
transport during this loop.

Rows distinguish GPU delivery, generic worker output, CPU fallback, silence,
passthrough, priming, and invalid rejection. A generic staged worker can produce
output without establishing that the selected payload came from successful GPU
execution; it is reported as `worker_output`. Numerical correctness and failed
GPU forwards remain independent checks. CPU-only runs keep the explicit delay
model's priming and CPU-baseline labels and mark transport accounting inapplicable.
Snapshots run outside the measured process-call interval, but their overhead is
still part of the paced loop and its scheduling environment.

The receipt reports `selection_evidence=transport_delivery_delta_v1` for transport
runs, all selected-output totals, accounting errors, and reconciliation status.
CSV column names remain unchanged; `selected` now uses these exact dispositions.
A same-block deadline miss is independent of the selected output classification.

## Earlier receipts

Earlier versions of `gpu_nam_stamped_paced.cpp` inferred GPU delivery when the
fallback-read counter did not advance and labeled the first lead blocks priming.
That heuristic cannot distinguish silence, rejected views, passthrough, or generic
worker output. Historical `gpu_callbacks` and `selected=gpu_delivered` fields
produced by that runner therefore need remeasurement or independent exact
selection evidence before being used as GPU-delivery proof. This does not erase
recorded timing or numerical comparisons, and does not invalidate separate
actual-plugin probes that already read the exact delivery counters. Preserve
source revisions when interpreting old receipts.

These changes have pure accounting and source validation. They do not establish
new hardware performance, deadline reliability, or installed-plugin acceptance.
