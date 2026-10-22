# CPU baseline validation

This diagnostic measures the repository CPU NAM implementation using the same
96-block stimulus and block-aligned prewarm as the shared-GPU diagnostic. It
adds zero input for the selected lead length to drain delayed output. The oracle
is prepared separately before timing; a linear reference array validates the
ring's output, leading silence, and final delayed blocks.

Build `gpu-nam-cpu-baseline` with the existing experimental-session option.
Run `python3 scripts/test_cpu_baseline.py /absolute/path/to/gpu-nam-cpu-baseline`.
The suite requires all 12 block/lead combinations and rejects a deliberately
wrong delay, invalid arguments, and a missing model.

Timing fields have distinct meanings:

- `elapsed_process_ns`: elapsed wall time spent inside model processing.
- `process_cpu_ticks` / `process_cpu_seconds`: process CPU usage for the measured
  loop, including delay-ring work and timer overhead. Tick resolution is emitted.
- `wall_ns`: elapsed time for that whole loop.

Loading, prewarming, stimulus/oracle generation, and output comparison are
outside the measured loop. Runs are unpaced and short. They do not establish
realtime reliability or steady-state thermal behavior. Measure GPU modes with
the same input and drain count before comparing. GPU process CPU measurements
must include the continuously running fallback and any worker-side CPU model;
inner GPU dispatch time cannot substitute for that total.

Current build uses the previous installed development SDK. Authenticated new
SDK and matched staged/shared GPU comparisons remain open. No speedup claim.
