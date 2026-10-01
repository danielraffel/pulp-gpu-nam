# Audio Work Interval experiment

The stamped validator accepts `--audio-work-interval` on macOS. It creates an
`AudioWorkIntervalCreate` interval, joins the callback-side master thread, and
passes the interval to Pulp's submission worker. Each paced callback brackets
its work with `os_workgroup_interval_start` and `os_workgroup_interval_finish`.
The worker join is reported in the receipt output.

This is a standalone approximation of the host/plugin contract. A real AU must
obtain the host-owned `os_workgroup_t` through its render-context observer and
hand that borrowed context to the transport. The experiment does not make
Dawn encode/submit realtime-safe, and it is not a hard-realtime guarantee.

On an M1 Max under load, 1,000 blocks at 32 frames and lead 8 produced:

- worker workgroup joined: 1, join failures: 0;
- GPU deliveries: 45;
- CPU fallbacks: 955;
- callback deadline misses: 0;
- numerical mismatches: 0.

The ordinary worker produced 21 GPU deliveries in an earlier matched sample;
the result is therefore not a clear improvement and needs a controlled quiet
host comparison. The switch remains experimental and is not in the normal test
matrix or plugin default.
