# Realtime worker experiment

GPU-NAM's stamped validator accepts `--realtime-worker` for an opt-in
scheduling experiment. It passes the public Pulp transport configuration
`join_audio_workgroup=true` with no host workgroup handle. On macOS that asks
the worker to use Pulp's fallback realtime-priority policy. It is not a host
Audio Workgroup, and it does not make Dawn encode or submit calls realtime-safe.

The flag is deliberately not part of the normal CTest matrix or plugin default.
It exists to compare admission behavior against the ordinary worker while the
host integration supplies a real `os_workgroup_t`.

Observed on an M1 Max at 32 frames, lead 8, under load:

- ordinary worker, 1,000 blocks: 21 GPU deliveries, 979 CPU fallbacks;
- realtime-worker, 1,000 blocks: 780 GPU deliveries, 220 CPU fallbacks, no
  callback deadline misses;
- realtime-worker, 100,000 blocks: 20 GPU deliveries, 99,980 CPU fallbacks,
  and 8 callback deadline misses.

All runs were numerically correct. The short improvement did not survive the
long run, so this remains an experiment rather than a default scheduling policy.
