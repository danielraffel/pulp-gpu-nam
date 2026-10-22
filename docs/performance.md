# CPU engine performance and output parity

## Summary

On an M5 Max, the CPU engine in this release, compared with GPU NAM 1.2.4:

| Capture | Time per 64-sample block, 1.2.4 | This release | Speedup |
|---|---|---|---|
| Standard WaveNet (real capture) | 194.7 us | 21.9 us | 8.9x |
| Lite WaveNet | 144.4 us | 20.2 us | 7.2x |
| Feather WaveNet | 89.8 us | 6.1 us | 14.7x |
| Nano WaveNet | 52.6 us | 3.0 us | 17.3x |
| 3x24 LSTM (real capture) | 139.8 us | 28.4 us | 4.9x |

A standard capture now uses about 1.6% of one core at 48 kHz.

The output matches the reference NAM engine
([NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)) with
fast tanh enabled to float rounding. Fast tanh is the setting NAM players ship
with. At the same output, this engine is 1.7-1.9x faster on the standard capture
and 1.1-1.2x faster on the LSTM.

## Method

- **Host.** M5 Max on AC power, macOS 27.0.1, Apple clang 21.0.0, `-O3 -DNDEBUG`,
  load average about 5 on 18 cores. One instance at 48 kHz, mono, on one thread at
  user-interactive QoS.
- **Input.** Band-limited noise with pluck envelopes. The first 0.5 s is not
  timed. Each cell is the median of 3 runs, and the engines are interleaved
  within each cell.
- **Comparison point.** The reference NAM engine is NeuralAmpModelerCore at
  commit 9c7b185 with its Eigen pin, built with the same compiler and flags. Its
  fast tanh is turned on through `nam::activations::Activation::enable_fast_tanh()`.
- **Captures.** The standard WaveNet and 3x24 LSTM are real MIT-licensed captures
  published with NAM. Lite, feather and nano use the trainer's preset shapes with
  seeded random weights, so they measure compute cost, not a real amp.
- **Block sizes.** Blocks 32 to 512 give the same speedups to within about 10%.

Full results for this release, at blocks 32-512 (median us per block):

| Capture | Block | 1.2.4 | This release (default) | Exact mode | Reference, fast tanh | Reference, exact tanh |
|---|---|---|---|---|---|---|
| Standard WaveNet | 32 | 102.9 | 12.0 | 38.2 | 22.5 | 49.0 |
| Standard WaveNet | 128 | 398.2 | 44.7 | 148.1 | 71.8 | 179.6 |
| Standard WaveNet | 512 | 1640.4 | 190.4 | 612.2 | 278.4 | 711.7 |
| 3x24 LSTM | 32 | 67.5 | 13.9 | 30.2 | 16.3 | 32.9 |
| 3x24 LSTM | 128 | 276.9 | 57.2 | 123.0 | 66.4 | 125.6 |
| 3x24 LSTM | 512 | 1187.5 | 236.9 | 514.7 | 278.0 | 514.1 |

## Output parity

Both engines were fed identical stimuli: sine sweeps at -6 and 0 dBFS, noise at
-18 dBFS, plucked-string transients, and impulses over silence. Each ran at 44.1,
48 and 96 kHz, in fixed and random block sizes, cold and prewarmed. The outputs
were then compared sample by sample with alignment.

- **Default mode vs reference with fast tanh.** The steady-state residual is
  -100 to -144 dB on WaveNet captures and -85 to -94 dB on the LSTM capture. At
  48 kHz the LSTM figure equals the reference's own float-rounding spread:
  building the reference with Eigen's vectorization off moves its output by the
  same amount.
- **Exact mode vs reference with exact tanh.** On every WaveNet capture the
  residual is at or below that rounding spread.
- **Audio Quality Lab.** Pulp's lab reports no material tonal, high-frequency,
  roughness, graininess or transient change for either pair.
- **Planted errors.** A 1-sample shift, a 0.1 dB gain error and a wrong capture
  are each detected, by at least 55 dB.

## Tanh modes

- **Default, `NamRational`.** The rational tanh approximation NAM players use
  with fast tanh on. It is the fastest mode.
- **`Exact`.** libm `tanh` and `exp`: the network as trained. It is slower than
  the default. It is faster than the reference engine with exact tanh on WaveNet
  captures, and as fast on the LSTM.
- **`Vector`.** Accelerate's vForce, a couple of ulps from `Exact` and faster
  than it. It is opt-in, because an LSTM's recurrence amplifies those ulps.

The GPU engine evaluates exact tanh. It matches the CPU engine's `Exact` mode
to -112 dB or better on the bundled and test captures.
