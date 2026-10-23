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

## Unreleased: lite WaveNet and LSTM speedups

Two changes since 1.3.0 make the CPU engine faster without changing its output.
Every sample is byte-identical to 1.3.0's, in the default and exact tanh modes and at block sizes
37, 64, 128 and 512 on all eight test captures, so the parity results above
still hold unchanged.

- **WaveNet widths above four are padded to a multiple of four.** Every product
  of the lite preset's 6-channel array now runs the four-lane kernels. Widths of
  four or less (nano, feather) keep their exact-width kernels.
- **LSTM layers run over a block of frames at a time.** The input half of the gate
  product is batched across the block, so only the recurrent half runs per
  sample. That half keeps all of a cell's gate rows in registers at once, and the
  default tanh mode evaluates the gates and the cell update in one pass.

Paired, interleaved runs on an M1 Max, at background QoS on a loaded shared host.
Times are median us per block. The ratios are reliable, but the absolute times are
not P-core figures. Single cells move by up to about 5% between runs.

| Capture | Block | 1.3.0 | This change | Speedup | Reference, fast tanh | vs reference |
|---|---|---|---|---|---|---|
| Lite WaveNet | 64 | 83.6 | 66.4 | 1.26x | 127.2 | 1.91x |
| Lite WaveNet | 128 | 168.1 | 131.9 | 1.27x | 233.8 | 1.77x |
| Lite WaveNet | 512 | 740.1 | 532.0 | 1.39x | 1155.5 | 2.12x |
| 3x24 LSTM | 64 | 118.3 | 103.8 | 1.14x | 159.0 | 1.53x |
| 3x24 LSTM | 128 | 240.6 | 210.4 | 1.15x | 320.2 | 1.52x |
| 3x24 LSTM | 512 | 1384.3 | 1179.8 | 1.21x | 2009.5 | 1.77x |
| 1x3 LSTM | 64 | 9.2 | 7.9 | 1.16x | 15.2 | 1.92x |
| 1x3 LSTM | 512 | 73.6 | 63.0 | 1.16x | 121.3 | 1.92x |

Standard, feather and nano captures run the same code as before and measure
within noise of 1.3.0.
