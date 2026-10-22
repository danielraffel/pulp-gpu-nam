# How GPU NAM works

## Signal chain

Input gain -> noise gate -> neural model -> tone stack (Bass / Middle / Treble)
-> cabinet IR (optional) -> DC blocker -> output level -> dry/wet mix. The dry
path is delayed to match the wet path's latency, so a blend stays aligned. See
[nam-support.md](nam-support.md#signal-chain) for the controls.

## The CPU engine

The CPU engine is the default. It re-blocks the host stream into fixed internal
blocks (or runs directly on host blocks when no GPU device, IR or resampling is
in play) and runs the model inline on the audio thread without allocating.

WaveNet A1 captures run on a layer-major block engine (`src/nam_wavenet_block.hpp`).
For each layer, every dilated-convolution tap, the input mix, the 1x1 and the head
is one small matrix product over a run of up to 128 frames
(`src/nam_kernels.hpp`). The products are register-blocked over several frames
at once, so each weight is loaded once per group. Bias, mix and residual are
fused into the product's epilogue. Each layer keeps its input history in a linear
buffer, so a tap reads a contiguous run without wrapping. Activations run over
whole arrays.

LSTM captures run a vectorized recurrent step: one tiled matrix-vector product
over `[x; h]` and the gate activations.

The original per-sample forward is kept as the correctness oracle. The block
engines agree with it to float rounding, and with any block size the output is
bit-identical.

Tanh can be evaluated three ways (`NamRuntime::set_tanh_mode`). The plugin's CPU
engine uses `NamRational`: the rational tanh approximation NAM players use when
fast tanh is on, which is also the fastest. `Exact` uses libm's tanh and exp and
matches the network as trained. `Vector` uses Accelerate's vForce for both, which
is faster than `Exact` and a couple of ulps away from it. The GPU engine always
evaluates exact tanh, so switching the Engine control moves between the two by
the approximation's error.

## The GPU engine

This is the case the skeptics are usually right about — and the one place the
conventional wisdom is *wrong* if you do it correctly.

**Done naively, GPU neural-amp inference is terrible.** If you run the network one
sample at a time, every sample pays a CPU↔GPU round-trip (hundreds of
microseconds against ~0.1 µs of actual work) — thousands of times slower than the
CPU. That's why "run NAM on the GPU" has a bad reputation.

**Done right, it wins — and the win grows with model size.** The WaveNet family is
feedforward, so a whole block's output samples can be computed in parallel; the
entire forward pass is fused into one GPU submission over device-resident weights,
and the CPU↔GPU round-trip is paid **once per block**, not per sample or per
layer. At small models the CPU is still faster (the round-trip dominates); as the
capture grows, the GPU's parallelism pulls ahead while the CPU falls behind
real time. The bundled example is small, so **the CPU engine is the honest default
here** — the GPU engine is there to switch on for large captures and to show the
architecture. The GPU engine matches the CPU engine's `Exact` mode to float
rounding (-112 to -139 dB on the bundled and test captures).

On the Pulp SDK before Generous-Corp/pulp#10635, WGSL `tanh()` on the Metal
backend returned NaN for large pre-activations, so the GPU engine could emit NaN
on some real captures. The current submodule pin includes that fix.

## How it's built on Pulp

GPU NAM is a plain Pulp plugin (`Processor` + a native view). The only Pulp
capability specific to this kind of work is the GPU inference primitive —
`pulp::render::GpuCompute::prepare_wavenet` / `wavenet_forward` — a general fused,
block-parallel, conditioned-WaveNet forward that lives in the Pulp SDK. This repo
owns the `.nam` / Keras format loaders, the CPU oracle, the tone stack and gate,
the editor, and the format packaging; the framework owns rendering, the audio
graph, and the GPU primitive. The integration boundary is deliberately thin: the
plugin depends only on Pulp's public targets (`pulp::render`, `pulp::gpu-audio`,
`pulp::signal`, `pulp::view`, `pulp::canvas`, `pulp::runtime`).

## Repository layout

```
pulp-gpu-nam/
├── pulp/              # Pulp SDK (git submodule, pinned)
├── src/               # the plugin: loaders, CPU oracle, GPU glue, editor, tests
│   ├── models/        # bundled example.nam
│   ├── assets/nam/    # editor image + font assets (see ATTRIBUTION.md)
│   └── docs/          # editor screenshot, CPU-vs-GPU comparison notes
├── CMakeLists.txt     # top-level: build Pulp lean, then the plugin
└── README.md
```
