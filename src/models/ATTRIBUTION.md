# Bundled model attribution

The `.nam` captures in this directory are the open, **MIT-licensed** Neural Amp
Modeler example models from **NeuralAmpModelerCore**
(`sdatkinson/NeuralAmpModelerCore`, `example_models/`), redistributed under
their MIT license with this attribution. They are included only so GPU NAM runs
out of the box and so you can test both engines; load your own `.nam` captures
for real amp tones.

| File | Source (NeuralAmpModelerCore/example_models) | Architecture |
|------|----------------------------------------------|--------------|
| `example.nam` | `wavenet.nam` | WaveNet (A1), 3 channels — the demo's default model |
| `wavenet_a1_standard.nam` | `wavenet_a1_standard.nam` | WaveNet (A1-Standard), 16 channels |
| `lstm.nam` | `lstm.nam` | LSTM |

`cabinet.wav` is a short **synthetic, original** guitar-cabinet impulse
response generated for this example — not a third-party capture.

`README-content.txt` is the note the installer places alongside the samples it
installs to `/Library/Application Support/GPU NAM/`.

The `.nam` file format and its architectures are the open, MIT-licensed Neural
Amp Modeler standard
(https://github.com/sdatkinson/neural-amp-modeler,
https://github.com/sdatkinson/NeuralAmpModelerCore). Pulp's inference
(`src/nam_model.hpp`, `nam_a2.hpp`, `nam_convnet.hpp`, `nam_lstm.hpp`,
`nam_linear.hpp`, `keras_runtime.hpp`) and GPU forward
(`GpuCompute::wavenet_forward`) are independent implementations of those
public architectures; these `.nam` files are the only third-party artifacts.

## Supported model architectures

GPU NAM's CPU engine loads the full `.nam` architecture set:

- **WaveNet A1** — the original architecture (CPU + an opt-in GPU engine).
- **WaveNet A2** — the newer architecture (LeakyReLU activations, a convolutional
  head, mixed kernel sizes per layer array, grouped convolutions, and
  packed-slimmable Full/Lite sizing). Detected at load and routed to a dedicated
  A2 forward — **A2 captures load and render**; the GPU engine covers A1 only, so
  A2 runs on the CPU engine.
- **ConvNet**, **LSTM**, and **Linear** `.nam` captures.
- **RTNeural / Keras** `.json` captures (GRU / LSTM / Dense) via a separate engine.

The models bundled here happen to be A1 and LSTM captures, but any of the above
load. TONE3000 captures in either the A1 or A2 architecture are compatible.
