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
32/64/128-frame blocks, fixed 1/2/4/8-block delay, exactly one callback CPU-model invocation per channel
per block, and parity after worker servicing stops. It requires at least one
post-priming callback without CPU fallback and correct nonzero output. Worker
production counts are reported separately from callback selection. Failed
preparation is a failure with an unclassified cause, not a hardware skip.

The executable accepts `[lead [frames [model_path]]]` so the same test can use
the standard WaveNet model as well as the bundled tiny model. Invalid arguments
fail before creating a provider.

The synthetic worker receives 10 ms between blocks. This checks correctness and
selection, not realtime reliability. Installed-SDK execution, paced performance,
model replacement, native host lifecycle, and packaging remain open.

Initial source compilation used the new node header with the existing development
SDK include dependencies. Both consumer translation units passed syntax checking.
This is not link, runtime, or authenticated installed-SDK proof.
