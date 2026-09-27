# Silent physical-device callback diagnostic

This manual test runs the built stamped GPU-NAM CLAP on Pulp's real audio-device
callback. It never opens microphone/input channels. The device output is cleared
before every callback branch and remains silent. Synthesized asymmetric stereo
input goes to the plugin; plugin output goes only to preallocated capture storage.

Build with `GPU_NAM_USE_INSTALLED_PULP=ON`, the stamped WaveNet option,
`GPU_NAM_NATIVE_HOST_PROBE=ON` and `GPU_NAM_SILENT_DEVICE_PROBE=ON`, against the
same exact installed SDK and consumer revision as the ordinary-thread host test.
The executable is excluded from the default build and is not registered in CTest.
Build target `gpu-nam-silent-device-probe` manually. List output IDs first:

```sh
gpu-nam-silent-device-probe --list-devices
gpu-nam-silent-device-probe /exact/plugin/binary DEVICE_ID cpu /path/cpu
gpu-nam-silent-device-probe /exact/plugin/binary DEVICE_ID shared /path/shared
python3 scripts/compare_silent_device_captures.py /path/cpu /path/shared \
  --output /path/comparison.json
```

Each invocation loads exactly one plugin, writes Engine while inactive, verifies
PDC, and opens the explicit output device at requested48kHz/128frames with zero
input channels. The actual device identity, reported rate and buffer size are
recorded. Callback frame sizes and sample positions come from the SDK callback;
128 is a request, never a fabricated observation. Blocks up to4096frames fit the
prepared plugin and capture storage; unexpected sizes/rates or sample-position
gaps fail. Each capture contains96256common input frames, plus any final physical
callback remainder. The offline comparison uses identical input sample indices
without fitting a delay. Differing observed partitions are reported explicitly.

The callback starts and stops CLAP processing on its actual device thread.
Capture and row storage are allocated before start. No oracle, CSV, logging,
waiting or plugin-state query occurs in that callback. Callback allocation
behavior of the plugin/backend is not measured by this probe. The observations
around `process()` use `steady_clock`; the SDK does not expose the CoreAudio
hardware host timestamp here. No scheduled deadline or workgroup membership is
inferred from those observations. The probe does not join an auxiliary thread
to an Audio Workgroup.

## Process lifetime is intentional

The current SDK's void `AudioDevice::stop()` clears its callback even if the
underlying stop fails; it does not expose an error-aware external callback join.
This diagnostic therefore **does not call device stop/close or unload the plugin**.
It retains the entire heap owner, library, device, callback and captures until
`_Exit`. After CLAP `stop_processing()` returns on the callback thread, the
callback publishes completion with release ordering. The control thread acquires
that flag before reading captures and the prepared plugin's delivery snapshot.
Later device callbacks only clear hardware output and return. The flag is a data
handoff, not proof that the callback returned. Retained lifetime makes that
unnecessary for this bounded experiment.

Failure and timeout paths also exit without destroying the retained graph.
Process termination lets the operating system reclaim the audio device. This
answers the real-callback execution question only. Device stop errors, callback
quiescence, plugin deactivation/unload and SDK shutdown remain separate acceptance
gates. Do not distribute this diagnostic plugin build.

A nonzero shared delivery count proves selected plugin GPU output; correct audio
with zero GPU selections fails this gate. The CPU capture must report no GPU
transport selections. The offline comparator rejects corruption, silence and
unmatched device/rate/PDC metadata. It reports observed process durations without
calling them end-to-end latency or deadline reliability.

The comparator pins48kHz and1024samples of PDC rather than accepting two matching
but incorrect reports. It rejects negative or inconsistent delivery counters,
requires all CPU transport counters to be zero, and requires shared GPU plus
fallback plus one priming block to equal the complete512-frame transport blocks
captured. Reported callback count must match the raw rows. Metadata mutation
controls verify these rejections while allowing explicitly reported partition
differences.

The xrun field comes from `AudioDevice::xrun_count`. Public API does not establish
whether the platform overload listener registered successfully, so metadata says
`xrun_listener_availability=unavailable`. A reported zero is not proof that no
hardware overruns occurred.
